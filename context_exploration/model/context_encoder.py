"""
Context encoder model

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import math

import numpy as np
import torch
import torch_scatter
from torch import nn
from torch.distributions import Distribution, Normal, register_kl
from torch.nn import functional as F

from context_exploration.utils import MLP


def get_context_encoder(class_name, state_dim, action_dim, context_dim, kwargs):
    return globals()[class_name](
        state_dim=state_dim, action_dim=action_dim, context_dim=context_dim, **kwargs
    )


class ContextSet:
    def __init__(self):
        self.x = None
        self.u = None
        self.x_next = None

    @staticmethod
    def create_empty():
        return ContextSet()

    @property
    def is_empty(self):
        return self.x is None or self.x.shape[0] == 0

    @staticmethod
    def from_array(x, u, x_next):
        """
        Parameters
        ----------
        x : np.ndarray or torch.Tensor, shape [B x <state_dim>]
        u : np.ndarray or torch.Tensor, shape [B x <action_dim>]
        x_next : np.ndarray or torch.Tensor, shape [B x <state_dim>]
        """
        assert x.shape[0] == u.shape[0] == x_next.shape[0]
        dim_fcn = lambda x: x.ndim if isinstance(x, np.ndarray) else x.dim()
        assert dim_fcn(x) == dim_fcn(u) == dim_fcn(x_next)
        context_set = ContextSet()
        context_set.x = x
        context_set.u = u
        context_set.x_next = x_next
        return context_set

    @staticmethod
    def from_trajectory(x, u, *, context_size=None, transition_idxs=None):
        """
        Generate ContextSet from trajectory.
        context_size and transition_idxs cannot be given at the same time.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor, shape [T x <state_dim>]
        u : np.ndarray or torch.Tensor, shape [T x <action_dim>]
        context_size : int, optional
        transition_idxs : array_like, optional

        Returns
        -------
        context_set: ContextSet
        """
        if transition_idxs is None:
            n_transitions = len(x) - 1
            if context_size is None:
                context_size = n_transitions
            transition_idxs = np.arange(n_transitions)
            np.random.shuffle(transition_idxs)
            transition_idxs = transition_idxs[:context_size]
        else:
            assert context_size is None
            transition_idxs = np.array(transition_idxs)
        return ContextSet.from_array(
            x[transition_idxs], u[transition_idxs], x[transition_idxs + 1]
        )

    def to(self, device):
        if self.is_empty:
            return self
        return ContextSet.from_array(
            self.x.to(device), self.u.to(device), self.x_next.to(device)
        )

    def as_torch(self):
        if self.is_empty:
            return self
        return ContextSet.from_array(
            torch.as_tensor(self.x).float(),
            torch.as_tensor(self.u).float(),
            torch.as_tensor(self.x_next).float(),
        )

    def __add__(self, other):
        """
        Append another ContextSet to this ContextSet

        Parameters
        ----------
        context_set: ContextSet

        Returns
        -------
        new_context_set: ContextSet
        """
        np_torch_cat = lambda x, y: (
            np.concatenate((x, y)) if isinstance(x, np.ndarray) else torch.cat((x, y))
        )
        return ContextSet.from_array(
            other.x if self.x is None else np_torch_cat(self.x, other.x),
            other.u if self.u is None else np_torch_cat(self.u, other.u),
            other.x_next
            if self.x_next is None
            else np_torch_cat(self.x_next, other.x_next),
        )


class LinearNonNegativeWeights(nn.Module):

    __constants__ = ["bias", "in_features", "out_features"]

    def __init__(self, in_features, out_features, bias=True):
        super(LinearNonNegativeWeights, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    @property
    def weight(self):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class LinearNonNegativeWeightsReLU(LinearNonNegativeWeights):
    def __init__(self, *args, **kwargs):
        super(LinearNonNegativeWeightsReLU, self).__init__(*args, **kwargs)

    @property
    def weight(self):
        return F.relu(self._weight)

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self._weight, a=math.sqrt(5))
            torch.abs_(self._weight)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


class ContextEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, context_dim):
        super(ContextEncoder, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.device = None

    def to(self, device):
        self.device = device
        return super().to(device)

    def encode_to_latent(self, x, u, x_next, assignments, batchsize):
        """
        Encode context to latent representation

        Parameters
        ----------
        x : torch.Tensor, shape [N x x_dim]
        u : torch.Tensor, shape [N x u_dim]
        x_next : torch.Tensor, shape [N x x_dim]
        assignments : torch.Tensor, shape [N,]
            Index assignments of context transitions to batch.

        Returns
        -------
        latent_mean: torch.Tensor, shape [bs x latent_dim]
        latent_diag_cov: torch.Tensor, shape [bs x latent_dim]
        """
        raise NotImplementedError

    def forward(self, x, u, x_next, assignments, batchsize):
        """
        Encode context set to belief over context variable

        Parameters
        ----------
        x : torch.Tensor, shape [N x x_dim]
        u : torch.Tensor, shape [N x u_dim]
        x_next : torch.Tensor, shape [N x x_dim]
        assignments : torch.Tensor, shape [N,]
            Index assignments of context transitions to batch.
        batchsize : int

        Returns
        -------
        context_belief: Normal, batch_shape [batchsize], event_shape [context_dim]
        """
        raise NotImplementedError

    def forward_set(self, context_set):
        if context_set.is_empty:
            return self.empty_set_context()
        torch_set = context_set.as_torch().to(self.device)
        x, u, x_next = torch_set.x, torch_set.u, torch_set.x_next
        assignments = torch.zeros(x.shape[0]).long().to(x.device)
        batchsize = 1
        return self.forward(x, u, x_next, assignments, batchsize)

    def empty_set_context(self, batch_dim=None):
        x = torch.zeros(0, self.state_dim).to(self.device)
        u = torch.zeros(0, self.action_dim).to(self.device)
        x_next = torch.zeros(0, self.state_dim).to(self.device)
        assignments = torch.zeros(0).long().to(self.device)
        batchsize = 1
        context = self.forward(x, u, x_next, assignments, batchsize)
        if batch_dim is None:
            return context[0]
        else:
            return context.expand((batch_dim, context.mean.shape[-1]))


class MLPContextEncoder(ContextEncoder):
    def __init__(
        self,
        state_dim,
        action_dim,
        context_dim,
        positive_weights,
        latent_dim=200,
        aggregation_type="sum",
        rescale_raw_stddev=False,
        clamp_softplus_weights=False,
        mean_feature_bidding="none",
    ):
        super(MLPContextEncoder, self).__init__(state_dim, action_dim, context_dim)

        self.aggregation_type = aggregation_type
        self.rescale_raw_stddev = rescale_raw_stddev
        self.mean_feature_bidding = mean_feature_bidding

        mlp_hidden_dim = 200

        self.latent_dim = latent_dim
        self.transition_encoder = MLP(
            input_dim=2 * state_dim + action_dim,
            output_dim=self.latent_dim,
            hidden_dims=[mlp_hidden_dim],
            hidden_nonlinearities="ReLU",
            output_nonlinearity="ReLU",
        )

        if aggregation_type != "max":
            raise NotImplementedError

        if not mean_feature_bidding in ["none", "softmax"]:
            raise ValueError
        if mean_feature_bidding != "none":
            self.mean_feature_encoder = MLP(
                input_dim=2 * state_dim + action_dim,
                output_dim=self.latent_dim,
                hidden_dims=[mlp_hidden_dim],
                hidden_nonlinearities="ReLU",
                output_nonlinearity=None,
            )

        self.mean_encoder = MLP(
            input_dim=self.latent_dim,
            output_dim=context_dim,
            hidden_dims=[mlp_hidden_dim],
            hidden_nonlinearities="ReLU",
            output_nonlinearity=None,
        )

        if positive_weights == "relu":
            layer_class = LinearNonNegativeWeightsReLU
        elif positive_weights == False:
            layer_class = nn.Linear
        else:
            raise ValueError

        self.log_stddev_encoder = MLP(
            input_dim=self.latent_dim,
            output_dim=context_dim,
            hidden_dims=[mlp_hidden_dim],
            hidden_nonlinearities="ReLU",
            output_nonlinearity=None,
            layer_class=layer_class,
        )
        # The actual variance is computed as softplus(-log_variance_encoder),
        # which ensures decreasing variance for an increasing number of
        # context points.

    def _encode_to_latent(self, x, u, x_next, assignments, batchsize):
        bids = self.transition_encoder(torch.cat((x, u, x_next), dim=-1))
        max_bids = torch.zeros(batchsize, self.latent_dim, device=x.device)
        torch_scatter.scatter_max(src=bids, index=assignments, dim=0, out=max_bids)
        if self.mean_feature_bidding == "none":
            mean_latent = max_bids
        elif self.mean_feature_bidding == "softmax":
            mean_features = self.mean_feature_encoder(torch.cat((x, u, x_next), dim=-1))
            softmax_bids = torch_scatter.composite.softmax.scatter_softmax(
                src=bids, index=assignments, dim=0
            )
            weighted_means = mean_features * softmax_bids
            aggregated_means = torch.zeros(batchsize, self.latent_dim, device=x.device)
            torch_scatter.scatter_add(
                src=weighted_means, index=assignments, dim=0, out=aggregated_means
            )
            mean_latent = aggregated_means
        else:
            raise ValueError
        std_latent = max_bids
        return mean_latent, std_latent

    def _forward_latent(self, mean_latent, std_latent):
        context_mean = self.mean_encoder(mean_latent)
        # we take -log_variance_encoder here, as log_variance_encoder
        # is monotonically increasing, but we want a monotonically
        # decreasing argument here (decreasing variance for increasing
        # number of context points)
        encoding = self.log_stddev_encoder(std_latent)
        if self.rescale_raw_stddev:
            context_std = F.softplus(-encoding * 0.3 + 0.5) + 0.01
        else:
            context_std = F.softplus(-encoding) + 0.01
        return Normal(context_mean, context_std)

    def forward(self, x, u, x_next, assignments, batchsize):
        mean_latent, std_latent = self._encode_to_latent(
            x, u, x_next, assignments, batchsize
        )
        return self._forward_latent(mean_latent, std_latent)

    def forward_tensor(self, x, u, x_next):
        """
        Encode ContextSets given as tensors, each
        ContextSet containing T elements.

        Parameters
        ----------
        x: torch.Tensor, shape [T x <bs> x state_dim]
        u: torch.Tensor, shape [T x <bs> x action_dim]
        x_next: torch.Tensor, shape [T x <bs> x state_dim]

        Returns
        -------
        context_distribution: Normal, shape [<bs> x context_dim]
        """
        if self.mean_feature_bidding != "none":
            raise NotImplementedError
        features = self.transition_encoder(torch.cat((x, u, x_next), dim=-1))
        mean_latent = torch.max(features, dim=0)[0]
        std_latent = mean_latent
        return self._forward_latent(mean_latent, std_latent)

    def forward_broadcast(self, x1, u1, x_next1, x2, u2, x_next2):
        """
        Encode pairs of ContextSets given as tensors,
        where the batch dimensions of the pairs are broadcastable.

        Parameters
        ----------
        x1: torch.Tensor, shape [T1 x <bs1> x state_dim]
        u1: torch.Tensor, shape [T1 x <bs1> x action_dim]
        x_next1: torch.Tensor, shape [T1 x <bs1> x state_dim]
        x2: torch.Tensor, shape [T2 x <bs2> x state_dim]
        u2: torch.Tensor, shape [T2 x <bs2> x action_dim]
        x_next2: torch.Tensor, shape [T2 x <bs2> x state_dim]

        Returns
        -------
        context_distribution: Normal, shape [<bs> x context_dim]
        """
        features_1 = self.transition_encoder(torch.cat((x1, u1, x_next1), dim=-1))
        max_1 = torch.max(features_1, dim=0)[0]
        features_2 = self.transition_encoder(torch.cat((x2, u2, x_next2), dim=-1))
        max_2 = torch.max(features_2, dim=0)[0]
        # compute maximum between '1' and '2'
        mean_latent = torch.max(max_1, max_2)
        std_latent = mean_latent
        return self._forward_latent(mean_latent, std_latent)


class ZeroDiracDistribution(Distribution):
    def __init__(self, shape, device):
        super().__init__()
        self._val = torch.zeros(*shape).to(device)

    @property
    def shape(self):
        return self._val.shape

    @property
    def mean(self):
        return self._val.clone()

    def expand(self, *target_shape):
        return ZeroDiracDistribution(
            self._val.expand(*target_shape).shape, device=self._val.device
        )

    def rsample(self):
        return self._val.clone()

    def entropy(self):
        # As for the Normal distribution, we assume the ZeroDiracDistribution
        # to be independent over all dimensions
        return self._val.clone()


@register_kl(ZeroDiracDistribution, ZeroDiracDistribution)
def kl_zerodirac_zerodirac(p, q):
    assert p._val.shape == q._val.shape
    # As for the Normal distribution, we assume the ZeroDiracDistribution
    # to be independent over all dimensions
    return torch.zeros_like(q._val)


class ConstantZeroContextEncoder(ContextEncoder):
    def __init__(
        self,
        state_dim,
        action_dim,
        context_dim,
    ):
        super(ConstantZeroContextEncoder, self).__init__(
            state_dim, action_dim, context_dim
        )

    def forward(self, x, u, x_next, assignments, batchsize):
        """
        Encode 'batchsize' context sets.

        Parameters
        ----------
        x: torch.Tensor, shape [B x state_dim]
            B is the cumulative size of all context sets to encode.
        u: torch.Tensor, shape [B x action_dim]
        x_next: torch.Tensor, shape [B x state_dim]
        assignments: torch.LongTensor, shape [B x state_dim]
            Assignments range from 0 (incl.) to batchsize-1 (incl.)
        batchsize: int
            Number of context sets
        """
        return ZeroDiracDistribution(
            shape=(batchsize, self.context_dim), device=x.device
        )

    def forward_tensor(self, x, u, x_next):
        """
        Encode ContextSets given as tensors, each
        ContextSet containing T elements.

        Parameters
        ----------
        x: torch.Tensor, shape [T x <bs> x state_dim]
        u: torch.Tensor, shape [T x <bs> x action_dim]
        x_next: torch.Tensor, shape [T x <bs> x state_dim]

        Returns
        -------
        context_distribution: ZeroDiracDistribution, shape [<bs> x context_dim]
        """
        return ZeroDiracDistribution(
            shape=x.shape[1:-1] + (self.context_dim,),
            device=x.device,
        )

    def forward_broadcast(self, x1, u1, x_next1, x2, u2, x_next2):
        """
        Encode pairs of ContextSets given as tensors,
        where the batch dimensions of the pairs are broadcastable.

        Parameters
        ----------
        x1: torch.Tensor, shape [T1 x <bs1> x state_dim]
        u1: torch.Tensor, shape [T1 x <bs1> x action_dim]
        x_next1: torch.Tensor, shape [T1 x <bs1> x state_dim]
        x2: torch.Tensor, shape [T2 x <bs2> x state_dim]
        u2: torch.Tensor, shape [T2 x <bs2> x action_dim]
        x_next2: torch.Tensor, shape [T2 x <bs2> x state_dim]

        Returns
        -------
        context_distribution: ZeroDiracDistribution, shape [<bs> x context_dim]
        """
        batchsize_1 = x1.shape[1:-1]
        batchsize_2 = x2.shape[1:-1]
        batchsize = torch.broadcast_shapes(batchsize_1, batchsize_2)
        return ZeroDiracDistribution(
            shape=batchsize
            + [
                self.context_dim,
            ],
            device=x1.device,
        )
