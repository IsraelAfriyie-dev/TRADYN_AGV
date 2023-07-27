"""
Multilayer perceptron utility

Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
from typing import List, Optional

import torch
from PIL import Image
from torch import nn


def gradient_relu(x):
    grads = torch.ones_like(x)
    grads[x < 0] = 0
    return grads


def gradient_sigmoid(x):
    return torch.sigmoid(x) * (1 - torch.sigmoid(x))


def gradient_tanh(x):
    return 1.0 - torch.tanh(x) ** 2


def activation_gradient(activation_fcn, input):
    if activation_fcn.__class__ == nn.ReLU:
        return gradient_relu(input)
    elif activation_fcn.__class__ == nn.Sigmoid:
        return gradient_sigmoid(input)
    elif activation_fcn.__class__ == nn.Tanh:
        return gradient_tanh(input)
    elif activation_fcn.__class__ == nn.Identity:
        return torch.ones_like(input)
    else:
        raise ValueError


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        hidden_nonlinearities: Optional[str] = "ReLU",
        output_nonlinearity: Optional[str] = None,
        bn_hidden: bool = False,
        bn_output: bool = False,
        layer_class: object = nn.Linear,
    ):
        """
        Multi-Layer perceptron

        Parameters
        ----------
        input_dim: Input dimensionality (int)
        output_dim: Output dimensionality (int)
        hidden_dims: List of dimensions of hidden layers (list[int]),
                     can be empty
        hidden_nonlinearities: List of non-linearities of hidden layers
                               (may contain 'None')
        output_nonlinearity: Output non-linearity
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bn_hidden = bn_hidden
        self.bn_output = bn_output

        if not isinstance(hidden_nonlinearities, (list, tuple)):
            hidden_nonlinearities = (hidden_nonlinearities,) * len(hidden_dims)
        else:
            if len(hidden_nonlinearities) != len(hidden_dims):
                raise ValueError(
                    "Length of hidden_nonlinearity list must equal len(hidden_dims)"
                )

        self.nonlinearities = nn.ModuleList()
        for nl in hidden_nonlinearities:
            nl_obj = getattr(nn, nl)() if nl is not None else nn.Identity()
            self.nonlinearities.append(nl_obj)

        if output_nonlinearity is None:
            self.nonlinearities.append(nn.Identity())
        else:
            self.nonlinearities.append(getattr(nn, output_nonlinearity)())

        self.layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        tmp_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(layer_class(tmp_dim, dim))
            self.batch_norm_layers.append(
                nn.BatchNorm1d(dim) if bn_hidden else nn.Identity()
            )
            tmp_dim = dim
        self.layers.append(layer_class(tmp_dim, output_dim))
        self.batch_norm_layers.append(
            nn.BatchNorm1d(output_dim) if bn_output else nn.Identity()
        )
        self.pre_activations = None

    def forward(self, x):
        pre_activations = []
        for layer, nonlinearity, batchnorm in zip(
            self.layers, self.nonlinearities, self.batch_norm_layers
        ):
            x_layer = layer(x)
            x_nonlinearity = nonlinearity(x_layer)
            x_batchnorm = batchnorm(x_nonlinearity)
            x = x_batchnorm
            pre_activations.append(x_layer)
        self.pre_activations = pre_activations
        return x

    def jacobian(self):
        """ Compute the jacobian wrt all inputs for the most recent forward pass """
        if self.bn_hidden or self.bn_output:
            raise NotImplementedError("BatchNorm layers not supported")
        jac = None
        for layer, nonlinearity, preact in list(
            zip(self.layers, self.nonlinearities, self.pre_activations)
        ):
            if nonlinearity is None:
                jac_factor = layer.weight.expand(preact.shape[0], *layer.weight.shape)
            else:
                act_grad = activation_gradient(nonlinearity, preact)
                act_grad = act_grad[..., None].expand(
                    *act_grad.shape, layer.weight.shape[1]
                )
                batch_weight = layer.weight.expand(preact.shape[0], *layer.weight.shape)
                jac_factor = act_grad * batch_weight
            if jac is None:
                jac = jac_factor
            else:
                if jac.shape[0] != jac_factor.shape[0] or jac.dim() != jac_factor.dim():
                    print("error")
                jac = torch.bmm(jac_factor, jac)
        return jac

    def l2_loss(self):
        """ Compute L2 loss for all layers """
        loss = 0
        for l in self.layers:
            loss += torch.sum(l.weight ** 2)
        return loss

    def __repr__(self):
        mlp_str = "MLP: "
        layer_str = []
        for layer, nonlinearity in zip(self.layers, self.nonlinearities):
            layer_str.append(
                "({}, {}) -> {}".format(
                    layer.in_features, layer.out_features, str(nonlinearity)
                )
            )
        return mlp_str + " -> ".join(layer_str)
