import torch
import torch.nn.functional as F
import math
import torch.nn as nn
from rational.torch import Rational  # Importing Rational Activation Function

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        scale_noise=0.001,
        scale_base=1.0,
        base_activation=torch.nn.GELU,
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.rational_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.base_activation = base_activation()
        self.rational_activation = Rational()  # Initializing Rational Activation Function
        
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.in_features, self.out_features) - 0.5) * self.scale_noise
            self.rational_weight.data.copy_(noise)

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        rational_output = F.linear(self.rational_activation(x), self.rational_weight)
        output = base_output + rational_output
        
        return output.reshape(*original_shape[:-1], self.out_features)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.rational_weight.abs().mean()
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        scale_noise=0.001,
        scale_base=1.0,
        base_activation=torch.nn.GELU,
        activation=True,
    ):
        super(KAN, self).__init__()

        assert len(layers_hidden) >= 2, "Must specify at least input and output layer sizes."
        self.activation = activation
        self.layers = torch.nn.ModuleList()
        
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    base_activation=base_activation,
                )
            )
        
        if self.activation:
            self.activations = nn.ModuleList([base_activation() for _ in range(len(self.layers) - 1)])

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.activation and i < len(self.layers) - 1:
                x = self.activations[i](x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
