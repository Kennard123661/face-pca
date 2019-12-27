import torch.nn as nn
import torch


class TripletNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs, network_name='mlp'):
        super(TripletNetwork, self).__init__()

        self.network_name = network_name
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        if self.network_name == network_name:
            self.net = MlpModel(self.n_inputs, self.n_outputs)
        else:
            raise ValueError('no such network exists')

    def predict(self, inputs):
        self.net(inputs)

    def forward(self, anchors, positives, negatives):
        anchor_feats = self.net(anchors)
        positive_feats = self.net(positives)
        negative_feats = self.net(negatives)
        return anchor_feats, positive_feats, negative_feats


class MlpModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MlpModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = [
            DenseBlock(n_inputs=self.input_dim, n_outputs=500, has_bias=False, has_bn=True, activation='relu'),
            DenseBlock(n_inputs=500, n_outputs=self.output_dim, has_bias=True, has_bn=False, activation='relu'),
            L2Norm()
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs)


class DenseBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, has_bias=False, has_bn=False, activation=None):
        super(DenseBlock, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # weighted layer.
        layers = [
            nn.Linear(in_features=self.n_inputs, out_features=self.n_outputs, bias=has_bias),
        ]

        # activations
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
            nn.init.kaiming_normal_(layers[0].weight, nonlinearity='relu')
        elif activation is None:
            nn.init.xavier_normal_(layers[0].weight)
        else:
            raise NotImplementedError

        # batch normalization
        if has_bn:
            layers.append(nn.BatchNorm1d(self.n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs)


class L2Norm(nn.Module):
    """
    This module only works for tensors with B x (num_features).
    """
    def __init__(self, eps=1e-10):
        super(L2Norm, self).__init__()
        self.eps = eps  # prevent it from divide by zero.

    def forward(self, inputs):
        norm = inputs.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        return inputs / norm
