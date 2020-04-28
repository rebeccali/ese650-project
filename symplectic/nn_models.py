# Symplectic ODE-Net | 2019
# Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty

# code structure follows the style of HNN by Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

import torch
import numpy as np
from symplectic.utils import choose_nonlinearity


class MLP(torch.nn.Module):
    """Just a MLP (Multilayer perceptron)"""
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh', bias_bool=True):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=bias_bool)

        for l in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.orthogonal_(l.weight) # use a principled initialization

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x, separate_fields=False):
        h = self.nonlinearity( self.linear1(x) )
        h = self.nonlinearity( self.linear2(h) )
        return self.linear3(h)


class PSD(torch.nn.Module):
    """A Neural Net which outputs a positive semi-definite matrix"""
    def __init__(self, input_dim, hidden_dim, diag_dim, nonlinearity='tanh'):
        super(PSD, self).__init__()
        self.diag_dim = diag_dim
        if diag_dim == 1:
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = torch.nn.Linear(hidden_dim, diag_dim)

            for l in [self.linear1, self.linear2, self.linear3]:
                torch.nn.init.orthogonal_(l.weight) # use a principled initialization

            self.nonlinearity = choose_nonlinearity(nonlinearity)
        else:
            assert diag_dim > 1
            self.diag_dim = diag_dim
            self.off_diag_dim = int(diag_dim * (diag_dim - 1) / 2)
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear4 = torch.nn.Linear(hidden_dim, self.diag_dim + self.off_diag_dim)

            for l in [self.linear1, self.linear2, self.linear3, self.linear4]:
                torch.nn.init.orthogonal_(l.weight) # use a principled initialization

            self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, q):
        if self.diag_dim == 1:
            h = self.nonlinearity( self.linear1(q) )
            h = self.nonlinearity( self.linear2(h) )
            h = self.nonlinearity( self.linear3(h) )
            return h*h + 0.1
        else:
            bs = q.shape[0]
            h = self.nonlinearity( self.linear1(q) )
            h = self.nonlinearity( self.linear2(h) )
            h = self.nonlinearity( self.linear3(h) )
            diag, off_diag = torch.split(self.linear4(h), [self.diag_dim, self.off_diag_dim], dim=1)
            # diag = torch.nn.functional.relu( self.linear4(h) )

            L = torch.diag_embed(diag)

            ind = np.tril_indices(self.diag_dim, k=-1)
            flat_ind = np.ravel_multi_index(ind, (self.diag_dim, self.diag_dim))
            L = torch.flatten(L, start_dim=1)
            L[:, flat_ind] = off_diag
            L = torch.reshape(L, (bs, self.diag_dim, self.diag_dim))

            D = torch.bmm(L, L.permute(0, 2, 1))
            D[:, 0, 0] = D[:, 0, 0] + 0.1
            D[:, 1, 1] = D[:, 1, 1] + 0.1
            return D


class MatrixNet(torch.nn.Module):
    """ a neural net which outputs a matrix"""
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh', bias_bool=True, shape=(2,2)):
        super(MatrixNet, self).__init__()
        self.mlp = MLP(input_dim, hidden_dim, output_dim, nonlinearity, bias_bool)
        self.shape = shape

    def forward(self, x):
        flatten = self.mlp(x)
        return flatten.view(-1, *self.shape)


class DampMatrix(torch.nn.Module):
    """(not used in the paper) A Neural Net which outputs a 2*2 damping matrix"""
    def __init__(self, input_dim, hidden_dim, diag_dim, device, nonlinearity='tanh'):
        super(DampMatrix, self).__init__()
        assert diag_dim > 1
        self.linear1 = torch.nn.Linear(int(input_dim/2), hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, 1, bias=None)

        for l in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.orthogonal_(l.weight) # use a principled initialization

        self.nonlinearity = choose_nonlinearity(nonlinearity)
        self.device = device

    def forward(self, x):
        bs = x.shape[0]
        q, p = torch.split(x, 1, dim=1)
        h = self.nonlinearity( self.linear1(torch.ones(1).to(self.device)) )
        h = self.nonlinearity( self.linear2(h) )
        d = self.nonlinearity( self.linear3(h) )
        # D = torch.reshape(D, (-1,2,2))
        D = torch.zeros(bs, 2, 2, device=self.device)
        D[:, 1, 1] = d*d * torch.ones(bs).to(self.device)
        return D