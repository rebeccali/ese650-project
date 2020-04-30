import torch.autograd
import torch.nn as nn
from au_functional import jacobian
import torch
from torchdiffeq import odeint

true_y0 = torch.tensor([[2., 0.]], requires_grad=True)
data_size=5
t = torch.linspace(0., 25., data_size)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])


class Lambda(nn.Module):

    # def forward(self, t, y):
    def forward(self, y):
        return torch.mm(y, true_A)


# with torch.no_grad():
f = Lambda()
# true_y = odeint(f, true_y0, t, method='dopri5')
# y0_hat = true_y[0]
# print(y0_hat)
# fx = f.forward(torch.tensor([0.]), true_y0)
# dfdy0 = torch.autograd.grad(fx.sum(), true_y0)
jac = jacobian(f, true_y0)
print(jac)





