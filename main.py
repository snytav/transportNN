import torch
import torch.nn as nn
import numpy as np
from convection_basic import linear_convection_solve


def f(x):
    return 0.


class PDEnet(nn.Module):
    def __init__(self,N,Lx,Lt,nt,c,dt):
        super(PDEnet,self).__init__()
        self.N = N
        fc1 = nn.Linear(2,self.N)
        fc2 = nn.Linear(self.N, 1)
        self.fc1 = fc1
        self.fc2 = fc2

        self.Lx = Lx
        self.Lt = Lt
        self.nt = nt
        self.c  = c
        self.dt = dt
        self.dx = self.Lx / (self.N - 1)

    def forward(self,x):
        x = x.reshape(1, 2)
        y = self.fc1(x)
        y = torch.sigmoid(y)
        y = self.fc2(y.reshape(1, self.N))
        return y

    def boundary(self,xx):
        x = xx[0] / self.Lx
        t = xx[1] / self.Lt
        return (t * torch.sin(np.pi * x))

    def trial(self,xx):
        x = xx[0] / self.Lx
        t = xx[1] / self.Lt
        f = A(xx) + x * (1 - x) * t * (1 - t) * self.forward(xx)
        return f

    def initial_condition(self):
        u = np.ones(self.N)  # numpy function ones()
        dx = self.dx
        u[int(.5 / dx):int(1 / dx + 1)] = 2  # setting u = 2 between 0.5 and 1 as per our I.C.s
        return u
    #
    # def boundary(self,xx):
    #     x = xx[0] / Lx
    #     t = xx[1] / Lt
    #     u = torch.zeros(self.nt,self.N)
    #     u[0,:] = self.initial_condition()

        return u

        # HERE: arrange solution from L.Barba as a method and make B.C. from this method
        # first u make from Barba's initial condition, all the params into class variables.
       # u, u2D = linear_convection_solve(u, self.c, self.dx, self.dt, self.Lx, self.nx, self.Lt, self.nt)

    # def trial(self, x):
    #     y = self.boundary(x)+self.forward(x)
    #     return y





Lx = 2.0
nx = 41  # try changing this number from 41 to 81 and Run All ... what happens?
dx = Lx / (nx - 1)
nt = 25  # nt is the number of timesteps we want to calculate
dt = .025  # dt is the amount of time each timestep covers (delta t)
Lt = nt*dt

def A(xx):
    x = xx[0] / Lx
    t = xx[1] / Lt
    return (t * torch.sin(np.pi * x))

def psy_trial(xx, net_out):
    x = xx[0] / Lx
    t = xx[1] / Lt
    f = A(xx) + x * (1 - x) * t * (1 - t) * net_out
    return f


if __name__ == '__main__':
    nx = 41  # try changing this number from 41 to 81 and Run All ... what happens?
    dx = 2 / (nx - 1)
    nt = 25  # nt is the number of timesteps we want to calculate
    dt = .025  # dt is the amount of time each timestep covers (delta t)
    c = 1  # assume wavespeed of c = 1
    x_space = torch.linspace(0,2,nx-1)
    t_space = np.arange(0,nt*dt,dt)
    # t_space = torch.from_numpy(t_space)


    pde = PDEnet(nx-1,Lx,Lt,nt,c,dt)
    y = pde.trial(torch.zeros(2))

    lmb = 0.01
    optimizer = torch.optim.Adam(pde.parameters(), lr=lmb)
    i = 0
    loss = 1e6*torch.ones(1)
    # for i in range(100):
    while loss.item() > 1e-1:
        optimizer.zero_grad()
        from loss import loss_function

        loss = loss_function(x_space, t_space, pde, psy_trial, f,c)

        loss.backward(retain_graph=True)

        optimizer.step()

        print(i, loss.item())
        i = i+1



    u_nn = torch.zeros((nt,nx-1))

    for i, x in enumerate(x_space):
        for j, t in enumerate(t_space):
            u_nn[j][i] = pde.trial(torch.Tensor([x, t]))
            # surface[j][i] = psy_trial([x, t], net_outt)

    from surface import draw_surf

    u = pde.initial_condition()
    u, u2D = linear_convection_solve(u, pde.c, pde.dx, pde.dt, pde.Lx, pde.N, pde.Lt, pde.nt)
    #draw_surf(pde.Lx, pde.N, pde.Lt, nt, u2D) #u_nn.detach().numpy())
    draw_surf(Lx, Lt, u_nn.detach().numpy(), 'T', 'X')



    qq = 0