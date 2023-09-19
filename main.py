import torch
import torch.nn as nn
import numpy as np

def f(x):
    return 0.


class PDEnet(nn.Module):
    def __init__(self,N):
        super(PDEnet,self).__init__()
        self.N = N
        fc1 = nn.Linear(2,self.N)
        fc2 = nn.Linear(self.N, 1)
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self,x):
        x = x.reshape(1, 2)
        y = self.fc1(x)
        y = torch.sigmoid(y)
        y = self.fc2(y.reshape(1, self.N))
        return y

    def boundary(self,xx):
        x = xx[0] / Lx
        t = xx[1] / Lt
        # HERE: arrange solution from L.Barba as a method and make B.C. from this method


    def trial(self, x):
        y = self.boundary(x)+self.forward(x)





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
    t_space = torch.from_numpy(t_space)


    pde = PDEnet(nx-1)

    lmb = 0.01
    optimizer = torch.optim.Adam(pde.parameters(), lr=lmb)
    i = 0
    loss = 1e6*torch.ones(1)
    #for i in range(100):
    while loss.item() > 1e-1:
        optimizer.zero_grad()
        from loss import loss_function

        loss = loss_function(x_space, t_space, pde, psy_trial, f,c)

        loss.backward(retain_graph=True)

        optimizer.step()

        print(i, loss.item())
        i = i+1



    surface = np.zeros((nt,nx-1))

    for i, x in enumerate(x_space):
        for j, t in enumerate(t_space):
            net_outt = pde.forward(torch.Tensor([x, t]))
            surface[j][i] = psy_trial([x, t], net_outt)

    import matplotlib.pyplot as plt
    from matplotlib import pyplot, cm
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    fig = plt.figure()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y = np.meshgrid(x_space.numpy(), t_space.numpy())
    surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    plt.xlabel('X')
    plt.ylabel('T')
    fig.colorbar(surf, ax=ax)
    plt.show()

    qq = 0