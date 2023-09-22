# from convection_basic import linear_convection_solve
import numpy
import matplotlib.pyplot as plt
from matplotlib import pyplot, cm

def draw_surf(Lx,Lt,u2D,tit_t,tit_x):
    nt = u2D.shape[0]
    nx = u2D.shape[1]
    xx = numpy.linspace(0,Lx,nx)
    tt = numpy.linspace(0,Lt,nt)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = numpy.meshgrid(xx, tt)
    surf = ax.plot_surface(X, Y, u2D, rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    plt.colorbar(surf)
    plt.xlabel(tit_x)
    plt.ylabel(tit_t)
    plt.title('1-D Linear Convection')
