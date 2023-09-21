# from convection_basic import linear_convection_solve
import numpy
import matplotlib.pyplot as plt
from matplotlib import pyplot, cm

def draw_surf(Lx,nx,Lt,nt,u2D):
    xx = numpy.linspace(0,Lx,nx)
    tt = numpy.linspace(0,Lt,nt)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = numpy.meshgrid(xx, tt)
    surf = ax.plot_surface(X, Y, u2D, rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    plt.colorbar(surf)
    plt.xlabel('X')
    plt.ylabel('T')
    plt.title('1-D Linear Convection')
