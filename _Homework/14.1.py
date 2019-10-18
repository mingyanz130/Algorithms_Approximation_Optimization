"""Volume 2
<Mingyan Zhao>
<Math 323>
<02/14/2019>
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def prob3():
    fig = plt.figure()

    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    plt.subplot(231)
    Z = X**3+X**2-Y**2
    plt.contour(X,Y, Z,levels=[0])
    plt.scatter([0],[0],c="red")
    plt.title("X**3+X**2-Y**2")

    plt.subplot(232)
    Z = X**3-X**2-Y**2
    plt.contour(X,Y, Z,levels=[-0.5, -0.4, -0.3, -0.2, -0.1,  0. ,  0.1,  0.2,  0.3,  0.4,  0.5])
    plt.scatter([0],[0],c="red")
    plt.title("X**3-X**2-Y**2")

    plt.subplot(233)
    Z = X**3-Y**2
    plt.contour(X,Y, Z,levels=[0])
    plt.scatter([0],[0],c="red")
    plt.title("X**3-Y**2")

    plt.subplot(234)
    F = lambda x,y,z:z**2-x**2+y**2
    plot_implicit(fn = F, f=fig, p=234)
    plt.scatter([0],[0],[0])
    plt.title("z**2-x**2+y**2")

    plt.subplot(235)
    G = lambda x,y,z:x**2*y-z**2
    plot_implicit(fn = G, bbox = [-20,20], f=fig, p=235)
    plt.scatter([0],[0],[0])
    plt.title("x**2*y-z**2")

    plt.show()

def plot_implicit(fn, bbox=(-2.5,2.5), p=111, f=0):
    ''' create a plot of an implicit function
    fn  ...implicit function (plot where fn==0)
    bbox ..the x,y,and z limits of plotted interval'''
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    #fig = plt.figure()
    ax = f.add_subplot(p, projection='3d')
    A = np.linspace(xmin, xmax, 100) # resolution of the contour
    B = np.linspace(xmin, xmax, 15) # number of slices
    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z')
        # [z] defines the only level to plot for this contour for this value of z

    for y in B: # plot contours in the XZ plane
        X,Z = A1,A2
        Y = fn(X,y,Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y')

    for x in B: # plot contours in the YZ plane
        Y,Z = A1,A2
        X = fn(x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x')

    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.
    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)

    #plt.show()
