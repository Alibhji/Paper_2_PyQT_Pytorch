# # from mpl_toolkits.mplot3d import axes3d
# # import matplotlib.pyplot as plt
# # from matplotlib import cm
# #
# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# # X, Y, Z = axes3d.get_test_data(0.05)
# # ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
# # cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
# # cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
# # cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
# #
# # ax.set_xlabel('X')
# # ax.set_xlim(-40, 40)
# # ax.set_ylabel('Y')
# # ax.set_ylim(-40, 40)
# # ax.set_zlabel('Z')
# # ax.set_zlim(-100, 100)
# #
# # plt.show()
#
#
#
# # # from mpl_toolkits.mplot3d import axes3d
# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # #
# # # fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8, 12), subplot_kw={'projection': '3d'})
# # # X, Y, Z = axes3d.get_test_data(0.05)
# # # ax1.plot_wireframe(X, Y, Z, rstride=10, cstride=0)
# # # ax1.set_title("Column stride 0")
# # # ax2.plot_wireframe(X, Y, Z, rstride=0, cstride=10)
# # # ax2.set_title("Row stride 0")
# # # plt.tight_layout()
# # # plt.show()
#
# # """
# # Demonstrates using custom hillshading in a 3D surface plot.
# # """
# #
# # from mpl_toolkits.mplot3d import Axes3D
# # from matplotlib import cbook
# # from matplotlib import cm
# # from matplotlib.colors import LightSource
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# # # Load and format data
# # filename = cbook.get_sample_data('jacksboro_fault_dem.npz', asfileobj=False)
# # with np.load(filename) as dem:
# #     z = dem['elevation']
# #     nrows, ncols = z.shape
# #     x = np.linspace(dem['xmin'], dem['xmax'], ncols)
# #     y = np.linspace(dem['ymin'], dem['ymax'], nrows)
# #     x, y = np.meshgrid(x, y)
# #
# # region = np.s_[5:50, 5:50]
# # x, y, z = x[region], y[region], z[region]
# #
# # # Set up plot
# # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
#
# # ls = LightSource(270, 45)
# # # To use a custom hillshading mode, override the built-in shading and pass
# # # in the rgb colors of the shaded surface calculated from "shade".
# # rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
# # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
# #                        linewidth=0, antialiased=False, shade=False)
# #
# # plt.show()
#
# # from numpy import *
# # import numpy as np
# # import matplotlib.pylab as plt
# # from mpl_toolkits.mplot3d import Axes3D
# # from mpl_toolkits.mplot3d.art3d import Poly3DCollection # New import
# #
# # #x,y = genfromtxt("data.dat",unpack=True)
# # # Generated some random data
# # w = 3
# # x,y = np.arange(100), np.random.randint(0,100+w,100)
# # y = np.array([y[i-w:i+w].mean() for i in range(3,100+w)])
# # z = np.zeros(x.shape)
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# #
# # #ax.add_collection3d(plt.fill_between(x,y,-0.1, color='orange', alpha=0.3,label="filled plot"),1, zdir='y')
# # verts = [(x[i],z[i],y[i]) for i in range(len(x))] + [(x.max(),0,0),(x.min(),0,0)]
# # ax.add_collection3d(Poly3DCollection([verts],color='orange')) # Add a polygon instead of fill_between
# #
# # ax.plot(x,z,y,label="line plot")
# # ax.legend()
# # ax.set_ylim(-1,1)
# # plt.show()
#
#
#
#
# #
# # import numpy as np
# # import matplotlib.pylab as plt
# #
# # # Fancy Formatter
# # # Plot a sine and cosine curve
# #
# # from mpl_toolkits import  mplot3d
# # fig = plt.figure()
# # ax = plt.axes(projection = '3d')
# #
# #
# # ax = plt.axes(projection = '3d')
# # zline = np.linspace(0, 15, 1000)
# # xline = np.sin(zline)
# # yline = np.cos(zline)
# # ax.plot3D(xline, yline, zline, 'gray')
# # # Data for three-dimensional scattered points
# # zdata = 15 * np.random.random(100)
# # xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# # ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# # ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
# # plt.style.use('classic')
# # plt.show()
# #
# # x = np.linspace(-6,6,30)
# # y = np.linspace(-6,6,30)
# #
# # X, Y = np.meshgrid(x,y)
# #
# # def func(x,y):
# #     return np.sin(np.sqrt(x**2 + y**2))
# # Z = func(X,Y)
# #
# # ax = plt.axes(projection ='3d')
# # ax.contour3D(X,Y,Z,50,cmap='Blues')
# # ax.view_init(60,45) # 调整观察的视角和方位角
# # plt.show()
# #
# # fig = plt.figure()
# # ax = plt.axes(projection='3d')
# # ax.plot_wireframe(X, Y, Z, color='black')
# # ax.set_title('wireframe');
# # plt.show()
# #
# # import numpy as np
# # from matplotlib import pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# #
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# #
# # x = np.linspace(-50,50,100)
# # y = np.arange(25)
# # X,Y = np.meshgrid(x,y)
# # Z = np.zeros((len(y),len(x)))
# #
# # for i in range(len(y)):
# #     damp = (i/float(len(y)))**2
# #     Z[i] = 5*damp*(1 - np.sqrt(np.abs(x/50)))
# #     Z[i] += np.random.uniform(0,.1,len(Z[i]))
# # ax.plot_surface(X, Y, Z, rstride=1, cstride=1000, color='w', shade=False, lw=.5)
# #
# # ax.set_zlim(0, 5)
# # ax.set_xlim(-51, 51)
# # ax.set_zlabel("Intensity")
# # ax.view_init(20,-120)
# # plt.show()
# #
# #
# #
# # import numpy as np
# # from matplotlib import pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# #
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# #
# # x = np.linspace(-50,50,100)
# # y = np.arange(25)
# # X,Y = np.meshgrid(x,y)
# # Z = np.zeros((len(y),len(x)))
# #
# # for i in range(len(y)):
# #     damp = (i/float(len(y)))**2
# #     Z[i] = 5*damp*(1 - np.sqrt(np.abs(x/50)))
# #     Z[i] += np.random.uniform(0,.1,len(Z[i]))
# # ax.plot_surface(X, Y, Z, rstride=1, cstride=1000, color='w', shade=False, lw=.5)
# #
# #
# # ax.set_zlim(0, 5)
# # ax.set_xlim(-51, 51)
# # ax.set_zlabel("Intensity")
# # ax.view_init(20,-120)
# # plt.show()
#
# #!/usr/bin/env python3
#
# from __future__ import print_function
#
# # Adapted from
# # https://stackoverflow.com/questions/13240633/matplotlib-plot-pulse-propagation-in-3d
# # and rewritten to make it clearer how to use it on real data.
#







# import numpy
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib.collections import PolyCollection
# from matplotlib import colors as mcolors
#
# import random
#
# def gen_data(xbins, numplots, lintest=False):
#     '''Generate a list of random histograms'''
#     data = []
#     ymin = 9999999999999
#     ymax = -ymin
#     for plot in range(numplots):
#         plotpoints = []
#         y = random.randint(0, 5)
#         for x in range(xbins):
#             # Optional: instead of random data, make each plot a constant
#             # to make it easier to tell which plot is which.
#             # Even if lintest isn't set, make the last 20% of the
#             # data predictable, to test whether matplotlib3d is
#             # re-ordering the plots (it isn't).
#             if lintest or x > xbins * .8:
#                 y = plot
#             else:
#                 y += random.uniform(-.8, 1)
#             ymin = min(ymin, y)
#             ymax = max(ymax, y)
#             plotpoints.append((x, y))
#         data.append(plotpoints)
#
#     return data, ymin, ymax
#
# def draw_3d(verts, ymin, ymax, line_at_zero=True, colors=True):
#     '''Given verts as a list of plots, each plot being a list
#        of (x, y) vertices, generate a 3-d figure where each plot
#        is shown as a translucent polygon.
#        If line_at_zero, a line will be drawn through the zero point
#        of each plot, otherwise the baseline will be at the bottom of
#        the plot regardless of where the zero line is.
#     '''
#     # add_collection3d() wants a collection of closed polygons;
#     # each polygon needs a base and won't generate it automatically.
#     # So for each subplot, add a base at ymin.
#     if line_at_zero:
#         zeroline = 0
#     else:
#         zeroline = ymin
#     for p in verts:
#         p.insert(0, (p[0][0], zeroline))
#         p.append((p[-1][0], zeroline))
#
#     if colors:
#         # All the matplotlib color sampling examples I can find,
#         # like cm.rainbow/linspace, make adjacent colors similar,
#         # the exact opposite of what most people would want.
#         # So cycle hue manually.
#         hue = 0
#         huejump = .27
#         facecolors = []
#         edgecolors = []
#         for v in verts:
#             hue = (hue + huejump) % 1
#             c = mcolors.hsv_to_rgb([hue, 1, 1])
#                                     # random.uniform(.8, 1),
#                                     # random.uniform(.7, 1)])
#             edgecolors.append(c)
#             # Make the facecolor translucent:
#             facecolors.append(mcolors.to_rgba(c, alpha=.7))
#     else:
#         facecolors = (1, 1, 1, .8)
#         edgecolors = (0, 0, 1, 1)
#
#     poly = PolyCollection(verts,
#                           facecolors=facecolors, edgecolors=edgecolors)
#
#     zs = range(len(data))
#     # zs = range(len(data)-1, -1, -1)
#
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#
#     plt.tight_layout(pad=2.0, w_pad=10.0, h_pad=3.0)
#
#     ax.add_collection3d(poly, zs=zs, zdir='y')
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#
#     ax.set_xlim3d(0, len(data[1]))
#     ax.set_ylim3d(-1, len(data))
#     ax.set_zlim3d(ymin, ymax)
#
#
# if __name__ == '__main__':
#     import argparse
#     import sys
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-l', "--lintest", dest="lintest", default=False,
#                         action="store_true",
#                         help="Ultra simple sample data for testing")
#     parser.add_argument('-c', "--color", dest="colors", default=False,
#                         action="store_true", help="Plot in multiple colors")
#     parser.add_argument('-x', action="store", dest="xbins",
#                         type=int, default=50,
#                         help='Number of points on the X axis')
#     parser.add_argument('-n', action="store", dest="numplots",
#                         type=int, default=5,
#                         help='Number of plots')
#     args = parser.parse_args(sys.argv[1:])
#
#     data, ymin, ymax = gen_data(args.xbins, args.numplots, lintest=args.lintest)
#     draw_3d(data, ymin, ymax, colors=args.colors)
#     plt.show()
#
# '''
# 1st green: 150, 100, 100 = .59, 1, 1    matplotlib hue .17 -> 0.   1.   0.04
# 2nd green: 152, 100, 94  = .60, 1, .94             hue .34 -> 0.   0.94 1.
#
# 1st green claims [0.   1.   0.04], GIMP says 0 1 . 5
# 2dn green claims [0.   0.94 1.  ], GIMP says 0 .94 .5
#
# '''



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
from mpl_toolkits.mplot3d import Axes3D

T = 60.
n = 512
t = np.linspace(-T / 2., T / 2., n + 1)
t = t[0:n]
# There's a function to set up the frequencies, but doing it by hand seems to help me think
# things through.
k = np.array([(2. * np.pi) * i if i < n / 2 else (2. * np.pi) * (i - n)
              for i in range(n)])

ks = np.fft.fftshift(k)
slc = np.arange(0, 10, 0.5)
# I haven't quite figured out how to use the meshgrid function in numpy
T, S = np.meshgrid(t, slc)
K, S = np.meshgrid(k, slc)

# Now, we have a plane flying back and forth in a sine wave and getting painted by a radar pulse
# which is a hyperbolic secant (1/cosh)
U = 1. / np.cosh(T - 10. * np.sin(S)) * np.exp(1j * 0. * T)





def waterfall(X, Y, Z, nslices):
    # Function to generate formats for facecolors
    cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.3)
    # This is just wrong. There must be some way to use the meshgrid or why bother.
    verts = []
    for i in range(nslices):
        verts.append(list(zip(X[i], Z[i])))

    xmin = np.floor(np.min(X))
    xmax = np.ceil(np.max(X))
    ymin = np.floor(np.min(Y))
    ymax = np.ceil(np.max(Y))
    zmin = np.floor(np.min(Z.real))
    zmax = np.ceil(np.max(np.abs(Z)))

    fig = plt.figure()
    ax = Axes3D(fig)

    poly = PolyCollection(verts, facecolors=[cc('g')])
    ax.add_collection3d(poly, zs=slc, zdir='y')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    plt.show()


waterfall(T, S, U.real, len(slc))


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)

xs = np.arange(0, 10, 0.4)
verts = []
zs = [0.0, 1.0, 2.0, 3.0]
for z in zs:
    ys = np.random.rand(len(xs))
    ys[0], ys[-1] = 0, 0
    verts.append(list(zip(xs, ys)))

poly = PolyCollection(verts, facecolors = [cc('r'), cc('g'), cc('b'),
                                           cc('y')])
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('X')
ax.set_xlim3d(0, 10)
ax.set_ylabel('Y')
ax.set_ylim3d(-1, 4)
ax.set_zlabel('Z')
ax.set_zlim3d(0, 1)

plt.show()