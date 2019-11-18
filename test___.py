# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
# from matplotlib import cm
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X, Y, Z = axes3d.get_test_data(0.05)
# ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
# cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
#
# ax.set_xlabel('X')
# ax.set_xlim(-40, 40)
# ax.set_ylabel('Y')
# ax.set_ylim(-40, 40)
# ax.set_zlabel('Z')
# ax.set_zlim(-100, 100)
#
# plt.show()



# # from mpl_toolkits.mplot3d import axes3d
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# # fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8, 12), subplot_kw={'projection': '3d'})
# # X, Y, Z = axes3d.get_test_data(0.05)
# # ax1.plot_wireframe(X, Y, Z, rstride=10, cstride=0)
# # ax1.set_title("Column stride 0")
# # ax2.plot_wireframe(X, Y, Z, rstride=0, cstride=10)
# # ax2.set_title("Row stride 0")
# # plt.tight_layout()
# # plt.show()

# """
# Demonstrates using custom hillshading in a 3D surface plot.
# """
#
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cbook
# from matplotlib import cm
# from matplotlib.colors import LightSource
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Load and format data
# filename = cbook.get_sample_data('jacksboro_fault_dem.npz', asfileobj=False)
# with np.load(filename) as dem:
#     z = dem['elevation']
#     nrows, ncols = z.shape
#     x = np.linspace(dem['xmin'], dem['xmax'], ncols)
#     y = np.linspace(dem['ymin'], dem['ymax'], nrows)
#     x, y = np.meshgrid(x, y)
#
# region = np.s_[5:50, 5:50]
# x, y, z = x[region], y[region], z[region]
#
# # Set up plot
# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

# ls = LightSource(270, 45)
# # To use a custom hillshading mode, override the built-in shading and pass
# # in the rgb colors of the shaded surface calculated from "shade".
# rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
# surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
#                        linewidth=0, antialiased=False, shade=False)
#
# plt.show()

# from numpy import *
# import numpy as np
# import matplotlib.pylab as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection # New import
#
# #x,y = genfromtxt("data.dat",unpack=True)
# # Generated some random data
# w = 3
# x,y = np.arange(100), np.random.randint(0,100+w,100)
# y = np.array([y[i-w:i+w].mean() for i in range(3,100+w)])
# z = np.zeros(x.shape)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# #ax.add_collection3d(plt.fill_between(x,y,-0.1, color='orange', alpha=0.3,label="filled plot"),1, zdir='y')
# verts = [(x[i],z[i],y[i]) for i in range(len(x))] + [(x.max(),0,0),(x.min(),0,0)]
# ax.add_collection3d(Poly3DCollection([verts],color='orange')) # Add a polygon instead of fill_between
#
# ax.plot(x,z,y,label="line plot")
# ax.legend()
# ax.set_ylim(-1,1)
# plt.show()





import numpy as np
import matplotlib.pylab as plt

# Fancy Formatter
# Plot a sine and cosine curve

from mpl_toolkits import  mplot3d
fig = plt.figure()
ax = plt.axes(projection = '3d')


ax = plt.axes(projection = '3d')
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')
# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
plt.style.use('classic')
plt.show()

x = np.linspace(-6,6,30)
y = np.linspace(-6,6,30)

X, Y = np.meshgrid(x,y)

def func(x,y):
    return np.sin(np.sqrt(x**2 + y**2))
Z = func(X,Y)

ax = plt.axes(projection ='3d')
ax.contour3D(X,Y,Z,50,cmap='Blues')
ax.view_init(60,45) # 调整观察的视角和方位角
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')
ax.set_title('wireframe');
plt.show()

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-50,50,100)
y = np.arange(25)
X,Y = np.meshgrid(x,y)
Z = np.zeros((len(y),len(x)))

for i in range(len(y)):
    damp = (i/float(len(y)))**2
    Z[i] = 5*damp*(1 - np.sqrt(np.abs(x/50)))
    Z[i] += np.random.uniform(0,.1,len(Z[i]))
ax.plot_surface(X, Y, Z, rstride=1, cstride=1000, color='w', shade=False, lw=.5)

ax.set_zlim(0, 5)
ax.set_xlim(-51, 51)
ax.set_zlabel("Intensity")
ax.view_init(20,-120)
plt.show()



import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-50,50,100)
y = np.arange(25)
X,Y = np.meshgrid(x,y)
Z = np.zeros((len(y),len(x)))

for i in range(len(y)):
    damp = (i/float(len(y)))**2
    Z[i] = 5*damp*(1 - np.sqrt(np.abs(x/50)))
    Z[i] += np.random.uniform(0,.1,len(Z[i]))
ax.plot_surface(X, Y, Z, rstride=1, cstride=1000, color='w', shade=False, lw=.5)


ax.set_zlim(0, 5)
ax.set_xlim(-51, 51)
ax.set_zlabel("Intensity")
ax.view_init(20,-120)
plt.show()