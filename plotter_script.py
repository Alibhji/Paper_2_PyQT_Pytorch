import os
# os.system('conda activate envtroch')
os.system('pyuic5 ./plotter_ui.ui > ./plotter_ui.py')

from PyQt5.QtWidgets import QApplication, QMainWindow ,QFileDialog ,QListWidgetItem
import sys

from plotter_ui import Ui_MainWindow
import pickle
# import pyjoyplot as pjp
import joypy
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt



class App(QMainWindow):
    def __init__(self):
        super(App,self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("The Loss plotter")
        self.confin_ui()



    def confin_ui(self):
        self.ui.btn_open.clicked.connect(self.btn_open)
        self.ui.btn_Plot_all.clicked.connect(self.plot_all)
        self.ui.listWidget.itemDoubleClicked.connect(self.double_tabel)
        self.ui.listWidget.clear()
        self.data={}
        self.params={}


    def load(self, path):
        with open(path, 'rb') as uiFile:
            # Step 3
            object = pickle.load(uiFile)
            return object

    def double_tabel(self,item):
        print(item.text())
        Module_name=item.text()
        l = self.data[Module_name]['loss']['loss_bce_train']
        # print(l)
        samples = [kk[0] for kk in l]
        # epoche = [k[1] for k in l]
        loss1 = [kk[2] for kk in l]
        loss2 = [kk[3] for kk in l]
        loss3 = [kk[4] for kk in l]
        lr = [kk[5] for kk in l]

        # fig=plt.figure()
        df_train = pd.DataFrame(l)
        df_train.columns = ['Sample', 'epoch', 'bce', 'loss_defined', 'total', 'lr']
        # fig, axes = joypy.joyplot(df_train, column=["bce", 'loss_defined', 'total'])
        # print(df_train)
        # fig.show()
        fig, axes = joypy.joyplot(df_train,column=["bce", 'loss_defined', 'total'])
        plt.show()
        # ax = plt.axes(projection='3d')
        # plt.show()


    def btn_open(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "losses Files (*.losses)", options=options)
        self.ui.in_file_address.setText(fileName)
        if fileName:
            print(fileName)
        self.filename=fileName
        self.data=self.load(fileName)
        # print(self.data)
        modules = [i for i in list(self.data.keys()) if i.startswith('Module')]

        k = self.data['params']['models_kernels']
        ch = self.data['params']['models_outputs']

        self.params.update({'models_kernels':k})
        self.params.update({'models_outputs': ch})



        for name_ in modules:
            self.ui.listWidget.addItem(name_)
        print(self.params )




    def plot_all(self):

        zs = list(range(self.ui.listWidget.count()))
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)


        verts = []
        names=[]

        for i in zs:
            item = self.ui.listWidget.item(i)
            Module_name = item.text()
            names.append(Module_name)
        filter=self.ui.in_filter.text()
        if not filter:
            filter='03'
        names=[i for i in names if (i.find(filter)>0)]

        print(names)

        for i in names:
            l = self.data[i]['loss']['loss_bce_train']

            samples = [kk[0] for kk in l]
            epoche  = [kk[1] for kk in l]
            loss1   = [kk[2] for kk in l]
            loss2   = [kk[3] for kk in l]
            loss3   = [kk[4] for kk in l]
            lr = [kk[5] for kk in l]

            xs = samples

            ys = loss1
            ys[0], ys[-1] = 0, 0
            verts.append(list(zip(xs, ys)))

        colours = plt.cm.Blues(np.linspace(0.2, 0.8, len(names)))
        poly = PolyCollection(verts, facecolors=colours,edgecolor="k", linewidth=2)
        # poly = PolyCollection(verts, facecolors=[cc('g')])

        poly.set_alpha(0.7)
        # zs=[eval('{}.0'.format(i)) for i in zs]

        zs = np.arange(0, len(names), 1.0)

        ax.add_collection3d(poly, zs=zs, zdir='y')



        ax.set_xlabel('X')
        ax.set_xlim3d(0, samples[-1])
        ax.set_ylabel('Y')
        ax.set_ylim3d(-1, zs[-1])
        ax.set_zlabel('Z')
        ax.set_zlim3d(0, 1)

        plt.show()
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # zs = [0.0, 1.0, 2.0, 3.0]
        #
        # for i in range(self.ui.listWidget.count()):
        #     item=self.ui.listWidget.item(i)
        #     print(item.text())
        #     Module_name=item.text()
        #
        #     l = self.data[Module_name]['loss']['loss_bce_train']
        #     # print(l)
        #     samples = [kk[0] for kk in l]
        #     epoche  = [kk[1] for kk in l]
        #     loss1   = [kk[2] for kk in l]
        #     loss2   = [kk[3] for kk in l]
        #     loss3   = [kk[4] for kk in l]
        #     lr = [kk[5] for kk in l]
        #
        #
        #
        #     xs = samples
        #     verts = []
        #
        #
        #     ys = loss1
        #     # ys[0], ys[-1] = 0, 0
        #     verts.append(list(zip(xs, ys)))
        #
        # cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)
        #
        # poly = PolyCollection(verts, facecolors=[cc('r'), cc('g'), cc('b'),
        #                                          cc('y')])
        # poly.set_alpha(0.7)
        # ax.add_collection3d(poly, zs=zs, zdir='y')














if __name__ == '__main__':
    app=QApplication(sys.argv)
    ex=App()
    ex.show()
    sys.exit(app.exec_())