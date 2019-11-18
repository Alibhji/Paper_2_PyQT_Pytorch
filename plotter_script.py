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
from matplotlib import pyplot as plt



class App(QMainWindow):
    def __init__(self):
        super(App,self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("The Loss plotter")
        self.confin_ui()



    def confin_ui(self):
        self.ui.btn_open.clicked.connect(self.btn_open)
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












if __name__ == '__main__':
    app=QApplication(sys.argv)
    ex=App()
    ex.show()
    sys.exit(app.exec_())