import os

from  PyQt5.QtWidgets import QApplication,QMainWindow
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import  QTableWidgetItem

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



class utils():
    def __init__(self,ui):
        ui.ui.textBrowser.setText("The program is started.")
        self.color()
        self.ui=ui

    def color(self):

        self.black = QColor(0, 0, 0)
        self.red = QColor(255, 0, 0)
        self.green = QColor(0, 100, 0)


    def logging(self,message=" ", type='info'):
        if(type=='info'):
            self.ui.ui.textBrowser.setTextColor(self.green)
            self.ui.ui.textBrowser.append(message)
        if(type=='red'):
            self.ui.ui.textBrowser.setTextColor(self.red)
            self.ui.ui.textBrowser.append(message)
            self.ui.ui.textBrowser.setTextColor(self.black)

    def plotting(self,ui):
        fig = plt.figure(figsize=(9, 15))
        gs = gridspec.GridSpec(nrows=4, ncols=4)

        for i in range(4):
            for j in range(4):
                ax = fig.add_subplot(gs[j, i])
                ax.imshow(list(ui.image_datasets['train'])[i][0].numpy()[1,:,:])
            # ax.set_title(title[i])
        plt.show()

    def check_dir(self,name, root='./' , create_dir=None):
        file = os.path.join(root,name)

        exist= [True,False][os.path.exists(file)]
        if create_dir and exist :
            os.makedirs(file)



    def fill_out_table(self, dict_data):
        # print(dict_data)

        self.ui.ui.tableWidget.setRowCount(len(dict_data))
        for row, row_data in enumerate(dict_data.items()):
            # for col , data in enumerate(row_data[1].items()):
            #     self.ui.ui.tableWidget.setItem(row,col,QTableWidgetItem(str(data[1])))
            self.ui.ui.tableWidget.setItem(row,0,QTableWidgetItem(str(row_data[1]['name'])))

