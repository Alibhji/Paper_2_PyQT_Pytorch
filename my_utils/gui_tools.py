from  PyQt5.QtWidgets import QApplication,QMainWindow
from PyQt5.QtGui import QColor

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
        


