import compile_ui
from main_ui import Ui_MainWindow
import sys ,  os

from PyQt5.QtWidgets import QApplication , QMainWindow

from learn import SimDataset,AliNet, train_model,training

import time
import copy

from torchsummary import summary



class AppWindow(QMainWindow):
    def __init__(self):
        super(AppWindow,self).__init__()
        self.ui= Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.train)
        
    def train(self):  
        training(self)
        



app=QApplication(sys.argv)
win=AppWindow()
win.show()
sys.exit(app.exec())
