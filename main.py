# A GUI for Pytorch and ptQT5
# Designed by ALI.BABOLHAVAEJI
# Detroit-MI-11/11/2019

# to conver UI to py file in Ununtu you need install the following package first
# sudo apt-get install qtcreator pyqt5-dev-tools
# then uncomment the the following line

import compile_ui
from main_ui import Ui_MainWindow
import sys ,  os

from PyQt5.QtWidgets import QApplication , QMainWindow

from learn import SimDataset,AliNet, train_model,training ,Dataset_create ,Model_create ,model_architecture

import time
import copy

from torchsummary import summary
from my_utils import gui_tools



class AppWindow(QMainWindow):
    def __init__(self):
        super(AppWindow,self).__init__()
        self.ui= Ui_MainWindow()
        self.ui.setupUi(self)
        self.UI_config()

        
    def UI_config(self):
        self.ui.btn_train.clicked.connect(self.train)
        self.ui.btn_dataset_gen.clicked.connect(self.dataset_generation)
        self.ui.btn_model_gen.clicked.connect(self.model_generation)
        self.ui.btn_model_arch.clicked.connect(self.model_archi_btn)
        self.ui.btn_plot_imgs.clicked.connect(self.plot_btn)
        
        self.config={}
        self.config_update()
        self.tools=gui_tools.utils(self)
    
    def config_update(self):
        self.config.update({'img_H': int(self.ui.in_img_H.text())})
        self.config.update({'img_W': int(self.ui.in_img_W.text())})
        self.config.update({'set_train': int(self.ui.in_train_dataset.text())})
        self.config.update({'set_val': int(self.ui.in_val_dataset.text())})
        
        self.ui.btn_train.setDisabled([True,False][hasattr(self,'image_datasets') and hasattr(self,'model')])
        self.ui.btn_plot_imgs.setDisabled([True,False][hasattr(self,'image_datasets')])


    

        
    def train(self):  
        self.config_update()
        training(self)
        print(len(self.image_datasets['train']))
        self.config_update()
        
        
    def dataset_generation(self):
        self.config_update()
        Dataset_create(self)
        self.config_update()
        
    

    def model_generation(self):
        self.config_update()
        Model_create(self)
        self.config_update()
        
 
    def model_archi_btn(self):
        model_architecture(self)    
        
    def plot_btn(self):
        self.tools.plotting(self)
        self.tools.logging(str(list(self.image_datasets['train'])[0][0].numpy().shape),'red')
        # self.tools.check_dir('test/123',create_dir=True)

        



app=QApplication(sys.argv)
win=AppWindow()
win.show()
sys.exit(app.exec())

