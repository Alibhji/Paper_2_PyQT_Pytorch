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
# from collections import OrderedDict


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
        self.ui.tableWidget.cellClicked.connect(self.showCell)

        # this is a place holder for the data set cofiguration --> img_H, img_W, num of train and val set ...
        self.modelList = {}
        # it holds the generated model by the program
        self.config={}
        self.config_update()
        # Gui tools such as display the logs and fill the table
        self.tools=gui_tools.utils(self)
        # state of the program
        # while it is training you can not change the model parameters
        self.ui_state='idle'

        self.ui.tableWidget.setColumnCount(4)

        self.ui.tableWidget.setColumnWidth(1, 250)
        self.ui.tableWidget.setColumnWidth(2, 250)
        self.ui.tableWidget.setColumnWidth(3, 250)
        self.ui.tableWidget.setRowCount(1)
        self.ui.tableWidget.setColumnWidth(0, 250)
    
    def config_update(self):
        self.config.update({'img_H': int(self.ui.in_img_H.text())})
        self.config.update({'img_W': int(self.ui.in_img_W.text())})
        self.config.update({'dataset_train_size': int(self.ui.in_train_dataset.text())})
        self.config.update({'dataset_val_size': int(self.ui.in_val_dataset.text())})

        self.config.update({'model_counter': 0})
        
        self.ui.btn_train.setDisabled(not (hasattr(self,'image_datasets') and hasattr(self,'model')))
        self.ui.btn_plot_imgs.setDisabled(not hasattr(self,'image_datasets'))




    
    def showCell(self,row,col):
        if(self.ui_state=='idle'):
            model_name=self.ui.tableWidget.item(row,col).text()
            print(row,col,self.ui.tableWidget.item(row,col).text())
            self.model=self.modelList[model_name]['model']
            self.model_name=self.modelList[model_name]['name']
            self.model_txt_file=self.modelList[model_name]['text_log']
            print(self.modelList[model_name]['trained'])



        
    def train(self):  
        self.config_update()
        training(self)
        print(len(self.image_datasets['train']))
        self.config_update()
        self.tools.fill_out_table(self.modelList)
        # print(self.modelList)
        
        
    def dataset_generation(self):
        self.config_update()
        Dataset_create(self)
        self.config_update()
        
    

    def model_generation(self):
        self.config_update()
        Model_create(self)
        self.config_update()
        
 
    def model_archi_btn(self):
        self.modelList={}
        model_architecture(self)
        self.tools.fill_out_table(self.modelList)
        self.config_update()
        
    def plot_btn(self):
        self.tools.plotting(self)
        self.tools.logging(str(list(self.image_datasets['train'])[0][0].numpy().shape),'red')
        # self.tools.check_dir('test/123',create_dir=True)

        



app=QApplication(sys.argv)
win=AppWindow()
win.show()
sys.exit(app.exec())

