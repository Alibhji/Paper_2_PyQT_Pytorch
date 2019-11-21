# A GUI for Pytorch and ptQT5
# Designed by ALI.BABOLHAVAEJI
# Detroit-MI-11/11/2019

# to conver UI to py file in Ununtu you need install the following package first
# sudo apt-get install qtcreator pyqt5-dev-tools
# then uncomment the the following line


import compile_ui
from main_ui import Ui_MainWindow
import sys ,  os
import pickle
from PyQt5.QtWidgets import QApplication , QMainWindow
from learn import SimDataset,AliNet, train_model,training ,Dataset_create ,Model_create ,model_architecture

import time
import copy

from torchsummary import summary
from my_utils import gui_tools
from my_utils import joy_plot_
from collections import OrderedDict


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
        self.ui.btn_3d_plot.clicked.connect(self.btn_plot_3d)

        # self.ui.in_cmbox_3dPlot.setItemText(0,'tesrt')
        # self.ui.in_cmbox_3dPlot.currentText()



        # self.ui.bt
        # self.ui.tra
        self.ui.btn_train_allmodels.clicked.connect(self.btn_train_all_models)





        # this is a place holder for the data set cofiguration --> img_H, img_W, num of train and val set ...
        self.modelList =OrderedDict()
        # it holds the generated model by the program
        self.config={}
        self.config_update()
        # Gui tools such as display the logs and fill the table
        self.tools=gui_tools.utils(self)
        # state of the program
        # while it is training you can not change the model parameters
        self.ui_state='idle'

        self.ui.tableWidget.setColumnCount(1)

        # self.ui.tableWidget.setColumnWidth(1, 250)
        # self.ui.tableWidget.setColumnWidth(2, 250)
        # self.ui.tableWidget.setColumnWidth(3, 250)
        # self.ui.tableWidget.setRowCount(1)
        self.ui.tableWidget.setColumnWidth(0, 300)


    def btn_plot_3d(self) :

        loss_file = os.path.join(os.getcwd(), self.module_dir_name, 'All_Results.losses')
        self.tools.plot_3d_(loss_file)




    def btn_train_all_models(self):

        for model in self.modelList.items():
            print(model[0])

            model_name=model[0]
            self.model=self.tools.load_object(self.modelList[model_name]['model_address'])
            self.model_name=self.modelList[model_name]['name']
            self.model_txt_file=self.modelList[model_name]['text_log']

            print("**--**" * 30)
            print(self.model)

            if (self.ui.chkbox_stop_training.isChecked()):
                break
            self.train()
            self.model = self.tools.save_object(object=self.model,path=self.model_txt_file)

        print((self.modelList))
        loss_file=os.path.join(os.getcwd(),self.module_dir_name,'All_Results.losses')

        params={'models_inputs':self.config['models_inputs'],
                'models_outputs':self.config['models_outputs'],
                'models_kernels':self.config['models_kernels'],
                'models_num_classes':self.config['models_num_classes']}
        self.modelList.update({'params' : params})

        self.tools.save_object(path=loss_file,object=self.modelList)
        # print(self.modelList)
        # self.tools.plot_3d_(loss_file)
        # del (self.modelList)
        
        # print((self.modelList))
            # print('----')



    
    def config_update(self):
        self.config.update({'img_H': int(self.ui.in_img_H.text())})
        self.config.update({'img_W': int(self.ui.in_img_W.text())})
        self.config.update({'dataset_train_size': int(self.ui.in_train_dataset.text())})
        self.config.update({'dataset_val_size': int(self.ui.in_val_dataset.text())})

        self.config.update({'dataset_obj_shape_triangle': self.ui.in_shape_triangle.isChecked()})
        self.config.update({'dataset_obj_shape_circle': self.ui.in_shape_circle.isChecked()})
        self.config.update({'dataset_obj_shape_mesh': self.ui.in_shape_mesh.isChecked()})
        self.config.update({'dataset_obj_shape_square': bool(self.ui.in_shape_squre.isChecked())})
        self.config.update({'dataset_obj_shape_plus': self.ui.in_shape_plus.isChecked()})

        self.config.update({'train_Epoch_number': int(self.ui.in_train_epoch_num.text())})
        self.config.update({'dataset_batch_size': int(self.ui.in_batch_size.text())})



        self.config.update({'model_counter': 0})
        
        self.ui.btn_train.setDisabled(not (hasattr(self,'image_datasets') and hasattr(self,'model')))
        self.ui.btn_plot_imgs.setDisabled(not hasattr(self,'image_datasets'))

        self.ui.in_models_layers.setEnabled(False)
        self.module_dir_name = self.ui.in_model_save_dir.text()

    #It generates event when you click on the tabel cells and give you row and column number
    def showCell(self,row,col):
        if(self.ui_state=='idle'):
            model_name=self.ui.tableWidget.item(row,col).text()


            # self.model=self.modelList[model_name]['model']
            # selected_name=self.ui.tableWidget.item(row,0).text()
            print(row, col, self.ui.tableWidget.item(row, col).text())
            self.model=self.tools.load_object(self.modelList[model_name]['model_address'])
            self.model_name=self.modelList[model_name]['name']
            self.model_txt_file=self.modelList[model_name]['text_log']
            print(self.modelList[model_name]['trained'])
            print("**--**" * 30)
            print(self.model)
            self.config_update()
            if(self.modelList[model_name]['trained']):
                joy_plot_.plot1(self.modelList[model_name]['loss'])



        
    def train(self):  
        self.config_update()
        training(self)
        print(len(self.image_datasets['train']))
        self.config_update()
        self.tools.fill_out_table(self.modelList)
        self.tools.fill_out_table_2(self.modelList)





        # print(self.modelList)
        
        
    def dataset_generation(self):
        self.config_update()
        Dataset_create(self)
        self.config_update()

        
    

    def model_generation(self):
        self.config_update()
        self.ui_state = 'model_generation'
        Model_create(self)
        self.ui_state = 'idle'
        self.config_update()

 
    def model_archi_btn(self):
        self.modelList={}
        # self.modelList =
        model_architecture(self)
        self.tools.fill_out_table(self.modelList)
        self.tools.fill_out_table_2(self.modelList)

        self.config_update()

        # self.ui.tableWidget_2.setColumnCount(len(self.config['models_kernels']))
        # self.ui.tableWidget_2.setRowCount   (len(self.config['models_outputs']))
        # print(len(self.config['models_kernels']))
        # print(len(self.config['models_outputs']))
        
    def plot_btn(self):
        self.tools.plotting(self)
        self.tools.logging(str(list(self.image_datasets['train'])[0][0].numpy().shape),'red')
        # self.tools.check_dir('test/123',create_dir=True)


    def closeEvent(self, event):
        pass
        # do stuff
        # if True:
        #     print('the win is closed')
        #     with open('last_saved.uiModel', 'wb') as uiFile:
        #         # Step 3
        #         pickle.dump(self, uiFile)
        #     event.accept() # let the window close
        # else:
        #     event.ignore()

        



app=QApplication(sys.argv)

# if(not os.path.exists('last_saved.uiModel')):
#     win=AppWindow()
# else :
#     with open('last_saved.uiModel', 'rb') as loaded_ui:
#         win = pickle.load(loaded_ui)

win=AppWindow()
win.show()
sys.exit(app.exec())

