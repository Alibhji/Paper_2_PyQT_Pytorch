

print("Ali Babolhavaeji # Extract features from image in PyTorch --> 11/19/2019")

import os
os.system("pyuic5 ./Feature_ext_ui.ui > ./Feature_ext_ui.py")

import sys
from Feature_ext.Feature_ext_ui import Ui_MainWindow
from  PyQt5.QtWidgets import QMainWindow, QApplication
import Feature_ext.utils_2 as util
import pandas as pd




class Appwindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Feature Extraction V1.00")

        
        cfg_file_path=os.path.join(os.getcwd(),'config.yml')
        cfg=util.import_yaml(cfg_file_path)
        util.Qlogging(self.ui.textBrowser, "{}".format(cfg_file_path))
        util.Qlogging(self.ui.textBrowser, "The config file is loaded.")

        # category_index=list(cfg['category_index'].split(','))
        # print(category_index)
        path=cfg['dataset']['path']

        if(util.check_VOC(path)):
            util.Qlogging(self.ui.textBrowser, "The PASCAL-VOC datase is found.")
        else:
            util.Qlogging(self.ui.textBrowser, "[ERROR] --> The PASCAL-VOC dataset is not exist!." ,type="red")


        data=util.Read_VOC(cfg)
        data_pd = pd.DataFrame.from_dict(data)
        model = util.PandasModel(data_pd)
        self.ui.tableView.setModel(model)
        
        util.Qlogging(self.ui.textBrowser,str(data_pd))

        # print(data)






        # self.ui.textBrowser

if __name__ == '__main__' :
    App= QApplication(sys.argv)
    window= Appwindow()
    window.show()
    sys.exit(App.exec())

