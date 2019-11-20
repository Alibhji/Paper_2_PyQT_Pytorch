

print("Ali Babolhavaeji # Extract features from image in PyTorch --> 11/19/2019")

import os
os.system("pyuic5 ./Feature_ext_ui.ui > ./Feature_ext_ui.py")

import sys
from Feature_ext.Feature_ext_ui import Ui_MainWindow
from  PyQt5.QtWidgets import QMainWindow, QApplication
from  PyQt5.QtCore import QModelIndex
import Feature_ext.utils_2 as util
import pandas as pd
import cv2
import ast




class Appwindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Feature Extraction V1.00")

        
        cfg_file_path=os.path.join(os.getcwd(),'config.yml')
        cfg=util.import_yaml(cfg_file_path)
        self.cfg=cfg



        util.Qlogging(self.ui.textBrowser, "{}".format(cfg_file_path))
        util.Qlogging(self.ui.textBrowser, "The config file is loaded.")

        # category_index=list(cfg['category_index'].split(','))
        # print(category_index)
        path=cfg['dataset']['path']

        if(util.check_VOC(path)):
            util.Qlogging(self.ui.textBrowser, "The PASCAL-VOC datase is found.")
        else:
            util.Qlogging(self.ui.textBrowser, "[ERROR] --> The PASCAL-VOC dataset is not exist!." ,type="red")


        data, self.category_index=util.Read_VOC(cfg)
        data_pd = pd.DataFrame.from_dict(data)
        data_pd['objects']=data_pd.apply(lambda row: len(row.cls),axis=1)
        self.model = util.PandasModel(data_pd)
        self.ui.tableView.setModel(self.model)
        
        util.Qlogging(self.ui.textBrowser,str(data_pd))

        # print(data)

        self.ui.tableView.clicked.connect(self.table_clicked)

    def table_clicked(self,index):
        index=QModelIndex(index)

        # ind.__setattr__('column',0)
        row , col=index.row() ,index.column()
        ind = self.model.index(row, 2)
        image= list(self.model.itemData(QModelIndex(ind)).values())[0]

        ind = self.model.index(row, 3)
        xyMax= list(self.model.itemData(QModelIndex(ind)).values())[0]
        xyMax=xyMax.replace('[', '').replace(']', '').split()

        ind = self.model.index(row, 4)
        xyMin = list(self.model.itemData(QModelIndex(ind)).values())[0]
        xyMin = xyMin.replace('[', '').replace(']', '').split()

        ind = self.model.index(row, 0)
        label = list(self.model.itemData(QModelIndex(ind)).values())[0]
        label = label.replace('[', '').replace(']', '').split()

        # xyMax=ast.literal_eval(xyMax.replace(' ', ','))

        # ind = self.model.index(row, 3)
        # kk=self.model.(index=ind)

        # ind = self.model.index(row, 4)
        # xyMin= list(self.model.itemData(QModelIndex(ind)).values())[0]
        # xyMin=ast.literal_eval(xyMin.replace(' ', ','))

        classes= self.cfg['category'].split(',')

        name=str(image.split('\\')[-1])
        image=cv2.imread(image)

        for i in range(int(len(xyMax) / 2)):
            x1=int(float(xyMin[i+1 ]))
            y1=int(float(xyMin[i]))
            x2=int(float(xyMax[i+1]))
            y2=int(float(xyMax[i]))


            start_point= (x1,y1)
            end_point = (x2,y2)
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(image, '{}-{}-({})'.format(i+1,list(self.category_index.keys())[i],label[i]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12),1)
            image = cv2.rectangle(image, start_point, end_point, color, thickness)






        # for box in (xyMax) :
        #     print('xyMax', box)






        # start_point = (5, 5)
        # end_point = (220, 220)
        # color = (255, 0, 0)
        # thickness = 2
        # image = cv2.rectangle(image, start_point, end_point, color, thickness)

        cv2.imshow(name, image)
        cv2.waitKey(4000)
        cv2.destroyAllWindows()
        # print(index.row() ,list(self.model.itemData(QModelIndex(ind)).values())[0])



        # util.Qlogging(self.ui.textBrowser,'image: {},{}'.format(index.data()))





        # self.ui.textBrowser

if __name__ == '__main__' :
    App= QApplication(sys.argv)
    window= Appwindow()
    window.show()
    sys.exit(App.exec())

