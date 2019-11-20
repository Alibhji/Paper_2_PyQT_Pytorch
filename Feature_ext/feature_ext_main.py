

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
        col_name = list(data_pd.columns)
        data_pd['area'] = data_pd.apply(
            lambda x: ((x[col_name[3]] - x[col_name[4]])[:, 1] * (x[col_name[3]] - x[col_name[4]])[:, 0]), axis=1)

        data_pd['center_point'] = data_pd.apply(
            lambda x: ((x[col_name[4]])+(x[col_name[3]]))/2.0, axis=1)
        # data_pd['area'] = data_pd.apply(util.calculate_area, axis=1)
        # print(data_pd[''])



        self.model = util.PandasModel(data_pd)
        self.ui.tableView.setModel(self.model)
        
        # util.Qlogging(self.ui.textBrowser,str(data_pd))

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
        xyMax = [int(float(i)) for i in xyMax]
        xyMax=[tuple(xyMax[i + 0:2 + i]) for i in range(0, len(xyMax), 2)]


        ind = self.model.index(row, 4)
        xyMin = list(self.model.itemData(QModelIndex(ind)).values())[0]
        xyMin = xyMin.replace('[', '').replace(']', '').split()
        xyMin = [int(float(i)) for i in xyMin ]
        xyMin = [tuple(xyMin[i + 0:2 + i]) for i in range(0, len(xyMin), 2)]

        ind = self.model.index(row, 7)
        center = list(self.model.itemData(QModelIndex(ind)).values())[0]
        center = center.replace('[', '').replace(']', '').split()
        center = [int(float(i)) for i in center ]
        center = [tuple(center[i + 0:2 + i]) for i in range(0, len(center), 2)]
        print(center)

        classes=self.cfg['category'].split(',')
        ind = self.model.index(row, 0)
        label = list(self.model.itemData(QModelIndex(ind)).values())[0]
        label = label.replace('[', '').replace(']', '').split()

        # xyMin=ast.literal_eval(xyMin.replace(' ', ','))

        classes= self.cfg['category'].split(',')

        name=str(image.split('\\')[-1])
        image=cv2.imread(image)

        objs=[]

        for i in range(len(xyMax)):
            y1, x1 = xyMin[i]
            y2, x2 = xyMax[i]
        #
        #
            print(i,'value: ',x1, y1 ,x2, y2 )
        #
        #
            start_point= (x1,y1)
            end_point = (x2,y2)
            color = (255, 0, 0)
            thickness = 2
            objs.append("{}-{}".format(i+1,classes[int(label[i])]))
            cv2.putText(image, '{}-{}({})'.format(i+1,classes[int(label[i])],label[i]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255),1)
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            image=cv2.circle(image, (center[i][1],center[i][0]), 2, (0, 0, 255), -1)

        util.Qlogging(self.ui.textBrowser, '{} has {} objects: \\n {}'.format(name,len(xyMax),objs),'red' )

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

