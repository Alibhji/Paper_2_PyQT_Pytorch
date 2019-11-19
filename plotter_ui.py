# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './plotter_ui.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(735, 446)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btn_open = QtWidgets.QPushButton(self.centralwidget)
        self.btn_open.setGeometry(QtCore.QRect(630, 10, 89, 25))
        self.btn_open.setObjectName("btn_open")
        self.in_file_address = QtWidgets.QLineEdit(self.centralwidget)
        self.in_file_address.setGeometry(QtCore.QRect(30, 10, 591, 25))
        self.in_file_address.setObjectName("in_file_address")
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(30, 40, 291, 361))
        self.listWidget.setObjectName("listWidget")
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        self.btn_Plot_all = QtWidgets.QPushButton(self.centralwidget)
        self.btn_Plot_all.setGeometry(QtCore.QRect(360, 90, 141, 25))
        self.btn_Plot_all.setObjectName("btn_Plot_all")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(370, 50, 186, 27))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lbl_filter = QtWidgets.QLabel(self.widget)
        self.lbl_filter.setObjectName("lbl_filter")
        self.horizontalLayout.addWidget(self.lbl_filter)
        self.in_filter = QtWidgets.QLineEdit(self.widget)
        self.in_filter.setObjectName("in_filter")
        self.horizontalLayout.addWidget(self.in_filter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 735, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_open.setText(_translate("MainWindow", "Open "))
        self.in_file_address.setText(_translate("MainWindow", "/home/ali/Alibhji/Paper_2_PyQT_Pytorch/designed_modules/All_Results.losses"))
        self.listWidget.setSortingEnabled(True)
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        item = self.listWidget.item(0)
        item.setText(_translate("MainWindow", "New Item"))
        item = self.listWidget.item(1)
        item.setText(_translate("MainWindow", "New Item"))
        item = self.listWidget.item(2)
        item.setText(_translate("MainWindow", "New Item"))
        item = self.listWidget.item(3)
        item.setText(_translate("MainWindow", "New Item"))
        self.listWidget.setSortingEnabled(__sortingEnabled)
        self.btn_Plot_all.setText(_translate("MainWindow", "Plot All Modules"))
        self.lbl_filter.setText(_translate("MainWindow", "Filter"))
        self.in_filter.setText(_translate("MainWindow", "002och"))
