# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:43:13 2019

@author: hstov
"""

from PyQt5 import QtGui, QtCore
import sys

import ImageViewerUI
 
class ImageViewer(QtGui.QMainWindow, ImageViewerUI.Ui_mainWindow):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.setupUi(self)
 
    def main(self):
        self.show()
 
if __name__=='__main__':
    app = QtGui.QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.main()
    app.exec_()