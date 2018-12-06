from PyQt5 import QtWidgets, uic, QtCore, QtGui
import sys
import cv2 as cv
from copy import copy
import numpy as np

form_class = uic.loadUiType("camera.ui")[0]

def form_frame(cam, queue, width, height, fps):
    
    pass

def contrast_pix(arr):
#    pix = np.zeros((256,), dtype=np.float)
    for i in range(256):
        yield np.sum(arr[0:i])
        
def contrast_norm(arr1, arr2):
    for j in arr2:
        for i in j:
            yield arr1[i]

class OwnImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()
        
class RangeSliderWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__()
        
        self.setMinimumSize(1, 30)
        self.value = 75
        
    def setValue(self, value):

        self.value = value


    def paintEvent(self, e):
      
        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawWidget(qp)
        qp.end()
      
      
    def drawWidget(self, qp):
        
        MAX_CAPACITY = 700
        OVER_CAPACITY = 750
      
        font = QtGui.QFont('Serif', 7, QtGui.QFont.Light)
        qp.setFont(font)

        size = self.size()
        w = size.width()
        h = size.height()

        step = int(round(w / 10))


        till = int(((w / OVER_CAPACITY) * self.value))
        full = int(((w / OVER_CAPACITY) * MAX_CAPACITY))
        
        with QtGui.QColor as QColor:
            if self.value >= MAX_CAPACITY:
                
                qp.setPen(QColor(255, 255, 255))
                qp.setBrush(QColor(255, 255, 184))
                qp.drawRect(0, 0, full, h)
                qp.setPen(QColor(255, 175, 175))
                qp.setBrush(QColor(255, 175, 175))
                qp.drawRect(full, 0, till-full, h)
                
            else:
                
                qp.setPen(QColor(255, 255, 255))
                qp.setBrush(QColor(255, 255, 184))
                qp.drawRect(0, 0, till, h)


            pen = QtGui.QPen(QColor(20, 20, 20), 1, 
                QtGui.SolidLine)
            
        qp.setPen(pen)
        qp.setBrush(QtGui.NoBrush)
        qp.drawRect(0, 0, w-1, h-1)

        j = 0

        for i in range(step, 10*step, step):
          
            qp.drawLine(i, 0, i, 5)
            metrics = qp.fontMetrics()
            fw = metrics.width(str(self.num[j]))
            qp.drawText(i-fw/2, h/2, str(self.num[j]))
            j = j + 1
        
class MyWindowClass(QtWidgets.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        
        self.factor = 1
        self.cap = cv.VideoCapture(0)
        
        self.phiRight = 1.0
        self.thetaRight = 1.0
        
        self.rightRangeWidget = RangeSliderWidget()
        
        self.window_width = self.DynamicWidget.frameSize().width()
        self.window_height = self.DynamicWidget.frameSize().height()
        self.DynamicWidget = OwnImageWidget(self.DynamicWidget)    
        self.StaticWidget = OwnImageWidget(self.StaticWidget)      
        
        self.contrastRightSlider.valueChanged.connect(self.contrastRightChange)
        self.brightnessRightSlider.valueChanged.connect(self.brightnessRightChange)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)
        
    def contrastRightChange(self):
        self.phiRight = self.contrastRightSlider.value()
        self.factor = np.float((259 * (self.phiRight + 255)) / (255 * (259 - self.phiRight)))
        print('__________________', self.phiRight)
        print(self.contrastRightSlider.value())
        
    def brightnessRightChange(self):
        self.thetaRight = int(self.brightnessRightSlider.value())
        print(self.brightnessRightSlider.value())

    def start_clicked(self):
        global cv_running
        
        if cv_running == False:
            cv_running = True
            self.startButton.setText('stop')
        else:
            cv_running = False
            self.startButton.setText('start')
            


    def update_frame(self):
        
        ret, img = self.cap.read()
        
        img_height, img_width, img_colors = img.shape
        scale_w = float(self.window_width) / float(img_width)
        scale_h = float(self.window_height) / float(img_height)
        #            print(scale_w, scale_h)
        scale = min([scale_w, scale_h])
        #            print(scale)
        
        if scale == 0:
            scale = 1
        
        img = cv.resize(img, None, fx=scale, fy=scale, interpolation = cv.INTER_CUBIC)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        
        norm_hist = hist/max(hist)
        
        pix = np.fromiter(contrast_pix(norm_hist), dtype=np.float)
        pix = (255/max(pix)) * pix
#        
#        for i in range(len(img)):
#            for j in range(len(img[i])):
#                img1[i, j] = pix[img[i, j]]
        
        
        img0 = np.fromiter(contrast_norm(pix, img), dtype=np.uint8).reshape(img.shape)
        
        img0 = np.array(np.clip(np.array(img0, dtype=np.uint16) + self.thetaRight, 0, 255), dtype=np.uint8)


        img0 = np.array(np.clip(np.array(self.factor * (np.array(img0, dtype=np.float) - 128), dtype=np.float) + 128, 0, 255), dtype=np.uint8)
#        img0 = cv.fastNlMeansDenoising(img0)        
        
        
        
        
        image = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], QtGui.QImage.Format_Grayscale8)
#        image = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], QtGui.QImage.Format_RGB888)
        image1 = QtGui.QImage(img0, img0.shape[1], img0.shape[0], img0.strides[0], QtGui.QImage.Format_Grayscale8)
#        image1 = QtGui.QImage(img0, img0.shape[1], img0.shape[0], img0.strides[0], QtGui.QImage.Format_RGB888)
        
        self.DynamicWidget.setImage(image)
        self.StaticWidget.setImage(image1)

    def closeEvent(self, event):
        global running
        running = False
        
        

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MyWindowClass()
    w.setWindowTitle('PyQT OpenCV USB camera test panel')
    w.show()
    
    quit()
    sys.exit(app.exec())
    
#     pix = np.array( [ sum([norm_hist[j] for j in range(i)]) for i in range(255) ] )
     
