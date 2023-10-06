import sys
from PyQt5.QtCore import Qt, QPoint, QPoint, QRect, QTimer, pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QMainWindow, QMenu, qApp, QMessageBox
import numpy as np
from PyQt5.QtGui import QPixmap, QPainter, QPen, QPixmap, QPainter, QPen, QKeySequence, QImage

class Menu(QMainWindow):

	def __init__(self):
		super().__init__()
		self.drawing = False
		self.lastPoint = QPoint()
		self._menu = QMenu()
		self._menu.addAction("&Quit", qApp.quit, QKeySequence.Quit)
		QTimer.singleShot(0, self.setProfile)
	
	@pyqtSlot()
	def setProfile(self):
		if QMessageBox.question(self, "Quit?", "Quit?") != QMessageBox.No:
			qApp.quit()
		self.hide()

	
class DrawingGraphics(QWidget):

	def __init__(self, numpy_array):
		super().__init__()
		self.setGeometry(0, 0, 1280, 800)
		qimage = QImage(numpy_array.data, numpy_array.shape[1], numpy_array.shape[0], 
                    QImage.Format_RGB888)
		self.pixmap = QPixmap.fromImage(qimage)
		self.resize(self.pixmap.width(), self.pixmap.height())
		self.show()
		self.setWindowTitle("Draw box(es) and/or point(s), close the window to continue the workflow")
		layout = QVBoxLayout()
		self.setLayout(layout)
		self.begin, self.destination = QPoint(), QPoint()
		self.boxes = []
		self.point = []	

	def paintEvent(self, event):
		painter = QPainter(self)
		painter.drawPixmap(self.rect(), self.pixmap)
		pen = QPen(Qt.red, 3)
		painter.setPen(pen)

		if not self.begin.isNull() and not self.destination.isNull():
			rect = QRect(self.begin, self.destination)
			painter.drawRect(rect.normalized())
	
	def mousePressEvent(self, event):
		if event.buttons() & Qt.LeftButton:
			
			self.begin = event.pos()
			self.destination = self.begin
			self.update()

	def mouseMoveEvent(self, event):
		if event.buttons() & Qt.LeftButton:		
			
			self.destination = event.pos()
			self.update()

	def mouseReleaseEvent(self, event):
	
		if event.button() & Qt.LeftButton:
			rect = QRect(self.begin, self.destination)
			painter = QPainter(self.pixmap)
			pen = QPen(Qt.red, 3)
			painter.setPen(pen)
			painter.drawRect(rect.normalized())
			self.begin, self.destination = QPoint(), QPoint()
			self.update()
			# Get the coordinates 
			x1 = rect.normalized().x()
			y1 = rect.normalized().y()
			x2 = rect.normalized().x() + rect.normalized().width()
			y2 = rect.normalized().y() + rect.normalized().height()
			if abs(x2-x1) and abs(y1-y2) == 1:
				print("Point selected")
				self.point = [x1, y1]
			else:
				print("Box selected")
				self.boxes.append([x1, y1, x2, y2])
    
	def get_coordinates(self):
		return self.boxes, self.point

if __name__ == '__main__':
	# don't auto scale when drag app to a different monitor.
	# QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
	
	app = QApplication(sys.argv)
	app.setStyleSheet('''
		QWidget {
			font-size: 30px;
		}
	''')
	
	myApp = DrawingGraphics()
	myApp.show()
	try:
		sys.exit(app.exec_())
	except SystemExit:
		print('Closing Window...')
