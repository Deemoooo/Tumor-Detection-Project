import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import *
from torchvision import utils,transforms,models
import torch
from PIL import Image
import numpy as np

class MyApp(QWidget):

	def __init__(self,args):
		super(MyApp, self).__init__()
		self.title = 'Patch Camelyon Project'
		self.args = args
		self.initUI()

	def initUI(self):
		self.setWindowTitle(self.title)
		self.setGeometry(50,50,800,600)

		self.vbox = QVBoxLayout()
		self.main_page()
		self.setLayout(self.vbox)
		self.show()

	def main_page(self):
		self.choose_button = QPushButton('Choose Image')
		self.choose_button.clicked.connect(self.openFileDirectory)
		self.choose_button.resize(self.choose_button.minimumSizeHint())

		self.intro = QLabel(self)
		self.intro.setText('Please select the image to predict.')
		self.intro.setFont(QFont('Helvetica',20))
		self.intro.setAlignment(Qt.AlignCenter)

		self.vbox.addWidget(self.intro)
		self.vbox.addStretch()
		self.vbox.addWidget(self.choose_button)


	def openFileDirectory(self):		
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Image Files (*.jpg *.png)", options=options)
		if fileName == '':
			self.intro.setText('Please select something!')
			return

		self.clearWidgets(self.vbox)
		device = torch.device('cpu')

		if fileName is not None:

			with torch.no_grad():
				
				self.img = QPixmap(fileName)
				self.img = self.img.scaled(400,400)

				model = models.resnet18(pretrained=True)
				model._modules['fc'] = torch.nn.Linear(in_features=512, out_features=2, bias=True) # 2048 for inception
				model = model.to(device)
				model.load_state_dict(torch.load('resnet_epoch_130.8466796875',map_location='cpu'))
				model.eval()

				trans = transforms.Compose([
					transforms.CenterCrop(40),
					transforms.Resize(224),
					transforms.ToTensor(),
					transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

				image = self.load_image(fileName,trans).float().to(device)
				pred = model(image)
				pred = pred.argmax().item()

				if pred == 0:
					pred = 'No Tumor'
				elif pred == 1:
					pred ='Contains Tumor!'

				self.choose_next = QPushButton('Choose Next Picture')
				self.choose_next.clicked.connect(self.openFileDirectory)
				self.choose_next.resize(self.choose_next.minimumSizeHint())

				self.img_label = QLabel(self)
				self.img_label.setPixmap(self.img)
				self.img_label.setAlignment(Qt.AlignCenter)

				self.prediction = QLabel(self)
				self.prediction.setText(pred)
				self.prediction.setAlignment(Qt.AlignCenter)

				self.vbox.addWidget(self.img_label)
				self.vbox.addWidget(self.prediction)
				self.vbox.addStretch()
				self.vbox.addWidget(self.choose_next)

			# except Exception as e:
			# 	print(e)


	def clearWidgets(self, layout):
		for i in reversed(range(layout.count())):
			try:
				toRemove = layout.itemAt(i).widget()
				layout.removeWidget(toRemove)
				toRemove.setParent(None)
			except:
				pass

	def load_image(self,fileName,trans):
		image = Image.open(fileName)
		if trans is not None:
			image = trans(image)
		return image.unsqueeze(0)


		
if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = MyApp('args')
	sys.exit(app.exec_())