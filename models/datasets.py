import os
from os.path import exists, join
import pandas as pd
import json

import torch
from torch.utils.data import Dataset

n_frames= 8

class continuous_IEMOCAP(Dataset):
	def __init__(self, roots='', csv_s='', fails_f= '',
					classes={}, mode='', sess=1, transform=None):
		super(continuous_IEMOCAP, self).__init__()
		self.DataRoots = roots
		self.AnnotationFiles = csv_s
		self.FailFiles = fails_f
		self.Transform = transform
		self.Mode = mode
		self.Classes = classes

		self.load_data(sess)

	def load_data(self, sess):
		self.Annotations = pd.read_csv(self.AnnotationFiles[sess])
		with open(self.FailFiles[sess], 'r') as jsfile:
			self.Fails = json.loads(jsfile.read())['list']

		self.Data = {}
		file_names = os.listdir(self.DataRoots[sess])
		for 1, row in self.Annotations.iterrows():
			if i in self.Fails:
				continue
			if row['emotion'] not in list(self.Classes.keys()):
				continue
			for j in range(n_frames):
				t_name = str(j+1) +row['file_name'] +'.npy'
				self.Data[true_name]=row['emotion']
