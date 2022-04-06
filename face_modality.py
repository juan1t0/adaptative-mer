import os
from os.path import exists,join
import random

import pandas as pd
import numpy as np
import cv2
import json

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

def my_collate(batch):
	batch = filter(lambda data: data is not None, batch)
	return default_collate(list(batch))

Classes = {'exc':0, 'neu':1, 'sad':2, 'hap':0, 'ang':3}

