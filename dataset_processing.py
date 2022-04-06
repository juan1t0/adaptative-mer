import os
import re
from os.path import join, exists
import argparse
import json

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa as lib
from pydub import AudioSegment

from PIL import Image
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip as extract


def getting_csvfiles(args):
	info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)
	start_times, end_times, file_names, emotions, vals, acts, doms = [], [], [], [], [], [], []
	for sess in args.sessions:
		emo_evaluation_dir = (args.dataset_path).format(sess)
		evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
		for file in evaluation_files:
			if file[0] == '.':
				continue
			with open(join(emo_evaluation_dir, file)) as f:
				content = f.read()
			info_lines = re.findall(info_line, content)

			for line in info_lines[1:]:
				start_end_time, file_name, emotion, val_act_dom = line.strip().split('\t')
				start_time, end_time = start_end_time[1:-1].split('-')
				start_time, end_time = float(start_time), float(end_time)
				val, act, dom = val_act_dom[1:-1].split('-')
				val, act, dom = float(val), float(act), float(dom)
				start_times += [start_time]
				end_times += [end_time]
				file_names += [file_name]
				emotions += [emotion]
				vals += [val]
				acts += [act]
				doms += [dom]

	df_iemocap = pd.DataFrame(columns=['start_time','end_time', 'file_name',
									'emotion', 'val', 'act', 'dom'])
	df_iemocap['start_time'] = start_times
	df_iemocap['end_time'] = end_times
	df_iemocap['file_name'] = file_names
	df_iemocap['emotion'] = emotions
	df_iemocap['val'] = vals
	df_iemocap['act'] = acts
	df_iemocap['dom'] = doms

	df_iemocap.to_csv(args.savename, index=False, header=True)

def getting_everyframe(args):
	df_iemocap = pd.read_csv(args.csvfile)
	os.system('mkdir IEMOCAP_subclips')
	os.system('mkdir temp')

	folder_save = 'IEMOCAP_subclip'
	temporal_folder = 'temp'
	for sess in args.sessions:
		videos_path = (args.dataset_path).format(sess)
		video_fails = []
		for i, row in df_iemocap.iterrows():
			file = row['file_name'][:-5] +'.avi'
			start = row['start_time'] #+2) /100
			end = row['end_time'] #+2) /100
			extract(join(videos_path, file), start, end, targetname=join(temporal_folder, 'tem.avi'))
			cap = cv2.VideoCapture(join(temporal_folder, 'temp.avi'))
			frames = []
			while True:
				ret, frame = cap.read()
				if not ret:
					break
				frames += [frame[150:275, 125:250, :]] # cuttin noise from frames
			if len(frames) == 0:
				video_fails += [i, file]
				continue
			np.save(join(folder_save, row['file_name'] +'.npy'),
			 				np.stack(frames))
		with open('wrong_videos_S{}.json'.format(sess), 'w') as fj:
			js = json.dumps(video_fails)
			json.dump(js, fj)

def reducing_n_frames(args):
	os.system('mkdir {}'.format(args.savename))
	n_frames = args.nframes
	for sess in args.sessions:
		videonames = os.listdir((args.dataset_path).format(sess))
		for i, file in enumerate(videonames):
			arr = np.load(join((args.dataset_path).format(sess), file))
			step = arr.shape[0] / n_frames
			selects = [int(i*step) for i in range(0,n_frames)]
			new_arr = np.take(arr, selects, 0)
			np.save(join((args.savename), file), new_arr)

def merging_videoframes(args):
	os.system('mkdir {}'.format(args.savename))
	for i, clipname in enumerate(os.listdir(args.dataset_path)):
		clip = np.load(join(args.dataset_path, clipname))
		for j, frame in enumerate(clip):
			np.save(join(args.savename, str(j+1)+clipname), frame)

def getting_spectograms(args):
	df_iemocap = pd.read_csv(args.csvfile)
	# audios_path = (args.dataset_path).format(sess)
	for i, row in df_iemocap.iterrows():
		file = row['file_name'][:-5] + '.wav'
		start = int(row['start_time']*1000)
		end = int(row['end_time']*1000)
		sess = int(file[4])
		newAudio = AudioSegment.from_wav(join(args.dataset_path).format(int(file[4])), file)
		newAudio = newAudio[start:end]
		samples = newAudio.get_array_of_samples()
		samples = np.array(samples).astype(np.float32)
		mfccs = np.mean(lib.feature.mfcc(y=samples, sr=44100, n_mfcc=40).T axis=0)
		np.save(join(args.savename, row['file_name']+'.npy'), mfccs)

def scale_minmax(X, min=0.0, max=1.0):
	X_std = (X - X.min()) / (X.max() - X.min())
	X_scaled = X_std * (max - min) + min
	return X_scaled

def getting_audio_images(args):
	os.system('mkdir {}'.format(args.savename))
	for j, n in enumerate(os.listdir(args.dataset_path)):
		smel = np.load(join(args.dataset_path, n))
		img = scale_minmax(smel, 0, 255).astype(np.uint8)
		img = np.flip(img, axis=0) # put low frequencies at the bottom in image
		img = 255-img # invert. make black==more energy
		im = Image.fromarray(img)
		im.save(join(file2save.format(i), n[:-3] + 'jpg'))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--option', default= 'getcsv')
	parser.add_argument('--sessions', nargs='+', help='put a list separated by spaces')
	parser.add_argument('--dataset_path', type=str)
	parser.add_argument('--savename', type=str)
	parser.add_argument('--csvfile', type=str)
	parser.add_argument('--nframes', type=int, default=8)

	args = parser.parse_args()

	match args.option:
		case 'getcsv':
			getting_csvfiles(args)
		case 'getframes':
			getting_everyframe(args)
		case 'reduceframes':
			reducing_n_frames(args)
		case 'mergevideos':
			merging_videoframes(args)
		case 'getspectograms':
			getting_spectograms(args)
		case 'getaudioimage':
			getting_audio_images(args)


