#!/usr/bin/env python
# coding: utf-8


import numpy as np 
import os 

def load_data(path):
	data = np.load(path)
	X = data['EEG']
	y = data['labels']
	return X, y


def cut_data(X, y, length=500):
	X_splits = np.split(X[:len(X)-len(X)%length], len(X) // length)
	y_splits = np.split(y[:len(y)-len(y)%length], len(y) // length)
	return np.array(X_splits), np.array(y_splits)


SEQ_LEN = 500 
PATH = '/mnt/ds3lab-scratch/veichta/Datasets/EEG/ICML_participant_streams_thresh_150_seqlen_4000_margin_2/' 
SAVE_PATH = PATH + f'tensors/'


for SET in ['train', 'val', 'test']:

	LOAD_PATH = PATH + SET + '/'
	X_all, y_all = [], []

	for file in os.listdir(LOAD_PATH):
		X, y = load_data(LOAD_PATH + file)
		#print(X.shape, y.shape)
		X_split, y_split = cut_data(X, y, SEQ_LEN)
		#print(X_split.shape, y_split.shape) 
		X_all.append(X_split)
		y_all.append(y_split)


	X_final = np.concatenate(X_all)
	y_final = np.concatenate(y_all)
	print(f"{SET} shapes: ", X_final.shape, y_final.shape)
	np.savez(SAVE_PATH + f'{SET}_{SEQ_LEN}.npz', EEG=X_final, labels=y_final)