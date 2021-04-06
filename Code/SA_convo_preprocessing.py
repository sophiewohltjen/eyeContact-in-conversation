#!/usr/bin/env python

#SA_convo_preprocessing.py
#created by Sophie Wohltjen, 4/02/21
#This is the wrapper script in which I call the functions contained in convo_preprocessing_functions.py

from SA_convo_preprocessing_functions import datacheck, lin_interpolate, median_filt, butter_lowpass_filter, detrending,pl_resample
import os
import glob

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

base_directory = '/Users/sophie/Dropbox/EyeContactinConversation'
#get the raw data files using a global variable. 
rawfiles = glob.glob("{0}/Data/raw_pupils/*/*.txt".format(base_directory))

#loop through the raw data files that I want to preprocess
for file in sorted(rawfiles):
	print("preprocessing data from {0}".format(file))
	#read data file into a pandas dataframe
	alldata = pd.read_csv(file)
	#read into a numpy array the variable from the dataframe that we care about
	if file[-6:-4] == 'pl':
		data = np.array(alldata['diameter'])
	else:
		data = np.delete(np.array(alldata['Pupil Diameter Left [mm]']),np.where(np.array(alldata['Pupil Diameter Left [mm]']) == '-'))
		data = np.ndarray.astype(data,float)
	
	#output a percentage of data that will need to be removed
	dataremoved = datacheck(data,threshold='0')
        
    #check whether the data exceeds 25% threshold
	if dataremoved > 25:
		data_right = np.delete(np.array(alldata['Pupil Diameter Right [mm]']),np.where(np.array(alldata['Pupil Diameter Right [mm]']) == '-'))
		data_right = np.ndarray.astype(data_right,float)
		dataremoved_right = datacheck(data_right,threshold='0')
		if dataremoved_right > 25:
        	#print the amount of data that will need to be removed
			print(dataremoved)
        	#prompt user for decision on whether to keep preprocessing or to skip the subject
			skip = input('This data is messy. should we keep preprocessing this subject? Type 1 for yes and 0 for no: ')
			if skip == '0':
				break
		else:
			data = dataremoved_right
	
	#linearly interpolate eye blinks and dropout 
	
	data_noEB = lin_interpolate(data,threshold='0',val=-2,padding=5,n_samples=10)
		
	#median filter data
	data_noEB_filt = median_filt(data_noEB,filtsize=5)
	
	#lowpass filter and detrend data (different steps for pupil labs vs smi)
	if file[-6:-4] == 'pl':
		#lowpass filter
		data_noEB_filt_10 = butter_lowpass_filter(data_noEB_filt, cutoff=10, fs=125, order=5)
		#now downsample the pupil labs data to match SMI
		print("resampling pupil labs data! This can take a while.")
		data_noEB_filt_10_ds = pl_resample(data_noEB_filt_10)
		#convert to mm to match SMI
		pixel_to_mm = 25.4/220
		data_noEB_filt_10_ds_mm = data_noEB_filt_10_ds * pixel_to_mm
		#detrend
		data_noEB_filt_10_ds_mm_dt = detrending(data_noEB_filt_10_ds_mm)
	else:
		#lowpass filter
		data_noEB_filt_10 = butter_lowpass_filter(data_noEB_filt, cutoff=10, fs=60, order=5)
		#detrend
		data_noEB_filt_10_dt = detrending(data_noEB_filt_10)

	
	if file[-6:-4] == 'pl':
		data2 = data_noEB_filt_10_ds_mm_dt[5:]
	else:
		data2 = data_noEB_filt_10_dt[5:]
		
	#Does the data need extra spike removal?
	fig, ax = plt.subplots(figsize=(15,3))
	plt.plot(data2)
	plt.show()
		
	clean = input('Does this data look alright? Type 1 for yes and 0 for no: ')
	
	
	while clean == '0':
		
		#what kind of spikes are we removing
		val = int(input('please type 3 to remove upper spikes and -3 to remove lower spikes'))
		
		#interpolate again
		data2 = lin_interpolate(data2,threshold='zscore',val=val,padding=5,n_samples=10)
		
		#subjects I055,I095, and I114 have REALLY pesky spikes, so we de-trend again to make sure we can get them
		if file[file.find('D0')+5:file.find('D0')+9] in ['I055','I095','I114']:
			data2 = detrending(data2)
		
		fig, ax = plt.subplots(figsize=(15,3))
		plt.plot(data2)
		plt.show()	
		#does it look good now?
		clean = input('Does this data look alright? Type 1 for yes and 0 for no: ')
	
	#save the file
	savedir="{0}/Analyses/preprocessed_pupils".format(base_directory)
	partner = file[file.find('D0')-2:file.find('D0')-1]
	dyad = file[file.find('D0'):file.find('D0')+4]
	subject = file[file.find('D0')+5:file.find('D0')+9]
	
	if file[-6:-4] == 'pl':
		np.savetxt("{0}/partner{1}/{2}_{3}_noEB_filt_10_ds_mm_dt.txt".format(savedir,partner,dyad,subject),data2)
	else:
		np.savetxt("{0}/partner{1}/{2}_{3}_noEB_filt_10_dt.txt".format(savedir,partner,dyad,subject),data2)

