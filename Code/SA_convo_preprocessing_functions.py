#!/usr/bin/env python

#convo_preprocessing_functions.py
#created by Sophie Wohltjen, 4/02/21
#These are all the functions I need to preprocess conversation pupil data

#Step 1 -- import everything I'll maybe need
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import medfilt, butter, lfilter, detrend, decimate, resample
from scipy.stats import binned_statistic, zscore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

#should we throw out the subject?
def datacheck(data,threshold='zscore'):
	if threshold == 'zscore':
		ind = np.array(np.where(zscore(data) <= -2)).flatten()#find samples recorded during blinks (2 SDs below mean)
		dataremoved = float(len(ind)) / len(data) * 100 #calculate percentage of data that is blink or noise
		if ind.size == 0:
			ind = np.array(np.where(data == 0)).flatten() #find samples recorded during blinks (data equals zero)
			dataremoved = float(len(ind)) / len(data) * 100
	elif threshold == '0':
		ind = np.array(np.where(data == 0)).flatten() #find samples recorded during blinks (data equals zero)
		dataremoved = float(len(ind)) / len(data) * 100 #calculate percentage of data removed 
	return dataremoved
	
#align pupil data and/or engagement ratings in time

	
##winsize is the size of the windows you want, in samples
##stepsize is how much overlap you want 
##(so like a stepsize of 0.5 would 'step' halfway across the first window to make the next window)
def window_stack(a, stepsize=1, winsize=10,pad_type='median'):
    #subtract the remainder of dividing the data evenly into windows 
    #to get an evenly divisible number for the data to be windowed
    padsize = winsize - np.mod(len(a),(winsize))
    a = np.pad(a, (0,padsize), pad_type)
    width=int(len(a))
    n = a.shape[0]
    return np.vstack( a[i:i+(winsize):1] for i in range(0,width,int(winsize*stepsize)))
	
#linearly interpolate
#accepts a data matrix, "val" is the zscore threshold the function uses to find bad data
#n_samples is the number of samples on either side of the noise that the function will use to interpolate
#padding is the number of samples interpolated on either side of the noise to avoid interpolation spikes
def lin_interpolate(data,threshold='zscore',val=-2,padding=5,n_samples=10):
	if threshold == 'zscore':
		if val < 0:
			ind = np.array(np.where(zscore(data) <= val)).flatten()#find samples recorded during blinks (2 SDs below mean, unless val is different)
		elif val > 0:
			ind = np.array(np.where(zscore(data) >= val)).flatten()#find samples recorded during blinks (2 SDs above mean, unless val is different)
		blink_ind = np.split(ind, np.where(np.diff(ind) > 15)[0]+1)#split indexed samples into groups of blinks
		data_noEB = np.copy(data) #copy data to interpolate over
	elif threshold == '0':
		ind = np.array(np.where(data == 0)).flatten()#find samples recorded during blinks (data equals zero)
		blink_ind = np.split(ind, np.where(np.diff(ind) > 15)[0]+1) #split indexed samples into groups of blinks
		data_noEB = np.copy(data) #copy data to interpolate over
    #loop through each group of blinks
	for blinks in blink_ind:
		if blinks.size == 0:
			continue
        #create a vector of data and sample numbers before and after the blink
		befores = np.arange((blinks[0] - (n_samples+padding)),(blinks[0]-padding))
		afters = np.arange(blinks[-1]+(1+padding),blinks[-1]+(1+n_samples+padding))
        #this if statement is a contingency for when the blinks occur at the end of the dataset. it deletes the blink rather than interpolating
		if any(afters > len(data)-1):
			data_noEB = data_noEB[0:blinks[0]-1]
		else:
            #this is the actual interpolation part. you create your model dataset to interpolate over
			x = np.append(befores,afters)
			y = np.append(data[befores],data[afters])
            #then interpolate it
			li = interp1d(x,y)
            #create indices for the interpolated data, so you can return it to the right segment of the data
			xs = range(blinks[0]-padding,blinks[-1]+(1+padding))
            #I'm actually not sure that you need these two variables anymore, but they're still in here for some reason.
			x_stitch = np.concatenate((x[0:n_samples],xs,x[n_samples:]))
			y_stitch = np.concatenate((y[0:n_samples],li(xs),y[n_samples:]))
            #put the interpolated vector into the data
			np.put(data_noEB,xs,li(xs))
	return data_noEB

#median filter 
#(kind of redundant, but I put it in so I don't have to import a bunch of disparate things in my actual preprocessing script)
def median_filt(data,filtsize=5):
    #this is a filtering function I got from the package "scipy"
	data_filt = medfilt(data, filtsize) 
	return data_filt

#lowpass butterworth filtering function
def butter_lowpass(cutoff, fs, order=5):
    #get nyquist frequency
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    #this is another filtering function from scipy
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff=10, fs=60, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #and another filtering function from scipy
    data_lp10 = lfilter(b, a, data)
    return data_lp10
    
#resampling function for Pupil Labs data
def pl_resample(data,old_fs=125,new_fs=1500,smi_fs=60):
	interp_vals = np.round(np.arange(0,len(data),1/(new_fs*smi_fs/old_fs)),decimals=3)
	data_interp = np.interp(interp_vals,np.arange(0,len(data),1),data)
	data_ds = resample(data_interp,int(len(data_interp)/new_fs))
	return data_ds



#detrend
#(again kind of redundant but makes things simpler)
def detrending(data):
	data_dt = detrend(data)
	return data_dt


  
