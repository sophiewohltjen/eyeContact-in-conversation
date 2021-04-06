#!/usr/bin/env python

#SA_DTW_timeseries.py
#created by Sophie Wohltjen, 4/02/21

#Here I take each dyad's preprocessed pupil files and compute DTW on them
#all dyads' synchrony is then outputted as a single csv, and all subjects' pupils are outputted as a single csv
#I output:
	#1. dyads' pupillary timeseries
	#2. dyads' synchrony timeseries
	
import os
import glob
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from SA_convo_preprocessing_functions import window_stack

base_directory='/Users/sophie/Dropbox/EyeContactinConversation'
	
datalen = 600
pupil_len = 36000
pupilwinsize = int(pupil_len / datalen)
dyadnum = 47
subjectnum = 94
	
os.chdir('{0}/Analyses/preprocessed_pupils/partner1/'.format(base_directory)
partner1 = glob.glob('*dt*.txt')

os.chdir('../partner2/')
partner2 = glob.glob('*dt*.txt')

pupils_p1 = np.zeros((dyadnum,pupil_len))
pupils_p2 = np.zeros((dyadnum,pupil_len))
pupils_p1_stack = np.zeros((dyadnum,datalen,pupilwinsize))
pupils_p2_stack = np.zeros((dyadnum,datalen,pupilwinsize))
allpupils_stack = np.zeros((subjectnum,datalen,pupilwinsize))
dyad = []
subject=[]

for num,(p1,p2) in enumerate(zip(sorted(partner1),sorted(partner2))):
    #save the dyad number
    dyad.append(p1[0:4])
    subject.append(p1[5:10])
    subject.append(p2[5:10])
    os.chdir('../partner1/')
    #read in the data
    p1data = np.genfromtxt('{0}'.format(p1),dtype='float', skip_header=1)
    #if the data is less than 10 minutes
    if len(p1data) < pupil_len:
        pupils_p1[num,0:len(p1data)] = p1data
    else:
        pupils_p1[num] = p1data[0:pupil_len]
    #stack it
    p1stacked = window_stack(pupils_p1[num],winsize=pupilwinsize)
    pupils_p1_stack[num] = p1stacked[0:datalen,0:pupilwinsize]
    allpupils_stack[num*2] = p1stacked[0:datalen,0:pupilwinsize]
    # do the same for partner 2
    os.chdir('../partner2/')
    p2data = np.genfromtxt('{0}'.format(p2),dtype='float',skip_header=1)
    if len(p2data) < pupil_len:
        pupils_p2[num,0:len(p2data)] = p2data
    else:
        pupils_p2[num] = p2data[0:pupil_len]
    p2stacked = window_stack(pupils_p2[num],winsize=pupilwinsize)
    pupils_p2_stack[num] = p2stacked[0:datalen,0:pupilwinsize]
    allpupils_stack[num*2+1] = p2stacked[0:datalen,0:pupilwinsize]
    

#preallocate a matrix of synchrony time series
dtw_matrix = np.zeros((dyadnum,datalen))

#loop through subjects in the stacked matrix & compute DTW
for subnum,(p1,p2) in enumerate(zip(pupils_p1_stack,pupils_p2_stack)):
    #loop through all stacks of data for each subject
    #this is collections of 60 data points per subject that correspond to seconds of their conversation
    for segnum,(p1_seg,p2_seg) in enumerate(zip(p1,p2)):
        #compute dynamic time warping. 
        #radius defines the additional number of cells besides the one at hand to be considered in the calc.
        #in this case, we're telling the algorithm to look up to .25 seconds away to find optimal costs. 
        distance,path = fastdtw(p1_seg,p2_seg,radius = 15)
        #save the distances into the final dtw matrix
        dtw_matrix[subnum,segnum] = distance
        
#save dtw timeseries 
dtw_long = np.reshape(dtw_matrix,(dyadnum*datalen))
dyad_long = np.repeat(dyad,datalen)
dtw_df = pd.DataFrame([dyad_long,dtw_long]).T
dtw_df.columns = ['dyad','dtw']
savedir = '{0}/Analyses'.format(base_directory)
dtw_df.to_csv('{0}/dyad_dtw_timeseries.csv'.format(savedir))

#save pupil timeseries 
pupil_long = np.mean(allpupils_stack,axis=2).reshape(subjectnum*datalen)
dyad_long = np.repeat(dyad,datalen*2)
sub_long = np.repeat(subject,datalen)
pupil_df = pd.DataFrame([dyad_long,sub_long,pupil_long]).T
pupil_df.columns = ['dyad','subject','psize']
savedir = '{0}/Analyses'.format(base_directory)
pupil_df.to_csv('{0}/dyad_subject_pupil_timeseries.csv'.format(savedir))
    
    