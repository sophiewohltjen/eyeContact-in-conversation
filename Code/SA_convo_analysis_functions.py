#!/usr/bin/env python

#convo_analysis_functions.py
#created by Sophie Wohltjen, 4/02/21

#These are some functions I use on the preprocessed data files to create matrices of 
#my different variables of interest, for all dyads

import glob
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from SA_convo_preprocessing_functions import window_stack
from scipy.stats import zscore

#engagement ratings
def get_mean_engagement_matrix(ratingfiles,dyadnum=47,datalen=6000,fs=10,save=False):
	# preallocate a matrix
	meanratings = np.zeros((dyadnum,datalen))
	dyads_engage = []
	# get sorted list of all files
	ref_file = ratingfiles[0]
	ratingfiles.sort(key = lambda x:x[ref_file.find('D0'):ref_file.find('D0')+4]) #sort by the dyad 

	# now read them in, keeping a list of dyads
	for num,file in enumerate(ratingfiles):
		data = pd.read_csv('{0}'.format(file),header=None)
		if num % 2 != 0:
			meanratings[int(num/2 - 0.5)] = np.mean((oldyad,data[0][0:datalen]),axis=0)
			dyads_engage.append('{0}'.format(file[file.find('D0'):file.find('D0')+4]))
		if num % 2 == 0:
			oldyad = data[0][0:datalen]
	#downsample to 1Hz
	engage_mean_ds = np.zeros((dyadnum,int(datalen/fs)))
	for num,engage in enumerate(meanratings):
		engage_mean_ds[num] = engage[0::fs]
	
	if save:
		#make a dataframe    
		engage_mean_ds_melt = np.reshape(engage_mean_ds,int(dyadnum*(datalen/fs)))
		dyad_long = np.repeat(dyads_engage,int(datalen/fs))
		engage_mean_df = pd.DataFrame([dyad_long,engage_mean_ds_melt]).T
		engage_mean_df.columns = ['dyad','engage_mean']
		savedir = input('please enter the full directory where you want to save this file, as a string: ')
		engage_mean_df.to_csv('{0}/dyad_mean_engagement.csv'.format(savedir))
	else:
		return engage_mean_ds, dyads_engage
    	
#eye contact time series
#ts_type can be 'binary' or 'average'
def get_ec_matrix(ecmutualfiles,dyadnum=47,datalen=600,origlen=600000,save=False,ts_type='binary'):
	
	ecwinsize = int(origlen / datalen)

	mutualec = np.zeros((dyadnum,origlen))
	mutualec_stacked = np.zeros((dyadnum,datalen,ecwinsize))
	ecdyad = []
     
	# now read them in, keeping a list of dyads
	for num,file in enumerate(sorted(ecmutualfiles)):
		ecdyad.append(file[file.find('D0'):file.find('D0')+4])
		data = pd.read_csv('{0}'.format(file),header=None,delimiter=' ')
		ectimeseries = np.zeros(origlen)
		for start,stop in zip(data[0],data[1]):
			ectimeseries[start:stop] = 1
		mutualec[num] = ectimeseries
		#stack the file in appropriate windows COMMENT OUT IF YOU WANT PRE DS DATA
		ec_stacked = window_stack(ectimeseries[0:origlen],winsize=ecwinsize,pad_type='minimum')
		mutualec_stacked[num] = ec_stacked[0:datalen,0:ecwinsize]
	
	#downsample mutual eye contact
	if datalen < origlen:
		mutualec_ds = np.zeros((dyadnum,datalen))
		for num,ec in enumerate(mutualec):
			ec_ds = ec
			while len(ec_ds) > mutualec_ds.shape[1]:
				ec_ds = ec_ds[0::10]
			mutualec_ds[num] = ec_ds
	else:
		mutualec_ds = mutualec
	
	#window mutual eye contact	
	mutualec_windowed = np.mean(mutualec_stacked,axis=2)
			
	if save: 
		mutualec_ds_long = np.reshape(mutualec_ds,(dyadnum*datalen))
		mutualec_windowed_long = np.reshape(mutualec_windowed,(dyadnum*datalen))
		#make the csv
		dyad_long = np.repeat(ecdyad,datalen)
		ec_df = pd.DataFrame([dyad_long,mutualec_ds_long,mutualec_windowed_long]).T
		ec_df.columns = ['dyad','ec','ec_avg']
		savedir = input('please enter the full directory where you want to save this file, as a string: ')
		ec_df.to_csv('{0}/dyad_ec_timeseries.csv'.format(savedir))
		
	else:
		if ts_type == 'binary':
			return mutualec_ds
		elif ts_type == 'average':
			return mutualec_windowed
			
def get_individual_ec(indec_files,subnum=94,dyadnum=47,datalen = 600,origlen = 600000,savedir='',save=False):
	
	ecwinsize = int(origlen / datalen)
	allec = np.zeros((subnum,origlen))
	allec_stacked = np.zeros((subnum,datalen,ecwinsize))
	ecdyad = []
	ecsub = []
	
	#get the files into the proper format
	for num,ecfile in enumerate(sorted(indec_files)):
		#save the dyad to make sure we have the right subs
		ecdyad.append(ecfile[ecfile.find('D0'):ecfile.find('D0')+4])
		ecsub.append(ecfile[ecfile.find('D0')+5:ecfile.find('D0')+9])
		#read in the file
		ec = np.genfromtxt('{0}'.format(ecfile))
		allec[num] = ec[0:origlen]
		
	#downsample individual eye contact
	if datalen < origlen:
		indec_ds = np.zeros((subnum,datalen))
		for num,ec in enumerate(allec):
			ec_ds = ec
			while len(ec_ds) > indec_ds.shape[1]:
				ec_ds = ec_ds[0::10]
			indec_ds[num] = ec_ds
	else:
		indec_ds = allec
	
	if save: 
		indec_ds_long = np.reshape(indec_ds,(subnum*datalen))
		#make the csv
		dyad_long = np.repeat(ecdyad,datalen)
		sub_long = np.repeat(ecsub,datalen)
		ec_df = pd.DataFrame([dyad_long,sub_long,indec_ds_long]).T
		ec_df.columns = ['dyad','subject','ind_ec']
		savedir = savedir
		ec_df.to_csv('{0}/individual_ec_timeseries.csv'.format(savedir))
		
	else:
		return indec_ds
			
#eye contact instances (for ERP analysis)
def get_ec_events(ec_matrix,ec_short=1,ec_long=100,ec_fs=1000,p_fs=60):
	#preallocate matrix of all instances of eye contact for all dyads
	ec_all = []
	#preallocate matrix of all instances of no eye contact for all dyads
	none_all = []
	
	#loop through dyads in eye contact matrix
	for ec in ec_matrix:
		#find eye contact chunks
		#split data into chunks where data changes from 1 to 0 and vice versa
		#note that this will give you chunks of BOTH eye contact and no eye contact
		ec_chunks = np.split(ec, np.where(np.diff(ec) != 0)[0]+1)
		#find the indices of those chunks. np.insert is included to add a zero to the beginning of the data
		#since the first instance of eye contact or none happens at 0
		ec_inds = np.insert((np.where(np.diff(ec) != 0)[0]+1),0,0)
		#create empty lists of eye contact instances and no eye contact instances
		#at the subject level
		ec_sub = []
		none_sub = []
		
		#loop through chunks and indices for eye contact and no eye contact
		for num,(chunk,ind) in enumerate(zip(ec_chunks,ec_inds)):
			#if the chunk corresponds to an instance of eye contact
			if np.all(chunk) == 1:
				#if the chunk fits within our pre-specified range of eye contact lengths:
				if ec_short*ec_fs <= len(chunk) <= ec_long*ec_fs:
					#append the second where that eye contact occurred
					#divide by the ec sampling rate to get seconds, 
					#then multiply by pupil sampling rate to get pupil indices
					#WE PROBABLY DON'T NEED TO DO THIS. COME BACK AND FIX IT ONCE YOU'VE IMPLEMENTED ALL THE CODE
					ec_sub.append(ind/ec_fs*p_fs)
			
			# if the chunk corresponds to an instance of no eye contact        
			elif np.all(chunk) == 0:
				#if the chunk fits within our pre-specified range of eye contact lengths:
				if ec_short*ec_fs <= len(chunk) <= ec_long*ec_fs:
					#append the second where that eye contact occurred
					#divide by the ec sampling rate to get seconds,
					#then multiply by pupil sampling rate to get pupil indices
					#WE PROBABLY DON'T NEED TO DO THIS. COME BACK AND FIX IT ONCE YOU'VE IMPLEMENTED ALL THE CODE
					none_sub.append(ind/ec_fs*p_fs)
		
		#now append the subject specific list to the larger list
		ec_all.append(ec_sub)
		none_all.append(none_sub)
	
	return ec_all,none_all
    
#compute 1s synchrony time series 
def get_dtw_timeseries(pupilfiles_p1,pupilfiles_p2,datalen=600,pupil_len=36000,dyadnum=47,subjectnum=94,save=False):
	
	pupilwinsize = int(pupil_len / datalen)
	
	#preallocate some matrices
	pupils_p1 = np.zeros((dyadnum,pupil_len))
	pupils_p2 = np.zeros((dyadnum,pupil_len))
	pupils_p1_stack = np.zeros((dyadnum,datalen,pupilwinsize))
	pupils_p2_stack = np.zeros((dyadnum,datalen,pupilwinsize))
	allpupils_stack = np.zeros((subjectnum,datalen,pupilwinsize))
	dyad = []
	subject=[]
	
	for num,(p1,p2) in enumerate(zip(sorted(pupilfiles_p1),sorted(pupilfiles_p2))):
		#save the dyad number
		dyad.append(p1[p1.find('D0'):p1.find('D0')+4])
		subject.append(p1[p1.find('D0')+5:p1.find('D0')+9])
		subject.append(p2[p2.find('D0')+5:p2.find('D0')+9])
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
	print('starting DTW computation. this usually takes ~20 minutes.')
	
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
	if save:
		#save dtw timeseries 
		dtw_long = np.reshape(dtw_matrix,(dyadnum*datalen))
		dyad_long = np.repeat(dyad,datalen)
		dtw_df = pd.DataFrame([dyad_long,dtw_long]).T
		dtw_df.columns = ['dyad','dtw']
		savedir = input('please enter the full directory where you want to save your files, as a string: ')
		dtw_df.to_csv('{0}/dyad_dtw_timeseries.csv'.format(savedir))
		
		#save pupil timeseries 
		pupil_long = np.mean(allpupils_stack,axis=2).reshape(subjectnum*datalen)
		dyad_long = np.repeat(dyad,datalen*2)
		sub_long = np.repeat(subject,datalen)
		pupil_df = pd.DataFrame([dyad_long,sub_long,pupil_long]).T
		pupil_df.columns = ['dyad','subject','psize']
		pupil_df.to_csv('{0}/dyad_subject_pupil_timeseries.csv'.format(savedir))
	
	else:
		sub_long = np.repeat(subject,datalen)
		return dtw_matrix,np.mean(allpupils_stack,axis=2),sub_long
		
def compute_erp_synchrony(dtw_matrix,ec_all,none_all,dyads,savedir='',before=4,after=4,dyadnum=47,datalen=600,pupil_fs=60,trialcutoff=30,drop_subs=False,spaced=False,save=False):
	#preallocate matrices for dyad level dtw average responses to onset and offset
	sec_onset_full = np.zeros((dyadnum,int(before)+int(after)+1))
	sec_offset_full = np.zeros((dyadnum,int(before)+int(after)+1))
	sec_onset_sublevel = []
	sec_offset_sublevel = []
	
	#grab the corresponding moments of synchrony 
	for num, dtw in enumerate(dtw_matrix):
		#save just the specific dyad into its own vector of eye contact indices
		ec = ec_all[num]
		#save just the specific dyad into its own vector of no eye contact indices
		none = none_all[num]
		
		#get the indices for the number of seconds before eye contact occurred
		#subtract the number of seconds you want from all of the indices
		#then divide it by 60 to get the actual sampling rate.
		# AGAIN, if you're not interested in pupil dilations you may not need this! 
		prec_inds = (np.subtract(np.round(ec),before*pupil_fs)/pupil_fs).astype(np.int32)
		# if any of these indices are negative, we can't use them, so we toss them
		prec_inds_filt = [ind for ind in prec_inds if ind > 0]
		#get the indices for the number of seconds after eye contact occurred
		#add the number of seconds you want to all of the indices
		#divide by 60 etc, potentially unneccessary but I'm leaving it in in case I want to add pupil dilations
		postec_inds = ((np.add(np.round(ec),after*pupil_fs)/pupil_fs)+1).astype(np.int32)
		# if any of these indices are less than the sum of before and after, 
		# or greater than the length of the conversation, we can't use them, so we toss them
		postec_inds_filt = [ind for ind in postec_inds if datalen > ind > int(before)+int(after)+1]
		
		if spaced:
			#now only take instances that are properly spaced from one another.
			count = 0
			prec_inds_spaced=[]
			postec_inds_spaced=[]
			for pre,post in zip(prec_inds_filt,postec_inds_filt):
				if count == 0:
					prec_inds_spaced.append(pre)
					postec_inds_spaced.append(post)
					old_post = post
					count=1
				elif pre >= old_post:
					prec_inds_spaced.append(pre)
					postec_inds_spaced.append(post)
					old_post=post
			prec = prec_inds_spaced
			postec = postec_inds_spaced
			
		else:
			prec= prec_inds_filt
			postec = postec_inds_filt
		
		
		#do the same for the no eye contact instances
		prenone_inds = (np.subtract(np.round(none),before*pupil_fs)/pupil_fs).astype(np.int32)
		prenone_inds_filt = [ind for ind in prenone_inds if ind > 0]
		postnone_inds = ((np.add(np.round(none),after*pupil_fs)/pupil_fs)+1).astype(np.int32)  
		postnone_inds_filt = [ind for ind in postnone_inds if datalen > ind > int(before)+int(after)+1]
		
		if spaced:
			#now only take instances that are properly spaced from one another.
			count = 0
			prenone_inds_spaced=[]
			postnone_inds_spaced=[]
			for pre,post in zip(prenone_inds_filt,postnone_inds_filt):
				if count == 0:
					prenone_inds_spaced.append(pre)
					postnone_inds_spaced.append(post)
					old_post = post
					count=1
				elif pre >= old_post:
					prenone_inds_spaced.append(pre)
					postnone_inds_spaced.append(post)
					old_post=post
					
			prenone = prenone_inds_spaced
			postnone = postnone_inds_spaced
			
		else:
			prenone= prenone_inds_filt
			postnone = postnone_inds_filt

		
		#grab dtw costs at the eye contact indices specified by the filtered pre and post values
		ecsync = [dtw[pre:post] for pre,post in zip(sorted(prec),sorted(postec))]
		#grab dtw costs at the no eye contact indices specified by the filtered pre and post values
		nonesync = [dtw[pre:post] for pre,post in zip(sorted(prenone),sorted(postnone))]
		sec_onset_sublevel.append(list(ecsync))
		sec_offset_sublevel.append(list(nonesync))
		#save an average of the subject's responses to the onset of eye contact
		sec_onset_full[num] = np.mean(ecsync,axis=0)
		#save an average of the subject's responses to the offset of eye contact
		sec_offset_full[num] = np.mean(nonesync,axis=0)
		
	if drop_subs:
		trialnums = [len(econsets) for econsets in sec_onset_sublevel]
		sec_onset_full[np.where(np.array(trialnums) < trialcutoff),:] = np.nan 
		sec_offset_full[np.where(np.array(trialnums) < trialcutoff),:] = np.nan 
		sec_onset_all = sec_onset_full
		sec_offset_all = sec_offset_full
	else:
		sec_onset_all = sec_onset_full
		sec_offset_all = sec_offset_full
    
	#zscore the dtw curves
	onset_zscore = np.array([zscore(on,nan_policy='omit') for on in sec_onset_all])
	offset_zscore = np.array([zscore(off,nan_policy='omit') for off in sec_offset_all])

	#change the values to negative to represent synchrony
	onset = -onset_zscore
	offset = -offset_zscore
	
	if save:
		onset_long = np.reshape(onset,((before+after+1)*dyadnum,1)).flatten()
		offset_long = np.reshape(offset,((before+after+1)*dyadnum,1)).flatten()
		dyad_long = np.repeat(dyads,before+after+1)
		time = np.array([np.arange(-before,after+1,1)])
		time_long = np.reshape(np.repeat(time,[dyadnum],axis=0),((before+after+1)*dyadnum,1)).flatten()
		sync_erp_df = pd.DataFrame([dyad_long,time_long,onset_long,offset_long]).T
		sync_erp_df.columns = ['dyad','time','onset','offset']
		if drop_subs:
			sync_erp_df.to_csv('{0}/SA_synchrony_ERPana_{1}trialmin.csv'.format(savedir,trialcutoff))
		else:
			sync_erp_df.to_csv('{0}/SA_synchrony_ERPana.csv'.format(savedir))
	else:
		return onset,offset

## function to get time series of turn information, for supplementary analyses
def get_convo_turns(turnfiles,fs=10,datalen=600,dyadnum=47):
	dyads = []
	#give it a sampling rate of 10hz for now
	turnseries = np.zeros((dyadnum,datalen*fs))
	turnspersec_counts = np.zeros((dyadnum))
	faketurns_counts = np.zeros((dyadnum))
	lateturns_counts = np.zeros((dyadnum))
	
	for num,turnfile in enumerate(sorted(turnfiles)):
		#add the dyad to the list of all dyads
		dyads.append(turnfile[turnfile.find('D0'):turnfile.find('D0')+4])
		#read in the csv of the turns
		turns = pd.read_csv('{0}'.format(turnfile))
		
		#figure out where the end is,and pad it if necessary
		if len(turns['0'].iloc[-1]) < 5:
			turnend = '0' + turns['0'].iloc[-1]
		else:
			turnend = turns['0'].iloc[-1]
			
		#get speakers subject IDs
		p1 = min(turns['1']) #partner 1 is always the first ID, or the lower number
		p2 = max(turns['1']) #partner 2 is always the second ID, or the higher number
		
		previousturn=[]
		previousspeaker=[]
		turnlist = []
		turntime=[]
		turnspersec=1 #there are usually <1 turn per second, so the baseline here is 1
		
		for turn in turns.iterrows():#iterate through the conversation turns
			if turn[1][1] == previousspeaker:
				faketurns_counts[num] = faketurns_counts[num] + 1
				
			#change the turn value to be bigger (so that the next line of code works):
			if len(turn[1][0])<5:
				turn[1][0] = '0'+turn[1][0]
				
			#check and make sure everything is a turn, not laughter or something like that.
			if turn[1][1] != p1 and turn[1][1] != p2:
				print('{0} says {1} at time {2}'.format(turnfile[47:],turn[1][1],turn[1][0]))
			else: 
				#calculate time in seconds when the turn starts
				turnstart = (int(turn[1][0][0:2])*60) + int(turn[1][0][3:5])
				if turnstart >= datalen:
					lateturns_counts[num] = lateturns_counts[num] + 1
				#check whether the turnlist should be cleared
				if turnstart != previousturn and turnspersec == 1:
					turnlist = []
					turntime = []
				#add subjects to the turn list that we'll need to divide between
				turnlist.append(turn[1][1])
				turntime.append(turn[1][0])
				
				if turnstart == previousturn:
					#increment the number of turns we have to account for in this second
					turnspersec = turnspersec + 1
					#if the last turn is also occuring in the same second as the one before
					if turn[1][0] == turnend:
						segment = 1/turnspersec  
						for num2,partner in enumerate(turnlist):
							if partner == p1:
								turnseries[num,(turnstart*fs)+int((num2*segment)*fs):] = 1
							elif partner == p2:
								turnseries[num,(turnstart*fs)+int((num2*segment)*fs):] = -1
						#reset everything
						turnspersec = 1
						turnlist=[]
						turntime=[]
				
				else:
					#if you have a pileup of turns happening in the same second
					if turnspersec > 1:
						segment = 1/turnspersec
						oldturnstart = (int(turntime[0][0:2])*60) + int(turntime[0][3:5])
						for num2,partner in enumerate(turnlist[0:-1]):
							if partner == p1:
								turnseries[num,(oldturnstart*fs)+int((num2*segment)*fs):] = 1
							elif partner == p2:
								turnseries[num,(oldturnstart*fs)+int((num2*segment)*fs):] = -1
						#deal with the current turn
						if turn[1][1] == p1:
							turnseries[num,turnstart*fs:] = 1
						elif turn[1][1] == p2:
							turnseries[num,turnstart*fs:] = -1
						#reset everything
						turnspersec_counts[num] = turnspersec_counts[num] + 1
						turnspersec = 1
					else:
						if turn[1][1] == p1:
							turnseries[num,turnstart*fs:] = 1
						elif turn[1][1] == p2:
							turnseries[num,turnstart*fs:] = -1
								
			previousturn=turnstart
			previousspeaker = turn[1][1]
			
	return turnseries
    
## function to get turn events, for supplementary analyses
def get_turn_events(turnseries,turn_fs=10,p_fs=60):
	#parameter for the shortest acceptable turn length, in seconds
	turn_short = 1
	#preallocate matrix of all turns for all dyads
	turn_all = []
	
	#loop through dyads in turnseries matrix
	
	for turn in turnseries:
		#find turn chunks
		#split data into chunks where data changes from 1 to -1 and vice versa
		#note that this will give you turns for BOTH subjects
		turn_chunks = np.split(turn, np.where(np.diff(turn) != 0)[0]+1)
		#find the indices of those chunks. np.insert is included to add a zero to the beginning of the data
		#since the first turn happens at 0
		turn_inds = np.insert((np.where(np.diff(turn) != 0)[0]+1),0,0)
		
		#create empty lists of turns at the subject level
		turn_sub = []
		
		#loop through chunks and indices for eye contact and no eye contact
		for num,(chunk,ind) in enumerate(zip(turn_chunks,turn_inds)):
			if turn_short*turn_fs <= len(chunk):
				#append the second where that turn occurred
				#divide by the turn sampling rate to get seconds, 
				#then multiply by pupil sampling rate to get pupil indices
				#WE PROBABLY DON'T NEED TO DO THIS. COME BACK AND FIX IT ONCE YOU'VE IMPLEMENTED ALL THE CODE
				turn_sub.append(ind/turn_fs*p_fs)
				
		#now append the subject specific list to the larger list
		turn_all.append(turn_sub)
	
	return turn_all

## function to separate eye contact instances based on turns, for supplementary analyses
def get_ec_turn_events(ecfiles,turn_all,dyadnum=47,ec_fs=1000,pupil_fs=60,datalen=600,):
	
	# preallocate a dyad matrix to make sure you're collecting dyads in the right order
	dyads = []
	# preallocate lists of onsets and offsets that occur during turns
	onsets = []
	offsets = []
	ecturns = []
	onsets_noturn = []
	offsets_noturn = []
	ecnoturns = []
	
	# now read them in, keeping a list of dyads
	for num,file in enumerate(sorted(ecfiles)):
		#read the file
		data = pd.read_csv('{0}'.format(file),header=None,delimiter=' ')
		#preallocate a single dyad eye contact vector
		ectimeseries = np.zeros(int(ec_fs * datalen))
		#initialize some vars 
		onsets_sub = []
		onsets_noturn_sub = []
		offsets_sub = []
		offsets_noturn_sub = []
		ecturns_sub = []
		ecnoturns_sub = []
		#loop through the start column (data[0]) and the stop column(data[1])
		for start,stop in zip(data[0],data[1]):
			#check to see whether this instance of eye contact occurs during a turn switch
			if stop-start > ec_fs:
				for turn in turn_all[num]:
					if (stop/ec_fs*pupil_fs)+60 >= turn >= (start/ec_fs*pupil_fs)-60:
						offsets_sub.append(stop/ec_fs*pupil_fs)
						onsets_sub.append(start/ec_fs*pupil_fs)
						ecturns_sub.append(turn)
					else:
						ecnoturns_sub.append(turn)
						
				if start/ec_fs*pupil_fs not in onsets_sub:
					onsets_noturn_sub.append(start/ec_fs*pupil_fs)
					offsets_noturn_sub.append(stop/ec_fs*pupil_fs)
					
		#append the dyad name to the dyads variable
		dyads.append('{0}'.format(file[file.find('D0'):file.find('D0')+4]))
		#add sub level turn information to full lists
		onsets.append(list(onsets_sub))
		onsets_noturn.append(list(onsets_noturn_sub))
		offsets.append(list(offsets_sub))
		offsets_noturn.append(list(offsets_noturn_sub))
		ecturns.append(list(ecturns_sub))
		ecnoturns.append(list(ecnoturns_sub))
	
	return onsets,onsets_noturn
    
def compute_erp_turn_synchrony(dtw_matrix,onsets,onsets_noturn,before=4,after=4,dyadnum=47,pupil_fs=60,datalen=600,savedir='',save=False):
	
	#preallocate matrices for dyad level dtw average responses to turn start
	secturn_onset_all = np.empty((dyadnum,int(before)+int(after)+1))
	secturn_onset_all[:] = np.nan
	
	secnoturn_onset_all = np.empty((dyadnum,int(before)+int(after)+1))
	secnoturn_onset_all[:] = np.nan
	
	#grab the corresponding moments of synchrony 
	for num, dtw in enumerate(dtw_matrix):
		#save just the specific dyad into its own vector of turn indices
		turn = onsets[num]
		noturn = onsets_noturn[num]
		
		preturn_inds = (np.subtract(np.round(turn),before*pupil_fs)/pupil_fs).astype(np.int32)
		# if any of these indices are negative, we can't use them, so we toss them
		preturn_inds_filt = [ind for ind in preturn_inds if ind > 0]
		#get the indices for the number of seconds after the turn occurred
		#add the number of seconds you want to all of the indices
		#divide by 60 etc, potentially unneccessary but I'm leaving it in in case I want to add pupil dilations
		postturn_inds = ((np.add(np.round(turn),after*pupil_fs)/pupil_fs)+1).astype(np.int32)
		# if any of these indices are less than the sum of before and after, 
		# or greater than the length of the conversation, we can't use them, so we toss them
		postturn_inds_filt = [ind for ind in postturn_inds if datalen > ind > int(before)+int(after)+1]
		prenoturn_inds = (np.subtract(np.round(noturn),before*pupil_fs)/pupil_fs).astype(np.int32)
		# if any of these indices are negative, we can't use them, so we toss them
		prenoturn_inds_filt = [ind for ind in prenoturn_inds if ind > 0]
		#get the indices for the number of seconds after the turn occurred
		#add the number of seconds you want to all of the indices
		#divide by 60 etc, potentially unneccessary but I'm leaving it in in case I want to add pupil dilations
		postnoturn_inds = ((np.add(np.round(noturn),after*pupil_fs)/pupil_fs)+1).astype(np.int32)
		# if any of these indices are less than the sum of before and after, 
		# or greater than the length of the conversation, we can't use them, so we toss them
		postnoturn_inds_filt = [ind for ind in postnoturn_inds if datalen > ind > int(before)+int(after)+1]
		
		#are you using the spaced out inds or not?
		preturn = preturn_inds_filt
		postturn = postturn_inds_filt
		prenoturn = prenoturn_inds_filt
		postnoturn = postnoturn_inds_filt
		
		#grab dtw costs at the turn indices specified by the filtered pre and post values
		turnsync = [dtw[pre:post] for pre,post in zip(preturn,postturn)]
		noturnsync = [dtw[pre:post] for pre,post in zip(prenoturn,postnoturn)]
		
		#save an average of the subject's responses to the onset of eye contact
		secturn_onset_all[num] = np.nanmean(turnsync,axis=0)
		secnoturn_onset_all[num] = np.nanmean(noturnsync,axis=0)
		
	turn_zscore = np.array([zscore(on,nan_policy='omit') for on in secturn_onset_all])
	noturn_zscore = np.array([zscore(off,nan_policy='omit') for off in secnoturn_onset_all])
	
	ec_turn = -turn_zscore
	ec_noturn = -noturn_zscore
	
	return ec_turn,ec_noturn
