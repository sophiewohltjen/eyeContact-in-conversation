#!/usr/bin/env python

#SA_ERP_permutations.py
#created by Sophie Wohltjen, 4/02/21

#This is the code to recreate the permutation test I performed on my event-related data

import glob
import random
import numpy as np
import pandas as pd
from SA_convo_analysis_functions import get_ec_matrix, get_ec_events, compute_erp_synchrony

base_directory='/Users/sophie/Dropbox/EyeContactinConversation'
#set some vars
dyadnum=47
dyads = np.arange(0,46)
before = 4
after = 4
nperm=1000
# initialize permutation array
perm_onset_all = np.zeros((nperm,47,int(before)+int(after)+1))
perm_offset_all = np.zeros((nperm,47,int(before)+int(after)+1))

#get mutual eye contact matrix
ec_dir = '{0}/Analyses/eye_contact/mutual'.format(base_directory)
ecmutualfiles = glob.glob('{0}/*_ecCorrespond.txt'.format(ec_dir))
ec_matrix = get_ec_matrix(ecmutualfiles,datalen=600000)
#get true ec events 
ec_all, none_all = get_ec_events(ec_matrix)

#read in pre-computed synchrony matrix (for speed -- to re-create, run SA_DTW_timeseries.py)
dtw_dir = '{0}/Analyses'.format(base_directory)
dtw_long = pd.read_csv('{0}/dyad_dtw_timeseries.csv'.format(dtw_dir))
dtw_matrix = np.reshape(np.array(dtw_long['dtw']),(47,600))

pseudo_sync_onset = np.zeros(nperm)
pseudo_sync_offset = np.zeros(nperm)

for i in range(nperm):
    #initialize permutation vars
    pseudo_onsets_all = []
    pseudo_offsets_all = []
    
    if i%100 == 0:
    	print('running permutation number {0}'.format(i))
    #grab the corresponding moments of synchrony
    for num, dtw in enumerate(dtw_matrix):
    	
    	#figure out how many instances we need to grab
    	num_onsets = len(ec_all[num])
    	num_offsets = len(none_all[num])
    	#set seed for onsets
    	random.seed(i)
    	#grab random "onsets"
    	pseudo_onsets = random.sample(range(600),num_onsets)
    	#set seed for offsets
    	random.seed(i+nperm)
    	#grab random "offsets"
    	pseudo_offsets = random.sample(range(600),num_offsets)
    	#append them to the final pseudo "ec_all" style var for this perm
    	pseudo_onsets_all.append(pseudo_onsets)
    	pseudo_offsets_all.append(pseudo_offsets)
    
    #compute pseudo synchrony curve 
    ps_on, ps_off = compute_erp_synchrony(dtw_matrix,pseudo_onsets_all,pseudo_offsets_all,dyads,pupil_fs=1)
    
    #add the value from the onsets and offsets of pseudo eye contact to our null distributions
    pseudo_sync_onset[i] = np.mean(ps_on,axis=0)[4]
    pseudo_sync_offset[i] = np.mean(ps_off,axis=0)[4]
    
#make a dataframe with these values
pseudo_onset_offset = pd.DataFrame([pseudo_sync_onset,pseudo_sync_offset]).T
pseudo_onset_offset.columns = ['pseudo_onset','pseudo_offset']
#write them to a csv
savedir = '{0}/Analyses'.format(base_dir)
pseudo_onset_offset.to_csv('{0}/SA_pseudo_ests_ERPana.csv'.format(savedir))
