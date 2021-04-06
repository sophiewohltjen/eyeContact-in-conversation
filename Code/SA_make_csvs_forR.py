#!/usr/bin/env python

#SA_make_csvs_forR.py
#created by Sophie Wohltjen, 4/02/21

#Here I call all the functions to prepare the data and output several csvs, for the following analyses:
#1. dyadic logistic regression and engagement analyses
#2. mlVAR analysis
#3. ERP analysis

import glob
import numpy as np
import pandas as pd
from SA_convo_analysis_functions import get_mean_engagement_matrix,get_ec_matrix,get_ec_events,get_dtw_timeseries,compute_erp_synchrony,get_individual_ec

#will need to be changed based on data location
base_directory = '/Users/sophie/Dropbox/EyeContactinConversation'

## first, the logistic regression / engagement analysis dataframe

#some vars we'll need
dyadnum=47
datalen=600

#mean engagement
engage_dir = '{0}/Data/engagement_ratings'.format(base_directory)

ratingpartner1files = glob.glob('{0}/partner1/*.csv'.format(engage_dir))
ratingpartner2files = glob.glob('{0}/partner2/*.csv'.format(engage_dir))
ratingfiles = ratingpartner1files + ratingpartner2files

engage_mean, dyads = get_mean_engagement_matrix(ratingfiles)

engage_long = np.reshape(engage_mean,(dyadnum*datalen))
dyad_long = np.repeat(dyads,datalen)

#binary eye contact
ec_dir = '{0}/Analyses/eye_contact/mutual'.format(base_directory)
ecmutualfiles = glob.glob('{0}/*_ecCorrespond.txt'.format(ec_dir))

ec_matrix = get_ec_matrix(ecmutualfiles)

ec_long = np.reshape(ec_matrix,(dyadnum*datalen))

# synchrony / pupil dilations (for later)
p1_dir = '{0}/Analyses/preprocessed_pupils/partner1'.format(base_directory)
pupilfiles1 = glob.glob('{0}/*dt*.txt'.format(p1_dir))

p2_dir = '{0}/Analyses/preprocessed_pupils/partner2'.format(base_directory)
pupilfiles2 = glob.glob('{0}/*dt*.txt'.format(p2_dir))

dtw_matrix, allpupils, sub_long = get_dtw_timeseries(pupilfiles1,pupilfiles2)

dtw_long = np.reshape(dtw_matrix,(dyadnum*datalen))

#format it into a dataframe and output
data_for_logreg_engage = pd.DataFrame([dyad_long,dtw_long,ec_long,engage_long]).T
data_for_logreg_engage.columns = ['dyad','dtw','ec','engage_mean']
savedir = '{0}/Analyses'.format(base_directory)
data_for_logreg_engage.to_csv('{0}/SA_data_for_logreg_engage_anas.csv'.format(savedir))

#get individual moments of eye contact for comparison
indec_dir = '{0}/Analyses/eye_contact/individual'.format(base_directory)
indec_files = glob.glob('{0}/*_*_ecTimeseries.txt'.format(indec_dir))

get_individual_ec(indec_files,savedir=savedir,save=True)



## next, the mlVAR analysis dataframe
subjectnum=94

#averaged eye contact
ec_dir = '{0}/Analyses/eye_contact/mutual'.format(base_directory)
ecmutualfiles = glob.glob('{0}/*_ecCorrespond.txt'.format(ec_dir))

ec_matrix = get_ec_matrix(ecmutualfiles,ts_type='average')

ec_rep = np.repeat(ec_matrix,2,axis=0)
ec_long = np.reshape(ec_rep,(subjectnum*datalen))

#dtw timeseries
dtw_rep = np.repeat(dtw_matrix,2,axis=0)
dtw_long = np.reshape(dtw_rep,(subjectnum*datalen))

#pupil timeseries
allpupils_long = np.reshape(allpupils,(subjectnum*datalen))

#long dyad list
dyad_sublevel = np.repeat(dyads,datalen*2)

#format it into a dataframe and output
data_for_mlvar = pd.DataFrame([dyad_sublevel,sub_long,dtw_long,ec_long,allpupils_long]).T
data_for_mlvar.columns = ['dyad','subject','dtw','ec','psize']
savedir = '{0}/Analyses'.format(base_directory)
data_for_mlvar.to_csv('{0}/SA_data_for_mlvar_ana.csv'.format(savedir))



## finally, the ERP analysis dataframe

#get the binary ec timeseries matrix first
ec_matrix = get_ec_matrix(ecmutualfiles,datalen=600000)

#next get the ec events
ec_all,none_all = get_ec_events(ec_matrix)

#find synchrony curves at those ec events and save the output
compute_erp_synchrony(dtw_matrix,ec_all,none_all,dyads,savedir=savedir,save=True)

#make csvs for a range of trial minimums, for supplementary materials analysis
#now save the onsets at a range of minimum trial thresholds
trial_minimums = [20,25,30,35,40,45]
savedir = '{0}/Supplement'.format(base_directory)
for trial_min in trial_minimums:
	compute_erp_synchrony(dtw_matrix,ec_all,none_all,dyads,trialcutoff=trial_min,drop_subs=True,spaced=True,savedir=savedir,save=True)





