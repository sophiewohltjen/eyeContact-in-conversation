import numpy as np
import pandas as pd

import glob
import os

base_directory = '/Users/sophie/Dropbox/EyeContactinConversation'
# get annotation files
annot_dir = '{0}/Data/eye_contact_annotations'.format(base_directory
annotation_1 = glob.glob('{0}/*notations.txt'.format(annot_dir))
annotation_2 = glob.glob('{0}/*notations_2.txt'.format(annot_dir))

# where are you saving the output?
savedir='{0}/Analyses/eye_contact/individual'.format(base_directory)

for annot1,annot2 in zip(sorted(annotation_1),sorted(annotation_2)):
	#annotation number 1
	annotations_p1 = pd.read_csv('{0}'.format(annot1),delimiter='\t',skiprows=1,header=None)
	starts_p1 = annotations_p1[4]
	stops_p1 = annotations_p1[7]

	#annotation number 2
	annotations_p2 = pd.read_csv('{0}'.format(annot2),delimiter='\t',skiprows=1,header=None)
	starts_p2 = annotations_p2[4]
	stops_p2 = annotations_p2[7]
	
	timeseries_p1 = np.zeros(max(stops_p2.iat[-1],stops_p1.iat[-1]))

	for start,stop in zip(starts_p1,stops_p1):
		timeseries_p1[start:stop] = 1
	
	timeseries_p2 = np.zeros(max(stops_p2.iat[-1],stops_p1.iat[-1]))

	for start,stop in zip(starts_p2,stops_p2):
		timeseries_p2[start:stop] = 1
	
	#annotater correspondence for eye contact
	eyecontactind = np.array(np.where([contact_p1 == 1 and contact_p2 == 1 for contact_p1,contact_p2 in zip(timeseries_p1,timeseries_p2)])).flatten()
	#chunks of corespondence for eye contact
	numcontacts = np.split(eyecontactind, np.where(np.diff(eyecontactind) > 1)[0]+1)
	
	eyecontact = np.zeros(len(timeseries_p2))
	eyecontact[eyecontactind] = 1
	
	np.savetxt('{0}/{1}_ecTimeseries.txt'.format(savedir,annot1[75:-16]),eyecontact)
