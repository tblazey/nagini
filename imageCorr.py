#!/usr/bin/python

###################
###Documentation###
###################

"""

imageCorr.py: Quick utility to calculate pearson correlation between two images

Requires the following inputs:
	imgOne -> First image
	imgTwo -> Second image

User can set the following options:
	mask -> Only compute correlation within specified mask. Assumed to be binary
	rank -> Computes spearman rank order correlation instead of pearson.

Requires the following modules:
	argparse, numpy, nibabel, nagini, sys, scipy

Tyler Blazey, Fall 2016
blazey@wustl.edu

"""

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Calculates pearson correlation between two images')
argParse.add_argument('imgOne',help='First image',nargs=1,type=str)
argParse.add_argument('imgTwo',help='Second image',nargs=1,type=str)
argParse.add_argument('-mask',help='Binary mask over which to compute correlation')
argParse.add_argument('-rank',help='Compute Spearman correlation instead of Pearson',action='store_const',const=1)
args = argParse.parse_args()


#Load needed libraries
import numpy as np, nibabel as nib, nagini, sys, scipy.stats as stats

#########################
###Data Pre-Processing###
#########################

#Load image headers
imgOne = nagini.loadHeader(args.imgOne[0])
imgTwo = nagini.loadHeader(args.imgTwo[0])

#Make sure images are 3D
if len(imgOne.shape) != 3 or len(imgTwo.shape) != 3:
	print 'ERROR: Images are not 3D.'
	sys.exit()

#Check to make sure dimensions match
if imgOne.shape != imgOne.shape:
	print 'ERROR: Data dimensions do not match. Please check...'
	sys.exit()

#Load in image data
oneData = imgOne.get_data().flatten()
twoData = imgTwo.get_data().flatten()

#Process mask if necessary
if args.mask is not None:

	#Load in mask header
	mask = nagini.loadHeader(args.mask)
	
	#Check to make sure mask is same shape
	if mask.shape != imgOne.shape:
		print 'ERROR: Mask does not match image shape. Please check...'
		sys.exit()

	#Load in mask data
	maskData = mask.get_data().flatten()

	#Mask images
	oneData = oneData[maskData==1]
	twoData = twoData[maskData==1]

#Compute correlation
if args.rank == 1:
	
	#Compute spearman correlation
	corr = stats.spearmanr(oneData,twoData)
	print corr[0]
else:

	#Compute pearson correlation
	corr = np.corrcoef(oneData,twoData)
	print corr[0,1]




