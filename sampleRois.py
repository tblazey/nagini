#!/usr/bin/python


#####################
###Parse Arguments###
#####################

import argparse
argParse = argparse.ArgumentParser(description='Samples a image at several ROIs:')
argParse.add_argument('img',help='Image to put into ROIs',nargs=1)
argParse.add_argument('roi',help='Image containing ROIs',nargs=1)
argParse.add_argument('out',help='Root for output file',nargs=1)
argParse.add_argument('-nii',help='Output ROI as a ROIx1x1xTime Nifti file. Default is text file',action='store_const',const=1)
argParse.add_argument('-stat',help='Statistic to compute.',choices=['mean','min','max','median'],default='mean')
args = argParse.parse_args()

#Load needed libraries
import numpy as np, nibabel as nib, nagini, sys

#Load image headers
img = nagini.loadHeader(args.img[0])
roi = nagini.loadHeader(args.roi[0])

#Check to make sure images have same dimensions
if img.shape[0:3] != roi.shape[0:3]:
	print 'ERROR: Images do not have same dimensions. Exiting...'
	sys.exit()

#Load image data
imgData = img.get_data()
roiData = roi.get_data()

#Reshape image data as necessary
if len(imgData.shape) == 4:
	imgData = nagini.reshape4d(imgData)
else:
	imgData = imgData.reshape((img.shape[0]*img.shape[1]*img.shape[2],1))
roiData = roiData.flatten()

#Sample into ROIs
avgData = nagini.roiAvg(imgData,roiData,stat=args.stat)

#Save roi averages in the format user wants.
if args.nii == 1:
	avg = nib.Nifti1Image(avgData.reshape((avgData.shape[0],1,1,avgData.shape[1])),np.identity(4))
	avg.to_filename('%s_roi_%s.nii.gz'%(args.out[0],args.stat))
else:
	nagini.writeText('%s_roi_%s.txt'%(args.out[0],args.stat),avgData)
	












