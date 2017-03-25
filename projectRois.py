#!/usr/bin/python


#####################
###Parse Arguments###
#####################

import argparse
argParse = argparse.ArgumentParser(description='Project ROI values back')
argParse.add_argument('val',help='ROI values to project',nargs=1)
argParse.add_argument('roi',help='Image containing ROIs',nargs=1)
argParse.add_argument('out',help='Root for output file',nargs=1)
args = argParse.parse_args()

#Load needed libraries
import numpy as np, nibabel as nib, nagini, sys

#Load images headers
val = nagini.loadHeader(args.val[0])
roi = nagini.loadHeader(args.roi[0])

#Load image data
valData = val.get_data()
roiData = roi.get_data()

#Reshape image data as necessary
if len(valData.shape) == 4:
	valData = nagini.reshape4d(valData)
else:
	valData = valData.reshape((val.shape[0]*val.shape[1]*val.shape[2],1))
roiData = roiData.flatten()

#Sample into ROIs
projData = nagini.roiBack(valData,roiData)

#Save projected ROI image
proj = nib.Nifti1Image(projData.reshape((roi.shape[0],roi.shape[1],roi.shape[2],projData.shape[-1])),roi.affine,header=roi.header)
proj.to_filename('%s_roiProj.nii.gz'%(args.out[0]))

	












