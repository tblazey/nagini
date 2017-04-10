#!/usr/bin/python

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Performes PCA on two 3D-images')
argParse.add_argument('one',help='First Nifti Image',nargs=1,type=str)
argParse.add_argument('two',help='Second Nifti Image',nargs=1,type=str)
argParse.add_argument('out',help='Root for output files',nargs=1,type=str)
argParse.add_argument('-brain',help='Brain mask for images',nargs=1,type=str)
args = argParse.parse_args()

#Load libraries
import numpy as np, nibabel as nib, nagini, sys

#Load image headers
one = nagini.loadHeader(args.one[0])
two = nagini.loadHeader(args.two[0])

#Make sure dimensions match
if one.shape != two.shape:
	print 'ERROR: Image dimensions do not match. Exiting...'
	sys.exit()

#Get image data and flatten
oneData = one.get_data().flatten()
twoData = two.get_data().flatten()

#Mask a mask of shared non-zero voxels
maskData = np.logical_and(oneData!=0,twoData!=0)

#Load mask if needed
if args.brain is not None:

	#Load brain mask header
	brain = nagini.loadHeader(args.brain[0])

	#Make sure its dimensions match
	if one.shape[0:3] != brain.shape[0:3]:
		print 'ERROR: Mask dimensions do not match data. Please check...'
		sys.exit()

	#Load in mask data
	brainData = brain.get_data()

	#Combine mask with non-zero mask
	maskData = np.logical_and(maskData,brainData.flatten()!=0)

#Get masked data
oneMasked = oneData[maskData]
twoMasked = twoData[maskData]

#Construct matrix for PCA
imgMat = np.vstack((oneMasked,twoMasked)).T

#Mean center
imgMeans = np.mean(imgMat,axis=0)
imgMat = imgMat - imgMeans

#Standardize
imgStd = np.std(imgMat,axis=0)
imgMat = imgMat / imgStd

#Run PCA
imgCov = np.cov(imgMat,rowvar=False)
u,s,v = np.linalg.svd(imgCov)

#Write out eigenvectors and eigenvalues
np.savetxt('%s_eigenVector.txt'%(args.out[0]),u)
np.savetxt('%s_eigenValue.txt'%(args.out[0]),s)

#Project data
imgProj = np.dot(imgMat,u)

#Write out projected image
projData = np.zeros((oneData.shape[0],2))
projData[maskData,:] = imgProj
projData = projData.reshape((one.shape[0],one.shape[1],one.shape[2],2))
projImg = nib.Nifti1Image(projData,one.affine,header=one.header)
projImg.to_filename('%s_proj.nii.gz'%(args.out[0]))




