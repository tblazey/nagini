#!/usr/bin/python

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Calculated a smoothed SUVR using cubic splines')
argParse.add_argument('pet',help='Nifti PET image',nargs=1,type=str)
argParse.add_argument('info',help='Yi Su style info file',nargs=1,type=str)
argParse.add_argument('mask',help='Nifti mask image',nargs=1,type=str)
argParse.add_argument('out',help='Root for output files',nargs=1,type=str)
argParse.add_argument('-window',help='Length of SUVR window (in seconds). Default is 60 seconds',default=[60],type=float)
argParse.add_argument('-nKnots',help='Number of knots for spline. Default is 8',default=[8],type=int)
args = argParse.parse_args()

#Load libraries
import numpy as np, nibabel as nib, matplotlib.pyplot as plt, nagini

#Load in pet and mask headers
pet = nib.load(args.pet[0]); mask = nib.load(args.mask[0]);

#Load in info file
info = nagini.loadInfo(args.info[0])

#Mask sure mask and pet dimensions mask
if pet.shape[0:3] != mask.shape[0:3]:
	print 'Error: Images do not have the same dimensions'
	sys.exit()

#Get image data
petData = nagini.reshape4d(pet.get_data())
maskData = mask.get_data().flatten()

#Mask pet data
petMasked = petData[maskData==1,:]

#Get PET time vetor
petTime = info[:,1]

#Caculate whole-brain timecourse
petMean = np.mean(petMasked,axis=0)

#Get spline knots and basis
petKnots = nagini.knotLoc(petTime,args.nKnots[0]); petKnots[0] = 10
petBasis,petDeriv = nagini.rSplineBasis(petTime,petKnots)

#Fit spline to whole brain
X,_,_,_ = np.linalg.lstsq(petBasis,petMean)

#Get spline predictions
timePred = np.linspace(0,petTime[-1],1000)
predBasis,predDeriv = nagini.rSplineBasis(timePred,petKnots)
petPred = np.dot(predBasis,X); petPredD = np.dot(predDeriv,X)

#Get timepoints for SUVR window
startTime = timePred[np.where(petPredD>10)[0][0]]
endTime = startTime + args.window[0]
suvrTime = np.linspace(startTime,endTime,100)

#Get new basis
suvrBasis,_ = nagini.rSplineBasis(suvrTime,petKnots)

#Get fits at each voxel
voxX,_,_,_ = np.linalg.lstsq(petBasis,petMasked.T)

#Get predictions at each voxel
voxPred = np.dot(suvrBasis,voxX)

#Get WB suvr image
voxSum = np.sum(voxPred,axis=0); voxSum = voxSum / np.mean(voxSum)

#Write out image
nagini.writeMaskedImage(voxSum,mask.shape,maskData,mask.affine,'%s_suvrSpline'%(args.out[0]))

#For comparision calculate non-smoothed SUVR
startFrame = np.where(petTime>=startTime)[0][0]
endFrame = np.where(petTime<=endTime)[0][-1]
petSum =  np.sum(petMasked[:,startFrame:(endFrame+1)],axis=1)
petSum = petSum / np.mean(petSum)
nagini.writeMaskedImage(petSum,mask.shape,maskData,mask.affine,'%s_suvr'%(args.out[0]))






