#!/usr/bin/python

###################
###Documentation###
###################

"""

cbfIdaifPoly.py: Calculates CBF using a 015 water scan and an image-derived input function

Uses model described by Videen et al, JCBFM 1987

	
Produces the following outputs:
	wbVals -> Text file with whole-brain flow
	wbTac -> Whole-brain tac
	wbFitted -> Text file with fit to whole-brain curve
	polyMat -> Text file with calculated Xs and Ys for polynomial regression
	polyCoef -> Text file with estimated regression coefficients

Requires the following modules:
	argparse, numpy, nibabel, nagini, tqdm, scipy

Tyler Blazey, Spring 2016
blazey@wustl.edu

"""

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Estimates cerebral blood flow using:')
argParse.add_argument('pet',help='Nifti water image',nargs=1,type=str)
argParse.add_argument('info',help='Yi Su style info file',nargs=1,type=str)
argParse.add_argument('idaif',help='Image-derived input function',nargs=1,type=str)
argParse.add_argument('brain',help='Brain mask in PET space',nargs=1,type=str)
argParse.add_argument('out',help='Root for outputed files',nargs=1,type=str)
argParse.add_argument('-lmbda',help='Value for the blood-brain partition coefficient (mL/g). Default is 0.9',type=float,default=0.9)
argParse.add_argument('-d',help='Density of brain tissue in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-range',help='Time range for kinetic estimation in seconds. Default is full scan. Can specify "start" or end" or use number is seconds',nargs=2,metavar='time')
argParse.add_argument('-wbOnly',action='store_const',const=1,help='Only perform whole-brain estimation')
args = argParse.parse_args()


#Load needed libraries
import numpy as np, nibabel as nib, nagini, sys, scipy.optimize as opt, scipy.interpolate as interp
from tqdm import tqdm

#########################
###Data Pre-Processing###
#########################
print ('Loading images...')

#Load image headers
pet = nagini.loadHeader(args.pet[0])
brain = nagini.loadHeader(args.brain[0]) 

#Load in the idaif.
idaif = nagini.loadIdaif(args.idaif[0])

#Load in the info file
info = nagini.loadInfo(args.info[0])

#Use middle times as pet time. Account for any offset
petTime = info[:,1] - info[0,0]

#Range logic
if args.range is not None:

	#So user doesn't have to know start or end
	if args.range[0] == "start":
		args.range[0] = petTime[0]
	if args.range[1] == "end":
		args.range[1] = petTime[-1]
	args.range = np.array(args.range,dtype=np.float64)

	#Check to see if the users range is actually within data
	if (args.range[0] < petTime[0]) or (args.range[1] > petTime[-1]):
		print 'Error: User selected range from %s to %s is outside of PET range of %f to %f.'%(args.range[0],args.range[1],petTime[0],petTime[-1])
		sys.exit()
	else:
		petRange = np.array([args.range[0],args.range[1]])
else:
	petRange = np.array([petTime[0],petTime[-1]])

#Check to make sure dimensions match
if pet.shape[0:3] != brain.shape[0:3] or pet.shape[3] != idaif.shape[0] or pet.shape[3] != info.shape[0]:
	print 'ERROR: Data dimensions do not match. Please check...'
	sys.exit()

#Get the image data
petData = pet.get_data()
brainData = brain.get_data()

#Flatten the PET images and then mask
petMasked = nagini.reshape4d(petData)[brainData.flatten()>0,:]

#Get interpolation times as start to end of pet range with aif sampling rate
minTime = np.min(np.diff(petTime[petTime<petRange[1]]))
interpTime = np.arange(petRange[0],np.ceil(petRange[1]+minTime),minTime)

#Interpolate the AIF
aifInterp = interp.interp1d(petTime,idaif,kind="linear",fill_value="extrapolate")(interpTime)

#Limit pet range
timeMask = np.logical_and(petTime>=petRange[0],petTime<=petRange[1])
petTime = petTime[timeMask]
petMasked = petMasked[:,timeMask]

###################
###Model Fitting###
###################
print ('Beginning fitting procedure...')

#Get input for polynomial regression
cbfPred = np.arange(10,100,10); nPred = cbfPred.shape[0]; petPred = np.zeros(nPred);
for pIdx in range(nPred):

	#Calculate PET model given flow, lmbda, and AIF
	modelPred = nagini.flowTwo(interpTime,aifInterp)(petTime,cbfPred[pIdx]/6000.0*args.d,args.lmbda*args.d)

	#Integrate it over specefied range 
	petPred[pIdx] = np.trapz(modelPred,petTime)

#Calculate polynomial regression coefficients
polyX = np.stack((petPred,np.power(petPred,2)),axis=1)
polyCoef,_,_,_ = np.linalg.lstsq(polyX,cbfPred)

#Get the whole brain tac
wbTac = np.mean(petMasked,axis=0)

#Calculate whole-brain CBF
wbInt = np.trapz(wbTac,petTime); wbFlow = wbInt*polyCoef[0] + wbInt**2*polyCoef[1]

#Get whole brain fitted values
wbFitted = nagini.flowTwo(interpTime,aifInterp)(petTime,wbFlow/6000.0*args.d,args.lmbda*args.d)

#Create string for whole-brain parameter estimates and polynomial regression
wbString = 'CBF=%f'%(wbFlow)
coefString = 'aOne=%e\naTwo=%e'%(polyCoef[0],polyCoef[1])

#Write out whole-brain results
try:
	#Parameter estimates
	wbOut = open('%s_wbVals.txt'%(args.out[0]), "w")
	wbOut.write(wbString)
	wbOut.close()

	#Whole-brain tac
	np.savetxt('%s_wbTac.txt'%(args.out[0]),wbTac)

	#Whole-brain fitted values
	np.savetxt('%s_wbFitted.txt'%(args.out[0]),wbFitted)

	#Write out polynomial matrix
	np.savetxt('%s_polyMat.txt'%(args.out[0]),np.hstack((polyX,cbfPred[:,np.newaxis])))
	
	#Polynomial regression coefficients
	coefOut = open('%s_polyCoef.txt'%(args.out[0]), "w")
	coefOut.write(coefString)
	coefOut.close()

except(IOError):
	print 'ERROR: Cannot write in output directory. Exiting...'
	sys.exit()

#Don't do voxelwise estimation if user says not to
if args.wbOnly == 1:
	sys.exit()

#Integrate all the pet data
petInt = np.trapz(petMasked,petTime,axis=1)

#Get matrix for flow predictions
petMat = np.stack((petInt,np.power(petInt,2)),axis=1)

#Get flow predictions at each voxel
petFlow = petMat.dot(polyCoef)

#Write out fow image
nagini.writeMaskedImage(petFlow,brain.shape,brainData,pet.affine,pet.header,'%s_flow'%(args.out[0]))



