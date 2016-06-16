#!/usr/bin/python

###################
###Documentation###
###################

"""

cbfIdaif.py: Calculates CBF using a 015 water scan and an image-derived input function

Uses model described by Mintun et al, Journal of Nuclear 1983

Requires the following inputs:
	pet -> PET H20 image.
	info -> Yi Su style PET info file. 
	idaif -> Image derivied arterial input function
	brain -> Brain mask in the same space as pet.
	out -> Root for all outputed file
	
User can set the following options:
	d -> Density of brain tissue in g/mL. Default is 1.05
	fBound -> Bounds for flow parameter. Default is 10 times the flow parameter.
	lBound -> Bounds for lambda (blood-brain-partition coefficient). Default is 0 to 1.
	
Produces the following outputs:
	cbf -> Voxelwise map of cerebral blood volume in mL/100g*min
	cbf_var -> Voxelwise map of the variance of the cbf estimate.
	lambda -> Voxelwise map of the blood brain parition coefficient. In ml/g
	lambda_var -> Voxelwise map of the variance fo the lambda estimate.
	nRmsd -> Normalized (to-mean) root-mean-square deviation of fit

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
argParse.add_argument('-d',help='Density of brain tissue in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-fBound',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds for flow parameter. Default is 10 times whole brain value')
argParse.add_argument('-lBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for lambda parameter. Default is 0 to 1.')
args = argParse.parse_args()

#Make sure sure user set bounds correctly
for bound in [args.fBound,args.lBound]:
	if bound is not None:
		if bound[1] <= bound[0]:
			print 'ERROR: Lower bound of %f is not lower than upper bound of %f'%(bound[0],bound[1])
			sys.exit()

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

#Check to make sure dimensions match
if pet.shape[0:3] != brain.shape[0:3] or pet.shape[3] != idaif.shape[0] or pet.shape[3] != info.shape[0]:
	print 'ERROR: Data dimensions do not match. Please check...'
	sys.exit()

#Get the image data
petData = pet.get_data()
brainData = brain.get_data()

#Flatten the PET images and then mask
petMasked = nagini.reshape4d(petData)[brainData.flatten()>0,:]

#Use middle times as pet time. Account for any offset
petTime = info[:,1] - info[0,0]

#Interpolate the aif to minimum sampling time
minTime = np.min(np.diff(petTime))
interpTime = np.arange(petTime[0],petTime[-1],minTime)
aifInterp = interp.interp1d(petTime,idaif,kind="linear")(interpTime)

#Get the whole brain tac and interpolate that
wbTac = np.mean(petMasked,axis=0)
wbInterp = interp.interp1d(petTime,wbTac,kind="linear")(interpTime)

###################
###Model Fitting###
###################
print ('Beginning fitting procedure...')

#Attempt to fit model to whole-brain curve
fitX = np.vstack((interpTime,aifInterp))
try:
	wbFit = opt.curve_fit(nagini.flowTwoIdaif,fitX,wbInterp,bounds=([0,0],[np.inf,1]))
except(RuntimeError):
	print 'ERROR: Cannot estimate two-parameter model on whole-brain curve. Exiting...'
	sys.exit()

#Use whole-brain values as initilization
init = wbFit[0]

#Set bounds 
bounds = np.array(([wbFit[0][0]/10,0],[wbFit[0][0]*10,1]),dtype=np.float); bIdx = 0
for bound in [args.fBound,args.lBound]:
	#If user wants different bounds, use them.
	if bound is not None:
		bounds[0,bIdx] = bound[0]
		bounds[1,bIdx] = bound[1]
		#Use midpoint between bounds as initial value if whole brain estimate is not in bounds
		if init[bIdx] < bound[0] or init[bIdx] > bound[1]:		
			init[bIdx] = (bound[0]+bound[1]) / 2
	bIdx += 1

#Loop through every voxel
nVox = petMasked.shape[0]; fitParams = np.zeros((nVox,5)); noC = 0;
for voxIdx in tqdm(range(nVox)):
	
	#Get voxel tac and then interpolate it
	voxTac = petMasked[voxIdx,:]
	voxInterp = interp.interp1d(petTime,voxTac,kind="linear")(interpTime)

	try:
		#Get voxel fit with two parameter model
		voxFit = opt.curve_fit(nagini.flowTwoIdaif,fitX,voxInterp,p0=init,bounds=bounds)
		
		#Save parameter estimates
		fitParams[voxIdx,0] = voxFit[0][0] * 6000.0/args.d
		fitParams[voxIdx,1] = voxFit[0][1] * args.d
		
		#Save parameter variance estimates
		fitVar = np.diag(voxFit[1])
		fitParams[voxIdx,2] = fitVar[0] * np.power(6000.0/args.d,2)
		fitParams[voxIdx,3] = fitVar[1] * np.power(args.d,2)

		#Get normalized root mean square deviation
	 	fitResid = voxInterp - nagini.flowTwoIdaif(fitX,voxFit[0][0],voxFit[0][1])
		fitRmsd = np.sqrt(np.sum(np.power(fitResid,2))/voxInterp.shape[0])
		fitParams[voxIdx,4] = fitRmsd / np.mean(voxInterp)

	except(RuntimeError):
		noC += 1

#Warn user about lack of convergence
if noC > 0:
	print('Warning: %i of %i voxels did not converge.'%(noC,nVox))

#############
###Output!###
#############
print('Writing out results...')

#Write out two parameter model images
imgNames = ['flow','lambda','flow_var','lambda_var','nRmsd']
for iIdx in range(len(imgNames)):
	nagini.writeMaskedImage(fitParams[:,iIdx],brain.shape,brainData,brain.affine,'%s_%s'%(args.out[0],imgNames[iIdx]))



