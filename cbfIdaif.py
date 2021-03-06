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
argParse.add_argument('-lBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for lambda parameter. Default is 0 to 2.')
argParse.add_argument('-range',help='Time range for kinetic estimation in seconds. Default is full scan. Can specify "start" or end" or use number is seconds',nargs=2,metavar='time')
argParse.add_argument('-wbOnly',action='store_const',const=1,help='Only perform whole-brain estimation')
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

#Limit pet range
timeMask = np.logical_and(petTime>=petRange[0],petTime<=petRange[1])
petTime = petTime[timeMask]
idaif = idaif[timeMask]
petMasked = petMasked[:,timeMask]

#Get interpolation times as start to end of pet scan with aif sampling rate
minTime = np.min(np.diff(petTime))
interpTime = np.arange(petTime[0],np.ceil(petTime[-1]+minTime),minTime)

#Interpolate the AIF
aifInterp = interp.interp1d(petTime,idaif,kind="linear",fill_value="extrapolate")(interpTime)

#Get the whole brain tac
wbTac = np.mean(petMasked,axis=0)

###################
###Model Fitting###
###################
print ('Beginning fitting procedure...')

#Attempt to fit model to whole-brain curve
wbFunc = nagini.flowTwo(interpTime,aifInterp)
try:
	wbFit = opt.curve_fit(wbFunc,petTime,wbTac,bounds=([0,0],[np.inf,2]))
except(RuntimeError):
	print 'ERROR: Cannot estimate two-parameter model on whole-brain curve. Exiting...'
	sys.exit()

#Get whole brain fitted values
wbFitted = wbFunc(petTime,wbFit[0][0],wbFit[0][1])

#Create string for whole-brain parameter estimates
wbString = 'CBF=%f\nLambda=%f'%(wbFit[0][0]*6000/args.d,wbFit[0][1]/args.d)

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
except(IOError):
	print 'ERROR: Cannot write in output directory. Exiting...'
	sys.exit()

#Don't do voxelwise estimation if user says not to
if args.wbOnly == 1:
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
	
	#Get voxel tac
	voxTac = petMasked[voxIdx,:]
	
	try:
		#Get voxel fit with two parameter model
		voxFit = opt.curve_fit(wbFunc,petTime,voxTac,p0=init,bounds=bounds)
		
		#Save parameter estimates
		fitParams[voxIdx,0] = voxFit[0][0] * 6000.0/args.d
		fitParams[voxIdx,1] = voxFit[0][1] / args.d
		
		#Save parameter variance estimates
		fitVar = np.diag(voxFit[1])
		fitParams[voxIdx,2] = fitVar[0] * np.power(6000.0/args.d,2)
		fitParams[voxIdx,3] = fitVar[1] * np.power(1/args.d,2)

		#Get normalized root mean square deviation
	 	fitResid = voxTac - wbFunc(petTime,voxFit[0][0],voxFit[0][1])
		fitRmsd = np.sqrt(np.sum(np.power(fitResid,2))/voxTac.shape[0])
		fitParams[voxIdx,4] = fitRmsd / np.mean(voxTac)

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
	nagini.writeMaskedImage(fitParams[:,iIdx],brain.shape,brainData,pet.affine,pet.header,'%s_%s'%(args.out[0],imgNames[iIdx]))



