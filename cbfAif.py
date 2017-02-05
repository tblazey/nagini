#!/usr/bin/python

###################
###Documentation###
###################

"""

cbfIdaif.py: Calculates CBF using a 015 water scan and an arterial sampled input function

Uses model described by Mintun et al, Journal of Nuclear 1983
	
Produces the following outputs:
	wb -> Text file with estimates to whole-brian curve
	wbPlot -> Plot of whole-brain curve, fit, and input function
	cbf -> Voxelwise map of cerebral blood volume in mL/100g*min
	cbfVar -> Voxelwise map of the variance of the cbf estimate.
	lambda -> Voxelwise map of the blood brain parition coefficient. In ml/g
	lambdaVar -> Voxelwise map of the variance fo the lambda estimate.
	nRmsd -> Normalized (to-mean) root-mean-square deviation of fit

If fModel is set it also produces:
	delay -> Voxelwise map of the shift of the input function (in seconds)
   	delayVar -> Voxelwise map of shift variance
   	disp -> Voxelwise map of dispersion
   	dispVar -> Voxelwise map of dispersion variance

Requires the following modules:
	argparse, numpy, nibabel, nagini, tqdm, scipy, matplotlib

Tyler Blazey, Winter 2017
blazey@wustl.edu

"""

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Estimates cerebral blood flow using:')
argParse.add_argument('pet',help='Nifti water image',nargs=1,type=str)
argParse.add_argument('info',help='Yi Su style info file',nargs=1,type=str)
argParse.add_argument('aif',help='Arterial-sampled input function',nargs=1,type=str)
argParse.add_argument('well',help='Well-counter calibration factor',nargs=1,type=float)
argParse.add_argument('pie',help='Pie calibration factor',nargs=1,type=float)
argParse.add_argument('brain',help='Brain mask in PET space',nargs=1,type=str)
argParse.add_argument('out',help='Root for outputed files',nargs=1,type=str)
argParse.add_argument('-d',help='Density of brain tissue in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-fModel',action='store_const',const=1,help='Does delay and dispersion estimate at each voxel.')
argParse.add_argument('-fBound',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds for flow parameter. Default is 10 times whole brain value')
argParse.add_argument('-lBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for lambda parameter. Default is 0 to 1.')
argParse.add_argument('-dBound',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds for delay parameter. Default is 10 times whole brain value. Note: -fModel must be set')
argParse.add_argument('-tBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for dispersion parameter. Default is 10 times whole brain value. Note: -fModel must be set')
argParse.add_argument('-nKnots',nargs=1,type=int,help='Number of knots for AIF spline. Default is 15',default=[15])
argParse.add_argument('-wbOnly',action='store_const',const=1,help='Only perform whole-brain estimation')
args = argParse.parse_args()


#Make sure sure user set bounds correctly
for bound in [args.fBound,args.lBound,args.dBound,args.tBound]:
	if bound is not None:
		if bound[1] <= bound[0]:
			print 'ERROR: Lower bound of %f is not lower than upper bound of %f'%(bound[0],bound[1])
			sys.exit()

#Load needed libraries
import numpy as np, nibabel as nib, nagini, sys, scipy.optimize as opt
import scipy.interpolate as interp, matplotlib.pyplot as plt, matplotlib.gridspec as grid
from tqdm import tqdm

#########################
###Data Pre-Processing###
#########################
print ('Loading images...')

#Load in the input function
aif = nagini.loadAif(args.aif[0])

#Load in the info file
info = nagini.loadInfo(args.info[0])

#Load image headers
pet = nagini.loadHeader(args.pet[0])
brain = nagini.loadHeader(args.brain[0]) 

#Check to make sure dimensions match
if pet.shape[0:3] != brain.shape[0:3] or pet.shape[3] != info.shape[0]:
	print 'ERROR: Data dimensions do not match. Please check...'
	sys.exit()

#Get the image data
petData = pet.get_data()
brainData = brain.get_data()

#Flatten the PET images and then mask
petMasked = nagini.reshape4d(petData)[brainData.flatten()>0,:]

#Use middle times as pet time. Account for offset
petTime = info[:,1] - info[0,0]

#Get aif time variable
aifTime = aif[:,0]

#Reset first two points in AIF which are not traditionally used
aifC= aif[:,1]; aifC[0:2] = aifC[2]

#Decay correct the AIF 
aifC = aifC * args.well[0] * np.exp(np.log(2)/122.24*aifTime) / args.pie[0] / 0.06 * np.exp(np.log(2)/122.24*info[0,0])

#Fit restricted cubic spline to AIF
aifKnots = nagini.knotLoc(aifTime,args.nKnots[0])
aifBasis,aifBasisD = nagini.rSplineBasis(aifTime,aifKnots)
aifCoefs,_,_,_ = np.linalg.lstsq(aifBasis,aifC)

#Get interpolation times as start to end of pet scan with aif sampling rate
sampTime = np.min(np.diff(aifTime))
interpTime = np.arange(petTime[0],petTime[-1]+sampTime,sampTime)

#Make sure we stay within boundaries of pet timing
if interpTime[-1] > petTime[-1]:
	interpTime = interpTime[:-1]

#Get whole-brain tac and interpolate it
wbTac = np.mean(petMasked,axis=0)
wbInterp = interp.interp1d(petTime,wbTac,kind="cubic")(interpTime)

###################
###Model Fitting###
###################
print ('Beginning fitting procedure...')

#Attempt to fit model to whole-brain curve
fourFunc = nagini.flowFour(aifCoefs,aifKnots)
try:
	wbOpt,wbCov = opt.curve_fit(fourFunc,interpTime,wbInterp,p0=[0.0105,0.9,-25,-25],bounds=([0,0,-50,-50],[0.035,1.5,50,50]))
except(RuntimeError):
	print 'ERROR: Cannot estimate four-parameter model on whole-brain curve. Exiting...'
	sys.exit()

#Write out whole-brain values
wbString = 'CBF = %f\nLambda = %f\nDelay = %f\nDispersion = %f'%(wbOpt[0]*6000.0/1.05,wbOpt[1]*1.05,wbOpt[2],wbOpt[3])
try:
	wbOut = open('%s_wbVals.txt'%(args.out[0]), "w")
	wbOut.write(wbString)
	wbOut.close()
except(IOError):
	print 'ERROR: Cannot write in output direcotry. Exiting...'
	sys.exit()

#Get whole brain fitted values
wbFitted = fourFunc(interpTime,wbOpt[0],wbOpt[1],wbOpt[2],wbOpt[3])

#Create whole brain fit figure
try:
	fig = plt.figure(1) 
	gs = grid.GridSpec(1,2)

	#Make fit plot
	axOne = plt.subplot(gs[0,0])
	axOne.scatter(petTime,wbTac,s=40,c="black")
	axOne.plot(interpTime,wbFitted,linewidth=3,label='Model Fit')
	axOne.set_xlabel('Time (seconds)')
	axOne.set_ylabel('Counts')
	axOne.set_title('Whole-Brain Time Activity Curve')
	axOne.legend(loc='upper left')

	#Get delay and dispersion corrected input function
	cBasis, cBasisD = nagini.rSplineBasis(aifTime+wbOpt[2],aifKnots)
	cAif = (np.dot(cBasis,aifCoefs) + np.dot(cBasisD,aifCoefs)*wbOpt[3]) * np.exp(np.log(2)/122.24*wbOpt[2])

	#Make input function plot
	axTwo = plt.subplot(gs[0,1])
	axTwo.scatter(aifTime,aifC,s=40,c="black")
	axTwo.plot(aifTime,np.dot(aifBasis,aifCoefs),linewidth=5,label='Spline Fit')
	axTwo.plot(aifTime,cAif,linewidth=5,c='green',label='Delay+Disp Corrected')
	axTwo.set_xlabel('Time (seconds)')
	axTwo.set_ylabel('Counts')
	axTwo.set_title('Arterial Sampled Input function')
	axTwo.legend(loc='upper right')

	#Add plots to figure
	fig.add_subplot(axOne); fig.add_subplot(axTwo)

	#Save plot
	plt.suptitle(args.out[0])
	fig.set_size_inches(15,5)
	fig.savefig('%s_wbPlot.jpeg'%(args.out[0]),bbox_inches='tight')
except(RuntimeError,IOError):
	print 'ERROR: Could not save figure. Moving on...'

#Don't do voxelwise estimation if user says not to
if args.wbOnly == 1:
	sys.exit()

#Use whole-brain values as initilization
init = wbOpt

#Set bounds 
lBounds = [wbOpt[0]/10.0,0,wbOpt[2]/10.0,wbOpt[3]/10.0]
hBounds = [wbOpt[0]*10.0,1,wbOpt[2]*10.0,wbOpt[3]*10.0]
bounds = np.array([lBounds,hBounds],dtype=np.float)
bIdx = 0
for bound in [args.fBound,args.lBound,args.dBound,args.tBound]:
	#If user wants different bounds, use them.
	if bound is not None:
		bounds[0,bIdx] = bound[0]
		bounds[1,bIdx] = bound[1]
		#Use midpoint between bounds as initial value if whole brain estimate is not in bounds
		if init[bIdx] < bound[0] or init[bIdx] > bound[1]:		
			init[bIdx] = (bound[0]+bound[1]) / 2
	bIdx += 1

#Setup for optimization
nVox = petMasked.shape[0]
if args.fModel == 1:
	fitParams = np.zeros((nVox,4)); fitVars = np.zeros((nVox,5))
	optFunc = fourFunc
else:
	fitParams = np.zeros((nVox,2)); fitVars = np.zeros((nVox,3))
	optFunc = nagini.flowTwoSet(aifCoefs,aifKnots,wbOpt[2],wbOpt[3])
	init = init[0:2]; bounds = bounds[:,0:2]

#Loop through every voxel
noC = 0
for voxIdx in tqdm(range(nVox)):
	
	#Get voxel tac and then interpolate it
	voxTac = petMasked[voxIdx,:]
	voxInterp = interp.interp1d(petTime,voxTac,kind="cubic")(interpTime)

	try:
		#Run fit
		voxOpt,voxCov = opt.curve_fit(optFunc,interpTime,voxInterp,p0=init,bounds=bounds)
		
		#Save common estimates 
		fitParams[voxIdx,0] = voxOpt[0] * 6000.0 / args.d
		fitParams[voxIdx,1] = voxOpt[1] * args.d

		#Save common variances
		fitVar = np.diag(voxCov)
		fitVars[voxIdx,0] = fitVar[0] * np.power(6000.0/args.d,2)
		fitVars[voxIdx,1] = fitVar[1] * np.power(args.d,2)

		#Do model specific saving
		if args.fModel == 1:
			
			#Save delay and disperison estimates
			fitParams[voxIdx,2] = voxOpt[2]
			fitParams[voxIdx,3] = voxOpt[3]

			#Save delay and dispersion errors
			fitVars[voxIdx,2:4] = fitVar[2:4]

			#Calculate model residual
	 		fitResid = voxInterp - optFunc(interpTime,voxOpt[0],voxOpt[1],voxOpt[2],voxOpt[3])
		
		else:
			fitResid = voxInterp - optFunc(interpTime,voxOpt[0],voxOpt[1])

		#Save normalized root mean square deviation
		fitRmsd = np.sqrt(np.sum(np.power(fitResid,2))/voxInterp.shape[0])
		fitParams[voxIdx,-1] = fitRmsd / np.mean(voxInterp)

	except(RuntimeError):
		noC += 1

#Warn user about lack of convergence
if noC > 0:
	print('Warning: %i of %i voxels did not converge.'%(noC,nVox))

#############
###Output!###
#############
print('Writing out results...')

#Set names for model images
if args.fModel == 1:
	paramNames = ['flow','lambda','delay','disp']
	varNames = ['flowVar','lambdaVar',"delayVar",'dispVar','nRmsd']
else:
	paramNames = ['flow','lambda']
	varNames = ['flowVar','lambdaVar','nRmsd']
for iIdx in range(len(paramNames)):
	nagini.writeMaskedImage(fitParams[:,iIdx],brain.shape,brainData,brain.affine,'%s_%s'%(args.out[0],paramNames[iIdx]))
	nagini.writeMaskedImage(fitVars[:,iIdx],brain.shape,brainData,brain.affine,'%s_%s'%(args.out[0],varNames[iIdx]))
nagini.writeMaskedImage(fitVars[:,-1],brain.shape,brainData,brain.affine,'%s_%s'%(args.out[0],varNames[-1]))


