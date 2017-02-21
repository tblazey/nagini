#!/usr/bin/python

###################
###Documentation###
###################

"""

cbfAif.py: Calculates CBF using a 015 water scan and an arterial sampled input function

Uses model described by Mintun et al, Journal of Nuclear 1983
	
Produces the following outputs:
	wb -> Text file with estimates to whole-brian curve
	wbPlot -> Plot of whole-brain curve, fit, and input function
	cbf -> Voxelwise map of cerebral blood volume in mL/100g*min
	cbfVar -> Voxelwise map of the variance of the cbf estimate.
	lambda -> Voxelwise map of the blood brain parition coefficient. In ml/g
	lambdaVar -> Voxelwise map of the variance fo the lambda estimate.
	nRmsd -> Normalized (to-mean) root-mean-square deviation of fit

If -fModel is set and -noDelay is not it will produce:
	delay -> Voxelwise map of the shift of the input function (in seconds)
   	delayVar -> Voxelwise map of shift variance
If -fModel is set and -noDisp is not, you also get:
   	disp -> Voxelwise map of dispersion
   	dispVar -> Voxelwise map of dispersion variance

Requires the following modules:
	argparse, numpy, nibabel, nagini, tqdm, scipy, matplotlib, scikit-learn

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
argParse.add_argument('-nKnots',nargs=1,type=int,help='Number of knots for AIF spline. Default is number of data points',metavar='n')
argParse.add_argument('-wbOnly',action='store_const',const=1,help='Only perform whole-brain estimation')
argParse.add_argument('-dcv',action='store_const',const=1,help='AIF is from a DCV file, not a CRV file. Using -noDisp or -noDelay is recommended.')
argParse.add_argument('-noDisp',action='store_const',const=1,help='Do not include AIF dispersion term in model.')
argParse.add_argument('-noDelay',action='store_const',const=1,help='Do not include AIF delay term in model. Implies -noDisp.')
argParse.add_argument('-fModel',action='store_const',const=1,help='Does delay and/or dispersion estimate at each voxel.')
argParse.add_argument('-d',help='Density of brain tissue in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-fBound',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds for voxelwise flow parameter. Default is 10 times whole-brain estimate')
argParse.add_argument('-lBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for voxelwise lambda parameter. Default is 0 to 2.')
argParse.add_argument('-dBound',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds for voxelwise delay parameter. Default is 10 times whole-brain estimate. Only relevant if -fModel is set and -noDelay is not.')
argParse.add_argument('-tBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for voxelwise dispersion parameter. Default is 10 times whole-brain estimate. Only relevant if -fModel is set and -noDisp is not.')
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
from sklearn import linear_model

#########################
###Data Pre-Processing###
#########################
print ('Loading images...')

#Load in the input function
if args.dcv != 1:
	aif = nagini.loadAif(args.aif[0])
else:
	aif = nagini.loadAif(args.aif[0],dcv=True)

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

#Decay correct the AIF to pet offset and apply pie correction
aifC = aif[:,1] / args.pie[0] / 0.06 * np.exp(np.log(2)/122.24*info[0,0])
if args.dcv != 1:

	#Reset first two points in AIF which are not traditionally used
	aifC[0:2] = aifC[2]

	#Add well counter and decay correction from start of sampling
	aifC = aifC * args.well[0] * np.exp(np.log(2)/122.24*aifTime) 

#Get basis for AIF spline
if args.nKnots is None:
	nKnots = aifC.shape[0]
else:
	nKnots = args.nKnots[0]
aifKnots = nagini.knotLoc(aifTime,nKnots,bounds=[5,95])
aifBasis = nagini.rSplineBasis(aifTime,aifKnots)

#Use scikit learn to do a bayesian style ridge regression for spline fit
bModel = linear_model.BayesianRidge(fit_intercept=True)
bFit = bModel.fit(aifBasis[:,1::],aifC)
aifCoefs = np.concatenate((bFit.intercept_[np.newaxis],bFit.coef_))

#Get interpolation times as start to end of pet scan with aif sampling rate
sampTime = np.min(np.diff(aifTime))
interpTime = np.arange(np.floor(petTime[0]),np.ceil(petTime[-1]+sampTime),sampTime)

#Make sure we stay within boundaries of pet timing
if interpTime[-1] > np.ceil(petTime[-1]):
	interpTime = interpTime[:-1]

#Get whole-brain tac
wbTac = np.mean(petMasked,axis=0)

###################
###Model Fitting###
###################
print ('Beginning fitting procedure...')

#Setup the proper model function
wbInit = [0.0105,0.9]; wbBounds = ([0,0],[0.035,2])
if args.noDelay == 1:
	
	#Interpolate AIF with restricted cubic spline
	interpBasis = nagini.rSplineBasis(interpTime,aifKnots)
	interpAif = np.dot(interpBasis,aifCoefs)

	#Model with just flow and lambda
	wbFunc = nagini.flowTwo(interpTime,interpAif)

elif args.noDisp == 1:
	
	#Model with just delay 
	wbFunc = nagini.flowThreeDelay(aifCoefs,aifKnots,interpTime)

	#Add in starting points and bounds
	wbInit.append(0); wbBounds[0].append(-50); wbBounds[1].append(50)
else:
	
	#Model with delay and dispersion:
	wbFunc = nagini.flowFour(aifCoefs,aifKnots,interpTime)

	#Add in starting points and bounds
	wbInit.extend([0,0]); wbBounds[0].extend([-50,0]); wbBounds[1].extend([50,50])


#Attempt to fit model to whole-brain curve
try:
	wbOpt,wbCov = opt.curve_fit(wbFunc,petTime,wbTac,p0=wbInit,bounds=wbBounds)
except(RuntimeError):
	print 'ERROR: Cannot estimate model on whole-brain curve. Exiting...'
	sys.exit()

#Create string for whole-brain parameter estimates
labels = ['CBF','Lambda','Delay','Dispersion']
scales = [6000.0/args.d,1/args.d,1.0,1.0]
wbString = ''
for pIdx in range(wbOpt.shape[0]):
	wbString += '%s = %f\n'%(labels[pIdx],wbOpt[pIdx]*scales[pIdx])

#Write out whole-brain results
try:
	wbOut = open('%s_wbVals.txt'%(args.out[0]), "w")
	wbOut.write(wbString)
	wbOut.close()
except(IOError):
	print 'ERROR: Cannot write in output directory. Exiting...'
	sys.exit()

#Get whole brain fitted values
if args.noDelay == 1:
	wbFitted = wbFunc(petTime,wbOpt[0],wbOpt[1])
elif args.noDisp == 1:
	wbFitted = wbFunc(petTime,wbOpt[0],wbOpt[1],wbOpt[2])
else:
	wbFitted = wbFunc(petTime,wbOpt[0],wbOpt[1],wbOpt[2],wbOpt[3])

#Create whole brain fit figure
try:
	fig = plt.figure(1) 
	gs = grid.GridSpec(1,2)

	#Make fit plot
	axOne = plt.subplot(gs[0,0])
	axOne.scatter(petTime,wbTac,s=40,c="black")
	axOne.plot(petTime,wbFitted,linewidth=3,label='Model Fit')
	axOne.set_xlabel('Time (seconds)')
	axOne.set_ylabel('Counts')
	axOne.set_title('Whole-Brain Time Activity Curve')
	axOne.legend(loc='upper left')

	#Make input function plot
	axTwo = plt.subplot(gs[0,1])
	axTwo.scatter(aifTime,aifC,s=40,c="black")
	axTwo.plot(aifTime,np.dot(aifBasis,aifCoefs),linewidth=5,label='Spline Fit')
	axTwo.set_xlabel('Time (seconds)')
	axTwo.set_ylabel('Counts')
	axTwo.set_title('Arterial Sampled Input function')

	#Show corrected input funciton as well
	if args.noDelay != 1:
		
		#Interpolate input function and correct for delay
		cBasis, cBasisD = nagini.rSplineBasis(aifTime+wbOpt[2],aifKnots,dot=True)
		cAif = np.dot(cBasis,aifCoefs)

		#Correct for dispersion if necessary
		if args.noDisp != 1:
			cAif += np.dot(cBasisD,aifCoefs)*wbOpt[3]
			lLabel = 'Delay+Disp Corrected'
		else:
			lLabel = 'Delay Corrected'

		#Create input function plot		
		axTwo.plot(aifTime,cAif,linewidth=5,c='green',label=lLabel)
	
	#Make sure we have legend for input function plot
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
init = wbOpt; nParam = init.shape[0]

#Set default voxelwise bounds
lBounds = [wbOpt[0]/10.0,0]; hBounds = [wbOpt[0]*10.0,2]
if args.fModel == 1:
	for pIdx in range(2,nParam):
		if wbOpt[pIdx] > 0:
			lBounds.append(wbOpt[pIdx]/10.0)
			hBounds.append(wbOpt[pIdx]*10.0)
		elif wbOpt[pIdx] < 0:
			lBounds.append(wbOpt[pIdx]*10.0)
			hBounds.append(wbOpt[pIdx]/10.0)
		else:
			lBounds.append(wbBounds[0][pIdx])
			hBounds.append(wbBounds[1][pIdx])
bounds = np.array([lBounds,hBounds],dtype=np.float)
 
#Create list of user bounds to check and set initlization
if args.noDelay == 1 or args.fModel != 1:
	bCheck = [args.fBound,args.lBound]
	init = init[0:2]
elif args.noDisp == 1:
	bCheck = [args.fBound,args.lBound,args.dBound]
	init = init[0:3]
else:
	bCheck = [args.fBound,args.lBound,args.dBound,args.tBound]
	init = init[0:4]

#Check user bounds
bIdx = 0
for bound in bCheck:
	#If user wants different bounds, use them.
	if bound is not None:
		bounds[0,bIdx] = bound[0]
		bounds[1,bIdx] = bound[1]
		#Use midpoint between bounds as initial value if whole brain estimate is not in bounds
		if init[bIdx] < bound[0] or init[bIdx] > bound[1]:		
			init[bIdx] = (bound[0]+bound[1]) / 2
	bIdx += 1

#Setup for voxelwise-optimization
nVox = petMasked.shape[0]
if args.fModel == 1 and args.noDelay != 1:
	fitParams = np.zeros((nVox,nParam)); fitVars = np.zeros((nVox,nParam+1))
	optFunc = wbFunc
else:
	fitParams = np.zeros((nVox,2)); fitVars = np.zeros((nVox,3))

	#Get proper funciton
	if args.noDelay == 1:
		optFunc = wbFunc
	else:
		#Interpolate input function using delay
		voxBasis,voxBasisD = nagini.rSplineBasis(interpTime+wbOpt[2],aifKnots,dot=True)
		voxAif = np.dot(voxBasis,aifCoefs)

		#Account for dispersion if necessary
		if args.noDisp != 1:
			voxAif += np.dot(voxBasisD,aifCoefs)*wbOpt[3]

		optFunc = nagini.flowTwo(interpTime,voxAif)

#Loop through every voxel
noC = 0
for voxIdx in tqdm(range(nVox)):
	
	#Get voxel tac 
	voxTac = petMasked[voxIdx,:]

	try:
		#Run fit
		voxOpt,voxCov = opt.curve_fit(optFunc,petTime,voxTac,p0=init,bounds=bounds)
		
		#Save common estimates 
		fitParams[voxIdx,0] = voxOpt[0] * 6000.0 / args.d
		fitParams[voxIdx,1] = voxOpt[1] / args.d

		#Save common variances
		fitVar = np.diag(voxCov)
		fitVars[voxIdx,0] = fitVar[0] * np.power(6000.0/args.d,2)
		fitVars[voxIdx,1] = fitVar[1] * np.power(1/args.d,2)

		#Do model specific saving
		if args.fModel == 1 and args.noDelay !=1:

			#Save additional parameter estimates
			fitParams[voxIdx,2:nParam] = voxOpt[2:nParam]
			fitVars[voxIdx,2:nParam] = fitVar[2:nParam]

			#Calculate model residual
			if args.noDisp == 1:
	 			fitResid = voxTac - optFunc(petTime,voxOpt[0],voxOpt[1],voxOpt[2])
			else:
				fitResid = voxTac - optFunc(petTime,voxOpt[0],voxOpt[1],voxOpt[2],voxOpt[3])
		
		else:
			fitResid = voxTac - optFunc(petTime,voxOpt[0],voxOpt[1])

		#Save normalized root mean square deviation
		fitRmsd = np.sqrt(np.sum(np.power(fitResid,2))/voxTac.shape[0])
		fitVars[voxIdx,-1] = fitRmsd / np.mean(voxTac)

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

#Do the writing
for iIdx in range(fitParams.shape[1]):
	nagini.writeMaskedImage(fitParams[:,iIdx],brain.shape,brainData,pet.affine,pet.header,'%s_%s'%(args.out[0],paramNames[iIdx]))
	nagini.writeMaskedImage(fitVars[:,iIdx],brain.shape,brainData,pet.affine,pet.header,'%s_%s'%(args.out[0],varNames[iIdx]))
nagini.writeMaskedImage(fitVars[:,-1],brain.shape,brainData,pet.affine,pet.header,'%s_%s'%(args.out[0],varNames[-1]))


