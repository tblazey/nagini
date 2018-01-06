#!/usr/bin/python

###################
###Documentation###
###################

"""

cbfAif.py: Calculates cerebral blood flow using an water scan

Requires the following modules:
	argparse, numpy, nibabel, nagini, tqdm, scipy, matplotlib

Tyler Blazey, Winter 2017
blazey@wustl.edu

"""

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Estimates blood flow using:')
argParse.add_argument('pet',help='Nifti image with decay corrected well counts',nargs=1,type=str)
argParse.add_argument('info',help='Yi Su style info file',nargs=1,type=str)
argParse.add_argument('aif',help='Arterial-sampled input function',nargs=1,type=str)
argParse.add_argument('well',help='Well-counter calibration factor',nargs=1,type=float)
argParse.add_argument('out',help='Root for outputed files',nargs=1,type=str)
argParse.add_argument('-brain',help='Brain mask in PET space',nargs=1,type=str)
argParse.add_argument('-wbOnly',action='store_const',const=1,help='Only perform whole-brain estimation')
argParse.add_argument('-dcv',action='store_const',const=1,help='AIF is from a DCV file, not a CRV file. Using -noDisp or -noDelay is recommended.')
argParse.add_argument('-kernel',nargs=1,help='Deconvolution kernel for AIF fit. Has no effect with -spline. Using along with -noDisp is strongly recommended.')
argParse.add_argument('-spline',help='Use a restricted cubic spline to fit AIF. Does not use deconvolution kernel.',action='store_true')
argParse.add_argument('-nKnots',nargs=1,help='Number of knots for restricted cubic spline. Defalut is number of data points.',type=int)
argParse.add_argument('-noDisp',action='store_const',const=1,help='Do not include AIF dispersion term in model.')
argParse.add_argument('-noDelay',action='store_const',const=1,help='Do not include AIF delay term in model. Implies -noDisp.')
argParse.add_argument('-fModel',action='store_const',const=1,help='Does delay and/or dispersion estimate at each voxel.')
argParse.add_argument('-extrap',action='store_const',const=1,help='Allow input function to be extrapolated.')
argParse.add_argument('-d',help='Density of brain tissue in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-fBound',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds for voxelwise flow parameter. Default is 10 times whole-brain estimate')
argParse.add_argument('-lBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for voxelwise lambda parameter. Default is 0 to 2.')
argParse.add_argument('-dBound',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds for voxelwise delay parameter. Default is 10 times whole-brain estimate. Only relevant if -fModel is set and -noDelay is not.')
argParse.add_argument('-tBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for voxelwise dispersion parameter. Default is 10 times whole-brain estimate. Only relevant if -fModel is set and -noDisp is not.')
argParse.add_argument('-weighted',action='store_const',const=1,help='Use weighted regression using John Lee style weights.')
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

#Ignore invalid  and overflow warnings
np.seterr(invalid='ignore',over='ignore')

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

#Check to make sure dimensions match
if pet.shape[3] != info.shape[0]:
	print 'ERROR: Data dimensions do not match. Please check...'
	sys.exit()

#Brain mask logic
if args.brain is not None:

	#Load brain mask header
	brain = nagini.loadHeader(args.brain[0])

	#Make sure its dimensions match
	if pet.shape[0:3] != brain.shape[0:3]:
		print 'ERROR: Mask dimensiosn do not match data. Please check...'
		sys.exit()

	#Load in  mask data
	brainData = brain.get_data()

else:
	#Make a fake mask
	brainData = np.ones(pet.shape[0:3])

#Load in the kernel if necessary
if args.kernel is not None:
	try:
		kernel = np.loadtxt(args.kernel[0])
	except(IOERROR):
		print 'ERORR: Cannot load kernel at %s'%(args.kernel[0])
else:
	kernel = None

#Get the image data
petData = pet.get_data()

#Flatten the PET images and then mask
petMasked = nagini.reshape4d(petData)[brainData.flatten()>0,:]

#Get pet start and end times. Assume start of first recorded frame is injection (has an offset which is subtracted out)
startTime = info[:,0] - info[0,0]
midTime = info[:,1] - info[0,0]
endTime = info[:,2] + startTime

#Get aif time variable. Assume clock starts at injection.
aifTime = aif[:,0]

#Extract AIF counts
aifC = aif[:,1] 

#Logic for preparing blood sucker curves
if args.dcv != 1:

	#Reset first two points in AIF which are not traditionally used
	aifC[0:2] = aifC[2]

	#Convert to well counter units
	aifC = aifC * args.well[0]

	#Decay correct to PET start
	aifC *= np.exp(np.log(2)/122.24*info[0,0]) * np.exp(np.log(2)/122.24*aifTime)
		
else:

	#Reset last point which appears to be consistently bad
	aifC[-1] = aifC[aifC.shape[0]-2]

	#Add in decay correction to PET start
	aifC *= np.exp(np.log(2)/122.24*info[0,0])

#Decide how to fit AIF
if args.spline is False:

	#Create inits and bounds for AIF fit
	initAif = [0.019059,0.0068475,2.937158,6.030855,35.5553015,11.652845]
	lBoundAif = [0,0,0,0,0,0]
	hBoundAif= [0.05,0.01,10,20,100,300]
	boundsAif = [lBoundAif,hBoundAif]

	#Create normalized AIF for fitting
	aifScale = np.sum(aifC)
	aifNorm = aifC / aifScale

	#Get function for fitting aif
	golishFitFunc = nagini.golishFunc(kernel)

	#Run fit
	aifFit,aifCov = opt.curve_fit(golishFitFunc,aifTime,aifNorm,initAif,bounds=boundsAif)

	#Create hacky AIF wrapper function
	def wrapFunc(cMax,cZero,alpha,beta,tZero,tau,scale):
		def interpFunc(time,deriv=False):
			if deriv is False:
				return nagini.golishFunc()(time,cMax,cZero,alpha,beta,tZero,tau)*scale
			else:
				return nagini.golishFunc()(time,cMax,cZero,alpha,beta,tZero,tau)*scale,nagini.golishDerivFunc()(time,cMax,cZero,alpha,beta,tZero,tau)*scale
		return interpFunc

	#Apply wrapper function
	aifFunc = wrapFunc(aifFit[0],aifFit[1],aifFit[2],aifFit[3],aifFit[4],aifFit[5],aifScale)

else:
	
	#Get basis for AIF spline
	if args.nKnots is None:
		nKnots = aifC.shape[0]
	else:
		nKnots = args.nKnots[0]
	aifKnots = nagini.knotLoc(aifTime,nKnots,bounds=[5,95])
	aifBasis = nagini.rSplineBasis(aifTime,aifKnots)

	#Use scikit learn to do a initial bayesian ridge fit
	bModel = linear_model.BayesianRidge(fit_intercept=True)
	initFit = bModel.fit(aifBasis[:,1::],aifC)
	initCoefs = np.concatenate((initFit.intercept_[np.newaxis],initFit.coef_))

	#Compute absolute studentized residuals
	initFitted = np.dot(aifBasis,initCoefs)
	initResid = aifC - initFitted
	initHat = np.diag(aifBasis.dot(np.linalg.inv(aifBasis.T.dot(aifBasis)).dot(aifBasis.T)))
	initVar = 1.0/(aifC.shape[0]-aifBasis.shape[1])*np.sum(np.power(initResid,2))
	initStudent = np.abs(initResid / np.sqrt(initVar*(1-initHat)))

	#Compute masks for high residual points
	residMask = initStudent < 2.5

	#Rerun fit without high residual points
	bFit = bModel.fit(aifBasis[residMask,1::],aifC[residMask])
	aifCoefs = np.concatenate((bFit.intercept_[np.newaxis],bFit.coef_))
	
	#Make a quick wrapper function for returning spline results
	def wrapFunc(coefs,knots):
		def interpFunc(time,deriv=False):
			if deriv is False:
				basis = nagini.rSplineBasis(time,knots)
				interp = np.dot(basis,coefs)
				return interp
			else:
				basis,basisD = nagini.rSplineBasis(time,knots,dot=True)
				interp = np.dot(basis,coefs)
				deriv = np.dot(basisD,coefs)
				return interp,deriv
		return interpFunc

	#Apply wrapper function
	aifFunc = wrapFunc(aifCoefs,aifKnots)
			
#Make sure we have uniform AIF sampling
sampTime = np.min(np.diff(aifTime))

#Allow 60 seconds worth of extrapolation
if args.extrap == 1:
	interpTime = np.arange(np.max((np.floor(startTime[0]),np.floor(aifTime[0]-60))),np.min((np.ceil(aifTime[-1]+sampTime+60),np.ceil(endTime)[-1])),sampTime)
else:
	interpTime = np.arange(np.floor(aifTime[0]),np.ceil(aifTime[-1]+sampTime),sampTime)

###################
###Model Fitting###
###################
print ('Beginning fitting procedure...')

#Get whole-brain tac
wbTac = np.mean(petMasked,axis=0)

#Setup the proper model function
wbInit = [1,1]; wbBounds = np.array([[0.2,5.0],[0.2,5.0]]); wbScales = np.array([0.007,0.52])
if args.noDelay == 1:

	#Interpolate AIF
	interpAif = aifFunc(interpTime)
	
	#Get mask for useable PET data
	wbPetMask = np.logical_and(midTime>=interpTime[0],midTime<=interpTime[-1])

	#Model with just flow and lambda
	wbFunc = nagini.flowTwo(interpTime,interpAif,wbTac,midTime,wbPetMask)

elif args.noDisp == 1:

	#Model with just delay
	wbFunc = nagini.flowThreeDelay(aifFunc,interpTime,wbTac,midTime)

	#Add in starting point, bounds, and scale
	if args.kernel is None:
		wbInit.append(1); wbBounds = np.vstack((wbBounds,[-4,4]));
	else:
		wbInit.append(0); wbBounds = np.vstack((wbBounds,[-2,2]));
	wbScales = np.hstack((wbScales,[10]))
else:
	#Model with delay and dispersion:
	wbFunc = nagini.flowFour(aifFunc,interpTime,wbTac,midTime)

	#Add in starting points, bounds, and scales
	if args.kernel is None:
		wbInit.extend([1,1]); wbBounds = np.vstack((wbBounds,[-4,4],[-5,5]));
	else:
		wbInit.extend([0,0]); wbBounds = np.vstack((wbBounds,[-2,2],[-2,2]));
	wbScales = np.hstack((wbScales,[10,5]))

#Setup number of iterations for whole brain fitting
if args.weighted is None:
	weights = np.ones_like(wbTac)
else:
	weights = 1.0 / (midTime*np.log(midTime[-1]/midTime[0]))
	weights /= np.sum(weights*info[:,2])

#Start out fitting with a very lenient tolerance
wbStart = opt.minimize(wbFunc,wbInit,method='L-BFGS-B',args=(weights),bounds=wbBounds,options={'eps':0.1,'maxls':100})

#Use first fit to initilize with more reasonable tolerance
wbOpt = opt.minimize(wbFunc,wbStart.x,method='L-BFGS-B',args=(weights),bounds=wbBounds,options={'eps':0.001,'maxls':100})

#Make sure we converged before moving on
if wbOpt.success is False:
	print 'ERROR: Model did not converge on whole-brain curve. Exiting...'
	sys.exit()

#Get whole-brian fitted values
if args.noDelay == 1:
	wbFitted = wbFunc(wbOpt.x,weights,pred=True)
else:
	wbFitted,wbPetMask,wbAifMask = wbFunc(wbOpt.x,weights,pred=True)

#Remove optmization scales
wbFit = wbOpt.x * wbScales

#Create string for whole-brain parameter estimates
labels = ['CBF','Lambda','Delay','Dispersion']
scales = [6000.0/args.d,1/args.d,1.0,1.0]
wbString = ''
for pIdx in range(wbFit.shape[0]):
	wbString += '%s = %f\n'%(labels[pIdx],wbFit[pIdx]*scales[pIdx])

#Write out whole-brain results
try:
	#Parameter estimates
	wbOut = open('%s_wbVals.txt'%(args.out[0]), "w")
	wbOut.write(wbString)
	wbOut.close()

except(IOError):
	print 'ERROR: Cannot write in output directory. Exiting...'
	sys.exit()

#Make aif mask if we need to
if args.noDelay == 1:
	wbAifMask = np.repeat(True,interpTime.shape)

#Create whole brain fit figure
try:
	fig = plt.figure(1)
	gs = grid.GridSpec(1,2)

	#Make fit plot
	axOne = plt.subplot(gs[0,0])
	axOne.scatter(midTime,wbTac,s=40,c="black")
	axOne.plot(midTime[wbPetMask],wbFitted,linewidth=3,label='Model Fit')
	axOne.set_xlabel('Time (seconds)')
	axOne.set_ylabel('Counts')
	axOne.set_title('Whole-Brain Time Activity Curve')
	axOne.legend(loc='upper left')

	#Get plain old input function fitted values. Without spline, this  applies the kernel so it "fits" the curve
	if args.spline is False:
		aifFitted = golishFitFunc(interpTime,aifFit[0],aifFit[1],aifFit[2],aifFit[3],aifFit[4],aifFit[5]) * aifScale
	else:
		aifFitted = aifFunc(interpTime)
	
	#Make input function plot
	axTwo = plt.subplot(gs[0,1])
	if args.spline is True:
		axTwo.scatter(aifTime,aifC,s=40,c=np.where(residMask,'black','red'))
	else:
		axTwo.scatter(aifTime,aifC,s=40,c="black")
	axTwo.plot(interpTime,aifFitted,linewidth=5,label='Model Fit')
	axTwo.set_xlabel('Time (seconds)')
	axTwo.set_ylabel('Counts')
	axTwo.set_title('Arterial Sampled Input function')

	#Logic for AIF correction
	if args.noDelay != 1 or args.kernel is not None:

		#Get input function corrected for delay and possibly dispersion.
		if  args.noDelay != 1:

			#Interpolate input function and correct for delay and despersion if necessary
			if args.noDisp != 1:
				cAif,cAifD = aifFunc(interpTime+wbFit[2],deriv=True); cAif += cAifD*wbFit[3]
				lLabel = 'Delay+Disp Corrected'
			else:
				cAif = aifFunc(interpTime+wbFit[2])
				lLabel = 'Delay Corrected'

			#If we used kernel, make sure we say so
			if args.kernel is not None and args.spline is False:
				lLabel += ' + Kernel'

		#Only kernel
		if args.noDelay == 1 and args.kernel is not None and spline is False:

			#Get AIF corrected with kernel
			cAif = aifFunc(interpTime)	
			lLabel = "Kernel Corrected"

		#Create input function plot
		axTwo.plot(interpTime[wbAifMask],cAif[wbAifMask],linewidth=5,c='green',label=lLabel)
	
	#Make sure we have legend for input function plot
	axTwo.legend(loc='lower right',fontsize=10)

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
	nagini.writeArgs(args,args.out[0])
	sys.exit()

#Use whole-brain values as initilization
init = wbOpt.x; nParam = init.shape[0]

#Set default voxelwise bounds
bounds = np.stack((init[0:2]/3.0,init[0:2]*3.0)).T
if args.fModel == 1:
	for pIdx in range(2,nParam):
		if init[pIdx] > 0:
			bounds = np.vstack((bounds,[init[pIdx]/3.0,init[pIdx]*3.0]))
		elif init[pIdx] < 0:
			bounds = np.vstack((bounds,[init[pIdx]*3.0,init[pIdx]/3.0]))
		else:
			bounds = np.vstack((bounds,wbBounds[pIdx]))

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
		bounds[bIdx,0] = bound[0]
		bounds[bIdx,1] = bound[1]
		#Use midpoint between bounds as initial value if whole brain estimate is not in bounds
		if init[bIdx] < bound[0] or init[bIdx] > bound[1]:
			init[bIdx] = (bound[0]+bound[1]) / 2
	bIdx += 1

#Setup for voxelwise-optimization
nVox = petMasked.shape[0]; nParam = init.shape[0]
fitParams = np.zeros((nVox,nParam+1))

#Get corrected input function if we need to
if args.fModel != 1 and args.noDelay !=1:

	#Get shifted aif
	if args.noDisp != 1:
		voxAif,voxAifD = aifFunc(interpTime+wbFit[2],deriv=True); voxAif += voxAifD*wbFit[3]
	else:
		voxAif = aifFunc(interpTime+wbFit[2])
elif args.noDelay == 1:
	voxAif = interpAif

#Loop through every voxel
noC = 0
for voxIdx in tqdm(range(nVox)):

	#Get voxel tac
	voxTac = petMasked[voxIdx,:]

	#Get proper model function.
	if args.fModel == 1 and args.noDelay !=1:

		if args.noDisp == 1:
			optFunc = nagini.flowThreeDelay(aifFunc,interpTime,voxTac,midTime)
		else:
			optFunc = nagini.flowFour(aifFunc,interpTime,voxTac,midTime)
	else:
		optFunc = nagini.flowTwo(interpTime[wbAifMask],voxAif[wbAifMask],voxTac,midTime,wbPetMask)


	try:
		#Run fit
		voxOpt = opt.minimize(optFunc,init,method='L-BFGS-B',args=(weights),bounds=bounds,options={'eps':0.001,'maxls':100})

		#Make sure we converged
		if voxOpt.success is False:
			noC += 1
			continue

		#Save common estimates
		fitParams[voxIdx,0] = voxOpt.x[0] * 6000.0 / args.d * wbScales[0]
		fitParams[voxIdx,1] = voxOpt.x[1] / args.d * wbScales[1]

		#Logic for model specific results
		if args.fModel == 1 and args.noDelay !=1:

			#Save additional parameter estimates
			fitParams[voxIdx,2:nParam] = voxOpt.x[2:nParam]*wbScales[2:nParam]

			#Get fitted values
			voxPred,voxMask,_ = optFunc(voxOpt.x,weights,pred=True)

			#Calculate residuals and mean of useable tac
			voxResid = voxPred - voxTac[voxMask]
			voxMean = np.mean(voxTac[voxMask])

		else:

			#Get fitted values
			voxPred = optFunc(voxOpt.x,weights,pred=True)

			#Calculate resisduals and mean of useable tac
			voxResid = voxPred - voxTac[wbPetMask]
			voxMean = np.mean(voxTac[wbPetMask])

		#Calculate normalized root mean square deviation
		fitParams[voxIdx,-1] = np.sqrt(np.sum(np.power(voxResid,2))/voxResid.shape[0]) / voxMean

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
	paramNames = ['flow','lambda','delay','disp','nRmsd']
else:
	paramNames = ['flow','lambda','nRmsd']

#Do out images
for iIdx in range(fitParams.shape[1]):
	nagini.writeMaskedImage(fitParams[:,iIdx],brainData.shape,brainData,pet.affine,pet.header,'%s_%s'%(args.out[0],paramNames[iIdx]))

#Write out arguments
nagini.writeArgs(args,args.out[0])
