#!/usr/bin/python

###################
###Documentation###
###################

"""

bayesCbf.py: Calculates CBF using a 015 water scan and an arterial sampled input function

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
	argparse, numpy, nibabel, nagini, tqdm, scikit-learn, pystan, os

Tyler Blazey, Winter 2017
blazey@wustl.edu

"""

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Estimates ROI-based CBF using:')
argParse.add_argument('pet',help='Nifti water image',nargs=1,type=str)
argParse.add_argument('info',help='Yi Su style info file',nargs=1,type=str)
argParse.add_argument('aif',help='Arterial-sampled input function',nargs=1,type=str)
argParse.add_argument('well',help='Well-counter calibration factor',nargs=1,type=float)
argParse.add_argument('pie',help='Pie calibration factor',nargs=1,type=float)
argParse.add_argument('brain',help='Brain mask image',nargs=1,type=str)
argParse.add_argument('roi',help='ROI image.',nargs=1,type=str)
argParse.add_argument('out',help='Root for outputed files',nargs=1,type=str)
argParse.add_argument('-cbv',help='CBV image for blood volume correction.',nargs=1,type=str)
argParse.add_argument('-decay',help='Perform decay correction before modeling. By default it occurs within  model',action='store_const',const=1)
argParse.add_argument('-nKnots',nargs=1,type=int,help='Number of knots for AIF spline. Default is number of data points',metavar='n')
argParse.add_argument('-wbOnly',action='store_const',const=1,help='Only perform whole-brain estimation')
argParse.add_argument('-dcv',action='store_const',const=1,help='AIF is from a DCV file, not a CRV file. Dispersion is not estimated')
argParse.add_argument('-d',help='Density of brain tissue in g/mL. Default is 1.05',default=[1.05],metavar='density',type=float)
argParse.add_argument('-nChains',help='Number of chains to run. Default is 4',default=4,type=int)
argParse.add_argument('-nSamples',help='Number of samples per chain. Default is 2000. Half will be discarded as burn in',default=2000,type=int,metavar='n')
argParse.add_argument('-nThin',help='Period for saving samples. Default is 1 (every sample)',default=1,type=int,metavar='n')
args = argParse.parse_args()

#Load needed libraries
import numpy as np, nibabel as nib, nagini, sys, os, pystan, subprocess
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
roi = nagini.loadHeader(args.roi[0])

#Check to make sure dimensions match
if pet.shape[0:3] != brain.shape[0:3] or pet.shape[3] != info.shape[0] or brain.shape[0:3] != roi.shape[0:3]:
	print 'ERROR: Data dimensions do not match. Please check...'
	sys.exit()

#Get the image data
petData = pet.get_data()
brainData = brain.get_data()
roiData = roi.get_data()

#Flatten the PET images and then mask
petFlat = nagini.reshape4d(petData)
petMasked = petFlat[brainData.flatten()>0,:]

#Get pet into rois
petRoi = nagini.roiAvg(petFlat,roiData.flatten())

#If cbv image is given, load it up
if args.cbv is not None:

	#Load in CBV image
	cbv = nagini.loadHeader(args.cbv[0])
	if cbv.shape[0:3] != pet.shape[0:3]:
		print 'ERROR: CBV image does not match PET resolution...'
		sys.exit()
	cbvData = cbv.get_data()

	#Get whole brian mean cbv
	wbCbv = np.mean(cbvData.flatten()[brainData.flatten()>0] / 100 * args.d)

	#Get CBV roi
	cbvRoi = nagini.roiAvg(cbvData.flatten() / 100 * args.d,roiData.flatten())
else:
	#Just use zeros if we don't add a cbv image
	wbCbv = 0
	cbvRoi = np.zeros(petRoi.shape[0])

#Get pet start and end times. Assume start of first recorded frame is injection (has an offset which is subtracted out)
startTime = info[:,0] - info[0,0]
midTime = info[:,1] - info[0,0]
endTime = info[:,2] + startTime

#Combine PET start and end times into one array
petTime = np.stack((startTime,endTime),axis=1)

#Get aif time variable. Assume clock starts at injection.
aifTime = aif[:,0]

#Apply pie factor and 4dfp offset factor
aifC = aif[:,1] / args.pie[0] / 0.06

#Logic for preparing blood sucker curves
if args.dcv != 1:

	#Reset first two points in AIF which are not traditionally used
	aifC[0:2] = aifC[2]

	#Add well counter and decay correction from start of sampling
	aifC = aifC * args.well[0] 

	#Decay correct each CRV point to start time reported in first saved PET frame
	if args.decay == 1:
		aifC *= np.exp(np.log(2)/122.24*info[0,0]) * np.exp(np.log(2)/122.24*aifTime)

else:
	#Decay correct DCV file to start frame
	if args.decay == 1:
		aifC *= np.exp(np.log(2)/122.24*info[0,0])
	#Remove decay correction for DCV file
	else:
		aifC /= np.exp(np.log(2)/122.24*aifTime)

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
aifSpline = np.dot(aifBasis,aifCoefs)

#Get interpolation times as start to end of pet scan with aif sampling rate
sampTime = np.min(np.diff(aifTime))
interpTime = np.arange(np.floor(startTime[0]),np.ceil(endTime[-1]+sampTime),sampTime)

#Make sure we stay within boundaries of pet timing
if interpTime[-1] > np.ceil(endTime[-1]):
	interpTime = interpTime[:-1]

#Setup proper decay constant based on whether or not input is decay corrected
if args.decay == 1:
	decayC = 0
else:
	decayC = np.log(2)/122.24

	#Remove decay correction from pet data
	petMasked /= info[:,3]
	petRoi /= info[:,3]

#Get whole-brain tac
wbTac = np.mean(petMasked,axis=0)

#Make data structure for whole-brain fit
wbData = {
	'nPet':petTime.shape[0],
	'nAif':interpTime.shape[0],
	'nKnot':aifKnots.shape[0],
	'aifTime':interpTime,
	'pet':wbTac,
	'petTime':petTime.T,
	'aifCoefs':aifCoefs,
	'aifKnots':aifKnots,
	'aifStep':sampTime,
	'dens':args.d[0],
	'meanPet':np.mean(wbTac),
	'decay':decayC,
	'cbv':wbCbv
}

#Define  whole-brain parameters and initilizations
wbPars = ['flow','lambda','delay','nu','sigma','petMu','petPost',"aifC","rmsd"]
if args.dcv is None:
	wbPars.append('disp')
if args.cbv is not None:
	wbPars.append('kOne')

#Define function for inilization
def initFunc(disp=True,delay=True):
	def ranInit():
		initDic = {'flow':np.random.uniform(10,60),
			'lambda':np.random.uniform(0.5,1),
			'sigma':np.random.uniform(100,500),
			'nu':np.random.uniform(1,10)}
		if disp is True:
			initDic['disp'] = np.random.uniform(2,20)
		if delay is True:
			initDic['delay'] = np.random.uniform(2,20)
		return initDic
	return ranInit

#Load the appropriate whole-brain model
if args.dcv is None:
	wbModelFile = os.path.join(os.path.dirname(os.path.realpath(nagini.__file__)),'stanFlowFourT.txt')
	wbInits = initFunc()

else:
	wbModelFile = os.path.join(os.path.dirname(os.path.realpath(nagini.__file__)),'stanFlowThreeT.txt')
	wbInits = initFunc(disp=False)

#Compile the whole-brain model
wbModel = pystan.StanModel(file=wbModelFile)

#Run whole-brain model
wbFit = wbModel.sampling(data=wbData,iter=args.nSamples,
				chains=args.nChains,thin=args.nThin,
				pars=wbPars,init=wbInits)

#Set which parameters to extract
wbParams = ['flow','lambda','delay','nu','sigma']
if args.dcv is None:
	wbParams.append('disp')
if args.cbv is not None:
	wbParams.append('kOne')

#Get parameter estimates in numpy format. 
wbEst = wbFit.extract(wbParams)
wbEst = np.array([list(i) for i in wbEst.values()])

#Get predictions in numpy format
wbPred = wbFit.extract(['petMu','petPost']);
wbPred = np.array([list(i) for i in wbPred.values()])

#Get aif predictions in numpy format
aifPred = wbFit.extract(['aifC']);
aifPred = np.array([list(i) for i in aifPred.values()])

#Get normalized root mean squared deviaion in numpy
rmsdPred = wbFit.extract(['rmsd']);
rmsdPred = np.array([list(i) for i in rmsdPred.values()])

#Extract rHat
wbSummary = wbFit.summary()
wbRhat = wbSummary['summary'][np.in1d(wbSummary['summary_rownames'],wbParams),-1]

#Write out combined dictionary. Inefficient. Think of something better.
nagini.saveRz({'wbEst':wbEst,'wbPred':wbPred,'aifPred':aifPred,'wbRhat':wbRhat,'rmsd':rmsdPred},'%s_wbDic.Rdump'%(args.out[0]))

#Write out input data. Add in raw aif stuff before we go
wbData['aifRaw'] = aifC; wbData['aifRawTime'] = aifTime; wbData['aifSpline'] = aifSpline
nagini.saveRz(wbData,'%s_wbData.Rdump'%(args.out[0]))

#Create string for whole-brain parameter estimates
wbString = ''
for pIdx in range(len(wbParams)):
	wbString += '%s = %f\n'%(wbParams[pIdx],np.mean(wbEst[pIdx,:]))

#Write out whole-brain results
try:
	wbOut = open('%s_wbVals.txt'%(args.out[0]), "w")
	wbOut.write(wbString)
	wbOut.close()
except(IOError):
	print 'ERROR: Cannot write in output directory. Exiting...'
	sys.exit()

#Stop before regional estimates
if args.wbOnly == 1:
	nagini.writeArgs(args,args.out[0])
	sys.exit()

#Correct AIF for estimated delay. Not the most Bayesian thing, but for now...
cBasis, cBasisD = nagini.rSplineBasis(interpTime+np.mean(wbEst[2,:]),aifKnots,dot=True)
aifFit = np.dot(cBasis,aifCoefs)

#Correct for dispersion if necessary
if args.dcv is None:
	aifFit += np.dot(cBasisD,aifCoefs)*np.mean(wbEst[3,:])

#Correct for decay during shift
aifFit *= np.exp(np.log(2)/122.24*np.mean(wbEst[2,:]))
		
#Make data structure for region fit
regData = {
	'nPet':petTime.shape[0],
	'nAif':interpTime.shape[0],
	'aifTime':interpTime,
	'aifC':aifFit,
	'petTime':petTime.T,
	'aifStep':sampTime,
	'dens':args.d[0],
	'decay':decayC
}

#Write out regional data now
nagini.saveRz(regData,'%s_regData.Rdump'%(args.out[0]))

#Define parameters for regional fits
regPars = ['flow','lambda','nu','sigma','petMu','petPost','rmsd']; regParams = regPars[0:4]
if args.cbv is not None:
	regPars.append('kOne'); regParams.append('kOne')

#Compile regional model
regModelFile = os.path.join(os.path.dirname(os.path.realpath(nagini.__file__)),'stanFlowTwoT.txt')
regModel = pystan.StanModel(file=regModelFile)

#Make empty data structures for saving all the simulation results
nRoi = petRoi.shape[0]; nSamples = wbEst.shape[1]; nParams = len(regParams)
regEsts = np.zeros((nRoi,nParams,nSamples))
regPreds = np.zeros((nRoi,2,nSamples,regData['nPet']))
regRmsds = np.zeros((nRoi,nSamples))
regRhats = np.zeros((nRoi,nParams))

#Loop through regions
for rIdx in range(nRoi):
	print 'Fitting ROI %i'%(rIdx)

	#Add region's data to stan data dictionary
	regData['pet'] = petRoi[rIdx,:]
	regData['meanPet'] = np.mean(petRoi[rIdx,:])
	regData['cbv'] = cbvRoi[rIdx,0]

	#Do ROI fit
	regFit = regModel.sampling(data=regData,iter=args.nSamples,
					chains=args.nChains,thin=args.nThin,
					pars=regPars,init=initFunc(disp=False,delay=False))

	#Extract regional parameters and predictions
	regEst = np.array([list(i) for i in regFit.extract(regParams).values()])
	regPred = np.array([list(i) for i in regFit.extract(['petMu','petPost']).values()])
	regRmsd = np.array([list(i) for i in regFit.extract(['rmsd']).values()])

	#Get rHat
	regSummary = regFit.summary()
	regRhat = regSummary['summary'][np.in1d(regSummary['summary_rownames'],regParams),-1]

	#Save regional parameters and predictions
	regEsts[rIdx,:,:] = regEst
	regPreds[rIdx,:,:,:] = regPred
	regRmsds[rIdx,:] = regRmsd
	regRhats[rIdx,:] = regRhat

#Write out regional parameters data
print 'Saving regional results...'
nagini.saveRz({'regEsts':regEsts,'regPreds':regPreds,'regRoi':petRoi,'regCbv':cbvRoi,'regRhats':regRhats,'regRmsds':regRmsds},'%s_regDic.Rdump'%(args.out[0]))
nagini.writeArgs(args,args.out[0])


