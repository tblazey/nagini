
#!/usr/bin/python

###################
###Documentation###
###################

"""

gluAif.py: Calculates cmrGlu using a C11 glucose  scan and an arterial sampled input function

Requires the following modules:
	argparse, numpy, nibabel, nagini, tqdm, scipy, matplotlib

Tyler Blazey, Winter 2017
blazey@wustl.edu

"""

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Estimates cmrGlu using:')
argParse.add_argument('pet',help='Nifti image with decay corrected well counts',nargs=1,type=str)
argParse.add_argument('info',help='Yi Su style info file',nargs=1,type=str)
argParse.add_argument('dta',help='Dta file with drawn samples',nargs=1,type=str)
argParse.add_argument('blood',help='Blood glucose concentration mg/dL',type=float)
argParse.add_argument('roi',help='ROIs for two stage procedure',nargs=1,type=str)
argParse.add_argument('cbf',help='CBF image. In mL/hg*min',nargs=1,type=str)
argParse.add_argument('cbv',help='CBV image for blood volume correction in mL/hg',nargs=1,type=str)
argParse.add_argument('out',help='Root for outputed files',nargs=1,type=str)
argParse.add_argument('-brain',help='Brain mask in PET space',nargs=1,type=str)
argParse.add_argument('-seg',help='Segmentation image used to create CBV averages.',nargs=1,type=str)
argParse.add_argument('-fwhm',help='Apply smoothing kernel to CBF and CBV',nargs=1,type=float)
argParse.add_argument('-fwhmSeg',help='Apply smoothing kerenl to CBV segmentation image',nargs=1,type=float)
argParse.add_argument('-noRoi',action='store_const',const=1,help='Do not perform region estimation. Implies -noVox')
argParse.add_argument('-dT',help='Density of brain tissue in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-dB',help='Density of blood in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-oBound',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds for beta one, where bounds are 0.005*[lower,upper]. Defalt is [0.2,5]')
argParse.add_argument('-tBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for beta two, where bounds are 0.00016*[lower,upper]. Defalt is [0.2,5]')
argParse.add_argument('-dBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for delay parameter. Default is [-50,50]')
argParse.add_argument('-weighted',action='store_const',const=1,help='Run with weighted regression')
argParse.add_argument('-interp',action='store_const',const=1,help='Interpolate AIF instead of fitting model')
argParse.add_argument('-noMet',action='store_const',const=1,help='Do not correct for input function metabolites')
args = argParse.parse_args()

#Setup bounds and inits
wbBounds = [[0.2,.2,-50],[5,5,50]]; wbInit = [1,1,0]

#Loop through bounds
uBounds = [args.oBound,args.tBound,args.dBound]
for bIdx in range(len(uBounds)):

	#Make sure bounds make sense
	bound = uBounds[bIdx]
	if bound is not None:
		if bound[1] <= bound[0]:
			print 'ERROR: Lower bound of %f is not lower than upper bound of %f'%(bound[0],bound[1])
			sys.exit()

		#If they do, use them
		wbBounds[0][bIdx] = bound[0]
		wbBounds[1][bIdx] = bound[1]

		#Make sure initlization is within bounds
		if wbInit[bIdx] < bound[0] or wbInit[bIdx] > bound[1]:
					wbInit[bIdx] = (bound[0]+bound[1]) / 2

#Load needed libraries
import numpy as np, nibabel as nib, nagini, sys, scipy.optimize as opt
import scipy.interpolate as interp, matplotlib.pyplot as plt, matplotlib.gridspec as grid
import scipy.ndimage.filters as filt, scipy.spatial as spat
from tqdm import tqdm

#Ignore invalid  and overflow warnings
np.seterr(invalid='ignore',over='ignore')

#########################
###Data Pre-Processing###
#########################
print ('Loading images...')

#Load in the dta file
dta = nagini.loadDta(args.dta[0])

#Load in the info file
info = nagini.loadInfo(args.info[0])

#Load image headers
pet = nagini.loadHeader(args.pet[0])
roi = nagini.loadHeader(args.roi[0])
cbf = nagini.loadHeader(args.cbf[0])
cbv = nagini.loadHeader(args.cbv[0])

#Check to make sure dimensions match
if pet.shape[3] != info.shape[0] or pet.shape[0:3] != roi.shape[0:3] or pet.shape[0:3] != cbf.shape[0:3] or pet.shape[0:3] != cbv.shape[0:3] :
	print 'ERROR: Data dimensions do not match. Please check...'
	sys.exit()

#Brain mask logic
if args.brain is not None:

	#Load brain mask header
	brain = nagini.loadHeader(args.brain[0])

	#Make sure its dimensions match
	if pet.shape[0:3] != brain.shape[0:3]:
		print 'ERROR: Mask dimensions do not match data. Please check...'
		sys.exit()

	#Load in  mask data
	brainData = brain.get_data()

else:
	#Make a fake mask
	brainData = np.ones(pet.shape[0:3])

#Get the image data
petData = pet.get_data()
cbfData = cbf.get_data().reshape(pet.shape[0:3])
cbvData = cbv.get_data().reshape(pet.shape[0:3])
roiData = roi.get_data().reshape(pet.shape[0:3])

#Segmentation logic
if args.seg is not None:

	#Load in segmentation header
	seg = nagini.loadHeader(args.seg[0])

	#Make sure its dimensions match
	if pet.shape[0:3] != seg.shape[0:3]:
		print 'ERROR: Segmentation dimensions do not match data. Please check...'
		sys.exit()

	#Load in segmentation data'
	segData = seg.get_data().reshape(pet.shape[0:3])

	#Make a new segmentation where we have high CBV
	segData[cbvData>=8.0] = np.max(segData)+1

	#Get CBV ROI averages
	cbvAvgs = nagini.roiAvg(cbvData.flatten(),segData.flatten(),min=0.0)

	#Remask CBV image from ROI averages
	cbvData = nagini.roiBack(cbvAvgs,segData.flatten()).reshape(cbvData.shape)

	#Smooth
	if args.fwhmSeg is not None:
			voxSize = pet.header.get_zooms()
			sigmas = np.divide(args.fwhmSeg[0]/np.sqrt(8.0*np.log(2.0)),voxSize[0:3])
			cbvData = filt.gaussian_filter(cbvData,sigmas).reshape(cbvData.shape)
			cbvData[segData==0] = 0.0

#Get mask where inputs are non_zero
maskData = np.where(np.logical_and(roiData!=0,np.logical_and(np.logical_and(brainData!=0,cbfData!=0),cbvData!=0)),1,0)

#Flatten the PET images and then mask
petMasked = nagini.reshape4d(petData)[maskData.flatten()>0,:]
roiMasked = roiData[maskData>0].flatten()

#Prep CBF and CBV data
if args.fwhm is None:
	#Do not smooth,just mask
	cbfMasked = cbfData[maskData>0].flatten()
	cbvMasked = cbvData[maskData>0].flatten()
else:
	#Prepare for smoothing
	voxSize = pet.header.get_zooms()
	sigmas = np.divide(args.fwhm[0]/np.sqrt(8.0*np.log(2.0)),voxSize[0:3])

	#Smooth and mask data
	cbfMasked = filt.gaussian_filter(cbfData,sigmas)[maskData>0].flatten()
	cbvMasked = filt.gaussian_filter(cbvData,sigmas)[maskData>0].flatten()

#Get flow in 1/s and cbv in mlB/mlT
flowMasked = cbfMasked / cbvMasked / 60.0
vbMasked = cbvMasked / 100.0 * args.dT

#Get whole brain medians of flow and fractional blood volume
flowWb = np.mean(flowMasked)
vbWb = np.mean(vbMasked)

#Get pet start and end times. Assume start of first recorded frame is injection (has an offset which is subtracted out)
startTime = info[:,0] - info[0,0]
midTime = info[:,1] - info[0,0]
endTime = info[:,2] + startTime

#Get decay corrected blood curve to injection
drawTime = dta[:,0]; corrCounts = dta[:,1]

#########################
###Data Pre-Processing###
#########################

#AIF fitting logic
if args.interp is None:

	#Update user
	print 'Fitting AIF...'

	#Run piecwise fit to start the initialization process for AIF fitting
	maxIdx = np.argmax(corrCounts)
	tRight = drawTime[maxIdx:]; cRight = np.log(corrCounts[maxIdx:])
	initOpt,initCov = opt.curve_fit(nagini.segModel,tRight,cRight,[10,0.1,0.01,-0.01,-0.001,-0.0001])

	#Save params for initialization
	aTwo = np.exp(initOpt[1]); aThree = np.exp(initOpt[2])
	eTwo = initOpt[4]; eThree = initOpt[5]

	#Get tau coefficient
	tMax= drawTime[np.argmax(corrCounts)]
	grad = np.gradient(corrCounts,drawTime)
	tau = drawTime[np.argmax(grad)-1]
	if tau >= tMax:
		tau = 0

	#Get inits for first component
	aOne = ((np.max(corrCounts)-aTwo-aThree)*np.exp(1) + aTwo+aThree)/(tMax-tau)
	eOne = -1 / ( (tMax-tau) - (aTwo+aThree)/aOne)

	#Setup bounds and inits
	inits = [np.min((np.max((0,tau)),60)),aOne,aTwo,aThree,eOne,eTwo,eThree]; lBound = [0]; hBound = [60]
	for pIdx in range(1,len(inits)):
		if inits[pIdx] > 0:
			lBound.append(inits[pIdx]/5.0)
			hBound.append(inits[pIdx]*5.0)
		else:
			lBound.append(inits[pIdx]*5.0)
			hBound.append(inits[pIdx]/5.0)
	bounds = [lBound,hBound]

	#Run fit
	aifFit,aifCov =  opt.curve_fit(nagini.fengModel,drawTime,corrCounts,inits,bounds=bounds)

	#Run global optimization
	aifGlobal = opt.basinhopping(nagini.fengModelGlobal,aifFit,minimizer_kwargs={'args':(drawTime,corrCounts),'bounds':np.array(bounds).T})

	#Get function for interpolating AIF
	aifFunc = nagini.fengFunc(aifGlobal.x[0],aifGlobal.x[1],aifGlobal.x[2],
	                          aifGlobal.x[3],aifGlobal.x[4],aifGlobal.x[5],
							  aifGlobal.x[6])

	#Label for future plot
	aifLabel = "Model Fit"

else:

	#Just do a straight up linear interpolation
	aifFunc = interp.PchipInterpolator(drawTime,corrCounts,extrapolate=True)

	#Label for future plot
	aifLabel = 'Interpolated'

#Get interpolation times as start to end of pet scan with aif sampling rate
sampTime = np.min(np.diff(np.sort(drawTime)))
interpTime = np.arange(np.floor(startTime[0]),np.ceil(endTime[-1]+sampTime),sampTime)

#Get whole-brain tac
wbTac = np.mean(petMasked,axis=0)

###################
###Model Fitting###
###################
print ('Fitting whole brain curve...')

#Use either uniform or John Lee Jeffery style weights
if args.weighted is None:
	weights = np.ones_like(wbTac)
else:
	weights = 1.0 / (midTime*np.log(midTime[-1]/midTime[0]))
	weights /= np.sum(weights*info[:,2])

#Decide about metabolite correction
if args.noMet is None:
	metBool = True
else:
	metBool = False

#Use cross-validation to get penalty for whole-brain curve
wbCv = opt.minimize_scalar(nagini.cvOpt,bounds=([0,15]),args=(aifFunc,interpTime,wbTac,midTime,flowWb,vbWb,weights,wbInit,np.array(wbBounds).T,metBool),options={'maxiter':50},method='Bounded',tol=0.1)

#Arguments for whole-brain model fit
wbArgs = (aifFunc,interpTime,wbTac,midTime,flowWb,vbWb,wbCv.x,1,weights,False,metBool)

#Attempt to fit model to whole-brain curve
wbFit = opt.minimize(nagini.gluDelayLstPen,wbInit,args=wbArgs,method='L-BFGS-B',bounds=np.array(wbBounds).T,options={'maxls':100})

#Make sure we converged
if wbFit.success is False:
	print 'ERROR: Cannot estimate model on whole-brain curve. Exiting...'
	sys.exit()

#Get estimated coefficients and fit
wbOpt = wbFit.x
wbFitted,wbCoef = nagini.gluDelayLstPen(wbOpt,aifFunc,interpTime,wbTac,midTime,flowWb,vbWb,wbCv.x,1,weights,True,metBool)

#Use coefficents to calculate all my parameters
wbParams = nagini.gluCalc(wbCoef,flowWb,vbWb,args.blood,args.dT)

#Create string for whole-brain parameter estimates
labels = ['gef','kOne','kTwo','kThree','kFour','cmrGlu',
          'alphaOne','alphaTwo','betaOne','betaTwo','netEx','influx','DV','conc','penalty']
values = [wbParams[0],wbParams[1],wbParams[2],wbParams[3],wbParams[4],wbParams[5],
          wbCoef[0],wbCoef[1],wbCoef[2],wbCoef[3],wbParams[6],wbParams[7],wbParams[8],wbParams[9],wbCv.x]
units = ['fraction','mLBlood/mLTissue/min','1/min','1/min','1/min','uMol/hg/min',
         '1/sec','1/sec','1/sec','1/sec','fraction','uMol/g/min','mLBlood/mLTissue','uMol/g','']
wbString = ''
for pIdx in range(len(labels)):
	wbString += '%s = %f (%s)\n'%(labels[pIdx],values[pIdx],units[pIdx])

#Add in delay estimate
wbString += 'delay = %f (min)\n'%(wbOpt[2]/60.0)

#Write out whole-brain results
try:
	#Write out values
	wbOut = open('%s_wbVals.txt'%(args.out[0]), "w")
	wbOut.write(wbString)
	wbOut.close()
except(IOError):
	print 'ERROR: Cannot write in output directory. Exiting...'
	sys.exit()

#Create whole brain fit figure
try:
	fig = plt.figure(1)
	gs = grid.GridSpec(1,2)

	#Make fit plot
	axOne = plt.subplot(gs[0,0])
	axOne.scatter(midTime,wbTac,s=40,c="black")
	axOne.plot(midTime,wbFitted,linewidth=3,label='Model Fit')
	axOne.set_xlabel('Time (seconds)')
	axOne.set_ylabel('Counts')
	axOne.set_title('Whole-Brain Time Activity Curve')
	axOne.legend(loc='upper left')

	#Make input function plot
	axTwo = plt.subplot(gs[0,1])
	axTwo.scatter(drawTime,corrCounts,s=40,c="black")
	aifFitted = aifFunc(drawTime)
	axTwo.plot(drawTime,aifFitted,linewidth=5,label=aifLabel)
	axTwo.set_xlabel('Time (seconds)')
	axTwo.set_ylabel('Counts')
	axTwo.set_title('Arterial Sampled Input function')

	#Interpolate input function and correct for delay
	cAif = aifFunc(interpTime+wbOpt[2])

	#Add shifted plot
	axTwo.plot(interpTime,cAif,linewidth=5,c='green',label='Delay Corrected')

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

#Don't do region estimation if user says not to
if args.noRoi == 1:
	nagini.writeArgs(args,args.out[0])
	sys.exit()

#Get number of parameters for roi optimization
nParam = wbOpt.shape[0]-1
roiInit = np.copy(wbOpt[0:nParam])

#Set default region bounds
lBounds = []; hBounds = []; bScale = [3,3]
for pIdx in range(nParam):
	if wbOpt[pIdx] > 0:
		lBounds.append(wbOpt[pIdx]/bScale[pIdx])
		hBounds.append(wbOpt[pIdx]*bScale[pIdx])
	elif wbOpt[pIdx] < 0:
		lBounds.append(wbOpt[pIdx]*bScale[pIdx])
		hBounds.append(wbOpt[pIdx]/bScale[pIdx])
	else:
		lBounds.append(wbBounds[0][pIdx])
		hBounds.append(wbBounds[1][pIdx])
roiBounds = np.array([lBounds,hBounds],dtype=np.float).T

#Setup for roi optmization
uRoi = np.unique(roiMasked)
nRoi = uRoi.shape[0]
roiParams = np.zeros((nRoi,nParam+14))

#Interpolate input function using delay
wbAif = aifFunc(interpTime+wbOpt[2])

#Get interpolated AIF integrated from start to end of pet times
wbAifPet = aifFunc(midTime+wbOpt[2])

#If necessary, convert AIF to plasma and correct for metabolites.
if metBool is True:
	pAif = wbAif*(1.19 + -0.002*interpTime/60.0)
	pAif *=  (1 - (4.983e-05*interpTime))
else:
	pAif = wbAif	

#Loop through every region
roiC = 0
for roiIdx in tqdm(range(nRoi),desc='Regions'):

	#Get regional pet data
	roiMask = roiMasked==uRoi[roiIdx]
	roiVoxels = petMasked[roiMask,:]
	roiTac = np.mean(roiVoxels,axis=0)

	#Get regional flow and vb
	roiVb = vbMasked[roiMask]
	roiFlow = flowMasked[roiMask]
	roiVbMean = np.mean(roiVb)
	roiFlowMean = np.mean(roiFlow)

	#Get concentration in compartment one
	cOneRoi = roiVbMean*wbAifPet

	try:

		#Run cross-valation to determine penalty
		roiCvArgs = (interpTime,wbAif,pAif,roiTac,midTime,cOneRoi,roiFlowMean,roiVbMean,wbOpt[0],weights,roiInit,roiBounds,metBool)
		roiCv = opt.minimize_scalar(nagini.cvOptRoi,bounds=([0,15]),args=roiCvArgs,options={'maxiter':50},method='Bounded',tol=0.1)

		#Run fit
		roiArgs = (interpTime,pAif,roiTac,midTime,cOneRoi,roiFlowMean,roiCv.x,wbOpt[0],weights,False)
		roiOpt = opt.minimize(nagini.gluAifLstPen,roiInit,args=roiArgs,method='L-BFGS-B',bounds=roiBounds,options={'maxls':100})

		#Extract coefficients
		roiFitted,roiCoef = nagini.gluAifLstPen(roiOpt.x,interpTime,pAif,roiTac,midTime,cOneRoi,roiFlowMean,roiCv.x,wbOpt[0],weights,True)

		#Caculate roi paramter values
		roiVals = nagini.gluCalc(roiCoef,roiFlowMean,roiVbMean,args.blood,args.dT)

		#Save common estimates
		roiParams[roiIdx,0:4] = roiCoef
		roiParams[roiIdx,4:14] = roiVals
		roiParams[roiIdx,14] = roiCv.x

		#Calculate residual
		roiResid = roiTac - roiFitted

		#Calculate normalized root mean square deviation
		roiRmsd = np.sqrt(np.sum(np.power(roiResid,2))/roiTac.shape[0]) / np.mean(roiTac)

		#Save residual
		roiParams[roiIdx,-1] = roiRmsd
		
	except(RuntimeError,ValueError):
		roiC += 1

#############
###Output!###
#############
print('Writing out results...')

#Set names for model images
paramNames = ['alphaOne','alphaTwo','betaOne','betaTwo','gef','kOne','kTwo','kThree','kFour','cmrGlu','netEx','influx','DV','conc','penalty','nRmsd']

#Do the writing for parameters
for iIdx in range(roiParams.shape[1]-1):

	#Write out regional data
	nib.Nifti1Image(roiParams[:,iIdx],affine=np.identity(4)).to_filename('%s_roiAvg_%s.nii.gz'%(args.out[0],paramNames[iIdx]))

#Write out root mean square images
nib.Nifti1Image(roiParams[:,-1],affine=np.identity(4)).to_filename('%s_%s.nii.gz'%(args.out[0],paramNames[-1]))

#Write out chosen arguments
nagini.writeArgs(args,args.out[0])

#Convergence output
try:
	#Open file
	cOut = open('%s_convergence.txt'%(args.out[0]), "w")

	#Write out ROI data
	cOut.write('%i of %i\n'%(roiC,nRoi))

	cOut.close()
except(IOError):
	print 'ERROR: Cannot write in output directory. Exiting...'
	sys.exit()
