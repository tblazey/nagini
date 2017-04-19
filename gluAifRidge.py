
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
argParse.add_argument('pet',help='Nifti cmrGlu image',nargs=1,type=str)
argParse.add_argument('info',help='Yi Su style info file',nargs=1,type=str)
argParse.add_argument('dta',help='Dta file with drawn samples',nargs=1,type=str)
argParse.add_argument('pie',help='Pie calibration factor',nargs=1,type=float)
argParse.add_argument('blood',help='Blood glucose concentration mg/dL',type=float)
argParse.add_argument('roi',help='ROIs for two stage procedure',nargs=1,type=str)
argParse.add_argument('cbf',help='CBF image. In mL/hg*min',nargs=1,type=str)
argParse.add_argument('cbv',help='CBV image for blood volume correction in mL/hg',nargs=1,type=str)
argParse.add_argument('out',help='Root for outputed files',nargs=1,type=str)
argParse.add_argument('-pen',help='Penalty term for whole-brain estimates. Default is 3.0',nargs=1,type=float,default=[3.0])
argParse.add_argument('-roiPen',help='Penalty term for ROI estimates. Default is 3.0',nargs=1,type=float,default=[3.0])
argParse.add_argument('-voxPen',help='Penalty term for voxel estimates. Default is 3.0',nargs=1,type=float,default=[3.0])
argParse.add_argument('-brain',help='Brain mask in PET space',nargs=1,type=str)
argParse.add_argument('-seg',help='Segmentation image used to create CBV averages.',nargs=1,type=str)
argParse.add_argument('-fwhm',help='Apply smoothing kernel to CBF and CBV',nargs=1,type=float)
argParse.add_argument('-fwhmSeg',help='Apply smoothing kerenl to CBV segmentation image',nargs=1,type=float)
argParse.add_argument('-noRoi',action='store_const',const=1,help='Do not perform region estimation. Implies -noVox')
argParse.add_argument('-noVox',action='store_const',const=1,help='Do not perform voxel estimation')
argParse.add_argument('-dT',help='Density of brain tissue in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-dB',help='Density of blood in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-oBound',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds for beta one, where bounds are 0.005*[lower,upper]. Defalt is [0.2,5]')
argParse.add_argument('-tBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for beta two, where bounds are 0.00016*[lower,upper]. Defalt is [0.2,5]')
argParse.add_argument('-dBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for delay parameter. Default is [-25,25]')
args = argParse.parse_args()

#Setup bounds and inits
wbBounds = [[0.2,.2,-25],[5,5,25]]; wbInit = [1,1,0]

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
import scipy.ndimage.filters as filt
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
cbfData = cbf.get_data()
cbvData = cbv.get_data()
roiData = roi.get_data()

#Segmentation logic
if args.seg is not None:

	#Load in segmentation header
	seg = nagini.loadHeader(args.seg[0])

	#Make sure its dimensions match
	if pet.shape[0:3] != seg.shape[0:3]:
		print 'ERROR: Segmentation dimensions do not match data. Please check...'
		sys.exit()

	#Load in segmentation data'
	segData = seg.get_data()

	#Make a new segmentation where we have high CBV
	segData[cbvData>=8.0] = np.max(segData)+1

	#Get CBV ROI averages
	cbvAvgs = nagini.roiAvg(cbvData.flatten(),segData.flatten(),min=0.0)

	#Remask CBV image from ROI averages
	cbvData = nagini.roiBack(cbvAvgs,segData.flatten()).reshape(cbvData.shape)

	#Smooth
	if args.fwhmSeg is not None:
			roiSize = pet.header.get_zooms()
			sigmas = np.divide(args.fwhmSeg[0]/np.sqrt(8.0*np.log(2.0)),roiSize[0:3])
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
	roiSize = pet.header.get_zooms()[0:3]
	sigmas = np.divide(args.fwhm[0]/np.sqrt(8.0*np.log(2.0)),roiSize[0:3])

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

#Combine PET start and end times into one array
petTime = np.stack((startTime,endTime),axis=1)

#Get decay corrected blood curve to injection
drawTime,corrCounts = nagini.corrDta(dta,1220.04,args.dB,toDraw=False)

#Apply time offset that was applied to images
corrCounts *= np.exp(np.log(2)/1220.04*info[0,0])

#Apply pie factor and scale factor to AIF
corrCounts /= (args.pie[0] * 0.06 )

#########################
###Data Pre-Processing###
#########################
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

#Get interpolation times as start to end of pet scan with aif sampling rate
sampTime = np.min(np.diff(drawTime))
interpTime = np.arange(np.floor(startTime[0]),np.ceil(endTime[-1]+sampTime),sampTime)

#Get whole-brain tac
wbTac = np.mean(petMasked,axis=0)
wbSum = np.sum(wbTac)

###################
###Model Fitting###
###################
print ('Beginning fitting procedure...')

#Arguments for whole-brain model fit
wbArgs = (aifGlobal.x,interpTime,wbTac,petTime,wbSum,flowWb,vbWb,args.pen[0],1)

#Attempt to fit model to whole-brain curve
wbFit = opt.minimize(nagini.gluDelayLstPen,wbInit,args=wbArgs,method='L-BFGS-B',bounds=np.array(wbBounds).T,options={'maxls':100})
if wbFit.success is False:
	print 'ERROR: Cannot estimate model on whole-brain curve. Exiting...'
	sys.exit()

#Get estimated coefficients and fit
wbOpt = wbFit.x
wbFitted,wbCoef = nagini.gluDelayLstPen(wbOpt,aifGlobal.x,interpTime,wbTac,petTime,wbSum,flowWb,vbWb,args.pen[0],1,coefs=True)

#Calculate rate constants from my alphas and betas
kOneWb = wbCoef[0] + (wbCoef[1]/(wbCoef[2]-wbCoef[3])) - ((wbCoef[1]*wbCoef[3])/((wbCoef[3]-wbCoef[2])*(flowWb-wbCoef[2])))
kThreeWb = wbCoef[1]/kOneWb
kTwoWb = wbCoef[2]-kThreeWb
kFourWb = wbCoef[3]

#Calculate gef
gefWb = kOneWb / (flowWb*vbWb)

#Calculate metabolic rate
gluScale = 333.0449 / args.dT
wbCmr = (kOneWb*kThreeWb*args.blood)/(kTwoWb+kThreeWb) * gluScale

#Calculate net extraction
wbNet = wbCoef[1]/(wbCoef[2]*flowWb*vbWb)

#Calculate influx
wbIn = kOneWb*args.blood*gluScale/100.0

#Calculate distrubtion volume
wbDv =  kOneWb/(wbCoef[2]*args.dT)

#Compute tissue concentration
wbConc = wbDv * args.blood * 0.05550748

#Create string for whole-brain parameter estimates
labels = ['gef','kOne','kTwo','kThree','kFour','cmrGlu','alphaOne','alphaTwo','betaOne','betaTwo','netEx','infux','DV','conc']
values = [gefWb,kOneWb*60,kTwoWb*60,kThreeWb*60,kFourWb*60,wbCmr,wbCoef[0],wbCoef[1],wbCoef[2],wbCoef[3],wbNet,wbIn,wbDv,wbConc]
units = ['fraction','mLBlood/mLTissue/min','1/min','1/min','1/min','uMol/hg/min','1/sec','1/sec','1/sec','1/sec','fraction','uMol/g/min','mLBlood/mLTissue','uMol/g']
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
	aifFitted = nagini.fengModel(drawTime,aifGlobal.x[0],aifGlobal.x[1],aifGlobal.x[2],
								  aifGlobal.x[3],aifGlobal.x[4],aifGlobal.x[5],aifGlobal.x[6])
	axTwo.plot(drawTime,aifFitted,linewidth=5,label='Spline Fit')
	axTwo.set_xlabel('Time (seconds)')
	axTwo.set_ylabel('Counts')
	axTwo.set_title('Arterial Sampled Input function')

	#Interpolate input function and correct for delay
	cAif = nagini.fengModel(drawTime+wbOpt[2],aifGlobal.x[0],aifGlobal.x[1],aifGlobal.x[2],
							aifGlobal.x[3],aifGlobal.x[4],aifGlobal.x[5],aifGlobal.x[6])
	cAif *= np.exp(np.log(2)/1220.04*wbOpt[2])

	#Add shifted plot
	axTwo.plot(drawTime,cAif,linewidth=5,c='green',label='Delay Corrected')

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

#Don't do region/roiel estimation if user says not to
if args.noRoi == 1:
	nagini.writeArgs(args,args.out[0])
	sys.exit()

#Get number of parameters for roi/voxel optmization
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
roiParams = np.zeros((nRoi,nParam+13))
if args.noVox != 1:
	voxParams = np.zeros((roiMasked.shape[0],nParam+13))

#Interpolate input function using delay
wbAif = nagini.fengModel(interpTime+wbOpt[2],aifGlobal.x[0],aifGlobal.x[1],aifGlobal.x[2],
						 aifGlobal.x[3],aifGlobal.x[4],aifGlobal.x[5],aifGlobal.x[6]) * np.exp(np.log(2)/1220.04*wbOpt[2])
wbAifStart = nagini.fengModel(petTime[:,0]+wbOpt[2],aifGlobal.x[0],aifGlobal.x[1],aifGlobal.x[2],
							  aifGlobal.x[3],aifGlobal.x[4],aifGlobal.x[5],aifGlobal.x[6]) * np.exp(np.log(2)/1220.04*wbOpt[2])
wbAifEnd =  nagini.fengModel(petTime[:,1]+wbOpt[2],aifGlobal.x[0],aifGlobal.x[1],aifGlobal.x[2],
							 aifGlobal.x[3],aifGlobal.x[4],aifGlobal.x[5],aifGlobal.x[6]) * np.exp(np.log(2)/1220.04*wbOpt[2])

#Get interpolated AIF integrated from start to end of pet times
wbAifPet = (wbAifStart+wbAifEnd)/2.0

#Convert AIF to plasma
pAif = wbAif*(1.19 + -0.002*interpTime/60.0)

#Remove metabolites from plasma input function
pAif *=  (1 - (4.983e-05*interpTime))

#Loop through every region
roiC = 0; voxC = 0
for roiIdx in tqdm(range(nRoi),desc='Regions'):

	#Get regional pet data
	roiMask = roiMasked==uRoi[roiIdx]
	roiVoxels = petMasked[roiMask,:]
	roiTac = np.mean(roiVoxels,axis=0)
	roiSum = np.sum(roiTac)

	#Get regional flow and vb
	roiVb = vbMasked[roiMask]
	roiFlow = flowMasked[roiMask]
	roiVbMean = np.mean(roiVb)
	roiFlowMean = np.mean(roiFlow)

	#Get concentration in compartment one
	cOneRoi = roiVbMean*wbAifPet

	try:
		#Run fit
		roiArgs = (interpTime,pAif,roiTac,petTime,roiSum,cOneRoi,roiFlowMean,args.roiPen[0],wbOpt[0])
		roiOpt = opt.minimize(nagini.gluAifLstPen,roiInit,args=roiArgs,method='L-BFGS-B',bounds=roiBounds,options={'maxls':100})

		#Extract coefficients
		roiFitted,roiCoef = nagini.gluAifLstPen(roiOpt.x,interpTime,pAif,roiTac,petTime,roiSum,cOneRoi,roiFlowMean,args.roiPen[0],wbOpt[0],coefs=True)

		#Calculate rate constants from my alphas and betas
		kOneRoi = roiCoef[0] + (roiCoef[1]/(roiCoef[2]-roiCoef[3])) - ((roiCoef[1]*roiCoef[3])/((roiCoef[3]-roiCoef[2])*(roiFlowMean-roiCoef[2])))
		kThreeRoi = roiCoef[1]/kOneRoi
		kTwoRoi = roiCoef[2]-kThreeRoi
		kFourRoi = roiCoef[3]

		#Calculate gef
		gefRoi = kOneRoi / (roiFlowMean*roiVbMean)

		#Calculate metabolic rate
		cmrRoi = (kOneRoi*kThreeRoi*args.blood)/(kTwoRoi+kThreeRoi) * gluScale

		#Calculate net extraction
		roiNet = roiCoef[1]/(roiCoef[2]*roiFlowMean*roiVbMean)

		#Calculate influx
		roiIn = kOneRoi*args.blood*gluScale/100.0

		#Calculate distrubtion volume
		roiDv =  kOneRoi/(roiCoef[2]*args.dT)

		#Compute tissue concentration
		roiConc = roiDv * args.blood * 0.05550748

		#Save common estimates
		roiParams[roiIdx,0] = gefRoi
		roiParams[roiIdx,1] = kOneRoi*60.0
		roiParams[roiIdx,2] = kTwoRoi*60.0
		roiParams[roiIdx,3] = kThreeRoi*60.0
		roiParams[roiIdx,4] = kFourRoi*60.0
		roiParams[roiIdx,5] = cmrRoi
		roiParams[roiIdx,6] = roiCoef[0]
		roiParams[roiIdx,7] = roiCoef[1]
		roiParams[roiIdx,8] = roiCoef[2]
		roiParams[roiIdx,9] = roiCoef[3]
		roiParams[roiIdx,10] = roiNet
		roiParams[roiIdx,11] = roiIn
		roiParams[roiIdx,12] = roiDv
		roiParams[roiIdx,13] = roiConc

		#Calculate residual
		roiResid = roiTac - roiFitted

		#Calculate normalized root mean square deviation
		roiRmsd = np.sqrt(np.sum(np.power(roiResid,2))/roiTac.shape[0]) / np.mean(roiTac)

		#Save residual
		roiParams[roiIdx,-1] = roiRmsd

	except(RuntimeError,ValueError):
		regC += 1

	#Don't go on if user doesn't want voxel results
	if args.noVox == 1:
		continue

	#Create data structure for voxels within roiOpt
	nVox = roiVoxels.shape[0]
	roiVoxParams = np.zeros((nVox,nParam+13))

	#If region optimization was a success use it for voxel
	if roiOpt.success is True:
		voxInit = np.copy(roiOpt.x)
	else:
		voxInit = np.copy(wbOpt)
		regC += 1

	#Set bounds for voxel
	lBounds = []; hBounds = []; bScale = [2,2]
	for pIdx in range(nParam):
		if wbOpt[pIdx] > 0:
			lBounds.append(voxInit[pIdx]/bScale[pIdx])
			hBounds.append(voxInit[pIdx]*bScale[pIdx])
		elif wbOpt[pIdx] < 0:
			lBounds.append(voxInit[pIdx]*bScale[pIdx])
			hBounds.append(voxInit[pIdx]/bScale[pIdx])
		else:
			lBounds.append(roiOpt.x[0][pIdx])
			hBounds.append(roiOpt.x[1][pIdx])
	voxBounds = np.array([lBounds,hBounds],dtype=np.float).T

	#Now loop through voxels
	for voxIdx in tqdm(range(nVox),desc='Voxels within region %i'%(roiIdx),leave=False):

		#Get voxel data
		voxTac = roiVoxels[voxIdx,:]
		voxSum = np.sum(voxTac)
		voxVb = roiVb[voxIdx]
		voxFlow = roiFlow[voxIdx]

		#Calculate concentration in compartment one
		cOneVox = voxVb*wbAifPet

		try:
			#Run fit
			voxArgs = (interpTime,pAif,voxTac,petTime,voxSum,cOneVox,voxFlow,args.voxPen[0],voxInit[0])
			voxOpt = opt.minimize(nagini.gluAifLstPen,voxInit,args=voxArgs,method='L-BFGS-B',bounds=voxBounds,options={'maxls':100})

			#Extract coefficients
			voxFitted,voxCoef = nagini.gluAifLstPen(voxOpt.x,interpTime,pAif,voxTac,petTime,voxSum,cOneVox,voxFlow,args.voxPen[0],voxInit[0],coefs=True)

			#Calculate rate constants from my alphas and betas
			kOneVox = voxCoef[0] + (voxCoef[1]/(voxCoef[2]-voxCoef[3])) - ((voxCoef[1]*voxCoef[3])/((voxCoef[3]-voxCoef[2])*(voxFlow-voxCoef[2])))
			kThreeVox = voxCoef[1]/kOneVox
			kTwoVox = voxCoef[2]-kThreeVox
			kFourVox = voxCoef[3]

			#Calculate gef
			gefVox = kOneVox / (voxFlow*voxVb)

			#Calculate metabolic rate
			cmrVox = (kOneVox*kThreeVox*args.blood)/(kTwoVox+kThreeVox) * gluScale

			#Calculate net extraction
			voxNet = voxCoef[1]/(voxCoef[2]*voxFlow*voxVb)

			#Calculate influx
			voxIn = kOneVox*args.blood*gluScale/100.0

			#Calculate distrubtion volume
			voxDv =  kOneVox/(voxCoef[2]*args.dT)

			#Compute tissue concentration
			voxConc = voxDv * args.blood * 0.05550748

			#Save common estimates
			roiVoxParams[voxIdx,0] = gefVox
			roiVoxParams[voxIdx,1] = kOneVox*60.0
			roiVoxParams[voxIdx,2] = kTwoVox*60.0
			roiVoxParams[voxIdx,3] = kThreeVox*60.0
			roiVoxParams[voxIdx,4] = kFourVox*60.0
			roiVoxParams[voxIdx,5] = cmrVox
			roiVoxParams[voxIdx,6] = voxCoef[0]
			roiVoxParams[voxIdx,7] = voxCoef[1]
			roiVoxParams[voxIdx,8] = voxCoef[2]
			roiVoxParams[voxIdx,9] = voxCoef[3]
			roiVoxParams[voxIdx,10] = voxNet
			roiVoxParams[voxIdx,11] = voxIn
			roiVoxParams[voxIdx,12] = voxDv
			roiVoxParams[voxIdx,13] = voxConc

			#Calculate residual
			voxResid = voxTac - voxFitted

			#Calculate normalized root mean square deviation
			voxRmsd = np.sqrt(np.sum(np.power(voxResid,2))/voxTac.shape[0]) / np.mean(voxTac)

			#Save residual
			roiVoxParams[voxIdx,-1] = voxRmsd

		except(RuntimeError,ValueError):
			voxC += 1

		#Check to see if we converged
		if voxOpt.success is False:
			voxC += 1

		#Save results from voxel loop
		voxParams[roiMask,:] = roiVoxParams

#############
###Output!###
#############
print('Writing out results...')

#Set names for model images
paramNames = ['gef','kOne','kTwo','kThree','kFour','cmrGlu','alphaOne','alphaTwo','betaOne','betaTwo','netEx','influx','DV','conc','nRmsd']

#Do the writing for parameters
for iIdx in range(roiParams.shape[1]-1):

	#Write out regional data
	nib.Nifti1Image(roiParams[:,iIdx],affine=np.identity(4)).to_filename('%s_roiAvg_%s.nii.gz'%(args.out[0],paramNames[iIdx]))

	#Write out voxelwise data
	if args.noVox != 1:
		nagini.writeMaskedImage(voxParams[:,iIdx],maskData.shape,maskData,pet.affine,pet.header,'%s_%s'%(args.out[0],paramNames[iIdx]))

#Write out root mean square images
nib.Nifti1Image(roiParams[:,-1],affine=np.identity(4)).to_filename('%s_%s.nii.gz'%(args.out[0],paramNames[-1]))
if args.noVox != 1:
	nagini.writeMaskedImage(voxParams[:,-1],maskData.shape,maskData,pet.affine,pet.header,'%s_%s'%(args.out[0],paramNames[-1]))

#Write out chosen arguments
nagini.writeArgs(args,args.out[0])

#Convergence output
try:
	#Open file
	cOut = open('%s_convergence.txt'%(args.out[0]), "w")

	#Write out ROI data
	cOut.write('%i of %i'%(roiC,nRoi))

	#Write out voxel data if necessary
	if args.noVox !=1:
		cOut.wirite('%i of %i'%(voxC,voxParams.shape[0]),"w")

	cOut.close()
except(IOError):
	print 'ERROR: Cannot write in output directory. Exiting...'
	sys.exit()

