
#!/usr/bin/python

###################
###Documentation###
###################

"""

fdgFourAifRidge.py: Calculates cmrGlu using a FDG scan and an arterial sampled input function

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
argParse.add_argument('cbv',help='CBV image for blood volume correction in mL/hg',nargs=1,type=str)
argParse.add_argument('out',help='Root for outputed files',nargs=1,type=str)
argParse.add_argument('-pen',help='Penalty term for whole-brain estimates. Default is 3.0',nargs=1,type=float,default=[3.0])
argParse.add_argument('-roiPen',help='Penalty term for ROI estimates. Default is 2.5',nargs=1,type=float,default=[2.5])
argParse.add_argument('-voxPen',help='Penalty term for voxel estimates. Default is 15.0',nargs=1,type=float,default=[15.0])
argParse.add_argument('-lc',help='Lumped Constant. Default is 0.52',nargs=1,type=float,default=[0.52])
argParse.add_argument('-cbf',help='CBF image. In mL/hg*min. If given, script will calculate GEF',nargs=1,type=str)
argParse.add_argument('-brain',help='Brain mask in PET space',nargs=1,type=str)
argParse.add_argument('-seg',help='Segmentation image used to create CBV averages.',nargs=1,type=str)
argParse.add_argument('-fwhm',help='Apply smoothing kernel to CBF and CBV',nargs=1,type=float)
argParse.add_argument('-fwhmSeg',help='Apply smoothing kerenl to CBV segmentation image',nargs=1,type=float)
argParse.add_argument('-noRoi',action='store_const',const=1,help='Do not perform region estimation. Implies -noVox')
argParse.add_argument('-noVox',action='store_const',const=1,help='Do not perform voxel estimation')
argParse.add_argument('-noFill',action='store_const',const=1,help='Do not replace nans with 6-neighbor average')
argParse.add_argument('-dT',help='Density of brain tissue in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-dB',help='Density of blood in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-oBound',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds for beta one, where bounds are 5.88E-5*[lower,upper]. Default is [0.2,5]')
argParse.add_argument('-tBound',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds for beta two, where bounds are 0.0038*[lower,upper]. Default is [0.2,5]')
argParse.add_argument('-dBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for delay parameter. Default is [-25,25]')
argParse.add_argument('-weighted',action='store_const',const=1,help='Run with weighted regression')
argParse.add_argument('-aifText',action='store_const',const=1,help='AIF is a simple 2 column file with sample times and decay corrected activity')
argParse.add_argument('-idaif',action='store_const',const=1,help='AIF is an image derived input function (1 column file with activity)')
argParse.add_argument('-interp',action='store_const',const=1,help='Interpolate AIF instead of fitting model')
argParse.add_argument('-noDelay',action='store_const',const=1,help='Do not estimate delay in AIF')
args = argParse.parse_args()

#Setup bounds and inits
if args.noDelay is None:
	wbBounds = [[0.1,0.1,-60],[10,10,60]]
else:
	wbBounds = [[0.1,0.1,0],[10,10,0]]
wbInit = [1,1,0]

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

#Load in the info file
info = nagini.loadInfo(args.info[0])

#Get pet start and end times. Assume start of first recorded frame is injection (has an offset which is subtracted out)
startTime = info[:,0] - info[0,0]
midTime = info[:,1] - info[0,0]
endTime = info[:,2] + startTime

#AIF loading logic
if args.aifText is None and args.idaif is None:

	#Load dta file
	dta = nagini.loadDta(args.dta[0])

	#Get decay correct activity
	drawTime,corrCounts = nagini.corrDta(dta,6586.26,args.dB,toDraw=False,dTime=False)

	#Apply pie factor and scale factor to AIF
	corrCounts /= (args.pie[0] * 0.06 )

elif args.idaif == 1:

	#Load in idaif
	corrCounts = np.loadtxt(args.dta[0])

	#Get time variable
	drawTime = np.copy(midTime)

else:

	#Load in simple aif text file
	aifData = np.loadtxt(args.dta[0])

	#Rename so variables have same names
	drawTime = aifData[:,0]; corrCounts = aifData[:,1]

#Apply time offset that was applied to images
corrCounts *= np.exp(np.log(2)/6586.26*info[0,0])

#Load image headers
pet = nagini.loadHeader(args.pet[0])
roi = nagini.loadHeader(args.roi[0])
cbv = nagini.loadHeader(args.cbv[0])

#Check to make sure dimensions match
if pet.shape[3] != info.shape[0] or pet.shape[0:3] != cbv.shape[0:3] or pet.shape[0:3] != roi.shape[0:3]:
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
maskData = np.where(np.logical_and(roiData!=0,np.logical_and(brainData!=0,cbvData!=0)),1,0)

#Prep CBV data
if args.fwhm is None:
	#Do not smooth,just mask
	cbvMasked = cbvData[maskData>0].flatten()
else:
	#Prepare for smoothing
	voxSize = pet.header.get_zooms()
	sigmas = np.divide(args.fwhm[0]/np.sqrt(8.0*np.log(2.0)),voxSize[0:3])

	#Smooth and mask data
	cbvMasked = filt.gaussian_filter(cbvData,sigmas)[maskData>0].flatten()

#Get cbv in units mlB/mlT
vbMasked = cbvMasked / 100.0 * args.dT
vbWb = np.mean(vbMasked)

#CBF logic
if args.cbf is not None:

	#Load in cbf header
	cbf = nagini.loadHeader(args.cbf[0]).reshape(pet.shape[0:3])

	#Make sure its dimensions match
	if pet.shape[0:3] != cbf.shape[0:3]:
		print 'ERROR: CBF dimensions do not match data. Please check...'
		sys.exit()

	#Load in cbf data
	cbfData = cbf.get_data()

	#Smoothing logic
	if args.fwhm is None:
		#Do not smooth,just mask
		cbfMasked = cbfData[maskData>0].flatten()
	else:
		#Smooth and mask data
		cbfMasked = filt.gaussian_filter(cbfData,sigmas)[maskData>0].flatten()

	#Get 1/s flow
	flowMasked = cbfMasked / cbvMasked / 60.0

#Flatten the PET images and then mask
petMasked = nagini.reshape4d(petData)[maskData.flatten()>0,:]
roiMasked = roiData[maskData>0].flatten()

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
print vbWb
###################
###Model Fitting###
###################
print ('Beginning fitting procedure...')

#Use either uniform or John Lee Jeffery style weights
if args.weighted is None:
	weights = np.ones_like(wbTac)
else:
	weights = 1.0 / (midTime*np.log(midTime[-1]/midTime[0]))
	weights /= np.sum(weights*info[:,2])

#Arguments for whole-brain model fit
wbArgs = (aifFunc,interpTime,wbTac,midTime,vbWb,args.pen[0],[1,1],weights)

#Attempt to fit model to whole-brain curve
wbFit = opt.minimize(nagini.fdgFourDelayLstPen,wbInit,args=wbArgs,method='L-BFGS-B',bounds=np.array(wbBounds).T,options={'maxls':100})

#Make sure we converged
if wbFit.success is False:
	print 'ERROR: Cannot estimate model on whole-brain curve. Exiting...'
	sys.exit()

#Get estimated coefficients and fit
wbOpt = wbFit.x
wbFitted,wbCoef = nagini.fdgFourDelayLstPen(wbOpt,aifFunc,interpTime,wbTac,midTime,vbWb,args.pen[0],[1,1],weights,coefs=True)

#Use coefficents to calculate all my parameters
if args.cbf is None:
	wbParams = nagini.fdgFourCalc(wbCoef,args.blood,args.dT,args.lc[0])
else:
	#Get mean flow
	flowWb = np.mean(flowMasked)
	wbParams = nagini.fdgFourCalc(wbCoef,args.blood,args.dT,args.lc[0],flow=flowWb,vb=vbWb)

#Create string for whole-brain parameter estimates
labels = ['kOne','kTwo','kThree','kFour','cmrGlu',
          'alphaOne','alphaTwo','betaOne','betaTwo',
	  'influx','DV','conc','delay','kI']
values = [wbParams[0],wbParams[1],wbParams[2],wbParams[3],wbParams[4],
          wbCoef[0],wbCoef[1],wbCoef[2],wbCoef[3],
	  wbParams[5],wbParams[6],wbParams[7],wbOpt[2]/60.0,
          (wbParams[0]*wbParams[2])/(wbParams[1]+wbParams[2])]
units = ['mLBlood/mLTissue/min','1/min','1/min','1/min','uMol/hg/min',
         '-','-','-','-',
	 'uMol/g/min','mLBlood/mLTissue','uMol/g','min',
         'mLBlood/mLTissue/min']

#Add in labels if we have cbf
if args.cbf is not None:
	labels.extend(['gef','netEx'])
	values.extend([gefWb,wbNet])
	units.extend(['fraction','fraction'])

#Create string for whole-brain parameter estimates
wbString = ''
for pIdx in range(len(labels)):
	wbString += '%s = %f (%s)\n'%(labels[pIdx],values[pIdx],units[pIdx])

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

#Don't do region/voxel estimation if user says not to
if args.noRoi == 1:
	nagini.writeArgs(args,args.out[0])
	sys.exit()

#Get number of parameters for roi/voxel optimization
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

#How many metabolic values are we goign to have to caculate?
if args.cbf is None:
	nMeas = 11
else:
	nMeas = 13

#Setup for roi optmization
uRoi = np.unique(roiMasked)
nRoi = uRoi.shape[0]
roiParams = np.zeros((nRoi,nParam+nMeas))
if args.noVox != 1:
	voxParams = np.zeros((roiMasked.shape[0],nParam+nMeas)); voxParams[:] = np.nan

#Interpolate input function using delay
if args.noDelay is None:
	shift = 0
else:
	shift = wbOpt[2]
wbAif = aifFunc(interpTime+shift)

#Get interpolated AIF at pet middle time
wbAifPet = aifFunc(midTime+shift)

#Convert AIF to plasma
pAif = (1.071966 + -1.07294E-5*interpTime)*wbAif

#Loop through every region
roiC = 0; voxC = 0
for roiIdx in tqdm(range(nRoi),desc='Regions'):

	#Get regional pet data
	roiMask = roiMasked==uRoi[roiIdx]
	roiVoxels = petMasked[roiMask,:]
	roiTac = np.mean(roiVoxels,axis=0)

	#Get regional blood volume
	roiVb = vbMasked[roiMask]
	roiVbMean = np.mean(roiVb)

	#Get concentration in compartment one
	cOneRoi = roiVbMean*wbAifPet

	try:
		#Run fit
		roiArgs = (interpTime,pAif,roiTac,midTime,cOneRoi,args.roiPen[0],wbOpt[0:2],weights)
		roiOpt = opt.minimize(nagini.fdgFourLstPen,roiInit,args=roiArgs,method='L-BFGS-B',bounds=roiBounds,options={'maxls':100})

		#Extract coefficients
		roiFitted,roiCoef = nagini.fdgFourLstPen(roiOpt.x,interpTime,pAif,roiTac,midTime,cOneRoi,args.roiPen[0],wbOpt[0:2],weights,coefs=True)

		#Save coefficients
		roiParams[roiIdx,0:4] = roiCoef

		#Caculate roi metabolic parameter values
		if args.cbf is None:
			roiParams[roiIdx,4:12] = nagini.fdgFourCalc(roiCoef,args.blood,args.dT,args.lc[0])
		else:
			roiParams[roiIdx,4:14] =  nagini.fdgFourCalc(roiCoef,args.blood,args.dT,args.lc[0],vb=roiVbMean,flow=np.mean(flowMasked[roiMask]))

		#Calculate residual
		roiResid = roiTac - roiFitted

		#Calculate normalized root mean square deviation
		roiRmsd = np.sqrt(np.sum(np.power(roiResid,2))/roiTac.shape[0]) / np.mean(roiTac)

		#Save residual
		roiParams[roiIdx,-1] = roiRmsd

	except(RuntimeError,ValueError):
		roiC += 1

	#Don't go on if user doesn't want voxel results
	if args.noVox == 1:
		continue

	#Create data structure for voxels within roiOpt
	nVox = roiVoxels.shape[0]
	roiVoxParams = np.zeros((nVox,roiParams.shape[1]))

	#If region optimization was a success use it for voxel
	if roiOpt.success is True:
		voxInit = np.copy(roiOpt.x)
	else:
		voxInit = np.copy(wbOpt)
		roiC += 1

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
		voxVb = roiVb[voxIdx]

		#Calculate concentration in compartment one
		cOneVox = voxVb*wbAifPet

		try:
			#Run fit
			voxArgs = (interpTime,pAif,voxTac,midTime,cOneVox,args.voxPen[0],voxInit[0:2],weights)
			voxOpt = opt.minimize(nagini.fdgFourLstPen,voxInit,args=voxArgs,method='L-BFGS-B',bounds=voxBounds,options={'maxls':100})

			#Extract coefficients
			voxFitted,voxCoef = nagini.fdgFourLstPen(voxOpt.x,interpTime,pAif,voxTac,midTime,cOneVox,args.voxPen[0],voxInit[0:2],weights,coefs=True)

			#Save coefficients
			roiVoxParams[voxIdx,0:3] = voxCoef

			#Caculate roi metabolic parameter values
			if args.cbf is None:
				roiVoxParams[voxIdx,4:12] = nagini.fdgFourCalc(voxCoef,args.blood,args.dT,args.lc[0])
			else:
				roiVoxParams[voxIdx,4:14] =  nagini.fdgFourCalc(voxCoef,args.blood,args.dT,args.lc[0],vb=voxVb,flow=flowMasked[roiMask][voxIdx])

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

#Prepare for voxelwise writing
if args.noVox !=1:

	#Put voxelwise parameters back into image space
	voxData = np.zeros((maskData.shape[0],maskData.shape[1],maskData.shape[2],voxParams.shape[1]))
	voxData[maskData>0,:] = voxParams

	#Fill nan logic
	if args.noFill != 1 and voxC > 0:

		#Loop through parameters that we need to replace the nans
		for cIdx in range(0,4):

			#Extract coefficinet data
			cData = voxData[:,:,:,cIdx]

			#Setup for getting nearest neighbors
			if cIdx == 0:

				#Get coordinates for nans
				nanBool = np.isnan(cData); nNan = np.sum(nanBool)
				cNanIdx = np.array(np.where(nanBool)).T

				#Get coordinates for useable data
				cUseIdx = np.array(np.where(np.logical_and(cData!=0.0,np.logical_not(nanBool)))).T

				#Make tree for finding nearest neighbours
				tree = spat.KDTree(cUseIdx)

				#Get 6 nearest neighbors for each nan
				nanD,nanNe = tree.query(cNanIdx,6)

				#Convert neighbors to image coordinates
				interpIdx = cUseIdx[nanNe]

			#Loop through nans
			for nIdx in range(nNan):

				#Get values for neighbors
				neVal = cData[interpIdx[nIdx,:,0],interpIdx[nIdx,:,1],interpIdx[nIdx,:,2]]

				#Calculate distance weighted average
				neAvg = np.sum(neVal/nanD[nIdx])/np.sum(1.0/nanD[nIdx])

				#Replace nan with weighted average
				cData[cNanIdx[nIdx,0],cNanIdx[nIdx,1],cNanIdx[nIdx,2]] = neAvg

			#Replace data
			voxData[:,:,:,cIdx] = cData

		#Extract values that replaced the nans
		coefReplace = voxData[cNanIdx[:,0],cNanIdx[:,1],cNanIdx[:,2],0:4].reshape((nNan,4))

		#Calculate parameter values for replaced values
		if args.cbf is None:
			paramReplace = nagini.fdgFourCalc(coefReplace,args.blood,args.dT,args.lc[0])
		else:

			#Get flow values needed for replacement
			flowData = np.zeros_like(maskData,dtype=np.float64); flowData[maskData>0] = flowMasked
			flowReplace = flowData[cNanIdx[:,0],cNanIdx[:,1],cNanIdx[:,2]].flatten()

			#Get blood volume values needed for replacement
			vbData = np.zeros_like(maskData,dtype=np.float64); vbData[maskData>0] = vbMasked
			vbReplace = vbData[cNanIdx[:,0],cNanIdx[:,1],cNanIdx[:,2]].flatten()

			#Get values fore replacement
			paramReplace = nagini.fdgFourCalc(voxCoef,args.blood,args.dT,args.lc[0],vb=vbReplace,flow=flowReplace)

		#Apply replaced values
		voxData[cNanIdx[:,0],cNanIdx[:,1],cNanIdx[:,2],4:14] = paramReplace

#############
###Output!###
#############
print('Writing out results...')

#Set names for model images
if args.cbf is None:
	paramNames = ['alphaOne','alphaTwo','betaOne','betaTwo','kOne','kTwo','kThree','kFour','cmrGlu','influx','DV','conc','nRmsd']
else:
	paramNames = ['alphaOne','alphaTwo','betaOne','betaTwo','kOne','kTwo','kThree','kFour','cmrGlu','influx','DV','conc','fef','netEx','nRmsd']

#Do the writing for parameters
for iIdx in range(roiParams.shape[1]-1):

	#Write out regional data
	nib.Nifti1Image(roiParams[:,iIdx],affine=np.identity(4)).to_filename('%s_roiAvg_%s.nii.gz'%(args.out[0],paramNames[iIdx]))

	#Write out voxelwise data
	if args.noVox != 1:

		#What to call the output image
		pName = '%s_%s'%(args.out[0],paramNames[iIdx])

		#Create image to write out
		outImg = nib.Nifti1Image(voxData[:,:,:,iIdx],pet.affine,header=pet.header)

		#Then do the writing
		try:
			outImg.to_filename('%s.nii.gz'%(pName))
		except (IOError):
			print 'ERROR: Cannot save image at %s.'%(pName)
			sys.exit()

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
	cOut.write('%i of %i\n'%(roiC,nRoi))

	#Write out voxel data if necessary
	if args.noVox !=1:
		cOut.write('%i of %i'%(voxC,voxParams.shape[0]))

	cOut.close()
except(IOError):
	print 'ERROR: Cannot write in output directory. Exiting...'
	sys.exit()
