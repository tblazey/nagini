
#!/usr/bin/python

###################
###Documentation###
###################

"""

gluAif.py: Calculates cmrGlu using a FDG scan and an arterial sampled input function

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
argParse.add_argument('-bBound',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds for beta one, where bounds are 0.0038*[lower,upper]. Defalt is [0.2,5]')
argParse.add_argument('-dBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for delay parameter. Default is [-25,25]')
argParse.add_argument('-weighted',action='store_const',const=1,help='Run with weighted regression')
argParse.add_argument('-aifText',action='store_const',const=1,help='AIF is a simple 2 column file with sample times and decay corrected activity')
args = argParse.parse_args()

#Setup bounds and inits
wbBounds = [[0.2,-25],[5,25]]; wbInit = [1,0]

#Loop through bounds
uBounds = [args.bBound,args.dBound]
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

#AIF loading logic
if args.aifText is None:

	#Load dta file
	dta = nagini.loadDta(args.dta[0])

	#Get decay correct activity
	drawTime,corrCounts = nagini.corrDta(dta,6586.26,args.dB,toDraw=False,dTime=False)

	#Apply pie factor and scale factor to AIF
	corrCounts /= (args.pie[0] * 0.06 )

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
maskData = np.where(np.logical_and(roiData!=0,np.logical_and(brainData!=0,cbvData!=0),1,0))

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

#Get pet start and end times. Assume start of first recorded frame is injection (has an offset which is subtracted out)
startTime = info[:,0] - info[0,0]
midTime = info[:,1] - info[0,0]
endTime = info[:,2] + startTime

#Combine PET start and end times into one array
petTime = np.stack((startTime,endTime),axis=1)

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
sampTime = np.min(np.diff(np.sort(drawTime)))
interpTime = np.arange(np.floor(startTime[0]),np.ceil(endTime[-1]+sampTime),sampTime)

#Get whole-brain tac
wbTac = np.mean(petMasked,axis=0)

#Construct initial weights
weights = np.ones_like(wbTac)

###################
###Model Fitting###
###################
print ('Beginning fitting procedure...')

#Set number of iterations based upon whether or not we are going to use weights
if args.weighted is None:
	wbIter = 1
else:
	wbIter = 5

#Loop through whole-brain fitting iterations
for wbIdx in range(wbIter):

	#Arguments for whole-brain model fit
	wbArgs = (aifGlobal.x,interpTime,wbTac,petTime,flowWb,vbWb,args.pen[0],1,weights)

	#Attempt to fit model to whole-brain curve
	wbFit = opt.minimize(nagini.fdgDelayLstPen,wbInit,args=wbArgs,method='L-BFGS-B',bounds=np.array(wbBounds).T,options={'maxls':100})

	#Make sure we converged
	if wbFit.success is False:
		print 'ERROR: Cannot estimate model on whole-brain curve. Exiting...'
		sys.exit()

	#Get estimated coefficients and fit
	wbOpt = wbFit.x
	wbFitted,wbCoef = nagini.fdgDelayLstPen(wbOpt,aifGlobal.x,interpTime,wbTac,petTime,flowWb,vbWb,args.pen[0],1,weights,coefs=True)

	#Recompute weights if necessary
	if wbIter !=1 and wbIdx != (wbIter - 1):

		#Calculate median absolute deviation
		wbResid = wbTac - wbFitted
		wbMad = np.median(np.abs(wbResid-np.median(wbResid))) / 0.6745

		#Create weights using Huber weight function
		wbU = np.abs(wbResid / wbMad); wbU[wbU<=1.345] = 1.345
		weights = 1.345 / wbU

#Use coefficents to calculate all my parameters
if args.cbf is None:
	wbParams = nagini.fdgCalc(wbCoef,args.blood,args.dT,args.lc)
else:
	#Get mean flow
	flowWb = np.mean(flowMasked)
	wbParams = nagini.fdgCalc(wbCoef,args.blood,args.dT,args.lc,flow=flowWb,vb=vbWb)

#Create string for whole-brain parameter estimates
labels = ['kOne','kTwo','kThree','cmrGlu',
          'alphaOne','alphaTwo','betaOne',
		  'influx','DV','conc','delay']
values = [wbParams[0],wbParams[1],wbParams[2],wbParams[3],
          wbCoef[0],wbCoef[1],wbCoef[2],
		  wbParams[4],wbParams[5],wbParams[6],wbOpt[1/60.0]]
units = ['mLBlood/mLTissue/min','1/min','1/min','1/min','uMol/hg/min',
         '1/sec','1/sec','mLBlood/mLTissue/sec',
		 'uMol/g/min','mLBlood/mLTissue','uMol/g','min']

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
	aifFitted = nagini.fengModel(drawTime,aifGlobal.x[0],aifGlobal.x[1],aifGlobal.x[2],
								  aifGlobal.x[3],aifGlobal.x[4],aifGlobal.x[5],aifGlobal.x[6])
	axTwo.plot(drawTime,aifFitted,linewidth=5,label='Spline Fit')
	axTwo.set_xlabel('Time (seconds)')
	axTwo.set_ylabel('Counts')
	axTwo.set_title('Arterial Sampled Input function')

	#Interpolate input function and correct for delay
	cAif = nagini.fengModel(drawTime+wbOpt[1],aifGlobal.x[0],aifGlobal.x[1],aifGlobal.x[2],
								  aifGlobal.x[3],aifGlobal.x[4],aifGlobal.x[5],aifGlobal.x[6])
	cAif *= np.exp(np.log(2)/6586.26*wbOpt[1])

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

#Don't do region/voxel estimation if user says not to
if args.noRoi == 1:
	nagini.writeArgs(args,args.out[0])
	sys.exit()

#Get number of parameters for voxelwise optimizations
if args.fModel == 1 or args.noDelay == 1:
	nParam = wbOpt.shape[0]
else:
	nParam = wbOpt.shape[0]-1
voxInit = np.copy(wbOpt[0:nParam])

#Set default voxelwise bounds
lBounds = []; hBounds = []; bScale = [3]
for pIdx in range(nParam):
	if wbOpt[pIdx] > 0:
		lBounds.append(np.max([wbOpt[pIdx]/bScale[pIdx],wbBounds[0][pIdx]]))
		hBounds.append(np.min([wbOpt[pIdx]*bScale[pIdx],wbBounds[1][pIdx]]))
	elif wbOpt[pIdx] < 0:
		lBounds.append(np.max([wbOpt[pIdx]*bScale[pIdx],wbBounds[0][pIdx]]))
		hBounds.append(np.min([wbOpt[pIdx]/bScale[pIdx],wbBounds[1][pIdx]]))
	else:
		lBounds.append(wbBounds[0][pIdx])
		hBounds.append(wbBounds[1][pIdx])
voxBounds = np.array([lBounds,hBounds],dtype=np.float)

#Setup for voxelwise-optimization
nVox = petMasked.shape[0]
if args.cbf is None:
	fitParams = np.zeros((nVox,nParam+11))
else:
	fitParams = np.zeros((nVox,nParam+13))

#If fModel and noDelay is not set, we need to process the AIF
if args.fModel != 1 and args.noDelay != 1:

	#Interpolate input function using delay
	wbAif = nagini.fengModel(interpTime+wbOpt[1],aifGlobal.x[0],aifGlobal.x[1],aifGlobal.x[2],
							 aifGlobal.x[3],aifGlobal.x[4],aifGlobal.x[5],aifGlobal.x[6]) * np.exp(np.log(2)/6586.26*wbOpt[1])
	wbAifStart = nagini.fengModel(petTime[:,0]+wbOpt[1],aifGlobal.x[0],aifGlobal.x[1],aifGlobal.x[2],
								 aifGlobal.x[3],aifGlobal.x[4],aifGlobal.x[5],aifGlobal.x[6]) * np.exp(np.log(2)/6586.26*wbOpt[1])
	wbAifEnd =  nagini.fengModel(petTime[:,1]+wbOpt[1],aifGlobal.x[0],aifGlobal.x[1],aifGlobal.x[2],
								 aifGlobal.x[3],aifGlobal.x[4],aifGlobal.x[5],aifGlobal.x[6]) * np.exp(np.log(2)/6586.26*wbOpt[1])

	#Get interpolated AIF integrated from start to end of pet times
	wbAifPet = (wbAifStart+wbAifEnd)/2.0

	#Convert AIF to plasma
	rbcToPlasma = 0.814101 + 0.000680*interpTime/60.0 + .103307*(1-np.exp(-interpTime/60.0/50.052431))
	pAif = wbAif / (args.hct*rbcToPlasma + 1 - args.hct)

#Loop through every voxel
noC = 0
for voxIdx in tqdm(range(nVox)):

	#Get voxel tac
	voxTac = petMasked[voxIdx,:]

	#Decide which function
	if args.fModel == 1 and args.noDelay !=1:
		#Full model function
		voxFunc = nagini.fdguDelayLst(aifGlobal.x,interpTime,voxTac,vbMasked[voxIdx])
	else:
		#Get concentration in compartment one
		cOneVox = vbMasked[voxIdx]*wbAifPet

		#No delay function
		#voxFunc = nagini.fdgAifLst(interpTime,pAif,voxTac,cOneVox)

	try:
		#Run fit
		voxOpt = opt.minimize(nagini.fdgAifLstPen,wbOpt[0],args=(interpTime,pAif,petTime,voxTac,cOneVox,5E8,wbOpt[0]),method='L-BFGS-B',bounds=voxBounds.T)

		#Extract coefficients
		if nParam > 1:
			voxFitted,voxCoef = voxFunc(petTime,voxOpt[0],voxOpt[1],coefs=True)
		else:
			voxFitted,voxCoef = nagini.fdgAifLstPen(voxOpt.x,interpTime,pAif,petTime,voxTac,cOneVox,5E8,wbOpt[0],coefs=True)

		#Calculate rate constants from my alphas and betas
		kOneVox = voxCoef[0]
		kThreeVox = voxCoef[1]*voxCoef[2]/voxCoef[0]
		kTwoVox = voxCoef[2]-kThreeVox

		#Calculate metabolic rate
		voxCmr = voxCoef[1] * args.blood * gluScale / args.lc[0]

		#Calculate influx
		voxIn = kOneVox*args.blood*gluScale/100.0

		#Calculate distrubtion volume
		voxDv =  kOneVox/(voxCoef[2]*args.dT)

		#Compute tissue concentration
		voxConc = voxDv * args.blood * 0.05550748

		#Save common estimates
		fitParams[voxIdx,0] = kOneVox*60.0
		fitParams[voxIdx,1] = kTwoVox*60.0
		fitParams[voxIdx,2] = kThreeVox*60.0
		fitParams[voxIdx,3] = voxCmr
		fitParams[voxIdx,4] = voxCoef[0]
		fitParams[voxIdx,5] = voxCoef[1]
		fitParams[voxIdx,6] = voxCoef[2]
		fitParams[voxIdx,7] = voxIn
		fitParams[voxIdx,8] = voxDv
		fitParams[voxIdx,9] = voxConc

		#Calculate gef nad net extraction if we have blood flow
		if args.cbf is not None:

			#Calculate gef
			fitParams[voxIdx,10] = kOneVox / (flowMasked[voxIdx]*vbMasked[voxIdx])

			#Calculate net extraction
			fitParams[voxIdx,11] = voxCoef[1] / (flowMasked[voxIdx]*vbMasked[voxIdx])


		#Save delay parameter if necessary
		if nParam > 2:
			fitParams[voxIdx,fitParams.shape[1]-2] = voxOpt[1]/60.0

		#Calculate residual with delay
		fitResid = voxTac - voxFitted

		#Calculate normalized root mean square deviation
		fitRmsd = np.sqrt(np.sum(np.power(fitResid,2))/voxTac.shape[0]) / np.mean(voxTac)

		#Save residual
		fitParams[voxIdx,-1] = fitRmsd

	except(RuntimeError,ValueError):
		noC += 1

#Warn user about lack of convergence
if noC > 0:
	print('Warning: %i of %i voxels did not converge.'%(noC,nVox))

#############
###Output!###
#############
print('Writing out results...')

#Set names for model images
if args.cbf is None:
	paramNames = ['kOne','kTwo','kThree','cmrGlu','alphaOne','alphaTwo','betaOne','influx','DV','conc','delay','nRmsd']
else:
	paramNames = ['kOne','kTwo','kThree','cmrGlu','alphaOne','alphaTwo','betaOne','influx','DV','conc','gef','netEx','delay','nRmsd']

#Do the writing.
for iIdx in range(fitParams.shape[1]-1):
	nagini.writeMaskedImage(fitParams[:,iIdx],maskData.shape,maskData,pet.affine,pet.header,'%s_%s'%(args.out[0],paramNames[iIdx]))
nagini.writeMaskedImage(fitParams[:,-1],maskData.shape,maskData,pet.affine,pet.header,'%s_%s'%(args.out[0],paramNames[-1]))
nagini.writeArgs(args,args.out[0])
