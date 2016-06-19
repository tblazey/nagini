#!/usr/bin/python

###################
###Documentation###
###################

"""

oxyIdaifOhta.py: Calculates metabolic rate of oxygen using a 015 oxygen scan and an image-derived input function

Uses model described by Mintun et al, Journal of Nuclear Medicine 1984

Requires the following inputs:
	pet -> PET oxygen image.
	cbf -> CBF image in mL/hg/min
	lmbda -> Water blood-brain paritition coefficient in mL/g
	cbv -> CBV image in mL/hg
	artOxy -> Arterial concentration of oxygen in uM/mL
	info -> Yi Su style PET info file. 
	idaif -> Image derivied arterial input function
	brain -> Brain mask in the same space as pet.
	out -> Root for all outputed file
	
User can set the following options:
	d -> Density of brain tissue in g/mL. Default is 1.05.
	delay -> Delay parameter for the estimation of the water component of the IDAIF. Default is 20 seconds.
	decay -> Decay parameter for the estimation of the water component of the IDAIF. Default is 0.0012 seconds.
	r -> Ratio of small-vessel to large-vessel hematrocrit. Default is 0.85'
	range -> Start and end time of PET data for estimation of OEF. Default is scan start to 45 seconds.
	kinetic -> Use full kinetic modeling to calculate OEF. Default is to use the autoradiographic method.
	eBound -> Bounds for OEF parameter when -kinetic is used. Default is 0 to 1.

Produces the following outputs:
	cmrOxy -> Voxelwise map of cerebral metabolic rate of oxygen in mL/hg/min
	oef -> Voxelwise map of oxygen extraction fraction

	When -kinetic is used script with also produce:
	
	cmrOxy_var -> Voxelwise map of the variance of cmrOxy
	oef_var -> Voxelwise map of the variance of the OEF estimate
	nRmsd -> Noramlized root mean square deviation for fit

Requires the following modules:
	argparse, numpy, nibabel, nagini, scipy, tqdm

Tyler Blazey, Spring 2016
blazey@wustl.edu

"""

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Estimates cerebral metabolic rate of oxygen using:')
argParse.add_argument('pet',help='Nifti oxygen image',nargs=1,type=str)
argParse.add_argument('cbf',help='Cerebral blood flow image in mL/hg/min',nargs=1,type=str)
argParse.add_argument('lmbda',help='Blood brain-parition coefficient in ml/g',nargs=1,type=str)
argParse.add_argument('cbv',help='Cerebral blodod volume image in mL/hg',nargs=1,type=str)
argParse.add_argument('artOxy',help='Measured arterial oxygen concentration in uM/mL',nargs=1,type=float)
argParse.add_argument('info',help='Yi Su style info file',nargs=1,type=str)
argParse.add_argument('idaif',help='Image-derived input function',nargs=1,type=str)
argParse.add_argument('brain',help='Brain mask in PET space',nargs=1,type=str)
argParse.add_argument('out',help='Root for outputed files',nargs=1,type=str)
argParse.add_argument('-d',help='Density of brain tissue in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-delay',help='Delay parameter in seconds for estimating water component of IDAIF. Default is 20',default=20,metavar='delay',type=float)
argParse.add_argument('-decay',help='Decay constant in seconds for estimating water component of IDAIF. Default is 0.0012',default=0.0722/60.0,metavar='decay',type=float)
argParse.add_argument('-r',help='Value for the mean ratio of small-vessel to large-vessel hematocrit. Default is 0.85',nargs=1,default=0.85,type=float,metavar='ratio')
argParse.add_argument('-range',help='Time range for OEF estimation in seconds. Default is scan start to 45 seconds.',nargs=2,type=float,metavar='time')
argParse.add_argument('-kinetic',help='Use full kinetic modeling instead of autoradiographic method',action='store_const',const=1)
argParse.add_argument('-eBound',help='Bounds for OEF when using kinetic modeling',nargs=2,type=float,metavar=('lower','upper'))
args = argParse.parse_args()

#Make sure sure user set bounds correctly
for bound in [args.eBound]:
	if bound is not None:
		if bound[1] <= bound[0]:
			print 'ERROR: Lower bound of %f is not lower than upper bound of %f'%(bound[0],bound[1])
			sys.exit()

#Load needed libraries
import numpy as np, nibabel as nib, nagini, sys, scipy.interpolate as interp, scipy.optimize as opt
from tqdm import tqdm

#########################
###Data Pre-Processing###
#########################
print ('Loading images...')

#Load image headers
pet = nagini.loadHeader(args.pet[0])
cbf = nagini.loadHeader(args.cbf[0])
lmbda = nagini.loadHeader(args.lmbda[0])
cbv = nagini.loadHeader(args.cbv[0])
brain = nagini.loadHeader(args.brain[0]) 

#Load in the idaif.
idaif = nagini.loadIdaif(args.idaif[0])

#Load in the info file
info = nagini.loadInfo(args.info[0])

#Use middle times as pet time. Account for any offset
petTime = info[:,1] - info[0,0]

#Check to see if the users range is actually within data
if args.range is not None:
	if args.range[0] < petTime[0] or args.range[1] > petTime[-1]:
		print 'Error: Users selected range from %f to %f is outside of PET range of %f to %f.'%(args.range[0],args.range[1],petTime[0],petTime[-1])
		sys.exit()
	else:
		petRange = np.array([args.range[0],args.range[1]])
else:
	petRange = np.array([petTime[0],45])

#Check to make sure dimensions match
if pet.shape[0:3] != brain.shape[0:3] or pet.shape[0:3] != cbf.shape[0:3] or \
   pet.shape[0:3] != lmbda.shape[0:3] or pet.shape[0:3] != cbv.shape[0:3] or \
   pet.shape[3] != idaif.shape[0] or pet.shape[3] != info.shape[0]:
	print 'ERROR: Data dimensions do not match. Please check...'
	sys.exit()

#Get the image data
petData = pet.get_data()
cbfData = cbf.get_data()
lmbdaData = lmbda.get_data()
cbvData = cbv.get_data()
brainData = brain.get_data()

#Flatten the PET images and then mask. Also convert parameteric images back to orginal PET units. 
brainMask = brainData.flatten()
petData = nagini.reshape4d(petData)[brainMask>0,:]
cbfData = cbfData.flatten()[brainMask>0] / 6000.0 * args.d
lmbdaData = lmbdaData.flatten()[brainMask>0] * args.d
cbvData = cbvData.flatten()[brainMask>0] / 100 * args.d

#Interpolate the aif to minimum sampling time
minTime = np.min(np.diff(petTime))
interpTime = np.arange(petRange[0],petRange[1],minTime)
nTime = interpTime.shape[0]
aifLinear = interp.interp1d(petTime,idaif,kind="linear",fill_value="extrapolate")
aifInterp = aifLinear(interpTime)

#Get input function for h20 and oxygen seperately
aifDelay = aifLinear(interpTime-args.delay); aifDelay[aifDelay<0] = 0
aifWater = args.decay*np.convolve(aifDelay,np.exp(-args.decay*interpTime))[0:nTime]*minTime
aifOxy = aifInterp - aifWater

#If user wants to do kinetic modeling
if args.kinetic == 1:
	
	#Get the whole brain tac and interpolate that
	wbTac = np.mean(petData,axis=0)
	wbInterp = interp.interp1d(petTime,wbTac,kind="linear")(interpTime)

	#Get the average whole-brain flow, cbv, and lambda. Not strictly speaking correct as I should get the tacs.
	wbFlow = np.mean(cbfData[cbfData!=0])
	wbLmbda = np.mean(lmbdaData[lmbdaData!=0])
	wbCbv = np.mean(cbvData[cbvData!=0])

	#Attempt to fit model to whole-brain curve
	fitX = np.vstack((interpTime,aifOxy,aifWater))
	try:
		wbFit = opt.curve_fit(nagini.oxyOneIdaif(wbFlow,wbLmbda,wbCbv,args.r),fitX,wbInterp,bounds=(0,1))
	except(RuntimeError):
		print 'ERROR: Cannot estimate one-parameter model on whole brian curve. Exiting...'
		sys.exit()

	#Use whole-brain values as initilization
	init = wbFit[0]

	#Set bounds 
	bounds = np.array([0,1],dtype=np.float)
	if args.eBound is not None:
		bounds[0] = args.eBound[0]
		bounds[1] = args.eBound[1]
		#Use midpoint between bounds as initial value if whole brain estimate is not in bounds
		if init < bounds[0] or init > bounds[1]:		
			init = (bounds[0]+bounds[1]) / 2

	#Set up output image names for later
	imgNames = ['oef','cmrOxy','oef_var','cmrOxy_var','nRmsd']

else:
	#Output names for autoradiographic method
	imgNames = ['oef','cmrOxy']


###################
###Model Fitting###
###################
print ('Calculating OEF and cmrOxy at each voxel...')

#Loop through every voxel
nVox = petData.shape[0]; oefParams = np.zeros((nVox,len(imgNames))); noC = 0
for voxIdx in tqdm(range(nVox)):

	#Only process data if input images are non-zero
	if cbfData[voxIdx] !=0  and lmbdaData[voxIdx] != 0 and cbvData[voxIdx] != 0:

		#Get voxel tac and then interpolate it
		voxTac = petData[voxIdx,:]
		voxInterp = interp.interp1d(petTime,voxTac,kind="linear")(interpTime)
	
		#Perform kinetic modeling if user wants
		if args.kinetic == 1:

			try:
		
				#Get voxel fit with two parameter model
				voxFunc = nagini.oxyOneIdaif(cbfData[voxIdx],lmbdaData[voxIdx],cbvData[voxIdx],args.r)
				voxFit = opt.curve_fit(voxFunc,fitX,voxInterp,bounds=(0,1),p0=init)
				
		
				#Save parameter estimates
				oxyScale = cbfData[voxIdx] * args.artOxy[0] * 6000.0 / args.d
				oefParams[voxIdx,0] = voxFit[0][0]
				oefParams[voxIdx,1] = voxFit[0][0] * oxyScale
		
				#Save parameter variance estimates
				oefParams[voxIdx,2] = voxFit[1][0]
				oefParams[voxIdx,3] = voxFit[1][0] * np.power(oxyScale,2)

				#Get normalized root mean square deviation
	 			fitResid = voxInterp - voxFunc(fitX,voxFit[0][0])
				fitRmsd = np.sqrt(np.sum(np.power(fitResid,2))/voxInterp.shape[0])
				oefParams[voxIdx,4] = fitRmsd / np.mean(voxInterp)

			except(RuntimeError):
				noC += 1
	
		#Otherwise do autoradiographic method
		else:
		
			#Calculate OEF
			oefParams[voxIdx,0] = nagini.oefCalcIdaif(voxInterp,interpTime,aifOxy,aifWater,cbfData[voxIdx],cbvData[voxIdx],lmbdaData[voxIdx],args.r)

			#Calculate the cerebral metabolic rate of oxygen
			oefParams[voxIdx,1] = oefParams[voxIdx,0] * cbfData[voxIdx] * args.artOxy[0] * 6000.0 / args.d 

#Warn user about lack of convergence
if noC > 0:
	print('Warning: %i of %i voxels did not converge.'%(noC,nVox))

#############
###Output!###
#############
print('Writing out results...')

#Write out two parameter model images
for iIdx in range(len(imgNames)):
	nagini.writeMaskedImage(oefParams[:,iIdx],brain.shape,brainData,brain.affine,'%s_%s'%(args.out[0],imgNames[iIdx]))


