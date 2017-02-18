#!/usr/bin/python

###################
###Documentation###
###################

"""

oxyIdaifOhta.py: Calculates metabolic rate of oxygen using a 015 oxygen scan and an image-derived input function

Uses model described by Mintun et al, Journal of Nuclear Medicine 1984

Produces the following outputs:
	wbVals.txt -> Estimates of whole-brain OEF and CMRO2
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
argParse.add_argument('-range',help='Time range for OEF estimation in seconds. Default is scan start to 45 seconds. Accepts start/end or numbers',nargs=2,metavar='time')
argParse.add_argument('-kinetic',help='Use full kinetic modeling instead of autoradiographic method',action='store_const',const=1)
argParse.add_argument('-eBound',help='Bounds for OEF when using kinetic modeling',nargs=2,type=float,metavar=('lower','upper'))
argParse.add_argument('-wbOnly',action='store_const',const=1,help='Only perform whole-brain estimation')
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
petMasked = nagini.reshape4d(petData)[brainMask>0,:]
cbfMasked = cbfData.flatten()[brainMask>0] / 6000.0 * args.d
lmbdaMasked = lmbdaData.flatten()[brainMask>0] * args.d
cbvMasked = cbvData.flatten()[brainMask>0] / 100 * args.d

#Limit pet range
timeMask = np.logical_and(petTime>=petRange[0],petTime<=petRange[1])
petTime = petTime[timeMask]
idaif = idaif[timeMask]
petMasked = petMasked[:,timeMask]

#Interpolate the aif to minimum sampling time
minTime = np.min(np.diff(petTime))
interpTime = np.arange(petTime[0],np.ceil(petTime[-1]+minTime),minTime)
nTime = interpTime.shape[0]
aifLinear = interp.interp1d(petTime,idaif,kind="linear",fill_value="extrapolate")
aifInterp = aifLinear(interpTime)

#Get input function for h20 and oxygen seperately
aifDelay = aifLinear(interpTime-args.delay); aifDelay[aifDelay<0] = 0
aifWater = args.decay*np.convolve(aifDelay,np.exp(-args.decay*interpTime))[0:nTime]*minTime
aifOxy = aifInterp - aifWater

#Scale for converting to CMRO02
oxyScale = args.artOxy[0] * 6000.0 / args.d

#Get the average whole-brain flow, cbv, and lambda. Not strictly speaking correct as I should get the tacs.
wbFlow = np.mean(cbfMasked[cbfMasked!=0])
wbLmbda = np.mean(lmbdaMasked[lmbdaMasked!=0])
wbCbv = np.mean(cbvMasked[cbvMasked!=0])

#Get the whole brain tac
wbTac = np.mean(petMasked,axis=0)

#If user wants to do kinetic modeling
if args.kinetic == 1:

	#Attempt to fit model to whole-brain curve
	wbFunc = nagini.oxyOne(interpTime,aifWater,aifOxy,wbFlow,wbLmbda,wbCbv,args.r)
	try:
		wbFit = opt.curve_fit(wbFunc,petTime,wbTac,bounds=(0,1))
	except(RuntimeError):
		print 'ERROR: Cannot estimate one-parameter model on whole brian curve. Exiting...'
		sys.exit()

	#Get whole brain fitted values
	wbFitted = wbFunc(petTime,wbFit[0][0])

	#Create string for whole-brain parameter estimates
	wbString = 'OEF=%f\nCMROxy=%f'%(wbFit[0][0],wbFit[0][0]*wbFlow*oxyScale)

	#Whole-brain tac
	np.savetxt('%s_wbTac.txt'%(args.out[0]),wbTac)

	#Whole-brain fitted values
	np.savetxt('%s_wbFitted.txt'%(args.out[0]),wbFitted)

	#Use whole-brain values as initilization
	init = wbFit[0]

	#Set bounds 
	bounds = np.array([0,1.5],dtype=np.float)
	if args.eBound is not None:
		bounds[0] = args.eBound[0]
		bounds[1] = args.eBound[1]
		#Use midpoint between bounds as initial value if whole brain estimate is not in bounds
		if init < bounds[0] or init > bounds[1]:		
			init = (bounds[0]+bounds[1]) / 2

	#Set up output image names for later
	imgNames = ['oef','cmrOxy','oef_var','cmrOxy_var','nRmsd']

else:
	#Calculate whole-brain OEF
	wbOef = nagini.oefCalc(wbTac,petTime,interpTime,aifOxy,aifWater,wbFlow,wbCbv,wbLmbda,args.r)
	
	#Create string for whole-brain parameter estimates
	wbString = 'OEF=%f\nCMROxy=%f'%(wbOef,wbOef*wbFlow*oxyScale)
	
	#Output names for autoradiographic method
	imgNames = ['oef','cmrOxy']

#Write out whole-brain results
try:
	#Parameter estimates
	wbOut = open('%s_wbVals.txt'%(args.out[0]), "w")
	wbOut.write(wbString)
	wbOut.close()
except(IOError):
	print 'ERROR: Cannot write in output directory. Exiting...'
	sys.exit()

#Don't do voxelwise estimation if user says not to
if args.wbOnly == 1:
	sys.exit()

###################
###Model Fitting###
###################
print ('Calculating OEF and cmrOxy at each voxel...')

#Create a mask of non-zero inputs
useMask = np.logical_and(np.logical_and(cbfMasked!=0,lmbdaMasked!=0),cbvMasked!=0)

#Loop through every voxel
nVox = petMasked.shape[0]; oefParams = np.zeros((nVox,len(imgNames))); noC = 0
for voxIdx in tqdm(range(nVox)):

	#Only process data if input images are non-zero
	if useMask[voxIdx] == True:

		#Get voxel tac and then interpolate it
		voxTac = petMasked[voxIdx,:]
	
		#Perform kinetic modeling if user wants
		if args.kinetic == 1:

			try:
				#Get voxel fit with two parameter model
				nagini.oxyOne(interpTime,aifWater,aifOxy,wbFlow,wbLmbda,wbCbv,args.r)
				voxFunc = nagini.oxyOne(interpTime,aifWater,aifOxy,\
						cbfMasked[voxIdx],lmbdaMasked[voxIdx],\
						cbvMasked[voxIdx],args.r)
				voxFit = opt.curve_fit(voxFunc,petTime,voxTac,bounds=bounds,p0=init)

				#Save parameter estimates
				voxScale = cbfMasked[voxIdx] * oxyScale
				oefParams[voxIdx,0] = voxFit[0][0]
				oefParams[voxIdx,1] = voxFit[0][0] * voxScale
		
				#Save parameter variance estimates
				oefParams[voxIdx,2] = voxFit[1][0]
				oefParams[voxIdx,3] = voxFit[1][0] * np.power(voxScale,2)

				#Get normalized root mean square deviation
	 			fitResid = voxTac - voxFunc(petTime,voxFit[0][0])
				fitRmsd = np.sqrt(np.sum(np.power(fitResid,2))/voxTac.shape[0])
				oefParams[voxIdx,4] = fitRmsd / np.mean(voxTac)

			except(RuntimeError):
				noC += 1
	
		#Otherwise do autoradiographic method
		else:
		
			#Calculate OEF
			oefParams[voxIdx,0] = nagini.oefCalc(voxTac,petTime,interpTime,aifOxy,aifWater,\
				cbfMasked[voxIdx],cbvMasked[voxIdx],lmbdaMasked[voxIdx],args.r)

			#Calculate the cerebral metabolic rate of oxygen
			oefParams[voxIdx,1] = oefParams[voxIdx,0] * cbfMasked[voxIdx] * oxyScale 

#Warn user about lack of convergence
if noC > 0:
	print('Warning: %i of %i voxels did not converge.'%(noC,nVox))

#############
###Output!###
#############
print('Writing out results...')
#Write out two parameter model images
for iIdx in range(len(imgNames)):
	nagini.writeMaskedImage(oefParams[:,iIdx],brain.shape,brainData,pet.affine,pet.header,'%s_%s'%(args.out[0],imgNames[iIdx]))


