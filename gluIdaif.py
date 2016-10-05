#!/usr/bin/python

###################
###Documentation###
###################

"""

gluIdaif.py: Calculates cmrGlu using a FDG water scan and an image-derived input function

Uses basic three-parameter two-compartment model (irreverisble binding)
See Carson, 2003 Positron Emission Tomography: Basic Science and Clinical Practice.

Can perform an optional CBV correction using method used in: Sasaki et al, JCBF&M 1986 

Requires the following inputs:
	pet -> PET FDG image.
	info -> Yi Su style PET info file. 
	idaif -> Image derivied arterial input function
	brain -> Brain mask in the same space as pet.
	blood -> Estimate value for blood glucose level in mg/dL.
	out -> Root for all outputed file
	
User can set the following options:
	d -> Density of brain tissue in g/mL. Default is 1.05
	lc -> Value for the lumped constant. Default is 0.52
	oneB -> Bounds for k1. Default is 10 times whole brain value. Ignored if -flow is used.
	twoB -> Bounds for k2. Default is 10 times whole brain value.
	thrB -> Bounds for k3. Default is 10 times whole brain value.
	gefB -> Bounds for glucose extraction fraction. Default is 0 to 1. Only valid if -flow is used. 
	cbv -> CBV pet image in mL/hg
	cbf -> CBF pet image in mL/hg/min
	omega -> Ratio of FDG radioactivity in whole blood and plasma for CBV correction. Default is 0.9.

Produces the following outputs:
	kOne -> Voxelwise map of k1 in 1/seconds.
	kTwo -> Voxelwise map of k2 in 1/seconds.
	kThree -> Voxelwise map of k3 in 1/seconds.
	cmrGlu -> Voxelwise map of cerebral metabolic rate of glucose in uMol/(hg*min)
	kOne_var -> Variance of k1 estimate.
	kTwo_var -> Variance of k2 estimate.
	kThree_var -> Variance of k3 estimate.
	cmrGlu_var -> Variance of cmrGlu estimate.
	nRmsd -> Normalized root-mean-square deviation for fit.

If -flow is set, script will also produce:
	gef -> Voxelwise map of glucose extraction fraction
	gef_var -> Variance of gef estimate

Requires the following modules:
	argparse, numpy, nibabel, nagini, tqdm, scipy

Tyler Blazey, Spring 2016
blazey@wustl.edu

"""

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Estimates metabolic rate of glucose using:')
argParse.add_argument('pet',help='Nifti FDG image',nargs=1,type=str)
argParse.add_argument('info',help='Yi Su style info file',nargs=1,type=str)
argParse.add_argument('idaif',help='Image-derived input function',nargs=1,type=str)
argParse.add_argument('brain',help='Brain mask in PET space',nargs=1,type=str)
argParse.add_argument('blood',help='Blood glucose level in mg/dL',nargs=1,type=float)
argParse.add_argument('out',help='Root for outputed files',nargs=1,type=str)
argParse.add_argument('-d',help='Density of brain tissue in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-lc',help='Value for the lumped constant. Default is 0.52.',default=0.52,metavar='lumped constant',type=float)
argParse.add_argument('-oneB',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds of k1. Default is 10 times whole brain value')
argParse.add_argument('-twoB',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds of k2. Default is 10 times whole brain value')
argParse.add_argument('-thrB',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds of k3. Default is 10 times whole brain value')
argParse.add_argument('-gefB',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds of glucose extraction fraction. Default is 0 to 1.')
argParse.add_argument('-cbv',nargs=1,help='Estimate of CBV in mL/hg. If given, corrects for blood volume.')
argParse.add_argument('-cbf',nargs=1,help='Estimate of CBF in mL/hg/min. If given, estimates GEF.')
argParse.add_argument('-omega',nargs=1,help='Ratio of FDG in whole brain and plasma for CBV correction. Default is 0.9',default=0.9)
args = argParse.parse_args()

#Make sure sure user set bounds correctly
for bound in [args.oneB,args.twoB,args.thrB,args.gefB]:
	if bound is not None:
		if bound[1] <= bound[0]:
			print 'ERROR: Lower bound of %f is not lower than upper bound of %f'%(bound[0],bound[1])
			sys.exit()

#Load needed libraries
import numpy as np, nibabel as nib, nagini, sys, scipy.optimize as opt, scipy.interpolate as interp
from tqdm import tqdm

#########################
###Data Pre-Processing###
#########################
print ('Loading images...')

#Load image headers
pet = nagini.loadHeader(args.pet[0])
brain = nagini.loadHeader(args.brain[0]) 

#Load in the idaif.
idaif = nagini.loadIdaif(args.idaif[0])

#Load in the info file
info = nagini.loadInfo(args.info[0])

#Check to make sure dimensions match
if pet.shape[0:3] != brain.shape[0:3] or pet.shape[3] != idaif.shape[0] or pet.shape[3] != info.shape[0]:
	print 'ERROR: Data dimensions do not match. Please check...'
	sys.exit()

#Get the image data
petData = pet.get_data()
brainData = brain.get_data()

#Flatten the PET images and then mask
petMasked = nagini.reshape4d(petData)[brainData.flatten()>0,:]

#Use middle times as pet time. Account for any offset
petTime = info[:,1] - info[0,0]

#If cbv image is given, correct for blood volume
if ( args.cbv is not None ):

	#Load in CBV image
	cbv = nagini.loadHeader(args.cbv[0])
	if cbv.shape[0:3] != pet.shape[0:3]:
		print 'ERROR: CBV image does not match PET resolution...'
		sys.exit()
	cbvData = cbv.get_data()

	#Mask it and convert it to original units
	cbvMasked = cbvData.flatten()[brainData.flatten()>0] / 100 * args.d

	#Correct all the tacs for blood volume
	petMasked = petMasked - (args.omega*cbvMasked[:,np.newaxis]*idaif)

#Load cbf image if necessary
if ( args.cbf is not None ):

	#Load in CFV image
	cbf = nagini.loadHeader(args.cbf[0])
	if cbf.shape[0:3] != pet.shape[0:3]:
		print 'ERROR: CBF image does not match PET resolution...'
		sys.exit()
	cbfData = cbf.get_data()

	#Mask it and convert it to original units
	cbfMasked = cbfData.flatten()[brainData.flatten()>0] / 6000.0 * args.d


#Interpolate the aif to minimum sampling time
minTime = np.min(np.diff(petTime))
interpTime = np.arange(petTime[0],petTime[-1],minTime)
aifInterp = interp.interp1d(petTime,idaif,kind="linear")(interpTime)

#Get the whole brain tac and interpolate it
wbTac = np.mean(petMasked,axis=0)
wbInterp = interp.interp1d(petTime,wbTac,kind="linear")(interpTime)

#Set scale factor to get cmrGlu to uMole / (hg*min)
gluScale = 333.0449 / args.d / args.lc * args.blood[0]


###################
###Model Fitting###
###################
print ('Beginning fitting procedure...')

#Attempt to fit model to whole-brain curve
fitX = np.vstack((interpTime,aifInterp))
try:
	if args.cbf is None:
		wbFit = opt.curve_fit(nagini.gluThreeIdaif,fitX,wbInterp,p0=[0.001,0.001,0.001],bounds=([0,0,0],[1,1,1]))
	else:
		wbFit = opt.curve_fit(nagini.gluGefIdaif(np.mean(cbfMasked)),fitX,wbInterp,p0=[0.2,0.001,0.001],bounds=([0,0,0],[1,1,1]))
except(RuntimeError):
	print 'ERROR: Cannot estimate three-parameter model on whole-brain curve. Exiting...'
	sys.exit()

#Use whole-brain values as initilization
init = wbFit[0]

#Setup voxelwise fit depending on model
nVox = petMasked.shape[0]
if args.cbf is None:

	#Set bounds for k1 based model
	bounds = np.array((init/25,init*25),dtype=np.float)
	bList = [args.oneB,args.twoB,args.thrB]

	#Create data structure for results and set output names
	fitParams = np.zeros((nVox,9))
	imgNames = ['kOne','kTwo','kThree','cmrGlu','kOne_var','kTwo_var','kThree_var','cmrGlu_var','nRmsd']
	
else:
	
	#Set bounds for gef based model
	bounds = np.array(([0,init[1]/25.0,init[2]/25.0],[1,init[1]*25.0,init[2]*25.0]),dtype=np.float)
	bList = [args.gefB,args.twoB,args.thrB]
	
	#Create data structure for results and set output names
	fitParams = np.zeros((nVox,11))
	imgNames = ['gef','kTwo','kThree','kOne','cmrGlu','gef_var','kTwo_var','kThree_var','kOne_var','cmrGlu_var','nRmsd']

#Make sure bounds are set properly
bIdx = 0
for bound in bList:
	#If user wants different bounds, use them.
	if bound is not None:
		bounds[0,bIdx] = bound[0]
		bounds[1,bIdx] = bound[1]
		#Use midpoint between bounds as initial value if whole brain estimate is not in bounds
		if init[bIdx] < bound[0] or init[bIdx] > bound[1]:		
			init[bIdx] = (bound[0]+bound[1]) / 2
	bIdx += 1

#Loop through every voxel
noC = 0; 
for voxIdx in tqdm(range(nVox)):
	
	#Get voxel tac and then interpolate it
	voxTac = petMasked[voxIdx,:]
	voxInterp = interp.interp1d(petTime,voxTac,kind="linear")(interpTime)

	try:
		#Run the appropriate kinetic model
		if args.cbf is None:
			
			#Fit model with k1 estimate
			voxFit = opt.curve_fit(nagini.gluThreeIdaif,fitX,voxInterp,p0=init,bounds=bounds)

			#Save parameter estimates.
			fitParams[voxIdx,0:3] = voxFit[0]
			fitParams[voxIdx,3] = ((voxFit[0][0]*voxFit[0][2])/(voxFit[0][1]+voxFit[0][2]))*gluScale

			#Save estimated parameter variances. Use delta method to get cmrGlu variance.
			fitParams[voxIdx,4:7] = np.diag(voxFit[1])
			gluGrad = np.array([(gluScale*voxFit[0][2])/(voxFit[0][1]+voxFit[0][2]), 
				    	   (-1*voxFit[0][0]*voxFit[0][2]*gluScale)/np.power(voxFit[0][1]+voxFit[0][2],2),
				           (voxFit[0][0]*voxFit[0][1]*gluScale)/np.power(voxFit[0][1]+voxFit[0][2],2)])
			fitParams[voxIdx,7] = np.dot(np.dot(gluGrad.T,voxFit[1]),gluGrad)

			#Get normalized root mean square deviation
	 		fitResid = voxInterp - nagini.gluThreeIdaif(fitX,voxFit[0][0],voxFit[0][1],voxFit[0][2])
			fitRmsd = np.sqrt(np.sum(np.power(fitResid,2))/voxInterp.shape[0])
			fitParams[voxIdx,8] = fitRmsd / np.mean(voxInterp)

			
		else:
			#Fit model with gef estimate
			voxFunc = nagini.gluGefIdaif(cbfMasked[voxIdx])
			voxFit = opt.curve_fit(voxFunc,fitX,voxInterp,p0=init,bounds=bounds)

			#Save parameter estimates
			fitParams[voxIdx,0:3] = voxFit[0]
			fitParams[voxIdx,3] = voxFit[0][0] * cbfMasked[voxIdx]
			fitParams[voxIdx,4] = ((voxFit[0][0]*cbfMasked[voxIdx]*voxFit[0][2])/(voxFit[0][1]+voxFit[0][2]))*gluScale

			#Save parameter variances. Use delta method to get cmrGlu and k1 variance
			fitParams[voxIdx,5:8] = np.diag(voxFit[1])
			fitParams[voxIdx,8] = voxFit[1][0][0] * np.power(cbfMasked[voxIdx],2)
			gluGrad = np.array([(gluScale*cbfMasked[voxIdx]*voxFit[0][2])/(voxFit[0][1]+voxFit[0][2]), 
				    	    (-1*voxFit[0][0]*voxFit[0][2]*gluScale*cbfMasked[voxIdx])/np.power(voxFit[0][1]+voxFit[0][2],2),
				            (voxFit[0][0]*voxFit[0][1]*gluScale*cbfMasked[voxIdx])/np.power(voxFit[0][1]+voxFit[0][2],2)])
			fitParams[voxIdx,9] = np.dot(np.dot(gluGrad.T,voxFit[1]),gluGrad)

			#Get normalized root mean square deviation
	 		fitResid = voxInterp - voxFunc(fitX,voxFit[0][0],voxFit[0][1],voxFit[0][2])
			fitRmsd = np.sqrt(np.sum(np.power(fitResid,2))/voxInterp.shape[0])
			fitParams[voxIdx,10] = fitRmsd / np.mean(voxInterp)	

	except(RuntimeError):
		noC += 1

#Warn user about lack of convergence
if noC > 0:
	print('Warning: %i of %i voxels did not converge.'%(noC,nVox))

#############
###Output!###
#############
print('Writing out results...')

#Write out two parameter model images
for iIdx in range(len(imgNames)):
	nagini.writeMaskedImage(fitParams[:,iIdx],brain.shape,brainData,brain.affine,'%s_%s'%(args.out[0],imgNames[iIdx]))

