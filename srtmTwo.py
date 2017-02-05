#!/usr/bin/python

###################
###Documentation###
###################

"""

srtmTwo.py: Python script to calculate binding potential, R1, k2, and k2' from dynamic PET data

Uses a simplified reference tissue model with one tissue compartment and two kinetic parameters
Method outlined in Wu and Carson, Journal of Cerebral Blood Flow & Metabolism 2002

Determines values for k2Ref using a multilinear reference tissue model (MRTM)
MRTM method from Ichise et al, Journal of Cerebral Blood Flow and Metabolism 2003

Requires the following inputs:
	pet -> PET timeseries image. 
	info -> Yi Su style pet timing info file.
	ref -> Rerence tissue mask in the same space as pet.
	brain -> Brain mask in the same space as pet.
	out -> Root for all outputed files
	
User can set the following options:
	thr -> Threshold to binarize reference and brain masks. Default is 1.
	mrtm -> Produces voxelwise maps of all paramters from the intial mrtm fits.
	rBound -> Lower and upper bound for SRTM Two R1 parameter. If not set, it is 10 times the whole-brain SRTM Two R1
	kBound -> Lower and upper bound for SRTM Two k2 parameter. If not set, it is 10 times the whole-brain SRTM Two k2.
	
Produces the following outputs:
	R1 -> Voxelwise map of the relative delivery in voxel compared to reference tissue
	k2 -> Voxelwise map of the clearance in voxel
	k2Ref ->  Text file giving the estimated clearance rate for the reference tissue
	BP -> Voxelwise map of binding potential (ratio at equilibrium between specifically bound tracer and free + nonspecifically bound tracer

Can also produce the following outputs if srtm is set:
   	mrtm_R1 -> Voxelwise map of R1 from initial mrtm estimation
   	mrtm_k2 -> Voxelwise map of k2 from initial mrtm estimation
   	mrtm_k2Ref -> Voxelwise map of k2Ref from initial mrtm estimation
   	mrtm_BP -> Voxelwise map of BP from initial mrtm estimation

Requires the following modules:
	argparse, numpy, nibabel, scipy, and tqdm

Tyler Blazey, Spring 2016
blazey@wustl.edu

"""

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Fit a simplified reference tissue model (SRTM) to dynamic PET data')
argParse.add_argument('pet',help='Nifti PET timeseries image',nargs=1)
argParse.add_argument('info',help='Info file for PET image',nargs=1)
argParse.add_argument('ref',help='Nifti mask for reference tissue',nargs=1)
argParse.add_argument('brain',help='Nifti mask for brain tissue',nargs=1)
argParse.add_argument('out',help='Root for outputed files',nargs=1)
argParse.add_argument('-thr',help='Threshold to binarize masks. Default is 1.',nargs=1,type=float,default=[1],metavar='threshold') 
argParse.add_argument('-mrtm',action='store_const',const=1,help='Output images from initial MRTM estimation')
argParse.add_argument('-rBound',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds for R1 parameter. Default is 10 times whole brain value')
argParse.add_argument('-kBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for k2 parameter. Default is 10 times whole brain value')
args = argParse.parse_args()

#Make sure sure user set bounds correctly
for bound in [args.rBound,args.kBound]:
	if bound is not None:
		if bound[1] <= bound[0]:
			print 'ERROR: Lower bound of %f is not lower than upper bound of %f'%(bound[0],bound[1])
			sys.exit()

###############
###Functions###
###############

#Import modules that we need
import numpy as np, nibabel as nib, scipy.integrate as integ, scipy.optimize as opt
from tqdm import tqdm

#Function to load in image headers
def loadHeader(path):
	try:
		header = nib.load(path)
	except (IOError,nib.spatialimages.ImageFileError):
		print 'ERROR: Cannot load image at %s.'%(path)
		sys.exit()
	return header


#Function to calculate residuals for srtm2 model
def srtmTwoResid(params,k2Ref,ref,roi,time,minTime):
   	srtmTwoConv = np.convolve(ref,np.exp(-1*params[1]*time))*minTime
	srtmTwoPred = (params[0]*ref) + (params[0]*(k2Ref-params[1])*srtmTwoConv[0:time.shape[0]])
	return roi - srtmTwoPred

#Function to write masked image to nifti
def writeMaskedImage(data,outDims,mask,affine,name):	
	#Get masked data array back into original dimensions
	outData = np.zeros_like(mask)
	outData[mask==1] = data
	outData = outData.reshape(outDims)

	#Create image to write out
	outImg = nib.Nifti1Image(outData,affine)
	
	#Then do the writing
	outName = '%s.nii.gz'%(name)
	try:
		outImg.to_filename(outName)
	except (IOError):
		print 'ERROR: Cannot save image at %s.'%(outName)
		sys.exit()

#########################
###Data Pre-Processing###
#########################
print('Loading image data...')

#Load image headers
pet = loadHeader(args.pet[0])
ref = loadHeader(args.ref[0])
brain = loadHeader(args.brain[0]) 

#Load in info file
try:
	info = np.loadtxt(args.info[0],usecols=[0,1])
except(IOError):
	print 'ERROR: Cannot load info file at %s.'%(args.info[0])

#Check to mask sure that all the image dimensions match
if pet.shape[0:3] != ref.shape[0:3] or pet.shape[0:3] != brain.shape[0:3] or pet.shape[3] != info.shape[0]:
	print 'ERROR: Mismatch in image dimensions. Please check data.'
	sys.exit()

#Load in image data and reshape
petData = pet.get_data(); petData = petData.reshape((pet.shape[0]*pet.shape[1]*pet.shape[2],pet.shape[3]))
refData = ref.get_data(); refData = refData.reshape((ref.shape[0]*ref.shape[1]*ref.shape[2]))
brainData = brain.get_data(); brainData = brainData.reshape((brain.shape[0]*brain.shape[1]*brain.shape[2]))

#Binarize masks
brainData[brainData>=args.thr[0]] = 1; brainData[brainData<args.thr[0]] = 0
refData[refData>=args.thr[0]] = 1; refData[refData<args.thr[0]] = 0

#Mask the pet data by the brain
petMasked = petData[brainData==1,:]
nVox = petMasked.shape[0]

#Get reference roi curve
refTac = np.mean(petData[brainData + refData==2,:],axis=0)

#Get PET times by using time at the middle of the frame. Also subtract any start time offset.
petTime = info[:,1] - info[0,0]

#Integration referene curve
refInte = integ.cumtrapz(refTac,petTime,initial=0)

#####################
###MRTM Estimation###
#####################
print('Performing initial MRTM fit...')

#Create empty data structure to store mrtm parameter estimates [R1,k2,k2Ref]
mrtmParams = np.zeros((nVox,3))

#Loop through each voxel
for voxIdx in tqdm(range(nVox)):

	#Get voxel tac and then integrate
	voxTac = petMasked[voxIdx,:]
	voxInte = integ.cumtrapz(voxTac,petTime,initial=0)

	#Run srtm fit on voxel
	voxX = np.column_stack((refInte,voxInte,refTac))
	voxFit = np.linalg.lstsq(voxX,voxTac)
	
	#Store results
	mrtmParams[voxIdx,0] = voxFit[0][2]
	mrtmParams[voxIdx,1] = -1 * voxFit[0][1]
	mrtmParams[voxIdx,2] = voxFit[0][0] / voxFit[0][2]

#Calculate median k2Ref for voxels not in the reference tissue
noRef = (refData*-1+1)*brainData; noRef = noRef[brainData==1]
k2RefHat = np.median(mrtmParams[noRef==1,2])

#########################
###SRTM Two Estimation###
#########################
print 'Calculating SRTM Two fit...'

#Create empty data structure to store srtmTwo parameter estimates [R1,k2]
srtmTwoParams = np.zeros((nVox,2))

#Get evenly spaced PET times series interpolation
minSamp = np.min(np.diff(petTime))
eTime = np.arange(petTime[0],petTime[-1],minSamp)

#Interpolate the reference curve
refInterp = np.interp(eTime,petTime,refTac)

#Get interpolated whole brain tac
wbInterp = np.interp(eTime,petTime,np.mean(petMasked,axis=0))

#Run SRTM two fit on whole-brian curve
wbFit = opt.least_squares(srtmTwoResid,[0,0],args=(k2RefHat,refInterp,wbInterp,eTime,minSamp),bounds=(0,np.inf))

#Use wbFit as backup intilization
bInit = np.array([wbFit.x[0],wbFit.x[1]])

#Set bounds 
bounds = np.array(([0,0],[0,0]),dtype=np.float); bIdx = 0
for bound in [args.rBound,args.kBound]:
	if bound is None:
		bounds[0,bIdx] = wbFit.x[bIdx]/10
		bounds[1,bIdx] = wbFit.x[bIdx]*10
	else:
		bounds[0,bIdx] = bound[0]
		bounds[1,bIdx] = bound[1]
		#Use midpoint between bounds if whole brain estimate is out of range
		if bInit[bIdx] < bound[0] or bInit[bIdx] > bound[1]:		
			bInit[bIdx] = (bound[0]+bound[1]) / 2
	bIdx += 1

#Loop through each voxel
noC = 0; init = np.zeros((2)); 
for voxIdx in tqdm(range(nVox)):
	
	#Get voxel tac and then interpolate it
	voxTac = petMasked[voxIdx,:]
	voxInterp = np.interp(eTime,petTime,voxTac)

	#Use MRTM values for initilization if they are within bounds. Otherwise use backup bounds.
	bIdx = 0
	for bound in [bounds[:,0],bounds[:,1]]:
		if mrtmParams[voxIdx,bIdx] < bound[0] or mrtmParams[voxIdx,bIdx] > bound[1]:
			init[bIdx] = bInit[bIdx]
		else:
			init[bIdx] = mrtmParams[voxIdx,bIdx]
		bIdx += 1 

	#Get voxel fit
	voxFit = opt.least_squares(srtmTwoResid,init,args=(k2RefHat,refInterp,voxInterp,eTime,minSamp),bounds=bounds)
	
	#Save results. Fall back to MRTM model if we did not converge
	if voxFit.status <= 0:
		noC += 1
		srtmTwoParams[voxIdx,:] = mrtmParams[voxIdx,0:2]
	else:
		srtmTwoParams[voxIdx,:] = voxFit.x

#Warn user about lack of convergence
if noC > 0:
	print('Warning: %i of %i voxels did not converge.'%(noC,nVox))

#############
###Output!###
#############
print('Writing out results...')

#First write out estimated k2Ref
k2RefOut = '%s_k2Ref.txt'%(args.out[0])
try:
	np.savetxt(k2RefOut,k2RefHat.reshape(1,1))
except(IOError):
	print 'ERROR: Cannot write k2Ref file at %s'%(k2RefOut)
	sys.exit()

#Write out SRTM Two images
writeMaskedImage(srtmTwoParams[:,0],brain.shape,brainData,brain.affine,'%s_R1'%(args.out[0]))
writeMaskedImage(srtmTwoParams[:,1],brain.shape,brainData,brain.affine,'%s_k2'%(args.out[0]))
writeMaskedImage((srtmTwoParams[:,0]/srtmTwoParams[:,1]*k2RefHat)-1,brain.shape,brainData,brain.affine,'%s_BP'%(args.out[0]))

#If user wants, output MRTM images
if args.mrtm == 1:
	writeMaskedImage(mrtmParams[:,0],brain.shape,brainData,brain.affine,'%s_mrtm_R1'%(args.out[0]))
	writeMaskedImage(mrtmParams[:,1],brain.shape,brainData,brain.affine,'%s_mrtm_k2'%(args.out[0]))
	writeMaskedImage(mrtmParams[:,2],brain.shape,brainData,brain.affine,'%s_mrtm_k2Ref'%(args.out[0]))
	writeMaskedImage((mrtmParams[:,0]/mrtmParams[:,1]*mrtmParams[:,2])-1,brain.shape,brainData,brain.affine,'%s_mrtm_BP'%(args.out[0]))



