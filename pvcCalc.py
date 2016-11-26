#!/usr/bin/python

#Parse arguments
import argparse
arg_parse = argparse.ArgumentParser(description='Run SGTM and RBV Partial Volume Correction')
arg_parse.add_argument('pet',help='PET image. Can be 4d',nargs=1)
arg_parse.add_argument('seg',help='3d segmentation image',nargs=1)
arg_parse.add_argument('out',help='Root for output image',nargs=1)
arg_parse.add_argument('-noZero',help='Do not count 0 as a seperate ROI',action='store_const',const=1,)
arg_parse.add_argument('-mask',help='Mask for RBV image',nargs=1)
args = arg_parse.parse_args()

#Import libraries
import numpy as np, nibabel as nib, scipy.ndimage.filters as filt
import scipy.optimize as opt, nagini, sys
from tqdm import tqdm

#Load headers
pet = nagini.loadHeader(args.pet[0])
seg = nagini.loadHeader(args.seg[0])

#Check that images have same dimensions
if pet.shape[0:3] != seg.shape[0:3]:
	print 'ERROR: Images do have not same dimensions...'
	sys.exit()

#Setup mask
if args.mask is not None:

	#Get mask header
	mask = nagini.loadHeader(args.mask[0])

	#Check to make sure dimensions mask
	if mask.shape[0:3] != pet.shape[0:3]:
		print 'ERROR: Mask image does not have same dimensions other data...'
		sys.exit()

	#Load mask data
	maskData = mask.get_data()

	#Remove 4th dimension of mask if necessary
	if len(maskData.shape) == 4:
		maskData = maskData[:,:,:,0]
else:
	maskData = np.ones((seg.shape[0],seg.shape[1],seg.shape[2]))
	

#Load in image data
petData = pet.get_data()
segData = seg.get_data()

#Remove 4th dimension of segmentation 
if len(segData.shape) == 4:
	segData = segData[:,:,:,0]

#Make a flattened version of the segmentation for use later
segFlat = segData.flatten()

#Reshape PET data if necessary
if len(petData.shape) == 4:
	petData = nagini.reshape4d(petData)
	nPet = petData.shape[1]
else:
	petData = petData.reshape((pet.shape[0]*pet.shape[1]*pet.shape[2],1))
	nPet = 1

#Get ROI list and number of ROIs
roiList = np.unique(segData)
if args.noZero == 1:
	roiList = roiList[roiList!=0]
nRoi = roiList.shape[0]

#Make weight matrices
wMatrix = np.zeros((nRoi,nRoi),dtype=np.float64)
tMatrix = np.zeros((nRoi,nPet),dtype=np.float64)

#Get weighted values
for iIdx in range(nRoi):
	print 'Processing ROI: %i'%(iIdx)

	#Make i ROI. Make sure it is float as well
	iRoi = np.float64(np.where(segData==roiList[iIdx],1.0,0.0))

	#Smooth i ROI for the first time
	iSmooth = filt.gaussian_filter(iRoi,3.397).flatten()

	#Get diagonal weight
	wMatrix[iIdx,iIdx] = iSmooth.dot(iSmooth)

	#Calculate t-values
	for petIdx in range(nPet):
		tMatrix[iIdx,petIdx] = iSmooth.dot(petData[:,petIdx])

	#Smooth i ROI again
	iSmooth = filt.gaussian_filter(iSmooth.reshape((seg.shape[0],seg.shape[1],seg.shape[2])),3.397).flatten()

	for jIdx in range(iIdx,nRoi):

		#Make j ROI
		jRoi = np.float64(np.where(segFlat==roiList[jIdx],1.0,0.0))

		#Get weight
		wMatrix[iIdx,jIdx] = iSmooth.dot(jRoi)
		wMatrix[jIdx,iIdx] = wMatrix[iIdx,jIdx]


#Save weight matrix
nagini.writeText('%s_wMatrix.txt'%(args.out[0]),wMatrix)

#Reshape pet data back
petData = petData.reshape((seg.shape[0],seg.shape[1],seg.shape[2],nPet))

#Loop through pet images
roiCoef = np.zeros(nRoi,dtype=np.float64)
rbvData = np.zeros(petData.shape,dtype=np.float64)
for petIdx in range(nPet):
	
	#Get regional coefficients
	roiCoef,_ = opt.nnls(wMatrix,tMatrix[:,petIdx])

	#Save coefficients
	nagini.writeText('%s_rsfAvg%03i.txt'%(args.out[0],petIdx),roiCoef)

	#Make rsf image
	rsfData = np.zeros((seg.shape[0],seg.shape[1],seg.shape[2]),dtype=np.float64)
	for roiIdx in range(nRoi):
		rsfData[segData==roiList[roiIdx]] = roiCoef[roiIdx]

	#Make rbv image
	rbvData[:,:,:,petIdx] = petData[:,:,:,petIdx] * rsfData / filt.gaussian_filter(rsfData,3.397) * maskData

#Save RBV image
rbv = nib.Nifti1Image(rbvData,seg.affine)
rbv.to_filename('%s_rbv.nii.gz'%(args.out[0]))




