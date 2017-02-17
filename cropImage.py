#!/usr/bin/python

#Parse arguments
import argparse
argParse = argparse.ArgumentParser(description='Crop out excess zeros from image')
argParse.add_argument('img',help='Image to crop',nargs=1,type=str)
argParse.add_argument('ref',help='Reference for cropping. Usually a head mask of some sort',nargs=1,type=str)
argParse.add_argument('-pad',help='Voxels to pad image in each dimension. Default is 3',default=3,type=int)
argParse.add_argument('-refCrop',help='Crop reference image as well',action='store_const',const=1)
args = argParse.parse_args()

#Load libraries
import numpy as np, nibabel as nib, sys, os

#Load in image headers
img = nib.load(args.img[0])
ref = nib.load(args.ref[0])

#Check image dimensions
if img.shape[0:3] != ref.shape[0:3]:
	print 'ERROR: Image dimensions do not match...'
	sys.exit()

#Load in images
imgData = img.get_data()
refData = ref.get_data()
	
#Remove 4th dimension from reference
if len(ref.shape) == 4:
	refData = refData[:,:,:,0]

#Reshape image data
if len(img.shape) == 3:
	imgData = imgData.reshape((img.shape[0],img.shape[1],img.shape[2],1))
	
#Get indices for non-zero rows, columns, and slices
xIdx = np.where(np.sum(refData,axis=(1,2))>0)
yIdx = np.where(np.sum(refData,axis=(0,2))>0)
zIdx = np.where(np.sum(refData,axis=(0,1))>0)

#Get cropping indicies
minIdx = np.zeros(3,dtype=np.int); maxIdx = np.zeros(3,dtype=np.int); idx = 0
for idxs in [xIdx,yIdx,zIdx]:
	minIdx[idx] = np.max([0,np.min(idxs)-args.pad])
	maxIdx[idx] = np.min([img.shape[idx],np.max(idxs)+args.pad])
	idx += 1	

#Crop image
imgCropData = imgData[minIdx[0]:maxIdx[0],minIdx[1]:maxIdx[1],minIdx[2]:maxIdx[2],:]

#Write out cropped image
imgCrop = nib.Nifti1Image(imgCropData,img.affine)
imgCrop.to_filename('%s_crop.nii.gz'%(os.path.splitext(os.path.splitext(args.img[0])[0])[0]))

#Write out transformation matrix
voxDims = np.array(img.header.get_zooms())
xfm = np.eye(4)
xfm[0,3] = (img.shape[0] - maxIdx[0]) * voxDims[0]
xfm[1,3] = minIdx[1]*voxDims[1]
xfm[2,3] = minIdx[2]*voxDims[2]
xfmPre = os.path.splitext(os.path.splitext(args.img[0])[0])[0]
np.savetxt('%s_crop_to_%s.mat'%(xfmPre,os.path.basename(xfmPre)),xfm,fmt='%10.5f')

#Crop reference image as well if user wants
if args.refCrop is not None:

	#Crop reference
	refCropData = refData[minIdx[0]:maxIdx[0],minIdx[1]:maxIdx[1],minIdx[2]:maxIdx[2]]

	#Write out cropped reference image
	refCrop = nib.Nifti1Image(refCropData,ref.affine,header=ref.header)
	refCrop.to_filename('%s_crop.nii.gz'%(os.path.splitext(os.path.splitext(args.ref[0])[0])[0]))

