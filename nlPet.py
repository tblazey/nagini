#!/usr/bin/python

###################
###Documentation###
###################

"""

nlPet.py: Attempts to denoise a 4D pet image using a non-local mean algorithm

Follows the algorithm described by Dutta et al., 2012 PLOS One.
This version is a bit slower than what they descibe, so performance could be improved. 
Also, I have implemented a 3d version (what they call 4d), but this quite slow.

"""""

###############
###Functions###
###############

#Define function for constructing kernel
def makeKernel(k_size, dim3=False):

	#Determine step size for kernel
	k_step = (0.5*k_size)-0.5

	#Get range of values for each dimension
	k_range = np.arange(-k_step, k_step+1, dtype=np.int)

	#Make empty kernel
	kernel = []
	
	#Switch for 2d or 3d kernel
	if dim3 is False:
		for i, j in itertools.product(k_range, k_range):
			kernel.append([i,j,0])
	else:
		for i, j, k in itertools.product(k_range, k_range, k_range):
			kernel.append([i,j,k])	
	 
	return np.vstack(kernel).T

#Function to produce set of indicies and mask given a kernel
def genPatchIdx(kernel,idx,mask):

	#Create indicies for patch/window
	pIdx = kernel + idx[:, np.newaxis]

	#Construct patch/window mask
	p_mask = mask[pIdx[0,:], pIdx[1,:], pIdx[2,:]]

	#Add patch indicies and mask
	return pIdx,p_mask


###########################
###Arguments and Loading###
###########################

#Create argument parser
import argparse
arg_parse = argparse.ArgumentParser(description='Denoise 4D PET data with non-local means:')
arg_parse.add_argument('pet', help='4D Nifti PET image.',nargs=1, type=str)
arg_parse.add_argument('mask', help='3D Nifti mask image.', nargs=1, type=str)
arg_parse.add_argument('out', help='Root for output image.', nargs=1, type=str)
arg_parse.add_argument('-p_size', help='Size for local patch. Default is 3.', nargs=1, type=int, default=[3], metavar='int')
arg_parse.add_argument('-w_size',help='Size for non-local window. Default is 11.', nargs=1, type=int, default=[11], metavar='int')
arg_parse.add_argument('-t_start', help='First timepoint to consider when computing weights.', nargs=1, type=int, metavar='int')
arg_parse.add_argument('-dim3', action='store_true' ,help='Use a 3D patch and window. Default is 2d (slice by slice).')
args = arg_parse.parse_args()

#Load libraries
import numpy as np
import nibabel as nib
import nagini
import itertools
import sys
from tqdm import tqdm

#Load pet and mask headers
pet = nagini.loadHeader(args.pet[0]); mask = nagini.loadHeader(args.mask[0])

#Make sure image dimensions match
if pet.shape[0:3] != mask.shape[0:3]:
	print 'Error: Image dimensions do not match. Please check...'
	sys.exit()

#Load image data
pet_data = pet.get_data(); mask_data = mask.get_data()

#Check for singleton dimensions
if pet_data.shape[-1] == 1:
	pet_data = np.squeeze(pet_data)
if mask_data.shape[-1] == 1:
	mask_data = np.squeeze(mask_data)

#Make sure we have a start time and it is set correctly
if args.t_start is None:
	
	#If we don't have a start time, the frame at about 60%
	t_start = int(pet.shape[3]*.60)

else:

	#Make sure user choice doesn't go beyond number of frames
	if args.t_start[0] >= pet.shape[3]:

		#If it doesn, override user
		t_start = int(pet.shape[3]*.60)
		print 'Cannot use t_start of %i. Using %i instead'%(args.t_start[0], t_start)

	else:

		#The user is right this time
		t_start = args.t_start[0]

#####################
###Data Processing###
#####################

#Make a combined mask
cMask = np.logical_and(mask_data!=0, np.sum(pet_data, axis=3)!=0)

#Make kernels
w_size = args.w_size[0]; p_size = args.p_size[0]
k_w = makeKernel(w_size, dim3=args.dim3); k_p = makeKernel(p_size, dim3=args.dim3)

#Created padded versions of image arrays so that we always stay within kernel
if args.dim3 is True:
	dim3 = w_size
else:
	dim3 = 0
petPad = np.pad(pet_data, [[w_size, w_size], [w_size, w_size],
				[dim3, dim3],[0, 0]], mode='constant')
mask_pad = np.pad(cMask,[[w_size, w_size],[w_size, w_size],
				[dim3, dim3]],mode='constant')

#Find coordinates of non-zero values
mask_coords = np.array(np.nonzero(mask_pad), dtype=np.int).T

#Get flattened version of mask
n_masked = mask_coords.shape[0]

#Make empty arrays for patches, windows, and their masks
patch_idxs = np.zeros((petPad.shape[0], petPad.shape[1], petPad.shape[2], k_p.shape[1], 3), dtype=np.int)
patch_masks = np.zeros((petPad.shape[0], petPad.shape[1], petPad.shape[2], k_p.shape[1]), dtype=np.bool)

#Get patch indices for each voxel.
#Note: This method won't consider patches where the center voxel is outside 
#of the brain mask but some patch voxels are inside mask. Also relies on 
#the padding because voxels that aren't in mask get indicies of 0,0,0 and so 0,0,0
#must not be within the mask.
for i in range(n_masked):
	
	#Get x,y,z, coordinates for masked value
	voxIdx = mask_coords[i,:]

	#Get patch indicies and mask
	patchIdx,patchMask = genPatchIdx(k_p,voxIdx,mask_pad)

	#Add data to arrays
	patch_idxs[voxIdx[0],voxIdx[1],voxIdx[2],:,:] = patchIdx.T
	patch_masks[voxIdx[0],voxIdx[1],voxIdx[2],:] = patchMask

#Make empty storage items for storing results
petSmooth = np.zeros(petPad.shape)

#Construct weights for voxel and its neighborhood
for i in tqdm(range(n_masked)):

	#Extract patch center and mask for current voxel
	iIdx = mask_coords[i,:]
	iPatch = patch_idxs[iIdx[0],iIdx[1],iIdx[2],:,:] 
	iPatchMask = patch_masks[iIdx[0],iIdx[1],iIdx[2],:]

	#Get window and its mask
	iWindow,iWindowMask = genPatchIdx(k_w,iIdx,mask_pad)

	#Mask window so we only consider valid voxels
	iWindowMasked = iWindow[:,iWindowMask]

	#Extract patchs and masks for paired voxels
	jPatchs = patch_idxs[iWindowMasked[0,:],iWindowMasked[1,:],iWindowMasked[2,:],:,:]
	jMasks = patch_masks[iWindowMasked[0,:],iWindowMasked[1,:],iWindowMasked[2,:],:]

	#Make combined mask
	ijPatchMasks = np.logical_and(iPatchMask,jMasks)

	#Extract PET data for patches
	iPet = petPad[iPatch[:,0],iPatch[:,1],iPatch[:,2],t_start:]
	jPets = petPad[jPatchs[:,:,0],jPatchs[:,:,1],jPatchs[:,:,2],t_start:]

	#Compute sum of squares across time
	ssT = np.sum(np.power(jPets-iPet,2),axis=2)

	#Compute sum of squares across each pair
	ssIj = np.sum(ssT*ijPatchMasks,axis=1)

	#Compute variance within iPatch
	iVar = np.var(iPet[iPatchMask,:])

	#Compute number of data points within each patch pair
	n = np.sum(ijPatchMasks,axis=1)*jPets.shape[2]

	#Compute weights
	iWeights = np.exp(-ssIj/iVar/n)

	#Extract pet data within window
	petWindow = petPad[iWindowMasked[0,:],iWindowMasked[1,:],iWindowMasked[2,:],:]

	#Compute and save smoothed PET values
	petSmooth[iIdx[0],iIdx[1],iIdx[2],:] = np.dot(iWeights[np.newaxis,:],petWindow) / np.sum(iWeights)

#Save smoothed image
petD = petSmooth[w_size:(petPad.shape[0]-w_size),w_size:(petPad.shape[1]-w_size),dim3:(petPad.shape[2]-dim3),:]
nib.Nifti1Image(petD,pet.affine,header=pet.header).to_filename('%s.nii.gz'%(args.out[0]))
		

