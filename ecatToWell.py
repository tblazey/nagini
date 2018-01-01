#!/usr/bin/python

#Parse arguments
import argparse, sys
argParse = argparse.ArgumentParser(description='Convert ECAT PET image to decay corrected well counts/mL/sec:')
argParse.add_argument('pet',help='Nifti ECAT count image',nargs=1,type=str)
argParse.add_argument('info',help='Yi Su style info file',nargs=1,type=str)
argParse.add_argument('pie',help='Pie calibration factor',nargs=1,type=float)
argParse.add_argument('out',help='Root for outputed files',nargs=1,type=str)
args = argParse.parse_args()

#Load in libraries we will need
import numpy as np, nibabel as nib, nagini

#Load in PET image
pet = nib.load(args.pet[0]); petData = nagini.reshape4d(pet.get_data())

#Load in info file
info = nagini.loadInfo(args.info[0])

#Loop through frames
for fIdx in range(petData.shape[1]):

	#Convert to decay corrected well counts
	petData[:,fIdx] *= 60.0 * args.pie[0] / info[fIdx,2] * info[fIdx,3]

#Write out result
well = nib.Nifti1Image(petData.reshape(pet.shape),pet.affine,header=pet.header)
well.to_filename(args.out[0]+'.nii.gz')

