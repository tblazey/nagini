#!/usr/bin/python

###################
###Documentation###
###################

"""

cbvIdaif.py: Calculates CBV using a CO scan and an image-derived input function

Uses model described by Mintun et al, Journal of Nuclear 1983

Requires the following inputs:
	pet -> CO image. 
	idaif -> Image derivied arterial input function
	brain -> Brain mask in the same space as pet.
	out -> Root for all outputed file
	
User can set the following options:
	d -> Density of brain tissue in g/mL. Default is 1.05
	r -> Mean ratio of small-vessel to large-vessel hematocrit
	
Produces the following outputs:
	cbv -> Voxelwise map of cerebral blood volume in ml/100g


Requires the following modules:
	argparse, numpy, nibabel, nagini

Tyler Blazey, Spring 2016
blazey@wustl.edu

"""

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Estimates cerebral blood volume using:')
argParse.add_argument('pet',help='Nifti CO image',nargs=1)
argParse.add_argument('idaif',help='Image-derived input functino',nargs=1)
argParse.add_argument('brain',help='Brain mask in PET space',nargs=1)
argParse.add_argument('out',help='Root for outputed files',nargs=1)
argParse.add_argument('-d',help='Density of brain tissue in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-r',help='Mean ratio of small-vessel to large-vessel hematocrit. Default is 0.85',default=0.85,metavar='ratio',type=float)
args = argParse.parse_args()

#Load needed libraries
import numpy as np, nibabel as nib, nagini, sys

#########################
###Data Pre-Processing###
#########################

#Load image headers
pet = nagini.loadHeader(args.pet[0])
brain = nagini.loadHeader(args.brain[0]) 

#Check to make sure image dimensions match
if pet.shape[0:3] != brain.shape[0:3]:
	print 'ERROR: Image dimensions do not match. Please check data...'
	sys.exit()

#Get the image data
petData = pet.get_data()
brainData = brain.get_data()

#Load in the idaif.
idaif = nagini.loadIdaif(args.idaif[0])

#Flatten the PET images and then mask
petMasked = petData.flatten()[brainData.flatten()>0]

############
###Output###
############

#Calculate CBV in mL/hG
cbvData = (petMasked * 100.0 ) / (args.r*args.d*idaif)

#Write out CBV image
nagini.writeMaskedImage(cbvData,brain.shape,brainData,brain.affine,'%s_cbv'%(args.out[0]))


