#!/usr/bin/python

###################
###Documentation###
###################

"""

varExtract.py: Extract a varible from a matlab '.m' file and writes it out to text

Requires the following inputs:
	mFile -> Location of matlab m file.
	var -> Name of variable to extract
	out -> Name of file output
	

Requires the following modules:
	argparse, numpy, scipy.io

Tyler Blazey, Spring 2016
blazey@wustl.edu

"""

#####################
###Parse Arguments###
#####################

import argparse
argParse = argparse.ArgumentParser(description='Extract variable from matlab .m file')
argParse.add_argument('mFile',help='Matlab .m file',nargs=1,type=str)
argParse.add_argument('var',help='Name of variable to extract',nargs=1,type=str)
argParse.add_argument('out',help='Name of text output file',nargs=1,type=str)
args = argParse.parse_args()

#####################
###Extract Variable##
#####################

#Import the needed libraries
import numpy as np, scipy.io as io, sys

#Load up only the variable user wants
try:
	mData = io.loadmat(args.mFile[0],variable_names=args.var[0])
except (IOError,ValueError):
	print 'ERROR: Cannot load .m file at %s'%(args.mFile[0])
	sys.exit()

#Make sure the variable is actually present
if args.var[0] not in mData:
	print 'ERROR: Variable: %s is not actually in %s'%(args.var[0],args.mFile[0])
	sys.exit()

#Write it out
try:
	np.savetxt(args.out[0],mData[args.var[0]])
except (IOError):
	print 'ERROR: Cannot save output file at %s'%(args.out[0])
	sys.exit()



