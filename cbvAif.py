#!/usr/bin/python

###################
###Documentation###
###################

"""

cbvAif.py: Calculates CBV using a CO15 water scan and an arterial sampled input function

Uses model described by Mintun et al, Journal of Nuclear 1983
	
Produces the following outputs:
	wbVal -> Text file with whole-brain CBV estimate
	aifPlot -> Plot input function
	cbv -> Voxelwise map of cerebral blood volume in mL/100g*min

Requires the following modules:
	argparse, numpy, nibabel, nagini, tqdm, matplotlib, scikit-learn

Tyler Blazey, Winter 2017
blazey@wustl.edu

"""

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Estimates cerebral blood flow using:')
argParse.add_argument('pet',help='Nifti oc image',nargs=1,type=str)
argParse.add_argument('info',help='Yi Su style info file',nargs=1,type=str)
argParse.add_argument('aif',help='Arterial-sampled input function',nargs=1,type=str)
argParse.add_argument('well',help='Well-counter calibration factor',nargs=1,type=float)
argParse.add_argument('pie',help='Pie calibration factor',nargs=1,type=float)
argParse.add_argument('out',help='Root for outputed files',nargs=1,type=str)
argParse.add_argument('-brain',help='Brain mask in PET space',nargs=1,type=str)
argParse.add_argument('-decay',help='Perform decay correction before CBV calcuation. By default it occurs within  calculation',action='store_const',const=1)
argParse.add_argument('-d',help='Density of brain tissue in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-r',help='Mean ratio of small-vessel to large-vessel hematocrit. Default is 0.85',default=0.85,metavar='ratio',type=float)
argParse.add_argument('-nKnots',nargs=1,type=int,help='Number of knots for AIF spline. Default is number of time points.',metavar='n')
argParse.add_argument('-wbOnly',action='store_const',const=1,help='Only perform whole-brain estimation')
argParse.add_argument('-dcv',action='store_const',const=1,help='AIF is from a DCV file, not a CRV file')
args = argParse.parse_args()

#Load needed libraries
import numpy as np, nibabel as nib, nagini, sys
import matplotlib.pyplot as plt, matplotlib.gridspec as grid
from tqdm import tqdm
from sklearn import linear_model

#########################
###Data Pre-Processing###
#########################
print ('Loading images...')

#Load in the input function
if args.dcv != 1:
	aif = nagini.loadAif(args.aif[0])
else:
	aif = nagini.loadAif(args.aif[0],dcv=True)

#Load in the info file
info = nagini.loadInfo(args.info[0])

#Load image headers
pet = nagini.loadHeader(args.pet[0])

#Check to make sure dimensions match
if pet.shape[3] != len(info.shape) or len(info.shape) != 1:
	print 'ERROR: Data dimensions do not match. Please check...'
	sys.exit()

#Get the image data
petData = pet.get_data()

#Brain mask logic
if args.brain is not None:

	#Load brain mask header
	brain = nagini.loadHeader(args.brain[0])

	#Make sure its dimensions match
	if pet.shape[0:3] != brain.shape[0:3]:
		print 'ERROR: Mask dimensiosn do not match data. Please check...'
		sys.exit()

	#Load in  mask data
	brainData = brain.get_data()

else:
	#Make a fake mask
	brainData = np.ones(pet.shape[0:3])

#Flatten the PET images and then mask
petMasked = petData.flatten()[brainData.flatten()>0]

#Get cbv ranges, using start of scan as zero point
cbvTime = np.array([0.0,info[2]])

#Get aif time variable
aifTime = aif[:,0]

#Apply pie factor and 4dfp offset factor
aifC = aif[:,1] / args.pie[0] / 0.06

#Logic for preparing blood sucker curves
if args.dcv != 1:

	#Reset first two points in AIF which are not traditionally used
	aifC[0:2] = aifC[2]

	#Add well counter and decay correction from start of sampling
	aifC = aifC * args.well[0] 

	#Decay correct each CRV point to start time reported in first saved PET frame
	if args.decay == 1:
		aifC *= np.exp(np.log(2)/122.24*info[0]) * np.exp(np.log(2)/122.24*aifTime)
	else:
		petMasked /= info[3]

else:
	if args.decay == 1:
		aifC *= np.exp(np.log(2)/122.24*info[0])
	else:
		aifC /= np.exp(np.log(2)/122.24*aifTime)
		petMasked /= info[3]

#Set number of knots
if args.nKnots is None:
	nKnots = aifTime.shape[0]
else:
	nKnots = args.nKnots[0]

#Calculate spline basis
aifKnots = nagini.knotLoc(aifTime,nKnots,bounds=[5,95])
aifBasis = nagini.rSplineBasis(aifTime,aifKnots)

#Use scikit learn to do a bayesian style ARD regression for fit
bModel = linear_model.ARDRegression(fit_intercept=True)
bFit = bModel.fit(aifBasis[:,1::],aifC)
aifCoefs = np.concatenate((bFit.intercept_[np.newaxis],bFit.coef_))

#Get basis for interpolating integral of spline at CBV time 
pBasis,pBasisI = nagini.rSplineBasis(cbvTime,aifKnots,dDot=True)

#Get integral of input function from start to end of CBV frame
aifInteg = np.dot(pBasisI,aifCoefs)
aifSum = aifInteg[1] - aifInteg[0]

#Get whole-brain pet value
wbPet = np.mean(petMasked)

#Get scale factor to convert imaging data from counts*s/mL to counts/hg of brain tissue.
cbvScale = (100.0*cbvTime[1]) / (args.r*args.d*aifSum)

#Calculate whole-brain CBV
wbCbv = wbPet * cbvScale

#Write out whole-brain results
try:
	wbOut = open('%s_wbVal.txt'%(args.out[0]), "w")
	wbOut.write('CBV = %f'%(wbCbv))
	wbOut.close()
except(IOError):
	print 'ERROR: Cannot write in output directory. Exiting...'
	sys.exit()

#Create aif figure
try:
	plt.clf()	
	fig = plt.figure(1) 
	plt.scatter(aifTime,aifC,s=40,c="black")
	plt.plot(aifTime,np.dot(aifBasis,aifCoefs),linewidth=5,label='Spline Fit')
	plt.xlabel('Time (seconds)')
	plt.ylabel('Counts')
	plt.title('Arterial Sampled Input function')
	plt.legend(loc='upper right')
	plt.suptitle(args.out[0])
	plt.savefig('%s_aifPlot.jpeg'%(args.out[0]),bbox_inches='tight')
except(RuntimeError,IOError):
	print 'ERROR: Could not save figure. Moving on...'

#Don't do voxelwise estimation if user says not to
if args.wbOnly == 1:
	nagini.writeArgs(args,args.out[0])
	sys.exit()

#Calculate voxelwise CBV
cbvData = petMasked * cbvScale

#Write out CBV image
nagini.writeMaskedImage(cbvData,brainData.shape,brainData,pet.affine,pet.header,'%s_cbv'%(args.out[0]))
nagini.writeArgs(args,args.out[0])

