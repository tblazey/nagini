#!/usr/bin/python

###################
###Documentation###
###################

"""

cbfIdaifPoly.py: Calculates OEF using a 015 oxygen scan and an image-derived input function

Uses model described by Videen et al, JCBFM 1987

	
Produces the following outputs:
	wbVals -> Text file with whole-brain OEF and CMRO2
	wbTac -> Whole-brain tac
	wbFitted -> Text file with fit to whole-brain curve
	water -> Text file with calculated Xs and Ys for water polynomial regression
	oxy -> Text file with calculated Xs and Ys for oxygen polynomial regression
	polyCoef -> Text file with estimated regression coefficients

Requires the following modules:
	argparse, numpy, nibabel, nagini, tqdm, scipy

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
argParse.add_argument('cbv',help='Cerebral blodod volume image in mL/hg',nargs=1,type=str)
argParse.add_argument('artOxy',help='Measured arterial oxygen concentration in uM/mL',nargs=1,type=float)
argParse.add_argument('info',help='Yi Su style info file',nargs=1,type=str)
argParse.add_argument('idaif',help='Image-derived input function',nargs=1,type=str)
argParse.add_argument('brain',help='Brain mask in PET space',nargs=1,type=str)
argParse.add_argument('out',help='Root for outputed files',nargs=1,type=str)
argParse.add_argument('-d',help='Density of brain tissue in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-lmbda',help='Value for the blood-brain partition coefficient (mL/g). Default is 0.9',type=float,default=0.9)
argParse.add_argument('-delay',help='Delay parameter in seconds for estimating water component of IDAIF. Default is 20',default=20,metavar='delay',type=float)
argParse.add_argument('-decay',help='Decay constant in seconds for estimating water component of IDAIF. Default is 0.0012',default=0.0722/60.0,metavar='decay',type=float)
argParse.add_argument('-r',help='Value for the mean ratio of small-vessel to large-vessel hematocrit. Default is 0.85',nargs=1,default=0.85,type=float,metavar='ratio')
argParse.add_argument('-range',help='Time range for OEF estimation in seconds. Default is scan start to 45 seconds. Accepts start/end or numbers',nargs=2,metavar='time')
argParse.add_argument('-wbOnly',action='store_const',const=1,help='Only perform whole-brain estimation')
args = argParse.parse_args()

#Load needed libraries
import numpy as np, nibabel as nib, nagini, sys, scipy.interpolate as interp
from tqdm import tqdm

#########################
###Data Pre-Processing###
#########################
print ('Loading images...')

#Load image headers
pet = nagini.loadHeader(args.pet[0])
cbf = nagini.loadHeader(args.cbf[0])
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
   pet.shape[0:3] != cbv.shape[0:3] or pet.shape[3] != idaif.shape[0] or \
   pet.shape[3] != info.shape[0]:
	print 'ERROR: Data dimensions do not match. Please check...'
	sys.exit()

#Get the image data
petData = pet.get_data()
cbfData = cbf.get_data()
cbvData = cbv.get_data()
brainData = brain.get_data()

#Flatten the PET images and then mask. Also convert parameteric images back to original PET units. 
brainData = np.logical_and(np.logical_and(cbfData!=0,cbvData!=0),brainData!=0)
brainMask = brainData.flatten()
petMasked = nagini.reshape4d(petData)[brainMask,:]
cbfMasked = cbfData.flatten()[brainMask] #Don't convert CBF as quadratic equations are in real units
cbvMasked = cbvData.flatten()[brainMask] / 100 * args.d * args.r

#Interpolate the aif to minimum sampling time. Make sure we get the last time even if we have to go over range.
minTime = np.min(np.diff(petTime[petTime<petRange[1]]))
interpTime = np.arange(petRange[0],np.ceil(petRange[1]+minTime),minTime)
nTime = interpTime.shape[0]
aifLinear = interp.interp1d(petTime,idaif,kind="linear",fill_value="extrapolate")
aifInterp = aifLinear(interpTime)

#Get input function for h20 and oxygen seperately
aifDelay = aifLinear(interpTime-args.delay); aifDelay[aifDelay<0] = 0
aifWater = args.decay*np.convolve(aifDelay,np.exp(-args.decay*interpTime))[0:nTime]*minTime
aifOxy = aifInterp - aifWater

#Limit pet range
timeMask = np.logical_and(petTime>=petRange[0],petTime<=petRange[1])
petTime = petTime[timeMask]
petMasked = petMasked[:,timeMask]

#Get the average whole-brain flow, cbv, and lambda. Not strictly speaking correct as I should get the tacs.
wbFlow = np.mean(cbfMasked[cbfMasked!=0])
wbCbv = np.mean(cbvMasked[cbvMasked!=0])

#Get the whole brain tac. Then integrate it
wbTac = np.mean(petMasked,axis=0)
wbInt = np.trapz(wbTac,petTime)

###################
###Model Fitting###
###################
print ('Beginning fitting procedure...')

#Setup for polynomial regressions
cbfPred = np.arange(10,100,10); nPred = cbfPred.shape[0]
waterPred = np.zeros(nPred); oxyPred = np.zeros(nPred)

#Loop through range of flows
for pIdx in range(nPred):

	#Calculate PET model given flow, lmbda, and AIF
	waterModel = nagini.flowTwo(interpTime,aifWater)(petTime,cbfPred[pIdx]/6000.0*args.d,args.lmbda*args.d)
	oxyModel = nagini.flowTwo(interpTime,aifOxy)(petTime,cbfPred[pIdx]/6000.0*args.d,args.lmbda*args.d)

	#Integrate it over specefied range 
	waterPred[pIdx] = np.trapz(waterModel,petTime)
	oxyPred[pIdx] = np.trapz(oxyModel,petTime)

#Create design matrix for polynomial regression
polyX = np.stack((cbfPred,np.power(cbfPred,2)),axis=1)

#Calculate regression coefficients
waterCoef,_,_,_ = np.linalg.lstsq(polyX,waterPred)
oxyCoef,_,_,_ = np.linalg.lstsq(polyX,oxyPred)

#Integrate oxygen input function. Interpolate to make sure ranges match.
aifOxyInt = np.trapz(interp.interp1d(interpTime,aifOxy,kind="linear")(petTime),petTime)

#Get regression matrix for whole-brain
wbX = np.array([wbFlow,wbFlow**2])

#Calculate whole-brain OEF
wbOef = ( wbInt - wbX.dot(waterCoef) - (aifOxyInt*wbCbv) ) / \
	( wbX.dot(oxyCoef) - (0.835*aifOxyInt*wbCbv) )

#Get whole brain fitted values. Note setting R=1 as we already correct cbv values.
wbFitted = nagini.oxyOne(interpTime,aifWater,aifOxy,wbFlow/6000*args.d,\
				args.lmbda*args.d,wbCbv,1)(petTime,wbOef)

#Create string for whole-brain parameter estimates and regression coefficients
wbString = 'OEF=%f\nCMROxy=%f'%(wbOef,wbOef*wbFlow*args.artOxy[0])
coefString = 'aOne=%e\naTwo=%e\naThree=%e\naFour=%e'%(waterCoef[0],waterCoef[1],oxyCoef[0],oxyCoef[1])

#Write out whole-brain results
try:
	#Parameter estimates
	wbOut = open('%s_wbVals.txt'%(args.out[0]), "w")
	wbOut.write(wbString)
	wbOut.close()

	#Whole-brain tac
	np.savetxt('%s_wbTac.txt'%(args.out[0]),wbTac)

	#Whole-brain fitted values
	np.savetxt('%s_wbFitted.txt'%(args.out[0]),wbFitted)

	#Write out polynomial matries
	np.savetxt('%s_waterMat.txt'%(args.out[0]),np.hstack((polyX,waterPred[:,np.newaxis])))
	np.savetxt('%s_oxyMat.txt'%(args.out[0]),np.hstack((polyX,oxyPred[:,np.newaxis])))

	#Polynomial regression coefficients
	coefOut = open('%s_polyCoef.txt'%(args.out[0]), "w")
	coefOut.write(coefString)
	coefOut.close()

except(IOError):
	print 'ERROR: Cannot write in output directory. Exiting...'
	sys.exit()

#Don't do voxelwise estimation if user says not to
if args.wbOnly == 1:
	sys.exit()

#Integreate pet tacs
petInt = np.trapz(petMasked,petTime,axis=1)

#CBF predictio matrix for each voxel
petMat = np.stack((cbfMasked,np.power(cbfMasked,2)),axis=1)

#Calculate whole-brain OEF and CMRO2
petOef = ( petInt - petMat.dot(waterCoef) - (aifOxyInt*cbvMasked) ) / \
	( petMat.dot(oxyCoef) - (0.835*aifOxyInt*cbvMasked) )
petOxy = petOef * args.artOxy[0] * cbfMasked

#Write out images
nagini.writeMaskedImage(petOef,brain.shape,brainData,pet.affine,pet.header,'%s_oef'%(args.out[0]))
nagini.writeMaskedImage(petOxy,brain.shape,brainData,pet.affine,pet.header,'%s_cmrOxy'%(args.out[0]))
