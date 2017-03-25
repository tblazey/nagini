
#!/usr/bin/python

###################
###Documentation###
###################

"""

gluAif.py: Calculates cmrGlu using a C11 glucose  scan and an arterial sampled input function

Requires the following modules:
	argparse, numpy, nibabel, nagini, tqdm, scipy, matplotlib

Tyler Blazey, Winter 2017
blazey@wustl.edu

"""

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Estimates cmrGlu using:')
argParse.add_argument('pet',help='Nifti cmrGlu image',nargs=1,type=str)
argParse.add_argument('info',help='Yi Su style info file',nargs=1,type=str)
argParse.add_argument('dta',help='Dta file with drawn samples',nargs=1,type=str)
argParse.add_argument('pie',help='Pie calibration factor',nargs=1,type=float)
argParse.add_argument('hct',help='Hematocrit',nargs=1,type=float)
argParse.add_argument('blood',help='Blood glucose concentration mg/dL',type=float)
argParse.add_argument('cbf',help='CBF image. In mL/hg*min',nargs=1,type=str)
argParse.add_argument('cbv',help='CBV image for blood volume correction in mL/hg',nargs=1,type=str)
argParse.add_argument('out',help='Root for outputed files',nargs=1,type=str)
argParse.add_argument('-brain',help='Brain mask in PET space',nargs=1,type=str)
argParse.add_argument('-noMet',help='Do not perform linear metabolite corretion',action='store_const',const=1)
argParse.add_argument('-noPlasma',help='Do not convert input function to plasma',action='store_const',const=1)
argParse.add_argument('-wbOnly',action='store_const',const=1,help='Only perform whole-brain estimation')
argParse.add_argument('-noDelay',action='store_const',const=1,help='Do not include AIF delay term in model.')
argParse.add_argument('-fModel',action='store_const',const=1,help='Does delay estimate at each voxel.')
argParse.add_argument('-dT',help='Density of brain tissue in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-dB',help='Density of bood in g/mL. Default is 1.05',default=1.05,metavar='density',type=float)
argParse.add_argument('-gBound',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds for GEF parameter. Default is 0 to 1')
argParse.add_argument('-twBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for kTwo parameter. Default is 10 times whole brain volume')
argParse.add_argument('-thBound',nargs=2,type=float,metavar=('lower', 'upper'),help='Bounds for voxelwise kThree parameter. Default is 10 times whole-brain estimate.')
argParse.add_argument('-fBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for voxelwise kFour parameter. Default is 10 times whole-brain estimate.')
argParse.add_argument('-dBound',nargs=2,type=float,metavar=('lower','upper'),help='Bounds for voxelwise delay parameter. Default is 10 times whole-brain estimate.')
args = argParse.parse_args()

#Make sure sure user set bounds correctly
for bound in [args.gBound,args.twBound,args.thBound,args.fBound,args.dBound]:
	if bound is not None:
		if bound[1] <= bound[0]:
			print 'ERROR: Lower bound of %f is not lower than upper bound of %f'%(bound[0],bound[1])
			sys.exit()

#Load needed libraries
import numpy as np, nibabel as nib, nagini, sys, scipy.optimize as opt
import scipy.interpolate as interp, matplotlib.pyplot as plt, matplotlib.gridspec as grid
from tqdm import tqdm

#Ignore invalid  and overflow warnings
np.seterr(invalid='ignore',over='ignore')

#########################
###Data Pre-Processing###
#########################
print ('Loading images...')

#Load in the dta file
dta = nagini.loadDta(args.dta[0])

#Load in the info file
info = nagini.loadInfo(args.info[0])

#Load image headers
pet = nagini.loadHeader(args.pet[0])
cbf = nagini.loadHeader(args.cbf[0])
cbv = nagini.loadHeader(args.cbv[0]) 

#Check to make sure dimensions match
if pet.shape[3] != info.shape[0] or pet.shape[0:3] != cbf.shape[0:3] or pet.shape[0:3] != cbv.shape[0:3] :
	print 'ERROR: Data dimensions do not match. Please check...'
	sys.exit()

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

#Get the image data
petData = pet.get_data()
cbfData = cbf.get_data()
cbvData = cbv.get_data()

#Get mask where inputs are non_zero
maskData = np.where(np.logical_and(np.logical_and(brainData!=0,cbfData!=0),cbvData!=0),1,0)

#Flatten the PET images and then mask
petMasked = nagini.reshape4d(petData)[maskData.flatten()>0,:]

#Make blood flow and blood volume
cbfMasked = cbfData[maskData>0].flatten()
cbvMasked = cbvData[maskData>0].flatten()

#Calculate kOne from flow and fractional blood volume
flowMasked = cbfMasked / cbvMasked / 60.0
vbMasked = cbvMasked / 100.0 * args.dT

#Get whole brain means of kOne and fractional blood volume
flowWb = np.mean(flowMasked)
vbWb = np.mean(vbMasked)

#Get pet start and end times. Assume start of first recorded frame is injection (has an offset which is subtracted out)
startTime = info[:,0] - info[0,0]
midTime = info[:,1] - info[0,0]
endTime = info[:,2] + startTime

#Combine PET start and end times into one array
petTime = np.stack((startTime,endTime),axis=1)

#Get decay corrected blood curve to injection
drawTime,corrCounts = nagini.corrDta(dta,1220.04,args.dB,toDraw=False)

#Apply time offset that was applied to images
corrCounts *= np.exp(np.log(2)/1220.04*info[0,0])

#Apply pie factor and scale factor to AIF
corrCounts /= (args.pie[0] * 0.06 )

#Apply basic lienar metabolite correction to input function
if args.noMet is None:
	corrCounts *= (1 - (4.983185e-05*drawTime))

#Apply simple conversion of whole blood tac to plasma tac
if args.noPlasma is None:
	corrCounts /= (1 - (0.3*args.hct[0]))
	plasma = True
else:
	args.blood *= (1 - (0.3*args.hct[0]))
	plasma = False

#########################
###Data Pre-Processing###
#########################
print 'Fitting AIF...'

#Run piecwise fit to start the initialization process for AIF fitting
maxIdx = np.argmax(corrCounts)
tRight = drawTime[maxIdx:]; cRight = np.log(corrCounts[maxIdx:])
initOpt,initCov = opt.curve_fit(nagini.segModel,tRight,cRight,[10,0.1,0.01,-0.01,-0.001,-0.0001])

#Save params for initialization
aTwo = np.exp(initOpt[1]); aThree = np.exp(initOpt[2])
eTwo = initOpt[4]; eThree = initOpt[5]

#Get delay coefficient
tMax= drawTime[np.argmax(corrCounts)]
grad = np.gradient(corrCounts,drawTime)
tau = drawTime[np.argmax(grad)-1]
if tau >= tMax:
	tau = 0

#Get inits for first component
aOne = ((np.max(corrCounts)-aTwo-aThree)*np.exp(1) + aTwo+aThree)/(tMax-tau)
eOne = -1 / ( (tMax-tau) - (aTwo+aThree)/aOne)

#Setup bounds and inits
inits = [np.min((np.max((0,tau)),60)),aOne,aTwo,aThree,eOne,eTwo,eThree]; lBound = [0]; hBound = [60]
for pIdx in range(1,len(inits)):
	if inits[pIdx] > 0:
		lBound.append(inits[pIdx]/5.0)
		hBound.append(inits[pIdx]*5.0)
	else:
		lBound.append(inits[pIdx]*5.0)
		hBound.append(inits[pIdx]/5.0)
bounds = [lBound,hBound]

#Run fit
aifFit,aifCov =  opt.curve_fit(nagini.fengModel,drawTime,corrCounts,inits,bounds=bounds)

#Run global optimization
aifGlobal = opt.basinhopping(nagini.fengModelGlobal,aifFit,minimizer_kwargs={'args':(drawTime,corrCounts),'bounds':np.array(bounds).T})

#Get interpolation times as start to end of pet scan with aif sampling rate
sampTime = np.min(np.diff(drawTime))
interpTime = np.arange(np.floor(startTime[0]),np.ceil(endTime[-1]+sampTime),sampTime)

#Get whole-brain tac
wbTac = np.mean(petMasked,axis=0)

###################
###Model Fitting###
###################
print ('Beginning fitting procedure...')

#Setup the proper model function
wbInit = [1,1,1,1]; wbBounds = [[0.2,0.2,.2,.2],[5,5,5,5]]
if args.noDelay == 1:

	#Interpolate Aif
	interpAif = nagini.fengModel(interpTime,aifGlobal.x[0],aifGlobal.x[1],aifGlobal.x[2],
								  aifGlobal.x[3],aifGlobal.x[4],aifGlobal.x[5],aifGlobal.x[6])

	#Setup optimization function for model with no delay
	wbFunc = nagini.gluAif(interpTime,interpAif,flowWb,vbWb,plasma,args.hct[0])
else:
	
	#Model with delay
	wbFunc = nagini.gluDelay(aifGlobal.x,interpTime,flowWb,vbWb,plasma,args.hct[0])

	#Add in starting points and bounds
	wbInit.extend([0]); wbBounds[0].extend([-25]); wbBounds[1].extend([25])

#Attempt to fit model to whole-brain curve
try:
	wbOpt,wbCov = opt.curve_fit(wbFunc,petTime,wbTac,p0=wbInit,bounds=wbBounds)
except(RuntimeError):
	print 'ERROR: Cannot estimate model on whole-brain curve. Exiting...'
	sys.exit()

#Calculate whole-brain correlation matrix
sdMat = np.diag(np.sqrt(np.diag(wbCov)))
sdMatI = np.linalg.inv(sdMat)
wbCor = sdMatI.dot(wbCov).dot(sdMatI)

#Remove scale
wbScale = [0.202,0.00389,0.00473,0.000303]
wbOpt[0:4] *= wbScale

#Calculate metabolic rate
gluScale = 333.0449 / args.dT; kOneWb = wbOpt[0]*flowWb
wbCmr = (kOneWb*vbWb*args.blood*wbOpt[2])/(wbOpt[1]+wbOpt[2]) * gluScale

#Calculate net extraction
wbNet = (vbWb/(vbWb*flowWb))*(kOneWb*wbOpt[2])/(wbOpt[1]+wbOpt[2])

#Calculate influx 
wbIn = args.blood*vbWb*kOneWb*gluScale

#Calculate free glucose
wbFree = wbCmr / (100.0*wbOpt[2]*60.0)

#Create string for whole-brain parameter estimates
labels = ['gef','kOne','kTwo','kThree','kFour','cmrGlu','eNet','influx','freeG','condition']
values = [wbOpt[0],kOneWb*60,wbOpt[1]*60.0,wbOpt[2]*60.0,wbOpt[3]*60.0,wbCmr,wbNet,wbIn,wbFree,np.linalg.cond(wbCov)]
units = ['fraction','1/min','1/min','1/min','1/min','uMol/hg/min','fraction','uMol/hg/min','uMol/g','unitless']
wbString = ''
for pIdx in range(len(labels)):
	wbString += '%s = %f (%s)\n'%(labels[pIdx],values[pIdx],units[pIdx])

#Add in delay estimate
if args.noDelay is None:
	wbString += 'delay = %f (min)\n'%(wbOpt[4]/60.0)

#Write out whole-brain results
try:
	#Write out values
	wbOut = open('%s_wbVals.txt'%(args.out[0]), "w")
	wbOut.write(wbString)
	wbOut.close()

	#Write out covariance matrix
	np.savetxt('%s_wbCov.txt'%(args.out[0]),wbCov)

	#Write out correlation matrix
	np.savetxt('%s_wbCor.txt'%(args.out[0]),wbCor)

except(IOError):
	print 'ERROR: Cannot write in output directory. Exiting...'
	sys.exit()

#Get whole brain fitted values
wbOpt[0:4] /= wbScale
if args.noDelay == 1:
	wbFitted = wbFunc(petTime,wbOpt[0],wbOpt[1],wbOpt[2],wbOpt[3])
else:
	wbFitted = wbFunc(petTime,wbOpt[0],wbOpt[1],wbOpt[2],wbOpt[3],wbOpt[4])

#Create whole brain fit figure
try:
	fig = plt.figure(1) 
	gs = grid.GridSpec(1,2)

	#Make fit plot
	axOne = plt.subplot(gs[0,0])
	axOne.scatter(midTime,wbTac,s=40,c="black")
	axOne.plot(midTime,wbFitted,linewidth=3,label='Model Fit')
	axOne.set_xlabel('Time (seconds)')
	axOne.set_ylabel('Counts')
	axOne.set_title('Whole-Brain Time Activity Curve')
	axOne.legend(loc='upper left')

	#Make input function plot
	axTwo = plt.subplot(gs[0,1])
	axTwo.scatter(drawTime,corrCounts,s=40,c="black")
	aifFitted = nagini.fengModel(drawTime,aifGlobal.x[0],aifGlobal.x[1],aifGlobal.x[2],
								  aifGlobal.x[3],aifGlobal.x[4],aifGlobal.x[5],aifGlobal.x[6])
	axTwo.plot(drawTime,aifFitted,linewidth=5,label='Spline Fit')
	axTwo.set_xlabel('Time (seconds)')
	axTwo.set_ylabel('Counts')
	axTwo.set_title('Arterial Sampled Input function')

	#Show corrected input funciton as well
	if args.noDelay != 1:
		
		#Interpolate input function and correct for delay
		cAif = nagini.fengModel(drawTime+wbOpt[4],aifGlobal.x[0],aifGlobal.x[1],aifGlobal.x[2],
								  aifGlobal.x[3],aifGlobal.x[4],aifGlobal.x[5],aifGlobal.x[6])
		cAif *= np.exp(np.log(2)/1220.04*wbOpt[4])

		#Add shifted plot		
		axTwo.plot(drawTime,cAif,linewidth=5,c='green',label='Delay Corrected')
	
	#Make sure we have legend for input function plot
	axTwo.legend(loc='upper right')
	
	#Add plots to figure
	fig.add_subplot(axOne); fig.add_subplot(axTwo)

	#Save plot
	plt.suptitle(args.out[0])
	fig.set_size_inches(15,5)
	fig.savefig('%s_wbPlot.jpeg'%(args.out[0]),bbox_inches='tight')
except(RuntimeError,IOError):
	print 'ERROR: Could not save figure. Moving on...'

#Don't do voxelwise estimation if user says not to
if args.wbOnly == 1:
	nagini.writeArgs(args,args.out[0])
	sys.exit()

#Get number of parameters for voxelwise optimizations
if args.fModel == 1 or args.noDelay == 1:
	nParam = wbOpt.shape[0]
else:
	nParam = wbOpt.shape[0]-1
init = wbOpt[0:nParam]

#Set default voxelwise bounds
lBounds = []; hBounds = [];
for pIdx in range(nParam):
	if wbOpt[pIdx] > 0:
		lBounds.append(wbOpt[pIdx]/2.5)
		hBounds.append(wbOpt[pIdx]*2.5)
	elif wbOpt[pIdx] < 0:
		lBounds.append(wbOpt[pIdx]*2.5)
		hBounds.append(wbOpt[pIdx]/2.5)
	else:
		lBounds.append(wbBounds[0][pIdx])
		hBounds.append(wbBounds[1][pIdx])
bounds = np.array([lBounds,hBounds],dtype=np.float)

#Create list of user bounds to check and set initlization
if args.fModel == 1 or args.noDelay is None:
	bCheck = [args.gBound,args.twBound,args.thBound,args.fBound,args.dBound]
else:
	bCheck = [args.gBound,args.twBound,args.thBound,args.fBound]

#Check user bounds
bIdx = 0
for bound in bCheck:
	#If user wants different bounds, use them.
	if bound is not None:
		bounds[0,bIdx] = bound[0]
		bounds[1,bIdx] = bound[1]
		#Use midpoint between bounds as initial value if whole brain estimate is not in bounds
		if init[bIdx] < bound[0] or init[bIdx] > bound[1]:		
			init[bIdx] = (bound[0]+bound[1]) / 2
	bIdx += 1

#Setup for voxelwise-optimization
nVox = petMasked.shape[0]
fitParams = np.zeros((nVox,nParam+6))

#Get corrected input function if we need to
if args.fModel != 1 and args.noDelay !=1:

	#Interpolate input function using delay
	voxAif = nagini.fengModel(interpTime+wbOpt[4],aifGlobal.x[0],aifGlobal.x[1],aifGlobal.x[2],
								  aifGlobal.x[3],aifGlobal.x[4],aifGlobal.x[5],aifGlobal.x[6])
	voxAif *= np.exp(np.log(2)/1220.04*wbOpt[4])

elif args.noDelay == 1:
	voxAif = interpAif

#Loop through every voxel
noC = 0
for voxIdx in tqdm(range(nVox)):
	
	#Get voxel tac 
	voxTac = petMasked[voxIdx,:]

	#Get proper model function. 	
	if args.fModel == 1 and args.noDelay !=1:
		optFunc = nagini.gluDelay(aifGlobal.x,interpTime,flowMasked[voxIdx],vbMasked[voxIdx],plasma,args.hct[0])
	else:
		optFunc = nagini.gluAif(interpTime,voxAif,flowMasked[voxIdx],vbMasked[voxIdx],plasma,args.hct[0])

	try:
		#Run fit
		voxOpt,voxCov = opt.curve_fit(optFunc,petTime,voxTac,p0=init,bounds=bounds)
		
		#Save common estimates 
		voxOpt[0:4] *= wbScale
		kOneVox = voxOpt[0]*flowMasked[voxIdx]
		fitParams[voxIdx,0] = voxOpt[0] 	#GEF
		fitParams[voxIdx,1] = kOneVox*60.0	#kOne
		fitParams[voxIdx,2] = voxOpt[1]*60.0	#kTwo
		fitParams[voxIdx,3] = voxOpt[2]*60.0	#kThree
		fitParams[voxIdx,4] = voxOpt[3]*60.0	#kFour
		fitParams[voxIdx,5] = (kOneVox*vbMasked[voxIdx]*args.blood*voxOpt[2]) \
					/ (voxOpt[1]+voxOpt[2]) * gluScale   #cmrGlu
		fitParams[voxIdx,6] = (vbMasked[voxIdx]/(vbMasked[voxIdx]*flowMasked[voxIdx])) \
					* (kOneVox*voxOpt[2])/(voxOpt[1]+voxOpt[2])   #enet
		fitParams[voxIdx,7] = args.blood*vbMasked[voxIdx]*kOneVox*gluScale   #influx
		fitParams[voxIdx,8] = fitParams[voxIdx,5]  / (100.0*voxOpt[2]*60.0)  #free

		#Do model specific saving
		if args.fModel == 1 and args.noDelay !=1:

			#Save delay parameter
			fitParams[voxIdx,9] = voxOpt[4]/60.0

			#Calculate residual with delay 
			fitResid = voxTac - optFunc(petTime,voxOpt[0],voxOpt[1],voxOpt[2],voxOpt[3],voxOpt[4])
		
		else:
			#Residual without delay
			fitResid = voxTac - optFunc(petTime,voxOpt[0],voxOpt[1],voxOpt[2],voxOpt[3])

		#Calculate normalized root mean square deviation
		fitRmsd = np.sqrt(np.sum(np.power(fitResid,2))/voxTac.shape[0]) / np.mean(voxTac)

		#Save residual
		fitParams[voxIdx,-1] = fitRmsd

	except(RuntimeError,ValueError):
		noC += 1

#Warn user about lack of convergence
if noC > 0:
	print('Warning: %i of %i voxels did not converge.'%(noC,nVox))

#############
###Output!###
#############
print('Writing out results...')

#Set names for model images
paramNames = ['gef','kOne','kTwo','kThree','kFour','cmrGlu','eNet','influx','freeG','delay','nRmsd']

#Do the writing. For now, doesn't write variance images.
for iIdx in range(fitParams.shape[1]-1):
	nagini.writeMaskedImage(fitParams[:,iIdx],maskData.shape,maskData,pet.affine,pet.header,'%s_%s'%(args.out[0],paramNames[iIdx]))
nagini.writeMaskedImage(fitParams[:,-1],maskData.shape,maskData,pet.affine,pet.header,'%s_%s'%(args.out[0],paramNames[-1]))
nagini.writeArgs(args,args.out[0])


