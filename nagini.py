#/usr/bin/python

###################
###Documentation###
###################

"""

Set of python functions for PET image processing. Currently there are the following functions:

loadHeader: Loads the header of a Nifiti Image
writeMaskedImage: Writes out an image that has been masked (first puts it back to original dimensions)
loadIdaif: Loads in a idaif from a one-column text file
reshape4d: Reshape a 4d image to a 2d image (perserves the last dimension)
loadInfo: Reads in Yi Su style info file
flowTwoIdaif: Produces model fit for two-paramter water model. For use with scipy.optimize.curvefit
flowThreeIdaif: Produces model fit for three-parameter water model. For use with scipy.optimize.curvefit
gluThreeIdaif: Produces model fit for three-parameter fdg model. For use with scipy.optimize.curvefit
gluFourIdaif: Produces model fit for four-parameter fdg model. For use with scipy.optimize.curvefit
oefCalcIdaif: Calculates OEF using the stanard Mintun model. 
oxyOneIdaif: Procuces model fit for one-parameter Mintun oxygen model. For use with sci.py.optimize.curvefit
tfceCalc: Calculates threshold free cluster enhancement for a statistic image
rSplineBasis: Produces a restricted spline basis and its derivatives
knotLoc: Determines location of knots for cubic spline using percentiles


"""

#What libraries do we need
import numpy as np, nibabel as nib, sys, scipy.ndimage as img

###############
###Functions###
###############

def loadHeader(path):
	"""
	
	Quick function to load in image header given a Nifti image

	Parameters
	----------
	path : string
	   Path to numpy image

	Returns
	-------
	header : Nibabel image header
	   Header data for nibabel image
	
	"""
	
	try:
		header = nib.load(path)
	except (IOError,nib.spatialimages.ImageFileError):
		print 'ERROR: Cannot load image at %s.'%(path)
		sys.exit()
	return header

def writeMaskedImage(data,outDims,mask,affine,name):
	"""
	
	Writes out an masked image

	Parameters
	----------
	data : array 
	   Numpy array representing masked image
	outDims : array
	   Dimensions out final output image
	mask : array
	   Numpy array that originally masked data
	affine : array
	   Nibabel affine matrix for mask image
	name : string
	   Output filename. Will append .nii.gz by default 

	"""
	
	#Get masked data array back into original dimensions
	outData = np.zeros_like(mask,dtype=np.float64)
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

def loadIdaif(iPath):
	"""
	
	Loads in text file with IDAIF values

	Parameters
	----------
	iPath : string
	   Path to IDAIF text file

	Returns
	-------
	idaif : array
	   Numpy array with idaif values
	
	"""
	
	try:
		idaif = np.loadtxt(iPath)
	except (IOError):
		print 'ERROR: Cannot load idaif at %s'%(iPath)
		sys.exit()
	return idaif

def reshape4d(array):
	"""
	
	Reshapes a 4d numpy array into a 2d array by flattening the first three dimensions

	Parameters
	----------
	array : array
	   4-D numpy array (a,b,c,d)

	Returns
	-------
	array : array
	   2-D numpy array with dimensions (a*b*c,d)
	
	"""
	
	return array.reshape((array.shape[0]*array.shape[1]*array.shape[2],array.shape[3]))

def loadInfo(iPath):
	"""
	
	Loads in Yi Su style info file.

	Parameters
	----------
	iPath : string
	   Path to Yi Su style info file

	Returns
	-------
	info : array
	   Returns numpy array with five columns (start time, middle time, duration, decay, and frame number)
	
	"""
	
	try:
		info = np.loadtxt(iPath)
	except (IOError):
		print 'ERROR: Cannot load info file at %s'%(iPath)
		sys.exit()
	return info

def flowTwoIdaif(X,flow,lmbda):
	"""
	
	scipy.optimize.curvefit model function for the two-paramter model flow model.
	Does not correct for delay or dispersion of input function, so only for use with an IDAIF

	Parameters
	----------
	X : array
	   A [2,n] numpy array where the first row is time and the second the input function
	flow : float
	   Flow parameter
	lmbda: float
	   Lambda (blood/brain partiion coefficient)

	Returns
	-------
	flowConv : array
	   A n length array with the model predictions given flow and lmbda
	
	"""
	
	flowConv = np.convolve(flow*X[1,:],np.exp(-flow/lmbda*X[0,:]))*(X[0,1]-X[0,0])
	return flowConv[0:X.shape[1]]

def flowThreeIdaif(X,flow,kTwo,vZero):
	"""
	
	scipy.optimize.curvefit model function for the three-paramter model flow model.
	Does not correct for delay or dispersion of input function, so only for use with an IDAIF

	Parameters
	----------
	X : array
	   A [2,n] numpy array where the first row is time and the second the input function
	flow : float
	   Flow parameter
	kTwo: float
	   Clearance of tracer from brain to blood
	vZero: float
	   Apparent vascular distrubtion volume of tracer

	Returns
	-------
	flowPred : array
	   A n length array with the model predictions given flow, lmbda, and vZero
	
	"""
	
	flowConv = np.convolve(flow*X[1,:],np.exp(-kTwo*X[0,:]))[0:X.shape[1]]*(X[0,1]-X[0,0])
	flowPred = flowConv + (vZero*X[1,:])
	return flowPred

#Function to fit fdg tracer data using three-parameter convolution model. X[0,:] is time and X[1,:] is idaif
def gluThreeIdaif(X,kOne,kTwo,kThree):
	"""
	
	scipy.optimize.curvefit model function for the three-parameter FDG model.
	Does not correct for delay or dispersion of input function, so only for use with an IDAIF

	Parameters
	----------
	X : array
	   A [2,n] numpy array where the first row is time and the second the input function
	kOne : float
	   kOne parameter
	kTwo: float
	   kTwo parameter
	kThree: float
	   kThree paramter

	Returns
	-------
	cT : array
	   A n length array with the model predictions given kOne,kTwo,and kThree
	
	"""
	minTime = X[0,1] - X[0,0]
   	cOne = np.convolve(X[1,:],kOne*np.exp(-(kTwo+kThree)*X[0,:]))*minTime
	cTwo = np.convolve(X[1,:],((kOne*kThree)/(kTwo+kThree))*(1-np.exp(-(kTwo+kThree)*X[0,:])))*minTime
	return cOne[0:X.shape[1]] + cTwo[0:X.shape[1]]

#Function to fit fdg tracer data using four-parameter convolution model
def gluFourIdaif(X,kOne,kTwo,kThree,kFour):
	"""
	
	scipy.optimize.curvefit model function for the four-parameter FDG model.
	Does not correct for delay or dispersion of input function, so only for use with an IDAIF

	Parameters
	----------
	X : array
	   A [2,n] numpy array where the first row is time and the second the input function
	kOne : float
	   kOne parameter
	kTwo: float
	   kTwo parameter
	kThree: float
	   kThree paramter
	kFour: float
	   kFour parameter

	Returns
	-------
	cT : array
	   A n length array with the model predictions given kOne, kTwo, kThree, and kFour
	
	"""
	minTime = X[0,1] - X[0,0]
	aLeft = kTwo + kThree + kFour
	aRight = np.sqrt(np.power(kTwo+kThree+kFour,2)-(4.0*kTwo*kFour))
	aOne = (aLeft - aRight) / 2.0
	aTwo = (aLeft + aRight) / 2.0
   	cLeft = kOne/(aTwo-aOne)
	cMiddle = (kThree+kFour-aOne) * np.exp(-aOne*X[0,:])
	cRight = (aTwo-kThree-kFour) * np.exp(-aTwo*X[0,:])
	cI = np.convolve(cLeft*(cMiddle+cRight),X[1,:])[0:X.shape[1]]*minTime
	return cI

#Function to calculate OEF given an oxygen timecourse and values for CBF, CBV and lambda
def oefCalcIdaif(pet,petTime,oxyIdaif,waterIdaif,cbf,cbv,lmbda,R):
	
	"""
	
	Simple function to calculate OEF according to the Mintun, 1984 model

	Parameters
	----------
	pet : array
	   A array of length n containing the pet timecourse values
	petTime: array
	   An array of length n containing the sampling times for PET. Must be evently spaced.
	oxyIdaif: array
	   An array of length n containing the input function for oxygen.
	waterIdaif: array
	   An array of length n containing the input fuction for water.
	cbf: float
	   CBF value in 1/seconds
	cbv: float
	   CBV value (unitless)
	lmbda: float
	   Blood-brain paritition coefficient (unitless)
	R: float
	   Ratio of small-vessel to large-vessel hematocrit (unitless)

	Returns
	-------
	oef : float
	   Estimated oxygen extraction fraction
	
	"""
	
	#Integrate the pet signal
	petInteg = np.trapz(pet,petTime)

	#Integrate the oxygen IDIAF
	oxyInteg = np.trapz(oxyIdaif,petTime)

	#Convolve oxygen and water with negative exponentials. Then integrate
	sampTime = petTime[1] - petTime[0]
	oxyExpInteg = np.trapz(np.convolve(oxyIdaif,np.exp(-cbf/lmbda*petTime))[0:petTime.shape[0]]*sampTime,petTime)
	waterExpInteg = np.trapz(np.convolve(waterIdaif,np.exp(-cbf/lmbda*petTime))[0:petTime.shape[0]]*sampTime,petTime)

	#Calculate EOF using the standard Mintun method
	return ( petInteg - (cbf*waterExpInteg) - (cbv*R*oxyInteg) ) / ( (cbf*oxyExpInteg) - (cbv*R*0.835*oxyInteg) )

#Function to calculate TFCE score for statistic image
def tfceScore(stat,mask,E=0.5,H=2,dH=0.1):
	
	"""
	
	Function to calculate Threshold-Free Cluster Enhancement for a statistic image. See Smith and Nichols 2009

	Parameters
	----------
	stat : array
	   A [n,m,q] statistic image
	mask: array 
	   A [n,m,q]  mask for the statistic image. Array elements greater than 0 are included
	E: float
	   TFCE extent parameter
	H: float  
	   TFCE height paramter
	dH: float
	    Step size for TFCE height integration

	Returns
	-------
	tfce : array
	   A [n,m,q] array of tfce scores at each voxel
	
	"""

	#Create a 6 neighbor connectivity matrix for clustering
	cMatrix = np.zeros((3,3,3))
	cMatrix[0:3,1,1] = 1; cMatrix[1,0:3,1] = 1; cMatrix[1,1,0:3] = 1

	#Flatten and mask the stat image
	maskFlat = mask.flatten() > 0
	statMasked = stat.flatten()[maskFlat]

	#Get array of integration steps to go through. Start with a value very close to zero so that we can ignore any 0 voxels are ignored
	dSteps = np.arange(1E-10,np.max(stat),dH)

	#Create empty array to store tfce scores
	tfceScore = np.zeros_like(statMasked)
	
	#Loop through every integration step
	for step in dSteps:

		#Threshold the image at current t-level
		stepInput = stat>=step

		#Get connected components at current step
		clust,nClust = img.measurements.label(stepInput,cMatrix)

		#Remove voxels outside mask
		clustMasked = clust.flatten()[maskFlat]
		stepMasked = stepInput.flatten()[maskFlat]

		#Get size and peak of each cluster
		clustSize = img.sum(stepMasked,clustMasked,range(nClust+1))
		clustPeak = img.maximum(statMasked,clustMasked,range(nClust+1))

		#Apply TFCE summation
		tfceScore = tfceScore + (np.power(clustSize[clustMasked],E) * np.power(clustPeak[clustMasked],H)*dH)


	#Return TFCE score at each voxel
	tfceFull = np.zeros_like(stat)
	tfceFull[mask>0] = tfceScore
	return tfceFull

#Produces a restricted cubic spline basis and its derivatives
def rSplineBasis(X,knots):
	
	"""
		
	Calculates a restricted cubic spline basis for X given a set of knots
	
	Parameters
	----------
	X : array
	   A array of length n containing the x-values for cubic spline basis
	knots: array
	   An array of length p containing knot locations

	Returns
	-------
	basis : matrix
		an nxp basis for a restricted cubic spine
	deriv : matrix
		an nxp matrix of the derivaties for the basis functions
	
	"""
	
	#Check number of knots
	nKnots = knots.shape[0]
	if nKnots <= 2:
		print 'Number of knots must be at least 3'
		sys.exit()	

	#Create array to store basis matrix and derivatives
	nPoints = X.shape[0]
	basis = np.ones((nPoints,nKnots))
	deriv = np.zeros((nPoints,nKnots))
	
	#Set second basis function to x-value
	basis[:,1] = X; deriv[:,1] = 1;
	
	#Loop through free knots
	for knotIdx in range(nKnots-2):

		#First part of basis function
		termOne = np.maximum(0,np.power(X-knots[knotIdx],3))
		termOneD = np.maximum(0,np.power(X-knots[knotIdx],2)*3) * np.sign(termOne)
		
		#Second part of basis function
		scaleD = (knots[nKnots-1]-knots[nKnots-2])
		twoScale = (knots[nKnots-1]-knots[knotIdx]) / scaleD 
		termTwo = np.maximum(0,np.power(X-knots[nKnots-2],3)) * twoScale          
		termTwoD = np.maximum(0,np.power(X-knots[nKnots-2],2)*3) * twoScale * np.sign(termTwo) 
		
		#You get the drill
		threeScale = (knots[nKnots-2]-knots[knotIdx]) / scaleD
		termThree = np.maximum(0,np.power(X-knots[nKnots-1],3)) * threeScale		    
		termThreeD = np.maximum(0,np.power(X-knots[nKnots-1],2)*3) * threeScale	* np.sign(termThree)
		
		#Compute the basis function. 
		basis[:,knotIdx+2] =  termOne - termTwo + termThree
		
		#Compute derivative.
		deriv[:,knotIdx+2] = termOneD - termTwoD + termThreeD
		
	
	return basis, deriv
	
#Calculates knot locations for cubic spline using percentiles
def knotLoc(X,nKnots):
	
	"""
		
	Calculates location for knots based on sample quantiles
	
	Parameters
	----------
	X : array
	   A array of length n containing the x-values
	nKnots: interger
	  Number of knots

	Returns
	-------
	knots : array
		A set of knot locations
		
	Notes
	-----
	Uses the same basic algorithm as Hmisc package in R:
		
		For 3 knots -> outer percentiles are 10 and 90%
		For 4-6 knots -> outer percentiels are 5% and 95%
		For >6 knots -> outer percentiles are 2.5% and 97.5%
		
		All other knots are linearly spaced between outer percentiles
	
	"""
	
	#Set boundary knot percentiles 
	if nKnots <= 2:
		print 'ERROR: Number of knots must be at least 3'
		sys.exit()	
	elif nKnots == 3:
		bKnots = [10,90]
	elif nKnots >= 4 and nKnots <= 6:
		bKnots = [5,95]
	elif nKnots >6 and nKnots <= X.shape[0]:
		bKnots = [2.5,97.5]
	else:
		'ERROR: Cannot determine knot locations for number of knots: %s'%(nKnots)
		sys.exit()
	
	#Get percentiles for all knots
	knotP = np.linspace(bKnots[0],bKnots[1],nKnots)
		
	#Get actual knot locations based upon percentiles
	knots = np.percentile(X,knotP)
	
	return knots

#Creates optimization function scipy curve fit
def oxyOneIdaif(flow,lmbda,cbv,R):
	
	"""
	
	Produces a model fit function for scipy curvefit

	Parameters
	----------

	flow: float
	   Estimate of cerebral blood flow
	lmbda: float
	   Estimate of blood brian partieint coeficient
	cbv: float
	   Estimate of CBV
	R: float
	   Ratio of small-vessel to large vessel hematocrit
	

	Returns
	-------
	oxyPred: function
		A function that will return the Mintun model predictions given inputs X and E 
	
	"""
	
	#Actual model prediction function
	def oxyPred(X,E):

		"""
	
		Calculates the model predictions

		Parameters
		----------

		X: array
	   	   A 3,n array containing pet times, oxygen input function, and water input function
		E: float
		   Oxygen extraction fraction
	

		Returns
		-------
		cT: array of length n
	  	   Model predictions for Mintun oxygen model give input parameters
	
		"""

		#Get sampling time and number of time points
		sampTime = X[0,1] - X[0,0]
		nTime = X.shape[1]

		#Calculate components of model
		cOne = cbv*R*(1-(E*0.835))*X[1,:]
		cTwo = flow*np.convolve(X[2,:],np.exp(-flow/lmbda*X[0,:]))[0:nTime]*sampTime
		cThree = flow*E*np.convolve(X[1,:],np.exp(-flow/lmbda*X[0,:]))[0:nTime]*sampTime

		#Return model predictions
		return cOne + cTwo + cThree

	return oxyPred



