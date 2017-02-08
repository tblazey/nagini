#/usr/bin/python

###################
###Documentation###
###################

"""

Set of python functions for PET image processing. Currently there are the following functions:

loadHeader: Loads the header of a Nifiti Image
writeMaskedImage: Writes out an image that has been masked (first puts it back to original dimensions)
loadIdaif: Loads in a idaif from a one-column text file
loadAif: Loads in standard Wash U .crt file
reshape4d: Reshape a 4d image to a 2d image (perserves the last dimension)
loadInfo: Reads in Yi Su style info file
writeText: Simple wrapper for numpy.savetxt
flowTwoIdaif: Produces model fit for two-paramter water model. 
flowThreeDelay: Produces model fit for two-paramter water model with delay. 
flowTwoSet: Produces model fit for two-parameter water model with fixed delay and dispersion.
flowFour: Produces model fit for two-parameter water model with variable delay and dispersion.
flowFourMl: Returns the negative log-likelihood for two-parameter model water model with variable delay and dispersion.
flowThreeIdaif: Produces model fit for three-parameter water model (volume component). 
gluThreeIdaif: Produces model fit for three-parameter fdg model. 
gluIdaifLin: Calculates model fit for linearized version of three-parameter fdg model
gluGefIdaif: COmputes model fit for three-parameter FDG model while using CBF to estimate GEF
gluFourIdaif: Produces model fit for four-parameter fdg model. 
oefCalcIdaif: Calculates OEF using the stanard Mintun model. 
oxyOneIdaif: Procuces model fit for one-parameter Mintun oxygen model given CBF, lambda, and CBV. 
tfceScore: Calculates threshold free cluster enhancement for a statistic image. Not complete
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

def loadAif(aPath):
	"""
	
	Loads in text file with AIF values

	Parameters
	----------
	aPath : string
	   Path to AIF text file

	Returns
	-------
	aif : n by 2 matrix
	   Numpy array with sampling times and aif
	
	"""
	
	try:
		aif = np.loadtxt(aPath,skiprows=2,usecols=[0,1])
	except (IOError):
		print 'ERROR: Cannot load idaif at %s'%(aPath)
		sys.exit()
	return aif

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

def writeText(out,X):
	"""

	Write out numpy array. Wrapper for numpy.savetxt

	Parameters
	----------
	out: string
	   A filename
	X: Numpy array
	   A numpy array to write out

	"""

	try:
		np.savetxt(out,X)
	except (IOError):
		print 'ERROR: Cannot save output file at %s'%(out)
		sys.exit()

def flowTwoIdaif(X,flow,lmbda=1):
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

def flowThreeDelay(aifInterp):

	"""
	
	Produces a three parameter blood flow model fit function for scipy curvefit

	Parameters
	----------

	aifCoef: function
		Interpolation function for AIF

	Returns
	-------
	flowPred: function
		A function that will return the three-parameter blood flow predictions 
		given pet time, flow, lambda, delta
	
	"""

	def flowPred(petTime,flow,lmbda,delta):

			
		"""
	
		Produces a three parameter blood flow model fit function for scipy curvefit

		Parameters
		----------

		petTime: array
	   		An n length array pet time points
		flow: float
	  		Blood flow parameter
		lmbda: float
			Blood brain paritition coefficient parameter
		delta: float
			Delay parameter

		Returns
		-------
		flowConv: array
			A n length array of model predictions given parameters
	
		"""
		
		#Remove delay form input function
		cAif = aifInterp(petTime+delta) * np.exp(np.log(2)/122.24*delta)

		#Calculate model prediciton
		flowConv = np.convolve(flow*cAif,np.exp(-flow/lmbda*petTime))*(petTime[1]-petTime[0])
		
		#Return predictions
		return flowConv[0:petTime.shape[0]]
	
	#Return function
	return flowPred

def flowTwoSet(aifCoef,aifKnots,delta,tau):

	"""
	
	Produces a four parameter blood flow model fit function for scipy curvefit

	Parameters
	----------

	aifCoef: array
	   An n length array of coefficients for natural cubic spline
	aifKnots: array
	   A n length array of knot locations for natural cubic spline
	delta: float
		Input function shift
	tau: float
		Input function disperison

	Returns
	-------
	flowPred: function
		A function that will return the four-parameter blood flow predictions 
		given pet time, flow, and lambda
	
	"""

	def flowPred(petTime,flow,lmbda):

			
		"""
	
		Produces a four parameter blood flow model fit function for scipy curvefit

		Parameters
		----------

		petTime: array
	   		An n length array pet time points
		flow: float
	  		Blood flow parameter
		lmbda: float
			Blood brain paritition coefficient parameter

		Returns
		-------
		flowConv: array
			A n length array of model predictions given parameters
	
		"""
		
		#Remove delay and dispersion from input function while using spline interpolation
		cBasis, cBasisD = rSplineBasis(petTime+delta,aifKnots)
		cAif = np.dot(cBasis,aifCoef) + np.dot(cBasisD,aifCoef)*tau

		#Calculate model prediciton
		flowConv = np.convolve(flow*cAif,np.exp(-flow/lmbda*petTime))*(petTime[1]-petTime[0])
		
		#Return predictions
		return flowConv[0:petTime.shape[0]]
	
	#Return function
	return flowPred

def flowFour(aifCoef,aifKnots):

	"""
	
	Produces a four parameter blood flow model fit function for scipy curvefit

	Parameters
	----------

	aifCoef: array
	   An n length array of coefficients for natural cubic spline
	aifKnots: array
	   A n length array of knot locations for natural cubic spline

	Returns
	-------
	flowPred: function
		A function that will return the four-parameter blood flow predictions 
		given pet time, flow, lambda, delta and tau
	
	"""

	def flowPred(petTime,flow,lmbda,delta,tau):

			
		"""
	
		Produces a four parameter blood flow model fit function for scipy curvefit

		Parameters
		----------

		petTime: array
	   		An n length array pet time points
		flow: float
	  		Blood flow parameter
		lmbda: float
			Blood brain paritition coefficient parameter
		delta: float
			Delay parameter
		tau: float
			Dispersion parameter

		Returns
		-------
		flowConv: array
			A n length array of model predictions given parameters
	
		"""
		
		#Remove delay and dispersion from input function while using spline interpolation
		cBasis, cBasisD = rSplineBasis(petTime+delta,aifKnots)
		cAif = np.dot(cBasis,aifCoef) + np.dot(cBasisD,aifCoef)*tau

		#Calculate model prediciton
		flowConv = np.convolve(flow*cAif,np.exp(-flow/lmbda*petTime))*(petTime[1]-petTime[0])
		
		#Return predictions
		return flowConv[0:petTime.shape[0]]
	
	#Return function
	return flowPred

def flowFourMl(aifCoef,aifKnots,petTime,petTac):

	"""
	
	Produces a four parameter blood flow log-likelihood function for scipy curvefit

	Parameters
	----------

	aifCoef: array
	   	An n length array of coefficients for natural cubic spline
	aifKnots: array
	   	A n length array of knot locations for natural cubic spline
	petTime: array
		A m lengtha rray containing pet times
	petTac: array
		A m length array containing pet timecourse

	Returns
	-------
	flowLl: function
		A function that will return the log likelihood for blood flow model given 
		given pet time, flow, lambda, delta and tau
	
	"""

	def flowLl(params):

			
		"""
	
		Produces a four parameter blood flow log likelihood function for scipy curvefit

		Parameters
		----------

		params: array
			A vector with 5 elements where:
				params[0] = flow
				params[1] = lmbda
				params[2] = delta
				params[3] = tau
				params[4] = sigma

		Returns
		-------
		logLik: float
			 Negative log likelihood of model given parameters
	
		"""
		
		#Rename params
		flow = params[0]; lmbda = params[1]; delta = params[2]; tau = params[3]; sigma = params[4]

		#Remove delay and dispersion from input function while using spline interpolation
		cBasis, cBasisD = rSplineBasis(petTime+delta,aifKnots)
		cAif = np.dot(cBasis,aifCoef) + np.dot(cBasisD,aifCoef)*tau

		#Calculate model prediciton
		flowConv = np.convolve(flow*cAif,np.exp(-flow/lmbda*petTime))[0:petTime.shape[0]]*(petTime[1]-petTime[0])

		#Calculate log-likelihood
		nData = flowConv.shape[0]
		logLik = (-nData/2.0*np.log(2*np.pi)) - (-nData/2.0*np.log(sigma)) \
			 - ((1.0/2.0*sigma)*np.sum(np.power(petTac-flowConv,2)))
		
		#Return predictions
		return -logLik
	
	#Return function
	return flowLl

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

def gluIdaifLin(X,bOne,bTwo,bThree):

	"""

	Produces a three parameter FDG model fit using the linearized form. 

	Parameters
	----------
	X : array
	   A [3,n] numpy array with the first row is first integral of the input function,
	   the second the second integral of the input function, and the third the integral of the 
	   time activity curve
	bZero : float
	   first parameter
	bTwo: float
	   second parameter
	bThree: float
	   third paramter

	Returns
	-------
	gluPred: array
	   An n length array with model predictions

	"""

	return bOne*X[0,:] + bTwo*X[1,:] + bThree*X[2,:]
	
def gluGefIdaif(cbf):

	"""
	
	Produces a three parameter FDG model fit function for scipy curvefit. Calculates GEF.

	Parameters
	----------

	cbf: float
	   Cerebral blood flow estimate. In 1/seconds.

	Returns
	-------
	gluPred: function
		A function that will return the three-parameter FDG predictions 
		given X, flow, gef, k2 and k3
	
	"""

	def gluPred(X,gef,kTwo,kThree):
	
		"""
	
		scipy.optimize.curvefit model function for the three-parameter FDG model.
		Does not correct for delay or dispersion of input function, so only for use with an IDAIF

		Parameters
		----------
		X : array
	   		A [2,n] numpy array where the first row is time and the second the input function
		gef : float
	   		Glucose extraction fraction
		kTwo: float
	  		kTwo parameter
		kThree: float
	   		kThree paramter

		Returns
		-------
		cT : array
	   		A n length array with the model predictions given flow, gef,kTwo,and kThree
	
		"""
	
		minTime = X[0,1] - X[0,0]
   		cOne = np.convolve(X[1,:],(cbf*gef)*np.exp(-(kTwo+kThree)*X[0,:]))*minTime
		cTwo = np.convolve(X[1,:],((cbf*gef*kThree)/(kTwo+kThree))*(1-np.exp(-(kTwo+kThree)*X[0,:])))*minTime
		return cOne[0:X.shape[1]] + cTwo[0:X.shape[1]]
	
	#Return function
	return gluPred

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

def rSplineBasis(X,knots,integ=False):
	
	"""
		
	Calculates a restricted cubic spline basis for X given a set of knots
	
	Parameters
	----------
	X : array
	   A array of length n containing the x-values for cubic spline basis
	knots: array
	   An array of length p containing knot locations
	integ: logical
		If True, returns integral of spline as well

	Returns
	-------
	basis : matrix
		an nxp basis for a restricted cubic spine
	deriv : matrix
		an nxp matrix of the derivaties for the basis functions
	integ : matrix
		If integ=True, an nxp matrix of the integrals for the basis function
	
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
	if integ is True: 
		
		#Matrix for storing integral results
		aDeriv = np.zeros((nPoints,nKnots))
				
		#Compute first integral terms
		aDeriv[:,0] = X 
		aDeriv[:,1] = np.power(X,2) * 0.5
			
	#Set second basis function to x-value
	basis[:,1] = X; deriv[:,1] = 1 
	
	#Loop through free knots
	for knotIdx in range(nKnots-2):

		#First part of basis function
		termOne = np.maximum(0,np.power(X-knots[knotIdx],3))
		signOne = np.sign(termOne)
		termOneD = np.power(X-knots[knotIdx],2) * 3.0 * signOne
		
		#Second part of basis function
		scaleD = (knots[nKnots-1]-knots[nKnots-2])
		twoScale = (knots[nKnots-1]-knots[knotIdx]) / scaleD 
		termTwo = np.maximum(0,np.power(X-knots[nKnots-2],3)) * twoScale
		signTwo = np.sign(termTwo)
		termTwoD = np.power(X-knots[nKnots-2],2) * 3.0 * twoScale * signTwo
		
		#You get the drill
		threeScale = (knots[nKnots-2]-knots[knotIdx]) / scaleD
		termThree = np.maximum(0,np.power(X-knots[nKnots-1],3)) * threeScale
		signThree = np.sign(termThree)
		termThreeD = np.power(X-knots[nKnots-1],2) * 3.0 * threeScale * signThree
		
		#Compute the basis function. 
		basis[:,knotIdx+2] =  termOne - termTwo + termThree
		
		#Compute derivative.
		deriv[:,knotIdx+2] = termOneD - termTwoD + termThreeD
		
		#Compute integral if necessary
		if integ is True: 
			termOneInt = np.power(X-knots[knotIdx],4) * 0.25 * signOne
			termTwoInt = np.power(X-knots[nKnots-2],4) * 0.25 * twoScale * signTwo
			termThreeInt = np.power(X-knots[nKnots-1],4) * 0.25 * threeScale * signThree
			aDeriv[:,knotIdx+2] = termOneInt - termTwoInt + termThreeInt
			
	#Return appropriate basis set
	if integ is True:
		return basis, deriv, aDeriv
	else:	
		return basis, deriv
		
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





