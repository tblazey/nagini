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
flowTwoIdaif: Produces model fit for two-parameter water model.
flowTwo: Produces model for for two-parameter water model. Allows for masking for PET data.
flowThreeDelay: Produces model fit for two-paramter water model with delay.
flowFour: Produces model fit for two-parameter water model with variable delay and dispersion.
mixedLl: Produces negative log-likelihood of linear mixed model
flowThreeIdaif: Produces model fit for three-parameter water model (volume component).
gluThree: Produces model fit for three-parameter fdg model.
gluIdaifLin: Calculates model fit for linearized version of three-parameter fdg model
gluFourIdaif: Produces model fit for four-parameter fdg model.
oefCalc: Calculates OEF using the stanard Mintun model.
oxyOne: Procuces model fit for one-parameter Mintun oxygen model given CBF, lambda, and CBV.
tfceScore: Calculates threshold free cluster enhancement for a statistic image. Not complete
rSplineBasis: Produces a restricted spline basis withs options for derivaties and integrals
knotLoc: Determines location of knots for cubic spline using percentiles
roiAvg: Calculates ROI averages
roiBack: Puts ROI averages back into an image
saveGz: Saves a gzipped .npy file
saveRz: Saves and then compresses a python directionary in R dump format
writeArgs: Writes input arguments for argparse to a text file
loadDta: Loads a DTA file
toSeconds: Converts time from minutes.seconds to seconds
corrData: Decay correction for hand-drawn counts from DTA files
gluDelayLst: Computes model prediction for Powers C11 glucose model, with delay
gluAifLst: Computes model prediction for Powers C11 glucose model, no delay
gluDelayLstPen: Version of gluDelayLst with a penalty term
gluAifLstPen: Version of gluAifLst with a penalty term
golishModel: Computes model for Golish AIF model
golishModelDeriv: Computes derivaties for Golish AIF model
fengModel: Returns model predictions for Feng AIF model
fengModelGlobal: Produces sum of squares error for Feng AIF model
segModel: Model predictions for piecewise linear fit (three segments)
fdgDelayLst: Computes model prediction for no k4 FDG model, with delay
fdgAiflst: Computes model predictiosn ofr no k4 FDG model, no delay
fdgAifLstPen: Version of fdgAifLst with penalty


"""

#What libraries do we need
import numpy as np, nibabel as nib, sys, scipy.ndimage as img
import scipy.interpolate as interp, subprocess as sub, scipy.special as spec
import scipy.integrate as integ

#Only import pystan if we can
try:
	import pystan
except(ImportError):
	pass

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

def writeMaskedImage(data,outDims,mask,affine,header,name):
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
	   Nibabel affine matrix for data
	header : struct
	   Nibabel header for data
	name : string
	   Output filename. Will append .nii.gz by default

	"""

	#Get masked data array back into original dimensions
	outData = np.zeros_like(mask,dtype=np.float64)
	outData[mask==1] = data
	outData = outData.reshape(outDims)

	#Create image to write out
	outImg = nib.Nifti1Image(outData,affine,header=header)

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

def loadAif(aPath,dcv=False):
	"""

	Loads in text file with AIF values

	Parameters
	----------
	aPath : string
	   Path to AIF text file
	dcv : logical
	   Indicates whether input function is DCV format.

	Returns
	-------
	aif : n by 2 matrix
	   Numpy array with sampling times and aif

	"""

	try:
		if dcv is False:
			aif = np.loadtxt(aPath,skiprows=2,usecols=[0,1])
		else:
			aif = np.loadtxt(aPath,skiprows=1,usecols=[0,1])
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
	Does not correct for delay or dispersion of input function

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

def flowTwo(aifTime,cAif,decayC,petTac,petTime,petMask):

	#Mask pet data now
	petTacM = petTac[petMask]
	petTimeM = petTime[petMask,:]

	"""

	Parameters
	----------

	aifTime: array
	   A n length array of AIF sample times
	cAif: array
	   A n length array of AIF values at aifTime
	decayC: float
	   Decay constant for model. Set to 0 for no decay correction.
	petTac: array
		An m length array containing PET time activity Curve
	petTime: array
		An n x 2 array containing PET start times and end times
	petMask: logical array
		An m length array containing useable points (those where we have AIF samples)

	Returns
	-------
	flowPred: function
		A function that will return the two-parameter blood flow predictions
		given pet time, flow and lambda

	"""

	def flowPred(param,pred=False):


		"""

		Produces a two parameter blood flow model fit function for scipy minimize

		Parameters
		----------

		param: array
			a 2 x 1 array containing flow and lambda parameter
		pred: logical
			Logical for what to return (see below)

		Returns
		-------
		If pred is True:

		petPred: array
			A array of model predictions

		If pred is False:

		sse: float
				Sum of squares error given parameters

		"""

		#Rename variables
		flow = param[0] * 0.007
		lmbda = param[1] * 0.52

		#Calculate model prediciton
		flowConv = np.convolve(flow*cAif,np.exp(-((flow/lmbda)+decayC)*aifTime))[0:aifTime.shape[0]]
		flowConv *= (aifTime[1]-aifTime[0])

		#Get interpolation function for model predictions
		predInterp = interp.interp1d(aifTime,flowConv[0:aifTime.shape[0]],kind="linear")

		#Interpolate predicted response at start and end times
		startPred = predInterp(petTimeM[:,0])
		endPred = predInterp(petTimeM[:,1])

		#Calculate average between start and end time. Implies average after trap. integration
		petPred = (startPred+endPred)/2

		#Return sum of squares error
		if pred is True:
			return petPred
		else:
			return np.sum(np.power(petPred-petTacM,2))

	#Return function
	return flowPred

def flowThreeDelay(aifCoef,aifScale,aifTime,decayC,petTac,petTime):

	"""

	Parameters
	----------

	aifCoef: array
	   An q length array of coefficients for Golish AIF model
	aifScale: float
	   A scale factor for AIF
	aifTime: array
	   A n length array of times to samples AIF at
	decayC: float
	   Decay constant for model. Set to 0 for no decay correction.
	petTac: array
		An m length array containing PET data
	petTime: array
		An m x 2 array containing PET start and stop times

	Returns
	-------
	flowPred: function
		A function that will return the three-parameter blood flow predictions
		given pet time, flow, lambda, delta

	"""

	def flowPred(param,pred=False):


		"""

		Produces a three parameter blood flow model fit function for scipy minimize

		Parameters
		----------

		param: array
			A 3 x 1 array containing flow, lambda, and delay parameters
		pred: logical
			Logical indicating what to return (see below)

		Returns
		-------
		If pred is False:

		sse: float
			Sum of squares errors given parameters

		If pred is True:

		petPred: array
			A p length array of pet predictions
		aifMask: logical array
			A n length array of valid AIF time points
		petMask: logical array
			A m length array of vlaid PET time points

		"""

		#Rename parameters
		flow = param[0] * 0.007
		lmbda = param[1] * 0.52
		delta = param[2] * 10

		#Calculate Aif
		cAif = golishModel(aifTime+delta,aifCoef[0],aifCoef[1],aifCoef[2],aifCoef[3],aifCoef[4],aifCoef[5])

		#Correct AIF for decay during shift and rescale
		cAif *= np.exp(np.log(2)/122.24*delta)* aifScale

		#Make valid AIF mask
		aifMask = (aifTime+delta)<=aifTime[-1]
		aifTimeM = aifTime[aifMask]
		cAifM = cAif[aifMask]

		#Calculate model prediction
		flowConv = np.convolve(flow*cAifM,np.exp(-((flow/lmbda)+decayC)*aifTimeM))[0:aifTimeM.shape[0]]
		flowConv *= (aifTimeM[1]-aifTimeM[0])

		#Get interpolation function for model predictions
		predInterp = interp.interp1d(aifTimeM,flowConv,kind="linear")

		#Get PET points where we have AIF data
		petMask = np.logical_and(petTime[:,0]>=aifTimeM[0],petTime[:,1]<=aifTimeM[-1])
		petTimeM = petTime[petMask,:]
		petTacM = petTac[petMask]

		#Interpolate predicted response at start and end times
		startPred = predInterp(petTimeM[:,0])
		endPred = predInterp(petTimeM[:,1])

		#Calculate average between start and end time. Implies trap. integration
		petPred = (startPred+endPred)/2

		#Return sum of squares error
		if pred is True:
			return petPred,petMask,aifMask
		else:
			return np.sum(np.power(petPred-petTacM,2))

	#Return function
	return flowPred

def flowFour(aifCoef,aifScale,aifTime,decayC,petTac,petTime):
	"""

	Parameters
	----------

	aifCoef: array
	   An q length array of coefficients for Golish AIF model
	aifScale: float
	   A scale factor for AIF
	aifTime: array
	   A n length array of times to samples AIF at
	decayC: float
	   Decay constant for model. Set to 0 for no decay correction.
	petTac: array
		An m length array containing PET data
	petTime: array
		An m x 2 array containing PET start and stop times

	Returns
	-------
	flowPred: function
		A function that will return the four-parameter blood flow predictions
		given flow, lambda, delta, and tau

	"""

	def flowPred(param,pred=False):

		"""

		Produces a four parameter blood flow model fit function for scipy minimize

		Parameters
		----------

		param: array
			A 4 x 1 array containing flow, lambda, delay, and tau
		pred: logical
			Logical indicating what to return (see below)

		Returns
		-------
		If pred is False:

		sse: float
			Sum of squares errors given parameters

		If pred is True:

		petPred: array
			A p length array of pet predictions
		aifMask: logical array
			A n length array of valid AIF time points
		petMask: logical array
			A m length array of vlaid PET time points

		"""

		#Rename and scale params
		flow = param[0] * 0.007
		lmbda = param[1] * 0.52
		delta = param[2] * 10.0
		tau = param[3] * 5.0

		#Calculate Aif
		cAif = golishModel(aifTime+delta,aifCoef[0],aifCoef[1],aifCoef[2],aifCoef[3],aifCoef[4],aifCoef[5])

		#Correct for disperison
		cAif += golishModelDeriv(aifTime+delta,aifCoef[0],aifCoef[1],aifCoef[2],aifCoef[3],aifCoef[4],aifCoef[5]) * tau

		#Correct AIF for decay during shift and rescale
		cAif *= np.exp(np.log(2)/122.24*delta) * aifScale

		#Make valid AIF mask
		aifMask = (aifTime+delta)<=aifTime[-1]
		aifTimeM = aifTime[aifMask]
		cAifM = cAif[aifMask]

		#Calculate model prediciton
		flowConv = np.convolve(flow*cAifM,np.exp(-(flow/lmbda)*aifTimeM))[0:aifTimeM.shape[0]]
		flowConv *= (aifTimeM[1]-aifTimeM[0])

		#Get PET points where we have AIF data
		petMask = np.logical_and(petTime[:,0]>=aifTimeM[0],petTime[:,1]<=aifTimeM[-1])
		petTimeM = petTime[petMask,:]
		petTacM = petTac[petMask]

		#Get interpolation function for model predictions
		predInterp = interp.interp1d(aifTimeM,flowConv,kind="linear")

		#Interpolate predicted response at start and end times
		startPred = predInterp(petTimeM[:,0])
		endPred = predInterp(petTimeM[:,1])

		#Calculate average between start and end time. Implies averaged after trap. integration
		petPred = (startPred+endPred)/2

		#Return sum of squares residuals or predictions
		if pred is True:
			return petPred,petMask,aifMask
		else:
			return np.sum(np.power(petPred-petTacM,2))

	#Return function
	return flowPred

def mixedLl(varCoefs,X,Z,y,coefs=False,const=False):
	"""

	Returns the restricted negative log-likelihood for a very, very basic linear mixed effects model

	Parameters
	----------
	varCoefs : vector
		A 2x1 numpy array for the variance parameters to be estimated
	X : matrix
		a mxn numpy matrix of fixed effects regressors
	Z : matrix
		a mxp numpy matrix of random effects regressors
	y : vector
                a mx1 numpy array of aif to fit
	coefs : logical
		If true, returns fixed and random effects along with log-likelihood
	const = logical
		If true, returns adds in constant part to log-likelihood

	Returns
	-------
	rLogLik : float
		 Restricted negative log-likelihood for model
	B : array
		If coefs = True, a n x 1 array of fixed effects
	b : array
		If coefs = True, a m x 1 array of random effects

	"""

	##Check to see data in input correctly
	print varCoefs
	nData = y.shape[0]
	if  X.shape[0] != nData or Z.shape[0] != nData:
		print 'ERROR: Number of rows in fixed and/or random effects does not match y'
		sys.exit()

	#Get random effects covariance matrix
	nR = Z.shape[1]
	D = np.identity(nR)*varCoefs[0]
	V = Z.dot(D).dot(Z.T) + np.identity(nData)*varCoefs[1]

	#Get inverse of total covariance matrix
	W = np.linalg.inv(V)

	#Get beta estimates
	B = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(y)

	#Calculate residuals
	resid = y - np.dot(X,B)

	#Calculate restricted log-likelihood.
	wSign,wLogDet = np.linalg.slogdet(W)
	hSign,hLogDet = np.linalg.slogdet(X.T.dot(W).dot(X))
	rLogLik = (0.5*wSign*wLogDet) - \
                  (0.5*resid.T.dot(W).dot(resid)) - \
		  (0.5*hSign*hLogDet)
	if const is True: rLogLik += -nData/2.0*np.log(2.0*np.pi)

	#Return log-likilihood and estimates if necessary
	if coefs == True:
		b = D.dot(Z.T).dot(W).dot(resid)
		return -rLogLik,B,b
	else:
		return -rLogLik

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

def gluThree(aifTime,aifC,cbf=None):
	"""

	Returns a prediction function for optimization of three-parameter glucose model

	Parameters
	----------
	aifTime: vector
		A nx1 vector of AIF samples times
	aifC: vector
		A nx1 vector of with the AIF data
	cbf: real
		Optional scale represnting CBF at voxel. If set, then we estimate GEF.

	Returns
	-------
	gluPred : function
	   A function that returns three-parameter model prediction given a vector of pet times, and k1,k2, and k3

	"""

	def gluPred(petTime,pOne,kTwo,kThree):
		"""

		Returns a prediction function for optimization of three-parameter glucose model

		Parameters
		----------
		petTime: vector
			A mx1 vector of pet sample times
		pOne: float
			kOne parameter if cbf is None. Otherwise it is GEF.
		kTwo: float
			kTwo parameter
		kThree: float
			kThree parameter

		Returns
		-------
		petPred : vector
	  		 A mx1 vector of model predictions given petTime and the rate constants kOne,kTwo, and kThree

		"""

		#Get minimum sampling time
		minTime = aifTime[1]-aifTime[0]

		#Convert kOne to GEF if necessary
		if cbf is None:
			kOne = pOne
		else:
			kOne = pOne*cbf

		#Calculate comparmental concentrations
   		cOne = np.convolve(aifC,kOne*np.exp(-(kTwo+kThree)*aifTime))*minTime
		cTwo = np.convolve(aifC,((kOne*kThree)/(kTwo+kThree))*(1-np.exp(-(kTwo+kThree)*aifTime)))*minTime
		cSum =  cOne[0:aifTime.shape[0]] + cTwo[0:aifTime.shape[0]]

		#Interpolate predicted response at pet times if necessary.
		if np.all(aifTime==petTime):
			petPred = cSum
		else:
			petPred = interp.interp1d(aifTime,cSum,kind="linear")(petTime)

		#Return predictions
		return petPred

	#Return prediction function
	return gluPred

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

def oefCalc(pet,petTime,aifTime,oxyAif,waterAif,cbf,cbv,lmbda,R):

	"""

	Simple function to calculate OEF according to the Mintun, 1984 model

	Parameters
	----------
	pet : array
	   A array of length n containing the pet timecourse values
	petTime: array
	   An array of length n containing the sampling times for PET. Must be evently spaced.
	aifTime : array
	   An array of length m containing aif times
	oxyAif: array
	   An array of length m containing the input function for oxygen.
	waterAif: array
	   An array of length m containing the input fuction for water.
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
	oxyInteg = np.trapz(oxyAif,aifTime)

	#Convolve oxygen and water with negative exponentials. Then integrate
	sampTime = aifTime[1] - aifTime[0]
	oxyExpInteg = np.trapz(np.convolve(oxyAif,np.exp(-cbf/lmbda*aifTime))[0:aifTime.shape[0]]*sampTime,aifTime)
	waterExpInteg = np.trapz(np.convolve(waterAif,np.exp(-cbf/lmbda*aifTime))[0:aifTime.shape[0]]*sampTime,aifTime)

	#Calculate OEF using the standard Mintun method
	return ( petInteg - (cbf*waterExpInteg) - (cbv*R*oxyInteg) ) / ( (cbf*oxyExpInteg) - (cbv*R*0.835*oxyInteg) )

def oxyOne(aifTime,aifWater,aifOxy,flow,lmbda,cbv,R):

	"""

	Produces a model fit function for scipy curvefit

	Parameters
	----------

	aifTime: vector
	   A n length vector of aif sample times
	aifWater: vector
	   A n length vector of aif samples for water
	aifOxy: vector
	   A n length vector of aif samples for oxygen
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

	#Get sampling time and number of time points
	sampTime = aifTime[1] - aifTime[0]
	nAif = aifTime.shape[0]

	#Actual model prediction function
	def oxyPred(petTime,E):

		"""

		Calculates the model predictions

		Parameters
		----------

		petTime: array
	   	   A m length array containing pet sample times
		E: float
		   Oxygen extraction fraction


		Returns
		-------
		petPred: array of length n
	  	   Model predictions for Mintun oxygen model give input parameters

		"""

		#Calculate components of model
		cOne = cbv*R*(1-(E*0.835))*aifOxy
		cTwo = flow*np.convolve(aifWater,np.exp(-flow/lmbda*aifTime))[0:nAif]*sampTime
		cThree = flow*E*np.convolve(aifOxy,np.exp(-flow/lmbda*aifTime))[0:nAif]*sampTime
		cSum = cOne + cTwo + cThree

		#Interpolate predicted response at pet times if necessary.
		if np.all(aifTime==petTime):
			petPred = cSum
		else:
			petPred = interp.interp1d(aifTime,cSum,kind="linear")(petTime)

		#Return predictions
		return petPred

	#Return prediction function
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

def rSplineBasis(X,knots,dot=False,dDot=False):

	"""

	Calculates a restricted cubic spline basis for X given a set of knots

	Parameters
	----------
	X : array
	   A array of length n containing the x-values for cubic spline basis
	knots: array
	   An array of length p containing knot locations
	dot: logical
	   If true returns derivative of spline
	dDot: logical
	   If true returns integral of spline

	Returns
	-------
	basis : matrix
		an nxp basis for a restricted cubic spine
	deriv : matrix
		If deriv=True, an nxp matrix of the derivaties for the basis functions
	integ : matrix
		If integ=True, an nxp matrix of the integrals for the basis function

	"""

	#Check number of knots
	nKnots = knots.shape[0]
	if nKnots <= 2:
		print 'Number of knots must be at least 3'
		sys.exit()

	#Create array to store basis matrix
	nPoints = X.shape[0]
	basis = np.ones((nPoints,nKnots))

	#Set second basis function to x-value
	basis[:,1] = X;

	#Setup for derivative if needed
	if dot is True:
		#Matrix for storing derivative results
		deriv = np.zeros((nPoints,nKnots))

		#Compute first derivative term
		deriv[:,1] = 1

	#Setup for integral if needed
	if dDot is True:

		#Matrix for storing integral results
		integ = np.zeros((nPoints,nKnots))

		#Compute first integral terms
		integ[:,0] = X
		integ[:,1] = np.power(X,2) * 0.5


	#Loop through free knots
	for knotIdx in range(nKnots-2):

		#First part of basis function
		termOne = np.maximum(0,np.power(X-knots[knotIdx],3))

		#Second part of basis function
		scaleD = (knots[nKnots-1]-knots[nKnots-2])
		twoScale = (knots[nKnots-1]-knots[knotIdx]) / scaleD
		termTwo = np.maximum(0,np.power(X-knots[nKnots-2],3)) * twoScale

		#You get the drill
		threeScale = (knots[nKnots-2]-knots[knotIdx]) / scaleD
		termThree = np.maximum(0,np.power(X-knots[nKnots-1],3)) * threeScale

		#Compute the basis function.
		basis[:,knotIdx+2] =  termOne - termTwo + termThree

		#Figure out signs of basis functions if further calculations are necessary
		if dot is True or dDot is True:
			signOne = np.sign(termOne)
			signTwo = np.sign(termTwo)
			signThree = np.sign(termThree)

		#Compute derivative if necessary
		if dot is True:
			termOneD = np.power(X-knots[knotIdx],2) * 3.0 * signOne
			termTwoD = np.power(X-knots[nKnots-2],2) * 3.0 * twoScale * signTwo
			termThreeD = np.power(X-knots[nKnots-1],2) * 3.0 * threeScale * signThree
			deriv[:,knotIdx+2] = termOneD - termTwoD + termThreeD

		#Compute integral if necessary
		if dDot is True:
			termOneInt = np.power(X-knots[knotIdx],4) * 0.25 * signOne
			termTwoInt = np.power(X-knots[nKnots-2],4) * 0.25 * twoScale * signTwo
			termThreeInt = np.power(X-knots[nKnots-1],4) * 0.25 * threeScale * signThree
			integ[:,knotIdx+2] = termOneInt - termTwoInt + termThreeInt

	#Return appropriate basis set
	if dot is True and dDot is True:
		return basis, deriv, integ
	elif dot is True:
		return basis, deriv
	elif dDot is True:
		return basis, integ
	else:
		return basis

def knotLoc(X,nKnots,bounds=None):

	"""

	Calculates location for knots based on sample quantiles

	Parameters
	----------
	X : array
	   A array of length n containing the x-values
	nKnots: interger
	  Number of knots
	bounds: array
		A 2 x 1 array containing percentile bounds.
		If not set then function uses method described below.

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
	elif bounds is not None:
		bKnots = [bounds[0],bounds[1]]
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

def roiAvg(imgData,roiData,min=None,max=None):

	"""

	Calculates ROI averages

	Parameters
	----------
	imgData : numpy array
		a n or nxm array of data to get into ROIs. Zeros are ignored in average.
	roiData: numpy array
		a n x 1 array of where each element is a ROI index. Zero is ignored
	min: float
		Minimum value to consider for average.
	max: float
		Maximum value to consider for average

	Returns
	-------
	avgData : numpy array
		a px1 or pxm array with averages for p ROIs


	"""

	#Reshape image data if necessary
	imgDim = len(imgData.shape)
	if imgDim == 1:
		imgData = imgData.reshape((imgData.shape[0],1))
	elif imgDim > 2:
		print 'ERROR: Input data is not a 1d or 2d array'
		sys.exit()

	#Get unique number of ROIs.
	uRoi = np.unique(roiData[roiData!=0])
	nRoi = uRoi.shape[0]

	#Create empty data array
	avgData = np.zeros((nRoi,imgData.shape[1]))

	#Loop though frames
	for fIdx in range(avgData.shape[1]):

		#Get rid of zeros in image
		imgMask = imgData[:,fIdx] != 0

		#Modify image mask if necessary
		if min is not None:
			imgMask = np.logical_and(imgMask,imgData[:,fIdx]>=min)
		if max is not None:
			imgMask = np.logical_and(imgMask,imgData[:,fIdx]<=max)

		#Loop through ROIs
		for rIdx in range(nRoi):

			#Get conjunction of ROI mask and image mask
			roiMask = np.logical_and(roiData == uRoi[rIdx],imgMask)

			#Compute mean within mask
			avgData[rIdx,fIdx] = np.mean(imgData[roiMask,fIdx])

	#Return data
	return avgData

def roiBack(avgData,roiData):

	"""

	Puts ROI averages back into image space

	Parameters
	----------
	avgData : numpy array
		a p x 1 or p x m array of  ROI averages
	roiData: numpy array
		a n x 1 array of where each element is a ROI index. Zero is ignored. Must contained m ROIs

	Returns
	-------
	backData : numpy array
		a nx1 or nxm array where each point is the average from its ROI


	"""

	#Reshape average data if necessary
	avgDim = len(avgData.shape)
	if avgDim == 1:
		avgData = avgData.reshape((avgData.shape[0],1))
	elif avgDim > 2:
		print 'ERROR: Input data is not a 1d or 2d array'
		sys.exit()

	#Get unique number of ROIs.
	uRoi = np.unique(roiData[roiData!=0])
	nRoi = uRoi.shape[0]

	#Make sure that the number of ROIs match
	if avgData.shape[0] != nRoi:
		print 'ERROR: Number of ROIs in roiData does not match dimensions of avgData.'
		sys.exit()

	#Create empty data array
	backData = np.zeros((roiData.shape[0],avgData.shape[1]))

	#Loop through ROIs
	for rIdx in range(nRoi):

		#Get ROI mask
		roiMask = roiData == uRoi[rIdx]

		#Set values within mask to ROI average
		backData[roiMask,:] = avgData[rIdx,:]

	#And we are done
	return(backData)

def saveGz(array,fName):

	"""

	Saves and then compresses a numpy array

	Parameters
	----------
	array : numpy array
	fName: file name


	"""

	try:
		np.save(fName,array)
		gZip = sub.call('gzip -f %s'%(fName),shell=True)
	except(IOError):
		print 'ERROR: Cannot save file at %s'%(fName)
		sys.exit()

def saveRz(dic,fName):

	"""

	Saves and then compresses a python dictionary in R dump format

	Parameters
	----------
	dic : python dictionary
	fName: file name


	"""

	try:
		pystan.stan_rdump(dic,fName)
		gZip = sub.call('gzip -f %s'%(fName),shell=True)
	except(IOError):
		print 'ERROR: Cannot save file at %s'%(fName)
		sys.exit()

def writeArgs(args,out):

	"""

	Saves arguments from argparse to a text file

	Parameters
	----------
	args : string
	   argparse arguments
	out: string
	   output path


	"""

	#Make string with all arguments
	argString = ''
	for arg,value in sorted(vars(args).items()):
		if type(value) is list:
			value = value[0]
		argString += '%s: %s\n'%(arg,value)

	#Write out arguments
	try:
		argOut = open('%s_args.txt'%(out), "w")
		argOut.write(argString)
		argOut.close()
	except(IOError):
		print 'ERROR: Cannot write in output directory. Exiting...'
		sys.exit()

def loadDta(dtaPath,header=True):

	"""

	Loads a old school DTA file for hand-drawn samples

	Parameters
	----------
	dtaPath : string
	   Path to DTA file
	header : logical
	   Does file contain a long header?

	Returns
	-------
	dta : numpy array
		a nSample x 8 array containing samples. Format is as follows

	Column 0: Corrected Time (seconds)
	Column 1: Corrected Counts(counts/mL-Blood sec). To injection (assumed to be zero)
	Column 2: Dry Weight (grams)
	Column 3: Wet Weight (grams)
	Column 4: Draw Time (minutes.seconds)
	Column 5: Count Time (minutes.seconds)
	Column 6: Counts
	Column 7: Counter Interval (seconds)

	"""

	#Get number of rows to skip
	if header is True:
		nSkip = 9
	else:
		nSkip = 2

	#Get the data
	try:
		dta = np.loadtxt(dtaPath,skiprows=nSkip)
	except(IOError):
		print 'ERROR: Cannot load file.at %s. Exiting...'%(dtaPath)
		sys.exit()

	return dta

def toSeconds(time):

	"""

	Function to convert times in minutes.seconds to seconds

	Parameters
	----------
	time : array
	   A array containing n time samples in minutes.seconds format

	Returns
	-------
	secs : array


	"""
	minutes = np.floor(time)
	seconds = time - minutes
	return (minutes*60) + (seconds*100)

def corrDta(dta,hLife,bDens,toDraw=True,dTime=True):

	"""

	Function to decay correct hand-drawn counts recorded in a DTA file

	Parameters
	----------
	dta : array
	   A nSample by 8 array. See nagini.loadDta for format
	hLife : float
	   Half-life of tracer in seconds
	bDens : float
	   Density of blood in g/ml
	toDraw: logical
	   If True, decay correct to draw time. If false, decay correct to injection
   	dTime: logical
	   If True, perform dead time corection

	Returns
	-------
	drawTime: array
	   A nSample by 1 array of draw times (sec)
	corrCounts : array
	   A nSample by 1 array of decay corrected counts (counts/mL-blood/sec)

	"""

	#Make sure dta array is the right size
	if dta.shape[1] != 8:
		print 'ERORR: DTA file does not have 8 columns. Exiting...'
		sys.exit()

	#If toDraw is true, decay correct to draw time. Otherwise to injection
	drawTime = toSeconds(dta[:,4])
	if toDraw is True:
		decayTime = toSeconds(dta[:,5])-drawTime
	else:
		decayTime = toSeconds(dta[:,5])

	#Get the decay constant
	dc = np.log(2)/hLife

	#Calculate sample weight in grams
	weight = dta[:,3] - dta[:,2]

	#Dead time correction. A bit mysterious.
	if dTime is True:
		x   = (0.001 * dta[:,6]) * (12.0 / dta[:,7])
		fac =  (0.000005298 * np.power(x,2)) + (0.0004575 * x) + 1.0
	else:
		fac = 1

	#Decay correct for time in well counter
	corrCounts = fac * dta[:,6] * dc / (1-np.exp(-dc*dta[:,7]))

	#Decay correct to injection or draw time. Also get in counts/mlB/sec
	corrCounts *= bDens * np.exp(dc*decayTime) / weight

	#Adjust for position and volume. Also mysterious
	corrCounts /= 1.026 + -0.0522*weight

	return drawTime,corrCounts

def gluDelayLst(aifCoef,aifTime,pet,flow,vb):

	"""

	Parameters
	----------

	aifCoef: array
	   An array of coefficients for Feng Model
	aifTime: array
	   A n length array of times to samples AIF at
	pet:
	   A m length array of pet data
	flow: float
	   flow from blood flow model (CBF/CBV). In seconds
	vb: float
	   Fractional blood volume. In mL-blood/mL-tissue

	Returns
	-------
	gluPred: function
		A function that will return the five-parameter glucose model prections

	"""

	def gluPred(petTime,p1,p2,delta,coefs=False):


		"""

		Produces a three parameter glucose model fit function for scipy curvefit

		Parameters
		----------

		petTime: array
	   	   An m x 2 array containing frame start and end times
		b1: float
	  	   First exponential parameter
		b2: float
		   Second exponential parameter
		delta: float
			Delay parameter
		coefs: logical
			If true, return estimated coefficients. If only returns predictions

		Returns
		-------
		petPred: array
			A n length array of model predictions given parameters
		petCoefs: array
		    A 4x1 array containing the estimated model coefficients

		"""

		#Scale beta parameters
		b1 = p1 * 0.015
		b2 = p2 * 0.00025

		#Get delay corrected input function
		wbAif = fengModel(aifTime+delta,aifCoef[0],aifCoef[1],aifCoef[2],aifCoef[3],aifCoef[4],
		                 aifCoef[5],aifCoef[6])

		#Correct input function for decay during delay
		wbAif *= np.exp(np.log(2)/1220.04*delta)

		#Calculate concentration in compartment one
		cOne = vb*wbAif

		#Interpolate compartment one
		cOneInterp = interp.interp1d(aifTime,cOne,kind='linear')
		cOnePet = (cOneInterp(petTime[:,0])+cOneInterp(petTime[:,1]))/2.0

		#Convert AIF to plasma
		pAif = wbAif*(1.19 + -0.002*aifTime/60.0)

		#Remove metabolites from plasma input function\
		pAif *=  (1 - (4.983e-05*aifTime))

		#Compute first basis function
		bfOne = np.convolve(pAif,np.exp(-b1*aifTime))[0:aifTime.shape[0]]

		#Interpolate first basis function
		aifStep = aifTime[1]-aifTime[0]
		bfOneInterp = interp.interp1d(aifTime,bfOne*aifStep,kind='linear')
		bfOnePet = (bfOneInterp(petTime[:,0])+bfOneInterp(petTime[:,1]))/2.0

		#Compute second basis function
		bfTwo = np.convolve(pAif,np.exp(-b2*aifTime)/(b1-b2) +
								   np.exp(-b2*aifTime)*b2/((b1-b2)*(flow-b2)) +
								   np.exp(-flow*aifTime)*b2/((b1-flow)*(b2-flow)))[0:aifTime.shape[0]]

		#Interpolate second basis function
		bfTwoInterp = interp.interp1d(aifTime,bfTwo*aifStep,kind='linear')
		bfTwoPet = (bfTwoInterp(petTime[:,0])+bfTwoInterp(petTime[:,1]))/2.0

		#Compute the alphas using linear least squares
		petX = np.stack((bfOnePet,bfTwoPet),axis=1)
		alpha,_,_,_ = np.linalg.lstsq(petX,pet-cOnePet)

		#Calculate model prediction
		petPred = cOnePet + alpha[0]*bfOnePet + alpha[1]*bfTwoPet

		#Return total tissue predictions
		if coefs is False:
				return petPred
		else:
			return petPred,np.array([alpha[0],alpha[1],b1,b2])

	#Return function
	return gluPred

def gluAifLst(aifTime,pAif,pet,cOne,flow):

	"""

	Parameters
	----------

	aifTime: array
	   A n length array of times to samples AIF at
	pAif: array
	   A n length array of plasma aif samples
	pet: array
	   A m length array of pet data.
	cOne: array
	   A m length array of pet concentration in vascular compartment
	flow: float
	   flow from blood flow model (CBF/CBV). In seconds

	Returns
	-------
	gluPred: function
		A function that will return the five-parameter glucose model prections

	"""

	#Remove vascular component from tac
	pet -= cOne

	def gluPred(petTime,p1,p2,coefs=False):


		"""

		Produces a two parameter glucose model fit function for scipy curvefit

		Parameters
		----------

		petTime: array
	   	   An m x 2 array containing frame start and end times
		b1: float
	  	   First exponential parameter
		b2: float
		   Second exponential parameter
		coefs: logical
			If true, return estimated coefficients. If false return predictions

		Returns
		-------
		petPred: array
			A n length array of model predictions given parameters
		petCoefs: array
	        A 4x1 array containing the estimated model coefficients

		"""

		#Scale beta parameters
		b1 = p1 * 0.015
		b2 = p2 * 0.00025

		#Compute first basis function
		bfOne = np.convolve(pAif,np.exp(-b1*aifTime))[0:aifTime.shape[0]]

		#Interpolate first basis function
		aifStep = aifTime[1]-aifTime[0]
		bfOneInterp = interp.interp1d(aifTime,bfOne*aifStep,kind='linear')
		bfOnePet = (bfOneInterp(petTime[:,0])+bfOneInterp(petTime[:,1]))/2.0

		#Compute second basis function
		bfTwo = np.convolve(pAif,np.exp(-b2*aifTime)/(b1-b2) +
								   np.exp(-b2*aifTime)*b2/((b1-b2)*(flow-b2)) +
								   np.exp(-flow*aifTime)*b2/((b1-flow)*(b2-flow)))[0:aifTime.shape[0]]

		#Interpolate second basis function
		bfTwoInterp = interp.interp1d(aifTime,bfTwo*aifStep,kind='linear')
		bfTwoPet = (bfTwoInterp(petTime[:,0])+bfTwoInterp(petTime[:,1]))/2.0

		#Compute the alphas using linear least squares
		petX = np.stack((bfOnePet,bfTwoPet),axis=1)
		alpha,_,_,_ = np.linalg.lstsq(petX,pet)

		#Calculate model prediction
		petPred = cOne + alpha[0]*bfOnePet + alpha[1]*bfTwoPet

		#Return total tissue predictions
		if coefs is False:
			return petPred
		else:
			return petPred,np.array([alpha[0],alpha[1],b1,b2])

	#Return function
	return gluPred

def gluDelayLstPen(params,aifCoef,aifTime,pet,petTime,petSum,flow,vb,pen,penC,coefs=False):

	"""

	Parameters
	----------

	params: array
		3x1 vector of paramters.
	aifTime: array
	   A n length array of times to samples AIF at
	aifCoef: array
	   A 7x1 array of coefficients for Feng Model AIF
	pet: array
		A m length array of pet data
	petTime: array
		A m x 2 array of PET start and end times
	petSum: float
		Sum of time activity curve. Used to normalize peanlty
	cOne: array
	   An m length array with concentration in compartment one
	flow: float
	   Flow in units 1/seconds.
	vb: float
		CBV in units mlB/mlT
	pen: float
 	   The penalty for least squares optimization
	penC: float
	   Value at which penalty = 0
	coefs: logical
		If true, returns predictions and coefficients

	Returns
	-------
	If coefs = False
		SSE: float
			Sum of squares error with penalty
	If coefs = True:
		petPred: array
			An m length array of PET predictions
		coefs: array
			A 4x1 array of coefficients

	"""

	#Rename and scale parameters
	b1 = params[0] * 0.015
	b2 = params[1] * 0.00025
	delta = params[2]

	#Get delay corrected input function
	wbAif = fengModel(aifTime+delta,aifCoef[0],aifCoef[1],aifCoef[2],aifCoef[3],aifCoef[4],
	                 aifCoef[5],aifCoef[6])

	#Correct input function for decay during delay
	wbAif *= np.exp(np.log(2)/1220.04*delta)

	#Calculate concentration in compartment one
	cOne = vb*wbAif

	#Interpolate compartment one
	cOneInterp = interp.interp1d(aifTime,cOne,kind='linear')
	cOnePet = (cOneInterp(petTime[:,0])+cOneInterp(petTime[:,1]))/2.0

	#Convert AIF to plasma
	pAif = wbAif*(1.19 + -0.002*aifTime/60.0)

	#Remove metabolites from plasma input function\
	pAif *=  (1 - (4.983e-05*aifTime))

	#Compute first basis function
	bfOne = np.convolve(pAif,np.exp(-b1*aifTime))[0:aifTime.shape[0]]

	#Interpolate first basis function
	aifStep = aifTime[1]-aifTime[0]
	bfOneInterp = interp.interp1d(aifTime,bfOne*aifStep,kind='linear')
	bfOnePet = (bfOneInterp(petTime[:,0])+bfOneInterp(petTime[:,1]))/2.0

	#Compute second basis function
	bfTwo = np.convolve(pAif,np.exp(-b2*aifTime)/(b1-b2) +
							 np.exp(-b2*aifTime)*b2/((b1-b2)*(flow-b2)) +
							 np.exp(-flow*aifTime)*b2/((b1-flow)*(b2-flow)))[0:aifTime.shape[0]]

	#Interpolate second basis function
	bfTwoInterp = interp.interp1d(aifTime,bfTwo*aifStep,kind='linear')
	bfTwoPet = (bfTwoInterp(petTime[:,0])+bfTwoInterp(petTime[:,1]))/2.0

	#Compute the alphas using linear least squares
	petX = np.stack((bfOnePet,bfTwoPet),axis=1)
	alpha,_,_,_ = np.linalg.lstsq(petX,pet-cOnePet)

	#Calculate model prediction and residual
	petPred = cOnePet + alpha[0]*bfOnePet + alpha[1]*bfTwoPet
	petResid = pet - petPred

	#Return penalized sum of sqaures or predictions/coefficients
	if coefs is False:
		return np.sum(np.power(petResid,2)) + pen*petSum*np.power(np.log(penC/params[0]),2)
	else:
		return petPred,np.array([alpha[0],alpha[1],b1,b2])

def gluAifLstPen(params,aifTime,pAif,pet,petTime,petSum,cOne,flow,pen,penC,coefs=False):

	"""

	Parameters
	----------

	params: array
		2x1 vector of paramters.
	aifTime: array
	   A n length array of times to samples AIF at
	pAif: array
	   A n length array containing plasma arterial input function
	pet: array
		A m length array of pet data
	petTime: array
		A m x 2 array of PET start and end times
	petSum: float
		Sum of time activity curve. Used to normalize peanlty
	cOne: array
	   An m length array with concentration in compartment one
	flow: float
	   Flow in units 1/seconds.
	pen: float
 	   The penalty for least squares optimization
	penC: float
	   Value at which penalty = 0
	coefs: logical
		If true, returns predictions and coefficients

	Returns
	-------
	If coefs = False
		SSE: float
			Sum of squares error with penalty
	If coefs = True:
		petPred: array
			An m length array of PET predictions
		coefs: array
			A 4x1 array of coefficients

	"""

	#Scale beta parameters
	b1 = params[0] * 0.015
	b2 = params[1] * 0.00025

	#Compute first basis function
	bfOne = np.convolve(pAif,np.exp(-b1*aifTime))[0:aifTime.shape[0]]

	#Interpolate first basis function
	aifStep = aifTime[1]-aifTime[0]
	bfOneInterp = interp.interp1d(aifTime,bfOne*aifStep,kind='linear')
	bfOnePet = (bfOneInterp(petTime[:,0])+bfOneInterp(petTime[:,1]))/2.0

	#Compute second basis function
	bfTwo = np.convolve(pAif,np.exp(-b2*aifTime)/(b1-b2) +
							   np.exp(-b2*aifTime)*b2/((b1-b2)*(flow-b2)) +
							   np.exp(-flow*aifTime)*b2/((b1-flow)*(b2-flow)))[0:aifTime.shape[0]]

	#Interpolate second basis function
	bfTwoInterp = interp.interp1d(aifTime,bfTwo*aifStep,kind='linear')
	bfTwoPet = (bfTwoInterp(petTime[:,0])+bfTwoInterp(petTime[:,1]))/2.0

	#Remove vascular component from tac
	petC = pet - cOne

	#Compute the alphas using linear least squares
	petX = np.stack((bfOnePet,bfTwoPet),axis=1)
	alpha,_,_,_ = np.linalg.lstsq(petX,petC)

	#Calculate model prediction and residual
	petPred = cOne + alpha[0]*bfOnePet + alpha[1]*bfTwoPet
	petResid = pet - petPred

	#Return penalized sum of sqaures or predictions/coefficients
	if coefs is False:
		return np.sum(np.power(petResid,2)) + pen*petSum*np.power(np.log(penC/params[0]),2)
	else:
		return petPred,np.array([alpha[0],alpha[1],b1,b2])

def golishModel(t,cMax,cZero,alpha,beta,tZero,tau):

	"""

	Produces AIF model fit from Golish et al., 2001 Journal of Medicine

	Parameters
	----------

	t: array
	   An array of n timepoints
	cMax: float
	   cMax parameter
	cZero: array
	   cZero parameter
	alpha: float
	   alpha parameter
	beta: float
	   beta parameter
	tZero: float
		tZero parameter
	tau: float
	   tau parameter

	Returns
	-------
	golishPred: array
		An array of n predictions given parameters

	"""

	#Compute gamma components
	gOne = cMax*np.power((np.exp(1)/(alpha*beta))*(t-tZero),alpha)
	gTwo = np.exp(-(t-tZero)/beta)
	gTotal = gOne*gTwo

	#Compute recirculation term
	rTotal = cZero*(1-np.exp(-(t-tZero)/tau))

	#Get sum
	golishPred = gTotal + rTotal

	#Zero out anything before tZero
	golishPred[t<tZero] = 0

	return golishPred

def golishModelDeriv(t,cMax,cZero,alpha,beta,tZero,tau):

	"""

	Produces temporal derivative of AIF model fit from Golish et al., 2001 Journal of Medicine

	Parameters
	----------

	t: array
	   An array of n timepoints
	cMax: float
	   cMax parameter
	cZero: array
	   cZero parameter
	alpha: float
	   alpha parameter
	beta: float
	   beta parameter
	tau: float
	   tau parameter

	Returns
	-------
	golishDeriv: array
		An array of n precicted derivatives given parameters

	"""

	#Compute repeatable units
	lC = cMax*np.exp(alpha+((tZero-t)/beta))/beta
	rC = (t-tZero)/(alpha*beta)

	#Compute terms of derivative
	tOne = lC*np.power(rC,alpha-1) - lC*np.power(rC,alpha)
	tTwo = cZero*np.exp((tZero-t)/tau)/tau

	#Get sum
	golishDeriv = tOne + tTwo

	#Zero out anything before tZero
	golishDeriv[t<tZero] = 0

	return golishDeriv

def fengModel(t,tau,aOne,aTwo,aThree,eOne,eTwo,eThree):

	"""

	Produces model predictions for Feng AIF model 1993

	Parameters
	----------

	t: array
	   An array of n timepoints
	tau: float
	   tau parameter
	aOne: float
	   aOne parameter
	aTwo: float
	   aTwo parameter
	aThree: float
	   aThree parameter
	eOne: float
	   eOne parameter
	eTwo: float
	   eTwo parameter
	eThree: float
	   eThree parameter
	tau: float
	   tau parameter

	Returns
	-------
	fengPred: array
		An array of n precicted derivatives given parameters

	"""

	#Get components
	one = (aOne*(t-tau)-aTwo-aThree)*np.exp(eOne*(t-tau))
	two = aTwo*np.exp(eTwo*(t-tau))
	three = aThree*np.exp(eThree*(t-tau))

	#Get sum
	fengPred = one+two+three

	#Set points where t is less than tau to zero
	fengPred[t<tau] = 0.0

	return fengPred

def fengModelGlobal(param,t,y):

	"""

	Produces sum of squares error for Feng AIF model 1993

	Parameters
	----------
	param: array
		An array of parameters:
			tau
			aOne
			aTwo
			aThree
			eOne
			eTwo
			eThree
	t: array
	   An array of n timepoints
	y: array
	   An array of y data points

	Returns
	-------
	fengError: float
		Sum of squares error given parameters

	"""

	#Get model predictions
	fengPred = fengModel(t,param[0],param[1],param[2],param[3],param[4],param[5],param[6])

	#Return sum of squares error
	return np.sum(np.power(y-fengPred,2))

def segModel(t,iOne,iTwo,iThree,sOne,sTwo,sThree,bOne=100,bTwo=1000):

	"""

	Rough function that produces model predictions for piecewise linear fit

	Parameters
	----------

	t: array
	   An array of n timepoints
	iOne: float
	   Intercept for first segment
	iTwo: float
	   Intercept for second segment
	iThree: float
	   Intercept for third segment
	sOne: float
	   Slope for first segment
	sTwo: float
	   Slope for second segment
	sThree: float
	   Slope for third segment
	bOne: float
	   First boundry knot
	bTwo: float
	   Second boundry knot

	Returns
	-------
	segPred: array
		An array of n precicted derivatives given parameters

	"""

	#Create empty container for predictions
	segPred = np.zeros(t.shape[0])

	#Loop through timepoints
	for i in range(t.shape[0]):
		if t[i] < bOne:
			segPred[i] = iOne + sOne*t[i]
		elif t[i] > bOne and t[i] < bTwo:
			segPred[i] = iTwo + sTwo*t[i]
		else:
			segPred[i] = iThree + sThree*t[i]

	return segPred

def fdgDelayLst(aifCoef,aifTime,pet,vb,hct):

	"""

	Parameters
	----------

	aifCoef: array
	   An array of coefficients for Feng Model
	aifTime: array
	   A n length array of times to samples AIF at
	pet:
	   A m length array of pet data
	vb: float
	   Fractional blood volume. In mL-blood/mL-tissue
	hct: float
		Hemmatocrit ratio

	Returns
	-------
	gluPred: function
		A function that will return the five-parameter glucose model prections

	"""

	def gluPred(petTime,b1,delta,coefs=False):


		"""

		Produces a three parameter glucose model fit function for scipy curvefit

		Parameters
		----------

		petTime: array
	   	   An m x 2 array containing frame start and end times
		b1: float
	  	   Expoential parameter
		delta: float
			Delay parameter
		coefs: logical
			If true, return estimated coefficients. If only returns predictions

		Returns
		-------
		petPred: array
			A n length array of model predictions given parameters
		petCoefs: array
		    A 4x1 array containing the estimated model coefficients

		"""

		#Scale beta parameters
		b1 *= 0.0038

		#Get delay corrected input function
		wbAif = fengModel(aifTime+delta,aifCoef[0],aifCoef[1],aifCoef[2],aifCoef[3],aifCoef[4],
		                 aifCoef[5],aifCoef[6])

		#Correct input function for decay during delay
		wbAif *= np.exp(np.log(2)/6586.26*delta)

		#Calculate concentration in compartment one
		cOne = vb*wbAif

		#Interpolate compartment one
		cOneInterp = interp.interp1d(aifTime,cOne,kind='linear')
		cOnePet = (cOneInterp(petTime[:,0])+cOneInterp(petTime[:,1]))/2.0

		#Convert AIF to plasma
		rbcToPlasma = 0.814101 + 0.000680*aifTime/60.0 + .103307*(1-np.exp(-aifTime/60.0/50.052431))
		pAif = wbAif / (hct*rbcToPlasma + 1 - hct)

		#Compute first basis function
		bfOne = np.convolve(pAif,np.exp(-b1*aifTime))[0:aifTime.shape[0]]

		#Interpolate first basis function
		aifStep = aifTime[1]-aifTime[0]
		bfOneInterp = interp.interp1d(aifTime,bfOne*aifStep,kind='linear')
		bfOnePet = (bfOneInterp(petTime[:,0])+bfOneInterp(petTime[:,1]))/2.0

		#Compute second basis function
		bfTwo = np.convolve(pAif,1.0-np.exp(-b1*aifTime))[0:aifTime.shape[0]]

		#Interpolate second basis function
		bfTwoInterp = interp.interp1d(aifTime,bfTwo*aifStep,kind='linear')
		bfTwoPet = (bfTwoInterp(petTime[:,0])+bfTwoInterp(petTime[:,1]))/2.0

		#Compute the alphas using linear least squares
		petX = np.stack((bfOnePet,bfTwoPet),axis=1)
		alpha,_,_,_ = np.linalg.lstsq(petX,pet-cOnePet)

		#Calculate model prediction
		petPred = cOnePet + alpha[0]*bfOnePet + alpha[1]*bfTwoPet

		#Return total tissue predictions
		if coefs is False:
				return petPred
		else:
			return petPred,np.array([alpha[0],alpha[1],b1])

	#Return function
	return gluPred

def fdgAifLst(aifTime,pAif,pet,cOne):

	"""

	Parameters
	----------

	aifTime: array
	   A n length array of times to samples AIF at
	pAif: array
	   A n length array containing plasma arterial input function
	pet:
	   A m length array of pet data
	cOne:
	   Concentration in compartment one

	Returns
	-------
	gluPred: function
		A function that will return the four-parameter glucose model prections

	"""

	def gluPred(petTime,b1,coefs=False):


		"""

		Produces a three parameter glucose model fit function for scipy curvefit

		Parameters
		----------

		petTime: array
	   	   An m x 2 array containing frame start and end times
		b1: float
	  	   Expoential parameter
		coefs: logical
			If true, return estimated coefficients. If only returns predictions

		Returns
		-------
		petPred: array
			A n length array of model predictions given parameters
		petCoefs: array
		    A 4x1 array containing the estimated model coefficients

		"""

		#Scale beta parameters
		b1 *= 0.0038

		#Compute first basis function
		bfOne = np.convolve(pAif,np.exp(-b1*aifTime))[0:aifTime.shape[0]]

		#Interpolate first basis function
		aifStep = aifTime[1]-aifTime[0]
		bfOneInterp = interp.interp1d(aifTime,bfOne*aifStep,kind='linear')
		bfOnePet = (bfOneInterp(petTime[:,0])+bfOneInterp(petTime[:,1]))/2.0

		#Compute second basis function
		bfTwo = np.convolve(pAif,1.0-np.exp(-b1*aifTime))[0:aifTime.shape[0]]

		#Interpolate second basis function
		bfTwoInterp = interp.interp1d(aifTime,bfTwo*aifStep,kind='linear')
		bfTwoPet = (bfTwoInterp(petTime[:,0])+bfTwoInterp(petTime[:,1]))/2.0

		#Compute the alphas using linear least squares
		petX = np.stack((bfOnePet,bfTwoPet),axis=1)
		alpha,_,_,_ = np.linalg.lstsq(petX,pet-cOne)

		#Calculate model prediction
		petPred = cOne + alpha[0]*bfOnePet + alpha[1]*bfTwoPet

		#Return total tissue predictions
		if coefs is False:
				return petPred
		else:
			return petPred,np.array([alpha[0],alpha[1],b1])

	#Return function
	return gluPred

def fdgAifLstPen(param,aifTime,pAif,petTime,pet,cOne,pen,penC,coefs=False):

	"""

	Parameters
	----------

	param: array
		1x1 vector of paramters.
	aifTime: array
	   A n length array of times to samples AIF at
	pAif: array
	   A n length array containing plasma arterial input function
	petTime: array
		A m x 2 array of PET start and end times
	pet: array
	   A m length array of pet data
	cOne: array
	   An m length array with concentration in compartment one
	pen: float
 	   The penalty for least squares optimization
	penC: float
	   Value at which penalty = 0
	coefs: logical
		If true, returns predictions and coefficients

	Returns
	-------
	If coefs = False
		SSE: float
			Sum of squares error with penalty
	If coefs = True:
		petPred: array
			An m length array of PET predictions
		coefs: array
			A 3x1 array of coefficients

	"""

	#Scale parameters
	b1 = param[0] * 0.0038

	#Compute first basis function
	bfOne = np.convolve(pAif,np.exp(-b1*aifTime))[0:aifTime.shape[0]]

	#Interpolate first basis function
	aifStep = aifTime[1]-aifTime[0]
	bfOneInterp = interp.interp1d(aifTime,bfOne*aifStep,kind='linear')
	bfOnePet = (bfOneInterp(petTime[:,0])+bfOneInterp(petTime[:,1]))/2.0

	#Compute second basis function
	bfTwo = np.convolve(pAif,1.0-np.exp(-b1*aifTime))[0:aifTime.shape[0]]

	#Interpolate second basis function
	bfTwoInterp = interp.interp1d(aifTime,bfTwo*aifStep,kind='linear')
	bfTwoPet = (bfTwoInterp(petTime[:,0])+bfTwoInterp(petTime[:,1]))/2.0

	#Compute the alphas using linear least squares
	petX = np.stack((bfOnePet,bfTwoPet),axis=1)
	alpha,_,_,_ = np.linalg.lstsq(petX,pet-cOne)

	#Calculate model prediction
	petPred = cOne + alpha[0]*bfOnePet + alpha[1]*bfTwoPet
	petResid = pet-petPred

	#Return sum of squares error
	if coefs is False:
		return np.sum(np.power(petResid,2)) + pen*np.sum(np.power(penC-param[0],2))
	else:
		return petPred,np.array([alpha[0],alpha[1],b1])
