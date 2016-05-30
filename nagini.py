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
flowThreeIdaif: Prodcues model fit for three-parameter water model. For use with scipy.optimize.curvefit
gluThreeIdaif: Produces model fit for three-paramter fdg model. For use with scipy.optimize.curvefit
oefCalcIdaif: Calculates OEF using the stanard Mintun model. 

"""

#What libraries do we need
import numpy as np, nibabel as nib, sys

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
	outData = np.zeros_like(mask)
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
	   A [2,n] numpy array where the first column is time and the second the input function
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
	   A [2,n] numpy array where the first column is time and the second the input function
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
	   A [2,n] numpy array where the first column is time and the second the input function
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


