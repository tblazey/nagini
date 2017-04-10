# nagini
A set of Python scripts for processing quantitative PET data.

Currently there are the following scripts:

	bayesCbf.py - Fits blood flow model to PET data using bayesian techniques. Super beta version.
	cbfAif.py - Fits a four-parameter flow model to PET data using an arterial-sampled input function.
	cbfIdaif.py - Fits a two-parameter flow model to PET data using an image-derived input function.
	cbfIdaifOhta.py - Fits a three-parameter flow model to PET data using an image-derived input function.
	cbfIdaifPoly.py - Uses polynomial approximation to get CBF from PET data and an image-derived input function
	cbvAif.py - Creates quantiative CBV images using an arterial-sampled input function.
	cbvIdaif.py - Creates quantiative CBV images using an image-derived input function.
	cropImage.py - Use a mask to crop image.
	extractVar.py - Extract a variable form a Matlab .mat file.
	gluAif.py - Fits four parameter C11 glucose model.
	gluIdaif.py - Fits a three-paramter FDG model to estimate the cerebral metabolic rate of glucose using an image-derived input function.
	gluIdaifFour.py - Fits a four-paramter FDG model to estimate the cerebral metabolic rate of glucose using an image-derived input function.
	imageCorr.py - Performs a spatial correlation between two images.
	imgPca.py - Performs a PCA on two images
	nagini.py - Module with various image processing functions.
	oxyIdaif.py - Calculates OEF and cerebral metabolic rate of oxygen using the original Mintun et al, 1984 model.
	oxyIdaifLin.py - Uses linear approximation of O15 model.
	oxyIdaifOhta.py - Fits a three-parameter oxygen metabolism model to PET data using an image-derived input function.
	oxyIdaifPoly.py - Uses polynomial approximation to fit 015 oxygen model.
	plotDiffImage.py - 3-view plot of volume image using FSL-style red/orange and blue/cyan color scale.
	plotImage.py - 3-view plot of volume image using user specified color scale.
	plotSigImage.py - 3-view plot of two volume significance images using FSL-style red/orange and blue/cyan color scale.
	projectRois.py - Takes a set of ROI values and puts them onto an image
	pvcCalc.py - Calcultes RSF and RBV partial volume corrlection.
	sampleRois.py - Calculates ROI averages
	splineSuvr.py - Calculate a SUVR after smoothing out PET timeseries with cubic splines.
	srtmTwo.py - Calculates a simplified reference tissue model (SRTM2)	
