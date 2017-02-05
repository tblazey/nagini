# nagini
A set of Python scripts for processing quantitative PET data.<br />

Currently there are the following scripts:<br />
	
	cbfIdaif.py - Fits a four-parameter flow model to PET data using an arterial-sampled input function.
	cbfIdaif.py - Fits a two-parameter flow model to PET data using an image-derived input function.
	cbfIdaifOhta.py - Fits a three-parameter flow model to PET data using an image-derived input function.
	cbvIdaif.py - Creates quantiative CBV images using an image-derived input function.
	cropImage.py - Use a mask to crop image.
	extractVar.py - Extract a variable form a Matlab .mat file.
	gluIdaif.py - Fits a three-paramter FDG model to estimate the cerebral metabolic rate of glucose using an image-derived input function.
	gluIdaif.py - Fits a four-paramter FDG model to estimate the cerebral metabolic rate of glucose using an image-derived input function.
	imageCorr.py - Calculate a correlation between two image.
	nagini.py - Module with various image processing functions.
	oxyIdaif.py - Calculates OEF and cerebral metabolic rate of oxygen using the orginal Mintun et al, 1984 model.
	oxyIdaifOhta.py - Fits a three-parameter oxygen metabolism model to PET data using an image-derived input function.
	plotDiffImage.py - 3-view plot of volume image using FSL-style red/orange and blue/cyan color scale.
	plotImage.py - 3-view plot of volume image using user specified color scale.
	plotSigImage.py - 3-view plot of two volume significance images using FSL-style red/orange and blue/cyan color scale.
	pvcCalc.py - Calcultes RSF and RBV partial volume corrlection.
	splineSuvr.py - Calculate a SUVR after smoothing out PET timeseries with cubic splines.
	
	

