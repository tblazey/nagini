#!/usr/bin/python

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Quick multiview image creator for FSL p-value images')
argParse.add_argument('pImg',help='Positive p-value image',nargs=1,type=str)
argParse.add_argument('nImg',help='Negative p-value image',nargs=1,type=str)
argParse.add_argument('out',help='Name for output image',nargs=1,type=str)
argParse.add_argument('-mVal',help='Value not to show when plotting. Default is 0',default=0.0,type=float)
argParse.add_argument('-pTitle',help='Title for positive colorbar',nargs=1,type=str)
argParse.add_argument('-nTitle',help='Title for negative colorbar',nargs=1,type=str)
argParse.add_argument('-fTitle',help='Title for figure',nargs=1,type=str)
argParse.add_argument('-thr',help='Minimum and maximum for color scale. Default is 0.95 and 0.9999',nargs=2,type=float)
argParse.add_argument('-showMin',help='Show values below minimum.',dest='showMin',action='store_true')
argParse.add_argument('-hideMax',help='Do not show values below maximum',dest='showMax',action='store_false')
argParse.add_argument('-x',help='Slice to plot in x dimension. Default is halfway point. Starts at 0',type=int)
argParse.add_argument('-y',help='Slice to plot in y dimension. Default is halfway point. Starts at 0',type=int)
argParse.add_argument('-z',help='Slice to plot in z dimension. Default is halfway point. Starts at 0',type=int)
argParse.add_argument('-f',help='Frame to plot. Starts at 0. Default is 0',type=int,default=0)
argParse.add_argument('-struct',help='Structural image for underlay',nargs=1,type=str)
argParse.add_argument('-alpha',help='Alpha value for plot, Default is 1',nargs=1,type=float,default=[1.0])
argParse.add_argument('-cSize',help='Font size for colorbar title. Default is 9.0',default=[9.0],nargs=1,type=float)
argParse.add_argument('-white',help='Use white background instead of black',action='store_true')
argParse.add_argument('-crop',help='Crop image based on structural image when taking shots',action='store_true')
argParse.set_defaults(showMin=False,showMax=True)
args = argParse.parse_args()

#Make sure user min is above maximum
if args.thr is not None:
	if args.thr[0] >= args.thr[1]:
		print 'Error: Minimum value of %f is not above maximum value of %f'%(args.thr[0],args.thr[1])
		sys.exit()

#Load in the libraries we will need
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, nibabel as nib, nagini, sys

#Load in image headers
pImg = nagini.loadHeader(args.pImg[0])
nImg = nagini.loadHeader(args.nImg[0])

#Make sure images have same dimensions
if pImg.shape != nImg.shape:
	print 'Error: P-value images do not have the same dimensions.'
	sys.exit()

#Check to make sure image is three or four dimensions
nImgDim = len(pImg.shape)
if nImgDim !=3 and nImgDim !=4:
	print 'Error: P-value images are three or four dimensional'
	sys.exit()

#Get the actual image data
pData = pImg.get_data()
nData = nImg.get_data()

#Setup colors
if args.white is False:
	faceColor = 'black'; textColor = 'white'
else:
	faceColor = 'white'; textColor = 'black'

#If we have a four dimensional image, extract the slice we want.
if nImgDim == 4:

	#Check to see if we can actually plot that time point
	if args.f < 0 or args.f >= pimg.shape[3]:
		print 'Error: Requested frame %i is out of range for image with %i frames. Remember zero based indexing...'%(args.f,pImg.shape[3])
		sys.exit() 
		
	#If we can go ahead and get it
	pData = pData[:,:,:,args.f]
	nData = nData[:,:,:,arg.f]
	
elif args.f != 0:
	print 'Error: Cannot plot selected frame %i because image is only 3D. Remember zero based indexing...'%(args.f)
	sys.exit()

#Check all the dimensions
plotDims = [0,0,0]
for dim,dimIdx in zip([args.x,args.y,args.z],range(3)):

	#If user didn't want a slice, just use midpoint
	if dim is None:
		plotDims[dimIdx] = np.int(pImg.shape[dimIdx] / 2)
	
	#Otherwise use user input if possible
	else:
		if dim <= 0 or dim >= pImg.shape[dimIdx]:
			print 'Error: Requested slice %i is out of range of 0 to %i'%(dim,pImg.shape[dimIdx]-1)
			sys.exit()
		else:
			plotDims[dimIdx] = dim

#Load structural underlay if necessary
if args.struct is not None:

	#Get structural image header
	struct = nagini.loadHeader(args.struct[0])
	
	#Check to make sure it is three or four dimensional
	nStructDim = len(struct.shape)
	if nStructDim != 3 and nStructDim != 4:
		print 'Error: Structural image is not 3 or 4 dimensional...'
		sys.exit()
		
	#Make sure structural image as the same first three dimensions as the image data
	if struct.shape[0:3] != pImg.shape[0:3]:
		print 'Error: Image dimensions do not match between overlay and structural underlay'
		sys.exit()
	
	#Get actual data
	structData = struct.get_data()
	
	#If the structural image is four dimensional, just plot the first
	if nStructDim == 4:
		structData = structData[:,:,:,0]
		print 'Warning: Structural image is four dimensional. Using first frame...'

		#Should we crop?
		if args.crop is True:
	
			#Get indices for non-zero rows, columns, and slices
			xIdx = np.where(np.sum(structData,axis=(1,2))>0)
			yIdx = np.where(np.sum(structData,axis=(0,2))>0)
			zIdx = np.where(np.sum(structData,axis=(0,1))>0)

			#Get cropping indicies
			minIdx = np.zeros(3,dtype=np.int); maxIdx = np.zeros(3,dtype=np.int); idx = 0
			for idxs in [xIdx,yIdx,zIdx]:
				minIdx[idx] = np.max([0,np.min(idxs)-1])
				maxIdx[idx] = np.min([struct.shape[idx],np.max(idxs)+1])
				idx += 1

			#Apply cropping
			pData = pData[minIdx[0]:maxIdx[0],minIdx[1]:maxIdx[1],minIdx[2]:maxIdx[2]]
			nData = nData[minIdx[0]:maxIdx[0],minIdx[1]:maxIdx[1],minIdx[2]:maxIdx[2]]
			structData = structData[minIdx[0]:maxIdx[0],minIdx[1]:maxIdx[1],minIdx[2]:maxIdx[2]]

			#Update slices
			plotDims[0] -= minIdx[0]
			plotDims[1] -= minIdx[1]
			plotDims[2] -= minIdx[2]

	#Get thresholds for structural image
	structThr = np.percentile(structData[structData!=0],[2,98])

	#Get colormap for structural image
	sMap = plt.get_cmap('gray')
	sMap.set_bad(faceColor); sMap.set_under(faceColor); sMap.set_over(sMap(255))
		
#Mask images
pMasked = np.ma.masked_where(pData==args.mVal,pData)
nMasked = np.ma.masked_where(nData==args.mVal,nData)

#Get insensity threshold limits if user doesn't set them.
if args.thr is  None:
	args.thr = [0.95,.9999]

#Get colormaps
pMap = plt.get_cmap('autumn')
nMap =  mpl.colors.ListedColormap(np.flipud(pMap(np.arange(256))[:,0:3]) * -1 + 1)
nDisplay = mpl.colors.ListedColormap(pMap(np.arange(256))[:,0:3] * -1 + 1)

#Set color map values
for cMap in [pMap,nMap]:

	#Don't show masked values 
	cMap.set_bad(faceColor,alpha=0)

	#Decide whether or not to show values below minimum
	if args.showMin is False:
		cMap.set_under(faceColor,alpha=0)
	else:
		cMap.set_under(cMap(0))
	
	#Do the same thing for above maximum
	if args.showMax is False:
		cMap.set_over(faceColor,alpha=0)
	else:
		cMap.set_over(cMap(255))

#Make figure
fig = plt.figure(facecolor=faceColor,figsize=(10,5))

#Add a title to the figure
if args.fTitle is not None:
	plt.suptitle(args.fTitle[0],color=textColor,x=0.44,y=0.72,size=14,weight='bold')

#Make the grid for the plotting
gs = mpl.gridspec.GridSpec(1, 5,width_ratios=[0.33,0.33,0.33,0.01,0])

#Figure of the x and y axis limits
axisLimits = np.max(pMasked.shape)

#Add sagittal view
axOne = plt.subplot(gs[0])  
if args.struct is not None: 
	 plt.imshow(structData[plotDims[0],:,:].T,cmap=sMap,vmin=structThr[0],vmax=structThr[1])
pOne = plt.imshow(pMasked[plotDims[0],:,:].T,cmap=pMap,vmin=args.thr[0],vmax=args.thr[1],alpha=args.alpha[0])
nOne = plt.imshow(nMasked[plotDims[0],:,:].T,cmap=nMap,vmin=args.thr[0],vmax=args.thr[1],alpha=args.alpha[0])
plt.xlim([0,axisLimits]); plt.ylim([0,axisLimits]); plt.axis('off')
axOneP = axOne.get_position()
axOne.set_position((0.115,0.05,axOneP.width,axOneP.height))

#Add horiziontal view
axTwo = plt.subplot(gs[1])
if args.struct is not None: 
	 plt.imshow(np.rot90(structData[:,:,plotDims[2]],3),cmap=sMap,vmin=structThr[0],vmax=structThr[1])
pTwo = plt.imshow(np.rot90(pMasked[:,:,plotDims[2]],3),cmap=pMap,vmin=args.thr[0],vmax=args.thr[1],alpha=args.alpha[0])
nTwo = plt.imshow(np.rot90(nMasked[:,:,plotDims[2]],3),cmap=nMap,vmin=args.thr[0],vmax=args.thr[1],alpha=args.alpha[0])
plt.xlim([0,axisLimits]); plt.ylim([0,axisLimits]); plt.axis('off')
axTwoP = axTwo.get_position()
axTwo.set_position((0.3725,0.025,axTwoP.width,axTwoP.height))

#Add coronal view
axThree = plt.subplot(gs[2])
if args.struct is not None: 
	 plt.imshow(np.rot90(structData[:,plotDims[1],:],3),cmap=sMap,vmin=structThr[0],vmax=structThr[1])
pThree = plt.imshow(np.rot90(pMasked[:,plotDims[1],:],3),cmap=pMap,vmin=args.thr[0],vmax=args.thr[1],alpha=args.alpha[0])
nThree = plt.imshow(np.rot90(nMasked[:,plotDims[1],:],3),cmap=nMap,vmin=args.thr[0],vmax=args.thr[1],alpha=args.alpha[0])
plt.xlim([0,axisLimits]); plt.ylim([0,axisLimits]); plt.axis('off')
axThreeP = axThree.get_position()
axThree.set_position((0.575,0.05,axThreeP.width,axThreeP.height))

#Add in positive colorbar
padLength = np.max([len(str(args.thr[0])),len(str(args.thr[1]))])*0.75
axFour = plt.subplot(gs[3])
pBar = mpl.colorbar.ColorbarBase(axFour,cmap=pMap,ticks=[0,1])
if args.pTitle is not None:
	pBar.set_label(args.pTitle[0],color=textColor,rotation=360,size=args.cSize[0],weight="bold",labelpad=padLength)
pBar.ax.set_position((0.775, 0.475, 0.0225, 0.2))
pBar.set_ticklabels([1-args.thr[0],1-args.thr[1]])
for tick in pBar.ax.yaxis.get_major_ticks():
    tick.label2.set_color(textColor)
    tick.label2.set_weight('bold')
    tick.label2.set_size(10)

#Add in negative colorbar
axFive = plt.subplot(gs[4])
nBar = mpl.colorbar.ColorbarBase(axFive,cmap=nDisplay,ticks=[0,1])
if args.nTitle is not None:
	nBar.set_label(args.nTitle[0],color=textColor,rotation=360,size=args.cSize[0],weight="bold",labelpad=padLength)
nBar.ax.set_position((0.775, 0.2, 0.0225, 0.2))
nBar.set_ticklabels([1-args.thr[1],1-args.thr[0]])
for tick in nBar.ax.yaxis.get_major_ticks():
    tick.label2.set_color(textColor)
    tick.label2.set_weight('bold')
    tick.label2.set_size(10)

#Write out figure
plt.savefig(args.out[0],transparent=True,faceColor=faceColor,bbox_inches='tight')

#Close all figures
plt.close('all')

