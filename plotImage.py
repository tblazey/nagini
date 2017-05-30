#!/usr/bin/python

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Quick multiview image creator')
argParse.add_argument('img',help='Image to plot',nargs=1,type=str)
argParse.add_argument('out',help='Name for output image',nargs=1,type=str)
argParse.add_argument('-cmap',help='Matplotlib colormap for ploting. Default is plasma',default='plasma',type=str)
argParse.add_argument('-mVal',help='Value not to show when plotting. Default is 0',default=0.0,type=float)
argParse.add_argument('-cTitle',help='Title for colorbar',nargs=1,type=str)
argParse.add_argument('-pTitle',help='Title for plot',nargs=1,type=str)
argParse.add_argument('-thr',help='Minimum and maximum for color scale. Default is 2nd and 98th percentile',nargs=2,type=float)
argParse.add_argument('-showMin',help='Show values below minimum.',dest='showMin',action='store_true')
argParse.add_argument('-hideMax',help='Do not show values above maximum',dest='showMax',action='store_false')
argParse.add_argument('-x',help='Slice to plot in x dimension. Default is halfway point. Starts at 0',type=int)
argParse.add_argument('-y',help='Slice to plot in y dimension. Default is halfway point. Starts at 0',type=int)
argParse.add_argument('-z',help='Slice to plot in z dimension. Default is halfway point. Starts at 0',type=int)
argParse.add_argument('-f',help='Frame to plot. Starts at 0. Default is 0',type=int,default=0)
argParse.add_argument('-struct',help='Structural image for underlay',nargs=1,type=str)
argParse.add_argument('-alpha',help='Alpha value for plot, Default is 1',nargs=1,type=float,default=[1.0])
argParse.add_argument('-scale',help='Scale image by specified amount',nargs=1,type=float)
argParse.add_argument('-sci',help='Use scientific notation for colorbar labels',dest='useSci',action='store_true')
argParse.add_argument('-custom',help='Custom color map RGB array. Overides -cmap',nargs='+',type=float) 
argParse.add_argument('-cSize',help='Font size for colorbar title. Default is 9.0',default=[9.0],nargs=1,type=float)
argParse.set_defaults(showMin=False,showMax=True,useSci=False)
args = argParse.parse_args()

#Make sure user min is above maximum
if args.thr is not None:
	if args.thr[0] >= args.thr[1]:
		print 'Error: Minimum value of %f is not above maximum value of %f'%(args.thr[0],args.thr[1])
		sys.exit()

import matplotlib
matplotlib.use('TkAgg')

#Load in the libraries we will need
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, nibabel as nib, nagini, sys
from matplotlib.colors import LinearSegmentedColormap


#Change default math font
mpl.rcParams['mathtext.default'] = 'regular'

#Load in image header
img = nagini.loadHeader(args.img[0])

#Check to make sure image is three or four dimensions
nImgDim = len(img.shape)
if nImgDim !=3 and nImgDim !=4:
	print 'Error: Plot image is not three or four dimensional'
	sys.exit()

#Get the actual image data
imgData = img.get_data()

#If we have a four dimensional image, extract the slice we want.
if nImgDim == 4:

	#Check to see if we can actually plot that time point
	if args.f < 0 or args.f >= imgData.shape[3]:
		print 'Error: Requested frame %i is out of range for image with %i frames. Remember zero based indexing...'%(args.f,imgData.shape[3])
		sys.exit() 
		
	#If we can go ahead and get it
	imgData = imgData[:,:,:,args.f]
elif args.f != 0:

	print 'Error: Cannot plot selected frame %i because image is only 3D. Remember zero based indexing...'%(args.f)
	sys.exit()

#Check all the dimensions
plotDims = [0,0,0]
for dim,dimIdx in zip([args.x,args.y,args.z],range(3)):

	#If user didn't want a slice, just use midpoint
	if dim is None:
		plotDims[dimIdx] = np.int(img.shape[dimIdx] / 2)
	
	#Otherwise use user input if possible
	else:
		if dim <= 0 or dim >= img.shape[dimIdx]:
			print 'Error: Requested slice %i is out of range of 0 to %i'%(dim,img.shape[dimIdx]-1)
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
	if struct.shape[0:3] != img.shape[0:3]:
		print 'Error: Image dimensions do not match between overlay and structural underlay'
		sys.exit()
	
	#Get actual data
	structData = struct.get_data()
	
	#If the structural image is four dimensional, just plot the first
	if nStructDim == 4:
		structData = structData[:,:,:,0]
		print 'Warning: Structural image is four dimensional. Using first frame...'

	#Get thresholds for structural image
	structThr = np.percentile(structData[structData!=0],[2,98])
		
#Mask image
maskedData = np.ma.masked_where(imgData==args.mVal,imgData)


#Scale the image if necessary
if args.scale is not None:
	maskedData = maskedData * args.scale[0]

#Get insensity threshold limits if user doesn't set them.
if args.thr is  None:
	args.thr = np.percentile(maskedData[maskedData!=0],[2,98])

#Get colormap and don't show masked values
if args.custom is None:
	cMap = plt.get_cmap(args.cmap)
else:
	rgbArray = np.array(args.custom)
	rgbArray = rgbArray.reshape((rgbArray.shape[0]/3,3))
	cMap = LinearSegmentedColormap.from_list('map',rgbArray,N=100)
cMap.set_bad('black',alpha=0)

#Decide whether or not to show values below minimum
if args.showMin is False:
	cMap.set_under('black',alpha=0)
else:
	cMap.set_under(cMap(0))
	
#Do the same thing for above maximum
if args.showMax is False:
	cMap.set_over('black',alpha=0)
else:
	cMap.set_over(cMap(255))

#Make figure
fig = plt.figure(facecolor='black',figsize=(10,5),frameon=False)

#Add a title to the figure
if args.pTitle is not None:
	plt.suptitle(args.pTitle[0],color='white',x=0.44,y=0.75,size=14,weight='bold')

#Make the grid for the plotting
gs = mpl.gridspec.GridSpec(1, 4,width_ratios=[0.33,0.33,0.33,0.01])

#Figure of the x and y axis limits
axisLimits = np.max(img.shape)

#Add sagittal view
axOne = plt.subplot(gs[0])  
if args.struct is not None: 
	 plt.imshow(structData[plotDims[0],:,:].T,cmap="gray",vmin=structThr[0],vmax=structThr[1])
imOne = plt.imshow(maskedData[plotDims[0],:,:].T,cmap=cMap,vmin=args.thr[0],vmax=args.thr[1],alpha=args.alpha[0])
plt.xlim([0,axisLimits]); plt.ylim([0,axisLimits]); plt.axis('off')
axOneP = axOne.get_position()
axOne.set_position((0.115,0.10,axOneP.width,axOneP.height))

#Add horiziontal view
axTwo = plt.subplot(gs[1])
if args.struct is not None: 
	 plt.imshow(np.rot90(structData[:,:,plotDims[2]],3),cmap="gray",vmin=structThr[0],vmax=structThr[1])
imTwo = plt.imshow(np.rot90(maskedData[:,:,plotDims[2]],3),cmap=cMap,vmin=args.thr[0],vmax=args.thr[1],alpha=args.alpha[0])
plt.xlim([0,axisLimits]); plt.ylim([0,axisLimits]); plt.axis('off')
axTwoP = axTwo.get_position()
axTwo.set_position((0.35,0.05,axTwoP.width,axTwoP.height))

#Add coronal view
axThree = plt.subplot(gs[2])
if args.struct is not None: 
	 plt.imshow(np.rot90(structData[:,plotDims[1],:],3),cmap="gray",vmin=structThr[0],vmax=structThr[1])
imThree = plt.imshow(np.rot90(maskedData[:,plotDims[1],:],3),cmap=cMap,vmin=args.thr[0],vmax=args.thr[1],alpha=args.alpha[0])
plt.xlim([0,axisLimits]); plt.ylim([0,axisLimits]); plt.axis('off')
axThreeP = axThree.get_position()
axThree.set_position((0.56,0.09,axThreeP.width,axThreeP.height))

#Add in colorbar
axFour = plt.subplot(gs[3])
cBar = mpl.colorbar.ColorbarBase(axFour,cmap=cMap,ticks=[0,1])
if args.cTitle is not None:
	cBar.set_label(r'$%s$'%(args.cTitle[0]),color='white',rotation='90',size=args.cSize[0],weight="bold",labelpad=-20)
cBar.ax.set_position((0.775, 0.29, 0.025, 0.35))
if args.useSci is True:
	cBar.set_ticklabels(['%.2e'%(args.thr[0]),'%.1e'%(args.thr[1])])
else:
	cBar.set_ticklabels([np.round(args.thr[0],2),np.round(args.thr[1],2)])
for tick in cBar.ax.yaxis.get_major_ticks():
    tick.label2.set_color('white')
    tick.label2.set_weight('bold')
    tick.label2.set_size(10)

#Write out figure
plt.savefig(args.out[0],transparent=True,facecolor='black',bbox_inches='tight')

#Close all figures
plt.close('all')

