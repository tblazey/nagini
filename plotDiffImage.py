#!/usr/bin/python

#####################
###Parse Arguments###
#####################

import argparse, sys
argParse = argparse.ArgumentParser(description='Quick multiview image creator for postive/negative images')
argParse.add_argument('img',help='Difference image',nargs=1,type=str)
argParse.add_argument('out',help='Name for output image',nargs=1,type=str)
argParse.add_argument('-mVal',help='Value not to show when plotting. Default is 0',default=0.0,type=float)
argParse.add_argument('-pTitle',help='Title for positive colorbar',nargs=1,type=str)
argParse.add_argument('-nTitle',help='Title for negative colorbar',nargs=1,type=str)
argParse.add_argument('-fTitle',help='Title for figure',nargs=1,type=str)
argParse.add_argument('-thr',help="Minimum and maximum for color scale. Default is 2 per and 98 per of absolute values.",nargs=2,type=float)
argParse.add_argument('-sThr',help='Minimum and maximum for structural image. Default is 2nd and 98th percentile',nargs=2,type=float)
argParse.add_argument('-showMin',help='Show values below minimum.',dest='showMin',action='store_true')
argParse.add_argument('-hideMax',help='Do not show values below maximum',dest='showMax',action='store_false')
argParse.add_argument('-x',help='Slice to plot in x dimension. Default is halfway point. Starts at 0',type=int)
argParse.add_argument('-y',help='Slice to plot in y dimension. Default is halfway point. Starts at 0',type=int)
argParse.add_argument('-z',help='Slice to plot in z dimension. Default is halfway point. Starts at 0',type=int)
argParse.add_argument('-f',help='Frame to plot. Starts at 0. Default is 0',type=int,default=0)
argParse.add_argument('-struct',help='Structural image for underlay',nargs=1,type=str)
argParse.add_argument('-alpha',help='Alpha value for plot, Default is 1',nargs=1,type=float,default=[1.0])
argParse.add_argument('-scale',help='Scale difference image by specified amount',nargs=1,type=float)
argParse.add_argument('-cSize',help='Font size for colorbar title. Default is 10.0',default=[10.0],nargs=1,type=float)
argParse.add_argument('-white',help='Use white background instead of black',action='store_true')
argParse.set_defaults(showMin=False,showMax=True)
args = argParse.parse_args()

#Make sure user min is above maximum
for thr in [args.thr,args.sThr]:
    if thr is not None:
        if thr[0] >= thr[1]:
            print 'Error: Minimum value of %f is not above maximum value of %f'%(thr[0],thr[1])
            sys.exit()

#Make sure user min is above maximum
if args.thr is not None:
	if args.thr[0] >= args.thr[1]:
		print 'Error: Minimum value of %f is not above maximum value of %f'%(args.thr[0],args.thr[1])
		sys.exit()

#Load in the libraries we will need
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, nibabel as nib, nagini, sys

#Quick function for cropping image
def crop_img(img,ref=None):
    
    #Find non-zero coordinates
    if ref is None:
        coords = np.argwhere(img!=0.0)
    else:
        coords = np.argwhere(ref!=0.0)

    #Get bounding box
    if coords.shape[0] > 0:
        mins = np.fmax(np.min(coords,axis=0) - 2, 0)
        maxs = np.fmin(np.max(coords,axis=0) + 2, img.shape)
    else:
        mins = np.array([0,0])
        maxs = img.shape

    #Return cropped image
    return img[mins[0]:maxs[0],mins[1]:maxs[1]]

#Load in image headers
img = nagini.loadHeader(args.img[0])

#Check to make sure image is three or four dimensions
nImgDim = len(img.shape)
if nImgDim !=3 and nImgDim !=4:
	print 'Error: Image is not three or four dimensional'
	sys.exit()

#Setup colors
if args.white is False:
	faceColor = 'black'; textColor = 'white'
else:
	faceColor = 'white'; textColor = 'black'

#Get the actual image data
imgData =img.get_data()

#Scale the data if necessary
if args.scale is not None:
	imgData = imgData * args.scale[0]

#If we have a four dimensional image, extract the volume we want.
if nImgDim == 4:

	#Check to see if we can actually plot that time point
	if args.f < 0 or args.f >= img.shape[3]:
		print 'Error: Requested frame %i is out of range for image with %i frames. Remember zero based indexing...'%(args.f,img.shape[3])
		sys.exit() 
	
	#Get frame we need	
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
	structThr = np.percentile(structData[structData!=0],[0,98])

	#Get colormap for structural image
	sMap = plt.get_cmap('gray')
	sMap.set_bad(faceColor); sMap.set_under(faceColor); sMap.set_over(sMap(255))

#Get insensity threshold limits if user doesn't set them.
if args.thr is  None:
	args.thr = np.percentile(np.abs(imgData[imgData!=0]),[2,98])

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
fig = plt.figure(facecolor=faceColor,figsize=(15,7.5),frameon=False)

#Add a title to the figure
if args.fTitle is not None:
	plt.suptitle(args.fTitle[0],color=textColor,x=0.35,y=0.33,size=12,weight='bold')

#Get data for different views
sag = imgData[plotDims[0],:,:].T
hor = np.rot90(imgData[:,:,plotDims[2]],3)
cor = np.rot90(imgData[:,plotDims[1],:],3)

#Get consistient cropping dimensions
if args.struct is not None:

    #Crop structural images
    struct_sag = crop_img(structData[plotDims[0],:,:].T)
    struct_hor = crop_img(np.rot90(structData[:,:,plotDims[2]],3))
    struct_cor = crop_img(np.rot90(structData[:,plotDims[1],:],3))

    #Crop images
    sag_crop = crop_img(sag,ref=structData[plotDims[0],:,:].T)
    hor_crop = crop_img(hor,ref=np.rot90(structData[:,:,plotDims[2]],3))
    cor_crop = crop_img(cor,ref=np.rot90(structData[:,plotDims[1],:],3))

    #Get dims for paddings
    xDims = np.array([struct_sag.shape[0],struct_hor.shape[0],struct_cor.shape[0]])
    yDims = np.array([struct_sag.shape[1],struct_hor.shape[1],struct_cor.shape[1]])
 
else:

    #Get cropped version of images
    sag_crop = crop_img(sag)
    hor_crop = crop_img(hor)
    cor_crop = crop_img(cor)

    #Get dims for paddings
    xDims = np.array([sag_crop.shape[0],hor_crop.shape[0],cor_crop.shape[0]])
    yDims = np.array([sag_crop.shape[1],hor_crop.shape[1],cor_crop.shape[1]])

#Get paddding dims
axisLimits = np.array([np.max(xDims),np.max(yDims)])
sag_x_pad = (axisLimits[0]-xDims[0]) / 2.0
sag_y_pad = (axisLimits[1]-yDims[0]) / 2.0
hor_x_pad = (axisLimits[0]-xDims[1]) / 2.0
hor_y_pad = (axisLimits[1]-yDims[1]) / 2.0
cor_x_pad = (axisLimits[0]-xDims[2]) / 2.0
cor_y_pad = (axisLimits[1]-yDims[2]) / 2.0

#Make padding arrays
sag_pad = np.array([(np.ceil(sag_x_pad),np.floor(sag_x_pad)),
                    (np.ceil(sag_y_pad),np.floor(sag_y_pad))],dtype=np.int)
hor_pad = np.array([(np.ceil(hor_x_pad),np.floor(hor_x_pad)),
                    (np.ceil(hor_y_pad),np.floor(hor_y_pad))],dtype=np.int)
cor_pad = np.array([(np.ceil(cor_x_pad),np.floor(cor_x_pad)),
                    (np.ceil(cor_y_pad),np.floor(cor_y_pad))],dtype=np.int)

#Zero pad so that all cropped images have same dimensions
sag_crop = np.pad(sag_crop,sag_pad,'constant',constant_values=0)
hor_crop = np.pad(hor_crop,hor_pad,'constant',constant_values=0)
cor_crop = np.pad(cor_crop,cor_pad,'constant',constant_values=0)

#Make masked arrays
sag_crop = np.ma.array(sag_crop, mask=sag_crop==0)
hor_crop = np.ma.array(hor_crop, mask=hor_crop==0)
cor_crop = np.ma.array(cor_crop, mask=cor_crop==0)

#Do the same thing for structural underlay if necessary
if args.struct is not None:

    #Pad structural images so that all dimensions are the same
    struct_sag = np.pad(struct_sag,sag_pad,'constant',constant_values=0)
    struct_hor = np.pad(struct_hor,hor_pad,'constant',constant_values=0)
    struct_cor = np.pad(struct_cor,cor_pad,'constant',constant_values=0)

    #Make masked versions
    struct_sag = np.ma.array(struct_sag, mask=struct_sag==0)
    struct_hor = np.ma.array(struct_hor, mask=struct_hor==0)
    struct_cor = np.ma.array(struct_cor, mask=struct_cor==0)

#Make the grid for the plotting
gs = mpl.gridspec.GridSpec(1, 5,height_ratios=[1.0,1.0,1.0,0.1,0.1],width_ratios=[1.0,1.0,1.0,0.2,0])

#Add sagittal view
axOne = plt.subplot(gs[0])  
if args.struct is not None: 
	 plt.imshow(struct_sag,cmap=sMap,vmin=structThr[0],vmax=structThr[1])
pOne = plt.imshow(sag_crop,cmap=pMap,vmin=args.thr[0],vmax=args.thr[1],alpha=args.alpha[0])
nOne = plt.imshow(sag_crop*-1.0,cmap=nMap,vmin=args.thr[0],vmax=args.thr[1],alpha=args.alpha[0])
plt.xlim([0,axisLimits[0]+2]); plt.ylim([0,axisLimits[1]]); plt.axis('off')
axOneP = axOne.get_position()
axOne.set_position((0.1,0.04,axOneP.width,axOneP.height))

#Add horiziontal view
axTwo = plt.subplot(gs[1])
if args.struct is not None: 
	 plt.imshow(struct_hor,cmap=sMap,vmin=structThr[0],vmax=structThr[1])
pTwo = plt.imshow(hor_crop,cmap=pMap,vmin=args.thr[0],vmax=args.thr[1],alpha=args.alpha[0])
nTwo = plt.imshow(hor_crop*-1.0,cmap=nMap,vmin=args.thr[0],vmax=args.thr[1],alpha=args.alpha[0])
plt.xlim([0,axisLimits[0]]); plt.ylim([0,axisLimits[1]]); plt.axis('off')
axTwoP = axTwo.get_position()
axTwo.set_position((0.245,0.025,axTwoP.width,axTwoP.height))

#Add coronal view
axThree = plt.subplot(gs[2])
if args.struct is not None: 
	 plt.imshow(struct_cor,cmap=sMap,vmin=structThr[0],vmax=structThr[1])
pThree = plt.imshow(cor_crop,cmap=pMap,vmin=args.thr[0],vmax=args.thr[1],alpha=args.alpha[0])
nThree = plt.imshow(cor_crop*-1.0,cmap=nMap,vmin=args.thr[0],vmax=args.thr[1],alpha=args.alpha[0])
plt.xlim([0,axisLimits[0]]); plt.ylim([0,axisLimits[1]]); plt.axis('off')
axThreeP = axThree.get_position()
axThree.set_position((0.38,0.05,axThreeP.width,axThreeP.height))

#Add in positive colorbar
axFour = plt.subplot(gs[3])
pBar = mpl.colorbar.ColorbarBase(axFour,cmap=pMap,ticks=[0,0.5,1])
if args.pTitle is not None:
	padLength = len(str(args.pTitle[0]))*4.0
	pBar.set_label(args.pTitle[0],color=textColor,rotation=360,size=args.cSize[0],weight="bold",labelpad=padLength)
pBar.ax.set_position((0.55, 0.2, 0.0125, 0.14))
midTick = (args.thr[1] - args.thr[0]) / 2.0 + args.thr[0]
pBar.set_ticklabels([args.thr[0],midTick,args.thr[1]])
for tick in pBar.ax.yaxis.get_major_ticks():
    tick.label2.set_color(textColor)
    tick.label2.set_weight('bold')
    tick.label2.set_size(9)

#Add in negative colorbar
axFive = plt.subplot(gs[4])
nBar = mpl.colorbar.ColorbarBase(axFive,cmap=nDisplay,ticks=[0,0.5,1])
if args.nTitle is not None:
	padLength = len(str(args.nTitle[0]))*4.0
	nBar.set_label(args.nTitle[0],color=textColor,rotation=360,size=args.cSize[0],weight="bold",labelpad=padLength)
nBar.ax.set_position((0.55, 0.0, 0.0125, 0.14))
nBar.set_ticklabels([-args.thr[1],-midTick,-args.thr[0]])
for tick in nBar.ax.yaxis.get_major_ticks():
    tick.label2.set_color(textColor)
    tick.label2.set_weight('bold')
    tick.label2.set_size(9)

#Write out figure
plt.savefig(args.out[0],transparent=True,facecolor=faceColor,bbox_inches='tight')

#Close all figures
plt.close('all')

