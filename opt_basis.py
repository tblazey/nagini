#!/usr/bin/python
"""
opt_basis.py:

Creates a syntheic shift map using basis functions

Tyler Blazey (blazey@wustl.edu)
"""

#Load libraries
import argparse
import numpy as np
import nibabel as nib
import scipy.ndimage.interpolation as interp
import scipy.optimize as opt

def parse_args():
    """
    Argument parser

    Returns
    -------
    args: argparse object
        User arguments
    """

    #Define parser
    parser = argparse.ArgumentParser(description='EPI distortion correction with '
                                                 'shift map basis functions',
                                     epilog='Note: All images must be in register')
    parser.add_argument('epi', type=str, nargs=1, help='3d EPI image to correct')
    parser.add_argument('struct', type=str, nargs=1, help='3d structural image')
    parser.add_argument('mean', type=str, nargs=1, help='3d mean shift map image')
    parser.add_argument('basis', type=str, nargs=1, help='4d image of shift map basis functions')
    parser.add_argument('mask', type=str, nargs=1, help='3d mask image')
    parser.add_argument('out', type=str, nargs=1, help='Trailer for output file')
    parser.add_argument('-n', type=int, default=[5], nargs=1,
                        help='Number of basis functions to use. Default is 5')
    parser.add_argument('-grid', default=[12, 5], type=float, nargs='+', metavar='dx',
                        help='Uniform grid spacing for optimization (in mm). Default is 12 and 5')
    parser.add_argument('-dir', default=['-y'], type=str, nargs=1,
                        choices=['-x', 'x', '-y', 'y', '-z', 'z'],
                        help='Define phase encoding direction. Default is -y')
    parser.add_argument('-pen', default=[None], type=float, nargs=1, metavar='penalty',
                        help='Add a simple ridge penalty to cost function. Default is none')
    parser.add_argument('-basin', default=[0], const=[1], action='store_const',
                        help='Run a global optimization using scipy.basinhopping')
    args = parser.parse_args()

    #Make sure number of basis functions is set correctly
    if args.n[0] < 1:
        raise argparse.ArgumentTypeError('Number of basis functions must be at least 1')

    #Make sure grid size is set correctly
    for i in range(len(args.grid)):
        if args.grid[i] < 0.25 or args.grid[i] > 25:
            raise argparse.ArgumentTypeError('All grid sizes must be between 0.25 and 25 mm')

    #Get sign of phase encoding direction
    if args.dir[0][0] == '-':
        args.sign = [-1.0]
    else:
        args.sign = [1.0]

    #Get phase encoding axis
    if args.dir[0][1] == 'x':
        args.axis = [0]
    elif args.dir[0][1] == 'y':
        args.axis = [1]
    else:
        args.axis = [2]

    return args

def load_hdr(img_path):
    """
    Loads in image header

    Parameters
    ----------
    img_path: string
        Path to image location

    Returns
    -------
    hdr: SpatialImage
        Nibabel image header object
    """

    try:
        hdr = nib.load(img_path)
    except:
        raise IOError('Cannot load image at %s'%(img_path))
    return hdr

def make_grid(grid_lens, grid_steps):
    """
    Constructs a 3d grid for interpolation

    Parameters
    ----------
    grid_lens: (3,) array
        Length of grid in each dimension
    grid_steps: (3,) array
        Step size for grid in each dimesion

    Returns
    -------
    grid: list
        Each list element contains the grid coordinates for that dimension
    grid_dims: (3,) array
        Dimensions of grid
    """

    #Get coodinates in each dimesion
    x_coord = np.arange(0, grid_lens[0], grid_steps[0])
    y_coord = np.arange(0, grid_lens[1], grid_steps[1])
    z_coord = np.arange(0, grid_lens[2], grid_steps[2])

    #Construct grid
    x_grid, y_grid, z_grid = np.meshgrid(x_coord, y_coord, z_coord, indexing='ij')

    return [x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], x_grid.shape

def calc_shift(mean, basis, basis_coef):
    """
    Computes a synthetic shift map

    Parameters
    ----------
    mean: (n, ) array
        Mean shift map image
    basis: (n, p) array
        Shift map basis functions
    basis_coef:(p, ) array
        Coefficients for basis functions

    Returns
    -------
    shift: (n,) array
        Synethic shift map computed as mean + basis weighted by basis_coef
    """

    return mean + basis.dot(basis_coef)

def apply_shift(img_data, coords, shift_data, axis=1, sign=1):
    """
    Interpolates image along phase encoding direction

    Parameters
    ----------
    img_data: (x, y, z) array
        Image to interpolate
    coords: (3, p) list
        Coordinates to sample img_data at
    shift_data: (p, ) array
        Shifts to apply to coodinates in coords
    axis: int
        Dimension to apply shift_data to. Must be 1, 2, or 3
    sign: int
        Direction to apply shift_data to. Must to 1 or -1.

    Returns
    -------
    img_interp: (p,) array
        Interpolated img_data
    """

    #Apply shift to coordinates
    shift_coords = np.float32(coords)
    shift_coords[axis] += shift_data * sign

    #Interolate image at shifted coordinates
    return interp.map_coordinates(img_data, shift_coords, order=1)

def calc_grad(img, img_mask, grid_size):
    """
    #Computes spatial gradient of a 3d image

    Parameters
    ----------
    img: (x, y, z) array
        Array of image values
    img_mask: (x*y*z, ) array
        Boolean mask where True indicates valid values
    grid_size: int
        Spacing between elements in img

    Returns
    -------
    img_grad: (x*y*z, 3) array
        Gradient of image in x, y, and z
    """

    #Get gradient
    img_grad = np.gradient(img, grid_size)

    #Make it into a matrix with mask
    img_grad = np.stack((img_grad[0].flatten()[img_mask],
                         img_grad[1].flatten()[img_mask],
                         img_grad[2].flatten()[img_mask]), axis=1)

    return img_grad

def calc_eta(img_one, img_two, img_mask, grid_size):
    """
    Computes spatial similarity between two images within a mask

    Parameters
    ----------
    img_one: (x, y, z) array
        3d image
    img_two: (x, y, z) array
        3d image
    img_mask: (x*y*z, ) array
        Boolean mask where True indicates valid values
    grid_size: float
        Spacing between elements in img_one and img_two

    Returns
    -------
    eta: float
        Spatial similarity metric between 0 and 1
    """

    #Calculate image gradients
    one_grad = calc_grad(img_one, img_mask, grid_size)
    two_grad = calc_grad(img_two, img_mask, grid_size)

    #Calculate gradient norm at each voxel
    one_norm = np.sum(one_grad * one_grad, axis=1)
    two_norm = np.sum(two_grad * two_grad, axis=1)

    #Compute dot product of the gradients
    cross_norm = np.sum(one_grad * two_grad, axis=1)

    #Make a product mask
    p_mask = cross_norm > 0.0

    #Compute eta
    eta_one = np.power(cross_norm[p_mask], 2)
    eta_two = np.sqrt(one_norm[p_mask] * two_norm[p_mask])
    eta_three = np.sqrt(np.sum(one_norm[p_mask]) * np.sum(two_norm[p_mask]))
    eta = np.sum(eta_one / eta_two) / eta_three

    return eta

#Function for computing cost of basis model
def cost_shift(param, epi, mean, basis, struct, mask, coords,
               grid_dims, grid_size, axis=1, sign=1, pen=None):
    """
    Compute cost function for basis function model

    Parameters
    ----------
    param: (n, ) array
        Weights for basis functions
    epi: (x, y, z) array
        3d image to undistort
    mean: (p, ) array
        Mean shift map sampled at p coordinates
    basis: (p, n) array
        Shift map basis fuctions at p coordinates for n functions
    struct: (p, ) array
        Structural image at p coordinates
    mask: (p, ) array
        Mask at p coordinates where True is a valid value
    coords: (3, p) list
        Coordinates to sample epi at
    grid_dims: (3) tuple
        Dimensions to map arrays of (p, ) to 3d arrays
    grid_size: float
        Spacing between elements in grid defiend by grid_dims
    axis: int
        Axis to apply shift to [1, 2, 3]
    sign: int
        Direction to apply shift [-1, 1]
    pen: float
        If set applies a simple ridge penalty to cost

    Returns
    -------
    eta: float
        Spatial simpliatry between epi and struct after warping epi with computed shift map
    """

    #Compute shift map
    shift_map = calc_shift(mean, basis, param)

    #Apply shift map to epi data
    unwarp = apply_shift(epi, coords, shift_map, axis=axis, sign=sign)

    #Compute the spatial similarity
    eta = calc_eta(unwarp.reshape(grid_dims), struct.reshape(grid_dims), mask, grid_size) * -1.0

    #Add penalty if necessary
    if pen is not None:
        eta += pen * np.sum(np.power(param, 2))

    #Return cost
    return eta

def write_args(args, out_path):
    """
    Saves arguments from argparse to a text file

    Parameters
    ----------
    args : argparse object
       Object returned from argparse
    out_path: string
       Name for output file
    """

    #Make string with all arguments
    arg_string = ''
    for arg, value in sorted(vars(args).items()):
        if isinstance(value, list):
            if not value:
                value = ','.join(map(str, value))
            else:
                value = value[0]
        arg_string += '%s: %s\n'%(arg, value)

    #Write out arguments
    try:
        arg_out = open(out_path, "w")
        arg_out.write(arg_string)
        arg_out.close()
    except IOError:
        raise IOError('Cannot write file at %s'%(out_path))

def main():
    """
    Main logic for program
    """

    #Get user input
    args = parse_args()

    #Load in image headers
    epi = load_hdr(args.epi[0])
    struct = load_hdr(args.struct[0])
    mean = load_hdr(args.mean[0])
    basis = load_hdr(args.basis[0])
    mask = load_hdr(args.mask[0])

    #Make sure epi image is only 3d
    n_epi_dim = len(epi.shape)
    if n_epi_dim != 3 and (n_epi_dim == 4 and epi.shape[-1] != 1):
        raise IOError('EPI image %s must be 3d'%(args.epi[0]))

    #Warn if we don't have enough basis functions
    if args.n[0] > basis.shape[3]:
        print('Warning: %i basis functions were requested but '
              'image only has %i. Using all available.'%(args.n[0], basis.shape[3]))

    #Loop through image headers
    for hdr in [struct, mean, basis, mask]:

        #Check to if image dimensions match those of epi
        if hdr.shape[0:3] != epi.shape[0:3]:
            raise IOError('Image %s does not match dimensions of EPI image'%(hdr.get_filename()))

        #Ensure all images but basis functions are 3d
        n_hdr_dim = len(hdr.shape)
        if hdr.get_filename() == args.basis[0]:
            if n_hdr_dim != 4:
                raise IOError('Basis function image %s must be 4d'%(args.basis[0]))
        elif n_hdr_dim != 3 and (n_hdr_dim == 4 and hdr.shape[-1] != 1):
            raise IOError('Image %s must be 3d'%(hdr.get_filename()))

    #Load all image data, removing singleton dimensions in 3d images
    epi_data = np.squeeze(epi.get_data())
    struct_data = np.squeeze(struct.get_data())
    mean_data = np.squeeze(mean.get_data())
    basis_data = basis.get_data()
    mask_data = np.squeeze(mask.get_data())

    #Don't consider voxels where map isn't defined
    use_mask = np.logical_and(mask_data != 0, mean_data != 0)

    #Apply mask
    struct_data *= use_mask
    mean_data *= use_mask
    basis_data *= use_mask[:, :, :, np.newaxis]

    #Create array for storing eta results
    n_grid = len(args.grid)
    eta_log = np.zeros((3, n_grid))
    eta_log[0, :] = args.grid

    #Set initial values for optimization
    init_opt = np.zeros(args.n[0])

    #Get voxel sizes
    vox_dims = np.array(epi.header.get_zooms())

    #Loop through grid levels
    for i in range(n_grid):

        #Update user
        grid_size = args.grid[i]
        print 'Optimizing on %.1f mm grid'%(grid_size)

        #Construct grid
        grid, grid_dims = make_grid(epi.shape[0:3], grid_size / vox_dims[0:3])

        #Interpolate images that remain fixed during optmization onto grid
        struct_grid = interp.map_coordinates(struct_data, grid, order=1)
        mean_grid = interp.map_coordinates(mean_data, grid, order=1)
        basis_grid = np.zeros((struct_grid.shape[0], args.n[0]))
        for j in range(args.n[0]):
            basis_grid[:, j] = interp.map_coordinates(basis_data[:, :, :, j], grid, order=1)
        mask_grid = interp.map_coordinates(use_mask, grid, order=0)

        #Get initial eta
        eta_log[1, i] = cost_shift(init_opt, epi_data, mean_grid, basis_grid, struct_grid,
                                   mask_grid, grid, grid_dims, grid_size, args.axis[0],
                                   args.sign[0], args.pen[0]) * -1.0

        #Define arguments for optmization
        args_opt = (epi_data, mean_grid, basis_grid, struct_grid, mask_grid, grid,
                    grid_dims, grid_size, args.axis[0], args.sign[0], args.pen[0])

        #Optimize the values for the basis fucntions
        opt_basis = opt.minimize(cost_shift, init_opt, method='Powell',
                                 args=args_opt, options={'maxiter':5000})

        #Run optional global optimization
        if args.basin[0] == 1:
            opt_dic = {'args':args_opt, 'options':{'eps':0.1}, 'method':'L-BFGS-B'}
            opt_basis = opt.basinhopping(cost_shift, init_opt, niter=10,
                                         minimizer_kwargs=opt_dic)

        #Save optimitized eta
        eta_log[2, i] = opt_basis.fun * -1.0

        #Use previous iteration as the intialization
        init_opt = opt_basis.x

    #Compute shift map in full resolution
    basis_flat_dim = (np.prod(epi.shape), args.n[0])
    shift_map_data = calc_shift(mean_data.flatten(),
                                basis_data[:, :, :, 0:args.n[0]].reshape(basis_flat_dim),
                                opt_basis.x)

    #Construct full resolution grid
    grid_f, _ = make_grid(epi.shape[0:3], [1, 1, 1])

    #Unwarp epi at full resolution
    unwarp_data = apply_shift(epi_data, grid_f, shift_map_data,
                              axis=args.axis[0], sign=args.sign[0])

    #Make unwarped data
    unwarp_data *= use_mask.flatten()

    #Make nibabel images for output
    shift_img = nib.Nifti1Image(shift_map_data.reshape(epi_data.shape), epi.affine)
    unwarp_img = nib.Nifti1Image(unwarp_data.reshape(epi_data.shape), epi.affine)

    #Write out nibabel images
    shift_img.to_filename('%s_shift.nii.gz'%(args.out[0]))
    unwarp_img.to_filename('%s_unwarp.nii.gz'%(args.out[0]))

    #Write out weights
    np.savetxt('%s_basis_weights.txt'%(args.out[0]), opt_basis.x, fmt='%.5f')

    #Write out eta progress
    np.savetxt('%s_eta.txt'%(args.out[0]), eta_log, fmt='%.5f', delimiter=',', comments='')

    #Write out arugments to file
    write_args(args, '%s_args.txt'%(args.out[0]))

#Call main function
if __name__ == '__main__':
    main()
