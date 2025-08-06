import os
import numpy as np
from scipy.ndimage import zoom
import json
from mrtwin import b0field, b1field, sensmap
import argparse
from ast import literal_eval

from pymrzeroxcat.MRXCAT_PHANTOM_LGE.compute_dynamic_parameters import compute_dynamic_parameters_maps


def parse_key_value(arg):
    try:
        key, value = arg.split('=')
        return key, literal_eval(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Arguments must be in key=value format: got '{arg}'")


def build_dynamic_phantom(
    bin_file, 
    log_file,
    concentrations_file,
    phantom_file='MRXCAT_phantom.npz', 
    field_strength=1.5, 
    plot=True, 
    bbox=np.array([[0., 1.]]*3), 
    resolution=None,
    ncoils=8,
    b0field_kwargs={}, 
    b1field_kwargs={},
    sensmap_kwargs={}, 
):
    """
    Build a dynamic phantom for LGE MRI.
    Parameters:
        bin_file (str): Path to the binary file containing the phantom data.
        log_file (str): Path to the log file containing the key 'array_size'.
        concentrations_file (str): file containing contrast agent concentrations for injected tube over time [mM].
        phantom_file (str): Output file path for the generated phantom.
        field_strength (float): Magnetic field strength in Tesla.
        plot (bool): Whether to plot the phantom after generation.
        bbox (np.ndarray): Bounding box for the phantom in the format [[x_min, x_max], [y_min, y_max], [z_min, z_max]].
        resolution (tuple): Resolution of the phantom in mm/pixel (Nx, Ny, Nz).
        ncoils (int): Number of coils for the sensitivity map.
        b0field_kwargs (dict): Keyword arguments for the B0 field computation.
        b1field_kwargs (dict): Keyword arguments for the B1 field computation.
        sensmap_kwargs (dict): Keyword arguments for the sensitivity map computation
    """
    t1_map, t2_map, t2dash_map, rho_map, chi_map, time_points = compute_dynamic_parameters_maps(bin_file, log_file, concentrations_file, bbox, resolution)
    
    # compute B0
    b0field_kwargs['chi'] = chi_map
    b0field_kwargs['B0'] = field_strength
    if 'b0range' not in b0field_kwargs:
        b0field_kwargs['b0range'] = (-200,200) # Default range of -200/+200Hz
    B0_map = b0field(**b0field_kwargs)
    
    # compute B1
    b1field_kwargs['shape'] = B0_map.squeeze().shape
    b1field_kwargs['mask'] = rho_map.squeeze()>0
    if 'b1range' not in b1field_kwargs:
        b1field_kwargs['b1range'] = (1., 1.)
    B1_map = b1field(**b1field_kwargs)
    if B1_map.ndim==2:
        B1_map = np.expand_dims(B1_map, axis=2)
    
    # Compute sensitivity map
    sensmap_kwargs['shape'] = (ncoils,) + rho_map.shape
    coil_sens = sensmap(**sensmap_kwargs)

    np.savez_compressed(
        phantom_file,
        PD_map=rho_map,
        T1_map=t1_map * 1e-3,
        T2_map=t2_map * 1e-3,
        T2dash_map=t2dash_map,
        D_map=np.zeros_like(rho_map),  # Diffusion is set to 0 everywhere
        B0_map=B0_map,
        B1_map=B1_map,
        FOV=(0.2, 0.2, 0.2),
        coil_sens=coil_sens,
        time_points=time_points,
    )
    print(f"Dynamic phantom saved to {phantom_file}")
    
    # Optionally plot the phantom
    if plot:
        import MRzeroCore as mr0
        phantom = mr0.DynamicVoxelPhantom.load(phantom_file)
        phantom.plot_dynamic(time_unit='ms', display_units=True)


def main():
    parser = argparse.ArgumentParser(description='Build MRXCAT Phantom from .bin file')
    
    parser.add_argument('bin_file', help='Input binary (.bin) file for phantom generation')
    parser.add_argument('-c', '--concentrations_file', help='File containing contrast agent concentrations over time [mM] for injected tubes', required=True)
    parser.add_argument('--log_file', help='Input log (_log) file', default=None)
    parser.add_argument('-p', '--phantom_file', help='Output phantom file (.npz)', default='MRXCAT_phantom.npz')
    parser.add_argument('-B0', '--field_strength', help='Field strength in Tesla (default: 1.5)', type=float, default=1.5)
    parser.add_argument('--plot', help='Whether to plot the phantom (default: True)', action=argparse.BooleanOptionalAction, default=True)
    
    parser.add_argument('--bbox', help="Bounding box (3x2 array), default: [0.2, 0.7, 0.6, 0.9, 0., 1.]", type=float, nargs='+', default=[0.2, 0.75, 0.55, 0.95, 0., 1.])
    parser.add_argument('-r', '--resolution', help='Resolution of the phantom (Nx, Ny, Nz)', type=int, nargs=3)
    
    parser.add_argument('--ncoils', help='Number of coils for sensitivity map', type=int, default=8)    
    parser.add_argument('--b0_kwargs', help="Keyword arguments passed to `mrtwin.b0field()`.", type=parse_key_value, default={}, nargs='+')
    parser.add_argument('--b1_kwargs', help="Keyword arguments passed to `mrtwin.b1field()`.", type=parse_key_value, default={}, nargs='+')
    parser.add_argument('--sensmap_kwargs', help="Keyword arguments passed to `mrtwin.sensmap()`.", type=parse_key_value, default={}, nargs='+')

    args = parser.parse_args()

    # Process bbox
    bbox = np.array(args.bbox).reshape((3, 2))
    
    # Convert key-value pairs to dictionaries
    b0_kwargs = dict(args.b0_kwargs)
    b1_kwargs = dict(args.b1_kwargs)
    sensmap_kwargs = dict(args.sensmap_kwargs)
    
    if args.log_file is None:
        if args.bin_file.endswith('_with_inf.bin'): # default naming suffix for mask with infarct
            log_file = log_file = '_'.join(args.bin_file.split('_')[:-4]) + '_log'
        else:
            log_file = '_'.join(args.bin_file.split('_')[:-2]) + '_log'
        if not os.path.isfile(log_file):
            raise FileNotFoundError(f"Auto-generated log file '{log_file}' does not exist. Please provide one using --log_file.")
    else:
        log_file = args.log_file
    
    # Call your phantom generator
    build_dynamic_phantom(
        bin_file=args.bin_file,
        log_file=log_file,
        concentrations_file=args.concentrations_file,
        phantom_file=args.phantom_file,
        field_strength=args.field_strength,
        plot=args.plot,
        bbox=bbox,
        resolution=args.resolution,
        b0field_kwargs=b0_kwargs,
        b1field_kwargs=b1_kwargs,
        sensmap_kwargs=sensmap_kwargs,
    )

if __name__ == "__main__":
    main()