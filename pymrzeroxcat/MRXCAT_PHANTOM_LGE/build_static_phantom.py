import os
import numpy as np
from scipy.ndimage import zoom
import json
from mrtwin import b0field, b1field, sensmap
import argparse
from ast import literal_eval

from pymrzeroxcat.read_mrxcat_raw_data import get_tissues_id, get_resolution, get_segmentation, resample_segmentation, crop_segmentation
from pymrzeroxcat.read_mrxcat_raw_data import resolve_log_file


DEFAULT_T1 = 900    # ms (muscle, organs)
DEFAULT_T2 = 50     # ms (muscle, soft tissue)
DEFAULT_T2dash = 30 # ms (typical T2' value for soft tissue)
DEFAULT_RHO = 85.0    # Between muscle (80) and liver (90)
DEFAULT_CHI = -9.0    # Typical soft tissue susceptibility (ppm)
DEFAULT_tissues_param_json = 'MRXCAT_raw_data/tissues.json'


def parse_key_value(arg):
    try:
        key, value = arg.split('=')
        return key, literal_eval(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Arguments must be in key=value format: got '{arg}'")


def compute_parameters_maps(bin_file, log_file, bbox=np.array([[0.,1.]]*3), new_resolution=None, tissues_param_json=DEFAULT_tissues_param_json, ):
    """
    Compute the T1, T2, T2dash, rho, and chi maps from the binary file.
    """
    tissues_parameters = json.load(open(tissues_param_json, 'r'))  # Load tissue parameters from JSON file
    segmentation = get_segmentation(bin_file, log_file)
    resolution = get_resolution(log_file)
    if new_resolution is not None:
        segmentation = resample_segmentation(segmentation, resolution, new_resolution)
        resolution = np.array(new_resolution)  # Update resolution to new resolution
    segmentation = crop_segmentation(segmentation, bbox)
    FOV = segmentation.shape * resolution
    print(f"Resolution: {resolution[0]} x {resolution[1]} x {resolution[2]} mm/pixel")
    print(f"  FOV : {FOV[0]} x {FOV[1]} x {FOV[2]} mm³")
    print(f"Matrix: {segmentation.shape}")

    default_values = {
        'T1': DEFAULT_T1,  # ms
        'T2': DEFAULT_T2,     # ms
        'T2dash': DEFAULT_T2dash,  # ms
        'rho': DEFAULT_RHO,   # arbitrary units
        'chi': DEFAULT_CHI,     # ppm
    }
    
    # initialize parameters
    t1_map = np.zeros(segmentation.shape, dtype=np.float32)
    t2_map = np.zeros(segmentation.shape, dtype=np.float32)
    t2dash_map = np.zeros(segmentation.shape, dtype=np.float32)
    rho_map = np.zeros(segmentation.shape, dtype=np.float32)
    chi_map = np.zeros(segmentation.shape, dtype=np.float32)
    
    tissues_ids = get_tissues_id(log_file)
    for tissue_id in tissues_ids:
        if str(tissue_id) in tissues_parameters:
            tissue_params = tissues_parameters[str(tissue_id)]
            t1_map[segmentation == tissue_id] = tissue_params['T1']
            t2_map[segmentation == tissue_id] = tissue_params['T2']
            rho_map[segmentation == tissue_id] = tissue_params['rho']
            chi_map[segmentation == tissue_id] = tissue_params['chi']
            if 'T2dash' in tissue_params:
                t2dash_map[segmentation == tissue_id] = tissue_params['T2dash']
            else:
                t2dash_map[segmentation == tissue_id] = default_values['T2dash'] # Default value if not specified in tissue parameters
        # if tissue_id not in tissues_parameters assign default values
        else:
            t1_map[segmentation == tissue_id] = default_values['T1']
            t2_map[segmentation == tissue_id] = default_values['T2']
            rho_map[segmentation == tissue_id] = default_values['rho']
            chi_map[segmentation == tissue_id] = default_values['chi']
            t2dash_map[segmentation == tissue_id] = default_values['T2dash']

    return t1_map, t2_map, t2dash_map, rho_map, chi_map


def build_static_phantom(
    bin_file, 
    log_file,
    phantom_file='MRXCAT_phantom.npz', 
    field_strength=1.5, 
    plot=True, 
    bbox=np.array([[0., 1.]]*3), 
    resolution=None,
    ncoils=8,
    tissues_param_json=DEFAULT_tissues_param_json,
    b0field_kwargs={}, 
    b1field_kwargs={},
    sensmap_kwargs={}, 
    plot_kwargs={}, 
):
    """
    Build a static phantom for LGE MRI.
    Parameters:
        bin_file (str): Path to the binary file containing the phantom data.
        log_file (str): Path to the log file containing the key 'array_size'.
        phantom_file (str): Output file path for the generated phantom.
        field_strength (float): Magnetic field strength in Tesla.
        plot (bool): Whether to plot the phantom after generation.
        bbox (np.ndarray): Bounding box for the phantom in the format [[x_min, x_max], [y_min, y_max], [z_min, z_max]].
        resolution (tuple): Resolution of the phantom in mm/pixel (Nx, Ny, Nz).
        ncoils (int): Number of coils for the sensitivity map.
        tissues_param_json (str): Path to a JSON file containing tissue parameters like T1, T2, T2dash, rho, and chi. Default to DEFAULT_tissues_param_json
        b0field_kwargs (dict): Keyword arguments for the B0 field computation.
        b1field_kwargs (dict): Keyword arguments for the B1 field computation.
        sensmap_kwargs (dict): Keyword arguments for the sensitivity map computation
        plot_kwargs (dict): Keyword arguments for the plot_dynamic
    """
    t1_map, t2_map, t2dash_map, rho_map, chi_map = compute_parameters_maps(bin_file, log_file, bbox, resolution, tissues_param_json=tissues_param_json)
    
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
        b1field_kwargs['b1range'] = (.99, 1.)
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
        T2dash_map=t2dash_map * 1e-3,
        D_map=np.zeros_like(rho_map),  # Diffusion is set to 0 everywhere
        B0_map=B0_map,
        B1_map=B1_map,
        FOV=(0.2, 0.2, 0.2),
        coil_sens=coil_sens,
    )
    print(f"Static phantom saved to {phantom_file}")
    
    # Optionally plot the phantom
    if plot:
        import MRzeroCore as mr0
        phantom = mr0.DynamicVoxelPhantom.load(phantom_file)
        if 'time_unit' not in plot_kwargs:
            plot_kwargs['time_unit'] = 'ms'
        if 'display_units' not in plot_kwargs:
            plot_kwargs['display_units'] = True
        phantom.plot(**plot_kwargs)


def main():
    parser = argparse.ArgumentParser(description='Build MRXCAT Phantom from .bin file')
    
    parser.add_argument('bin_file', help='Input binary (.bin) file for phantom generation')
    parser.add_argument('--log_file', help='Input log (_log) file', default=None)
    parser.add_argument('-p', '--phantom_file', help='Output phantom file (.npz)', default='MRXCAT_phantom.npz')
    parser.add_argument('-B0', '--field_strength', help='Field strength in Tesla (default: 1.5)', type=float, default=1.5)
    parser.add_argument('--plot', help='Whether to plot the phantom (default: True)', action=argparse.BooleanOptionalAction, default=True)
    
    parser.add_argument('--bbox', help="Bounding box (3x2 array), default: [0.2, 0.7, 0.6, 0.9, 0., 1.]", type=float, nargs='+', default=[0.2, 0.75, 0.55, 0.95, 0., 1.])
    parser.add_argument('-r', '--resolution', help='Resolution of the phantom (Nx, Ny, Nz)', type=int, nargs=3)
    
    parser.add_argument('--ncoils', help='Number of coils for sensitivity map', type=int, default=8)
    parser.add_argument('--param_json', help="Tissues parameters file (.json)", default=DEFAULT_tissues_param_json)
        
    parser.add_argument('--b0_kwargs', help="Keyword arguments passed to `mrtwin.b0field()`.", type=parse_key_value, default={}, nargs='+')
    parser.add_argument('--b1_kwargs', help="Keyword arguments passed to `mrtwin.b1field()`.", type=parse_key_value, default={}, nargs='+')
    parser.add_argument('--sensmap_kwargs', help="Keyword arguments passed to `mrtwin.sensmap()`.", type=parse_key_value, default={}, nargs='+')
    parser.add_argument('--plot_kwargs', help="Keyword arguments passed to `DynamicVoxelPhantom.plot_dynamic()`.", type=parse_key_value, default={}, nargs='+')

    args = parser.parse_args()

    # Process bbox
    bbox = np.array(args.bbox).reshape((3, 2))
    
    # Convert key-value pairs to dictionaries
    b0_kwargs = dict(args.b0_kwargs)
    b1_kwargs = dict(args.b1_kwargs)
    sensmap_kwargs = dict(args.sensmap_kwargs)
    plot_kwargs = dict(args.plot_kwargs)
    
    if args.log_file is None:
        log_file = resolve_log_file(args.bin_file)
    else:
        log_file = args.log_file
    
    # Call your phantom generator
    build_static_phantom(
        bin_file=args.bin_file,
        log_file=log_file,
        phantom_file=args.phantom_file,
        field_strength=args.field_strength,
        plot=args.plot,
        bbox=bbox,
        resolution=args.resolution,
        ncoils=args.ncoils,
        tissues_param_json=args.param_json,
        b0field_kwargs=b0_kwargs,
        b1field_kwargs=b1_kwargs,
        sensmap_kwargs=sensmap_kwargs,
        plot_kwargs=plot_kwargs,
    )

if __name__ == "__main__":
    main()