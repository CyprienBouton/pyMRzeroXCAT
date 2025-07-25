import numpy as np
from scipy.ndimage import zoom
import re
import json
from mrtwin import b0field, b1field, sensmap
import argparse
from ast import literal_eval

DEFAULT_T1 = 900    # ms (muscle, organs)
DEFAULT_T2 = 50     # ms (muscle, soft tissue)
DEFAULT_T2dash = 30 # ms (typical T2' value for soft tissue)
DEFAULT_RHO = 85.0    # Between muscle (80) and liver (90)
DEFAULT_CHI = -9.0    # Typical soft tissue susceptibility (ppm)
DEFAULT_tissues_param_json = 'pymrzeroxcat/MRXCAT_PHANTOM_LGE/tissues.json'

def parse_key_value(arg):
    try:
        key, value = arg.split('=')
        return key, literal_eval(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Arguments must be in key=value format: got '{arg}'")
    

def search_key_log(log_file, key):
    """
    Search for a key in the log file and return its value.
    """
    with open(log_file, 'r') as fid:
        for line in fid:
            if key in line:
                # Remove text in parentheses at the end, e.g., (cm/pixel)
                line = re.sub(r'\s*\([^)]*\)\s*$', '', line)
                value =  line.split()[-1]
                if value.isdigit():
                    return int(value)
                elif value.replace('.', '', 1).isdigit():
                    return float(value)
                else:
                    return value
    raise ValueError(f"Key '{key}' not found in log file.")


def get_segmentation(bin_file, log_file, flip_horizontal=True):
    matrix = search_key_log(log_file, 'array_size')
    seg = np.fromfile(bin_file, dtype=np.float32).astype(np.int32)
    seg = seg.reshape(-1, matrix, matrix).transpose(1, 2, 0)
    if flip_horizontal:
        seg = np.flip(seg, axis=1)
    return seg


def get_tissues_id(log_file):
    tissues_ids = []
    # keep lines containing _act or _activity
    pattern = r'^(?=.*(?:_act(?:ivity)?))((?!_act_).)*$'
    with open(log_file, 'r') as fid:
        for line in fid:
            if re.search(pattern, line):
                tissue_name = line.split()[0].split('_')[0]
                tissue_id = int(float(line.split()[-1]))
                if tissue_id not in tissues_ids:
                    tissues_ids.append(tissue_id)
    return tissues_ids


def compute_parameters_maps(bin_file, bbox=np.array([[0.,1.]]*3), new_resolution=None, tissues_param_json=DEFAULT_tissues_param_json, ):
    """
    Compute the T1, T2, T2dash, rho, and chi maps from the binary file.
    """
    tissues_parameters = json.load(open(tissues_param_json, 'r'))  # Load tissue parameters from JSON file
    log_file = '_'.join(bin_file.split('_')[:-2]) + '_log'
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


def crop_segmentation(segmentation, bbox):
    """
    Crop the segmentation according to the bounding box.
    bbox: np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
    """
    if bbox.shape != (3, 2):
        raise ValueError("Bounding box must be a 3x2 array.")
    
    x_min, x_max = int(bbox[0, 0] * segmentation.shape[0]), int(bbox[0, 1] * segmentation.shape[0])
    y_min, y_max = int(bbox[1, 0] * segmentation.shape[1]), int(bbox[1, 1] * segmentation.shape[1])
    z_min, z_max = int(bbox[2, 0] * segmentation.shape[2]), int(bbox[2, 1] * segmentation.shape[2])
    
    return segmentation[x_min:x_max, y_min:y_max, z_min:z_max]


def resample_segmentation(segmentation, orig_res, new_res, order=0):
    """
    Resamples a 3D segmentation to a new resolution.

    Parameters:
        segmentation (np.ndarray): 3D array of shape (Nx, Ny, Nz)
        orig_res (tuple or list): Original resolution (rx, ry, rz) in cm/pixel
        new_res (tuple or list): Desired resolution (res_x, res_y, res_z) in cm/pixel
        order (int): Interpolation order (0=nearest, 1=linear, 3=cubic)
    
    Returns:
        np.ndarray: Resampled segmentation
    """
    orig_res = np.array(orig_res)
    new_res = np.array(new_res)

    # Calculate zoom factors: how much to scale each dimension
    zoom_factors = orig_res / new_res

    # Interpolate the segmentation
    resampled = zoom(segmentation, zoom=zoom_factors, order=order)

    return resampled


def get_resolution(log_file):
    """
    Get the resolution from the log file.
    """
    rx_cm = search_key_log(log_file, "pixel width")
    rz_cm = search_key_log(log_file, "slice width")
    return np.array([rx_cm, rx_cm, rz_cm])*10  # Convert to mm


def build_static_phantom(
    bin_file, 
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
    Build a static phantom for LGE MRI.
    Parameters:
        bin_file (str): Path to the binary file containing the phantom data.
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
    t1_map, t2_map, t2dash_map, rho_map, chi_map = compute_parameters_maps(bin_file, bbox, resolution)
    
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
    )
    print(f"Static phantom saved to {phantom_file}")
    
    # Optionally plot the phantom
    if plot:
        import MRzeroCore as mr0
        phantom = mr0.DynamicVoxelPhantom.load(phantom_file)
        phantom.plot()


def main():
    parser = argparse.ArgumentParser(description='Build MRXCAT Phantom from .bin file')
    
    parser.add_argument('bin_file', help='Input binary (.bin) file for phantom generation')
    parser.add_argument('-p', '--phantom_file', help='Output phantom file (.npz)', default='MRXCAT_phantom.npz')
    parser.add_argument('-B0', '--field_strength', help='Field strength in Tesla (default: 1.5)', type=float, default=1.5)
    parser.add_argument('--plot', help='Whether to plot the phantom (default: True)', action=argparse.BooleanOptionalAction, default=True)
    
    parser.add_argument('--bbox', help="Bounding box (3x2 array), default: [[0.,1.]]*3", type=float, nargs='+', default=[0.2, 0.7, 0.6, 0.9, 0., 1.])
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
    
    # Call your phantom generator
    build_static_phantom(
        bin_file=args.bin_file,
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