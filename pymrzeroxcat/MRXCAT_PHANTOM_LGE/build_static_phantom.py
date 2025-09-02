import argparse
import json
import importlib.resources as pkg_resources
import os

import numpy as np
import MRzeroCore as mr0
from mrtwin import b1field, b0field, sensmap

import tkinter as tk
from tkinter import filedialog

from pymrzeroxcat.convert_arguments import parse_key_value_or_json_file, complete_imaging_args
from pymrzeroxcat.read_mrxcat_raw_data import get_crop_segmentation_resampled, resolve_log_file, get_tissues_id

JSON_PARAMETERS = pkg_resources.files("pymrzeroxcat").joinpath("tissues.json").as_posix()
DEFAULT_T1 = 900    # ms (muscle, organs)
DEFAULT_T2 = 50     # ms (muscle, soft tissue)
DEFAULT_T2dash = 30 # ms (typical T2' value for soft tissue)
DEFAULT_RHO = 85.0    # Between muscle (80) and liver (90)
DEFAULT_CHI = -9.0    # Typical soft tissue susceptibility (ppm)
DEFAULT_D = 0.0       # Typical diffusion coefficient (mm^2/s)


def get_phantom_values(
    bin_file,
    log_file,
    FOV=(200, 200, 200),
    matrix=(128,128,5),
    center_segmentation=[0.4, 0.75, 0.5],
    field_strength=1.5,
    ncoils=8,
    param_json=JSON_PARAMETERS,
    b0field_kwargs={},
    b1field_kwargs={},
    sensmap_kwargs={}, 
    phantom_file='T1MES_default.npz',
    plot=True, 
    plot_kwargs={},
):
    """
    Create and save a T1MES phantom using values from doi:10.1186/s12968-016-0280-z.

    Args:
        bin_file (str): Path to the binary segmentation file.
        log_file (str): Path to the log file containing the 'array_size' key.
        FOV (list, tuple): field of view of the phantom [mm].
        matrix (list, tuple): Resolution matrix.
        center_segmentation (list, tuple): Center of the segmentation [x, y, z] (relative to FOV).
        field_strength (float): Field strength in Tesla. Only 1.5 or 3.0 supported.
        ncoils (int): Number of receiver coils for sensitivity map generation. Defaults to 8.
        param_json (str): Path to JSON file with phantom parameters. Defaults to 't1mes/T1MES_values.json'.
        b0field_kwargs (dict, optional): Keyword arguments passed to `mrtwin.b0field()`.
        b1field_kwargs (dict, optional): Keyword arguments passed to `mrtwin.b1field()`.
        sensmap_kwargs (dict, optional): Keyword arguments passed to `mrtwin.sensmap()`.
        phantom_file (str): Path to save the generated phantom .npz file. Defaults to 'T1MES_default.npz'.
        plot (bool): If True, displays the phantom using MRzeroCore. Defaults to True.
        plot_kwargs (dict, optional): Keyword arguments passed to `mr0.DynamicVoxelPhantom.plot()`.
    """
    # Load T1MES phantom tissue parameters from JSON
    phantom_data = json.load(open(param_json, 'r'))  # Values from Bruker minispec (1.4 T, 22°C)

    seg = get_crop_segmentation_resampled(bin_file, log_file, FOV, matrix, center_segmentation)
    tissues_ID = get_tissues_id(log_file)
    # Initialize property maps
    T1_map = np.zeros_like(seg, dtype=float)
    T2_map = np.zeros_like(seg, dtype=float)
    PD_map = np.zeros_like(seg, dtype=float)
    chi_map = np.zeros_like(seg, dtype=float)
    D_map = np.zeros_like(seg, dtype=float)
    T2dash_map = np.zeros_like(seg, dtype=float)

    # Fill values for each tissue
    for tissue_id in tissues_ID:
        if str(tissue_id) not in phantom_data:
            T1_map[seg == tissue_id] = DEFAULT_T1
            T2_map[seg == tissue_id] = DEFAULT_T2
            T2dash_map[seg == tissue_id] = DEFAULT_T2dash
            PD_map[seg == tissue_id] = DEFAULT_RHO
            chi_map[seg == tissue_id] = DEFAULT_CHI
            D_map[seg == tissue_id] = DEFAULT_D
        else:
            T1_map[seg == tissue_id] = phantom_data[str(tissue_id)]['T1']
            T2_map[seg == tissue_id] = phantom_data[str(tissue_id)]['T2']
            T2dash_map[seg == tissue_id] = phantom_data[str(tissue_id)]['T2dash']
            PD_map[seg == tissue_id] = phantom_data[str(tissue_id)]['PD']
            chi_map[seg == tissue_id] = phantom_data[str(tissue_id)]['chi']
            D_map[seg == tissue_id] = phantom_data[str(tissue_id)]['D']

    # compute B0
    b0field_kwargs['chi'] = chi_map
    b0field_kwargs['B0'] = field_strength
    if 'b0range' not in b0field_kwargs:
        b0field_kwargs['b0range'] = (-200,200) # Default range of -200/+200Hz
    B0_map = b0field(**b0field_kwargs)
    
    # Compute B1
    b1field_kwargs['shape'] = B0_map.squeeze().shape
    b1field_kwargs['mask'] = PD_map.squeeze()>0
    B1_map = b1field(**b1field_kwargs)
    if B1_map.ndim==2:
        B1_map = np.expand_dims(B1_map, axis=2)
    
    # Compute sensitivity map
    sensmap_kwargs['shape'] = (ncoils,) + PD_map.shape
    coil_sens = sensmap(**sensmap_kwargs)
    
    # Add tissue masks
    tissues_mask = {phantom_data[tissue_id]['description']: seg==int(tissue_id)  for tissue_id in phantom_data.keys()}
    tissues_mask = {'tissue_'+k: v for k,v in tissues_mask.items()}
    
    # Save maps to compressed .npz file (convert T1, T2 to seconds)
    np.savez_compressed(
        phantom_file,
        PD_map=PD_map,
        T1_map=T1_map * 1e-3,
        T2_map=T2_map * 1e-3,
        T2dash_map=T2dash_map,
        D_map=D_map,  # Diffusion is set to 0 everywhere
        B0_map=B0_map,
        B1_map=B1_map,
        FOV=np.array(FOV, dtype=float)*1e-3, # in meters
        coil_sens=coil_sens,
        **tissues_mask,
    )

    # Optional plot: display center slice using MRzeroCore
    if plot:
        phantom = mr0.VoxelGridPhantom.load(phantom_file)
        if 'time_unit' not in plot_kwargs:
            plot_kwargs['time_unit'] = 'ms'
        if 'display_units' not in plot_kwargs:
            plot_kwargs['display_units'] = True
        phantom.plot(**plot_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Build Static T1MES Phantom")
    # Binary segmentation file
    binary_group = parser.add_argument_group("Binary Segmentation File")
    binary_group.add_argument('--bin_file', help='Input binary (.bin) file for phantom generation', default=None)
    binary_group.add_argument('--log_file', help='Input log (_log) file', default=None)
    
    # Imaging Parameters
    imaging_group = parser.add_argument_group("Imaging Parameters")
    imaging_group.add_argument('-FOV', type=float, nargs=3, help='Field of view in mm. Default to (300,300,50)')   
    imaging_group.add_argument('-r', '--resolution', type=float, nargs=3, help='Voxel resolution. Default to 2x2x5 mm')
    imaging_group.add_argument('-m', '--matrix', type=int, nargs=3, help='Matrix size. Default to 150x150x10')
    imaging_group.add_argument('-c', '--center_segmentation', type=float, nargs=3, 
                               help='Center of the segmentation [x,y,z] (relative to FOV). Default to [0.4,0.75,0.5]', default=[0.4,0.75,0.5])
    imaging_group.add_argument('-B0', '--field_strength', type=float, help='Main magnetic field strength in Tesla. Default to 1.5', default=1.5)
    imaging_group.add_argument('--ncoils', type=int, help='Number of receiver coils. Default to 8', default=8)

    # Phantom Configuration
    phantom_group = parser.add_argument_group("Phantom Configuration")
    phantom_group.add_argument('--param_json', type=str, help='Path to JSON file used to generate phantom', default=JSON_PARAMETERS)
    phantom_group.add_argument('--b0_kwargs', type=parse_key_value_or_json_file, nargs='+', help='JSON string or file path for B0 simulation params', default={})
    phantom_group.add_argument('--b1_kwargs', type=parse_key_value_or_json_file, nargs='+', help='JSON string or file path for B1 simulation params', default={})
    phantom_group.add_argument('--sensmap_kwargs', type=parse_key_value_or_json_file, nargs='+', help='JSON string or file path for sensitivity map params', default={})

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument('-p', '--phantom_file', type=str, help='Path to output phantom file', default='T1MES_default.npz')

    # Visualization
    vis_group = parser.add_argument_group("Visualization")
    vis_group.add_argument('--plot', action=argparse.BooleanOptionalAction, help="Whether to plot resulting phantom. Default to True", default=True)
    vis_group.add_argument('--plot_kwargs', type=parse_key_value_or_json_file, nargs='+', help='JSON string or file path with plot customization options', default={})

    args = parser.parse_args()
    
    args = complete_imaging_args(args)
    
    print(f"FOV: {args.FOV}")
    print(f"Resolution: {args.resolution}")
    print(f"Matrix: {args.matrix}")

    for attr in ['b0_kwargs', 'b1_kwargs', 'sensmap_kwargs', 'plot_kwargs']:
        val = getattr(args, attr)
        if val:
            setattr(args, attr, {k: v for d in val for k, v in d.items()})

    if args.bin_file is None:
        root = tk.Tk()
        root.withdraw()

        args.bin_file = filedialog.askopenfilename()
        
    if args.log_file is None:
        args.log_file = resolve_log_file(args.bin_file)
        
    get_phantom_values(
        bin_file=args.bin_file,
        log_file=args.log_file,
        FOV=args.FOV,
        matrix=args.matrix,
        center_segmentation=args.center_segmentation,
        field_strength=args.field_strength,
        ncoils=args.ncoils,
        param_json=args.param_json,
        b0field_kwargs=args.b0_kwargs,
        b1field_kwargs=args.b1_kwargs,
        sensmap_kwargs=args.sensmap_kwargs,
        phantom_file=args.phantom_file,
        plot=args.plot,
        plot_kwargs=args.plot_kwargs,
    )


if __name__ == "__main__":
    main()