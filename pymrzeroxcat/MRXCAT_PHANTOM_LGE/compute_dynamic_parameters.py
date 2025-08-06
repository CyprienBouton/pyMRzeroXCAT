import json
import numpy as np

from pymrzeroxcat.read_mrxcat_raw_data import get_tissues_id, get_resolution, get_segmentation, resample_segmentation, crop_segmentation


DEFAULT_T1 = 900    # ms (muscle, organs)
DEFAULT_T2 = 50     # ms (muscle, soft tissue)
DEFAULT_T2dash = 30 # ms (typical T2' value for soft tissue)
DEFAULT_RHO = 85.0    # Between muscle (80) and liver (90)
DEFAULT_CHI = -9.0    # Typical soft tissue susceptibility (ppm)
DEFAULT_tissues_param_json = 'pymrzeroxcat/MRXCAT_PHANTOM_LGE/tissues.json'


def compute_dynamic_parameters_maps(bin_file, log_file, concentrations_file, 
                            bbox=np.array([[0.,1.]]*3), new_resolution=None, tissues_param_json=DEFAULT_tissues_param_json, ):
    """
    Compute T1, T2, T2dash, rho, and chi parameter maps based on segmentation and tissue parameters.

    This function processes segmentation data from binary and log files, resamples it if required, 
    crops it using a bounding box, and computes tissue-specific parameter maps using provided or 
    default tissue properties. Dynamic parameters (T1, T2) can vary over time depending on contrast agent concentrations.

    Parameters:
    ----------
    bin_file : str
        Path to the binary segmentation file.
    log_file : str
        Path to the log file containing scan metadata and tissue labels.
    concentrations_file : str
        Path to the file containing concentration data over time for contrast-enhanced simulation.
    bbox : np.ndarray, optional
        3x2 array defining the cropping bounding box in normalized coordinates (default: full volume).
    new_resolution : tuple or list, optional
        Desired spatial resolution (in mm) for resampling the segmentation volume. If None, original resolution is used.
    tissues_param_json : str, optional
        Path to a JSON file containing tissue parameters like T1, T2, T2dash, rho, and chi. 
        If not specified, uses the default parameter file.

    Returns:
    -------
    t1_map : np.ndarray
        3D array representing the T1 relaxation map (in ms).
    t2_map : np.ndarray
        3D array representing the T2 relaxation map (in ms).
    t2dash_map : np.ndarray
        3D array representing the T2* (T2dash) map (in ms).
    rho_map : np.ndarray
        3D array representing the proton density (arbitrary units).
    chi_map : np.ndarray
        3D array representing the magnetic susceptibility map (in ppm).
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
    
    tissues_ids = get_tissues_id(log_file)
    T1, T2, times_post = get_T1_T2_over_time(concentrations_file, tissues_parameters)
    time_frames = len(times_post)
    
    # initialize parameters
    t1_map = np.zeros((time_frames, *segmentation.shape), dtype=np.float32)
    t2_map = np.zeros((time_frames, *segmentation.shape), dtype=np.float32)
    t2dash_map = np.zeros(segmentation.shape, dtype=np.float32)
    rho_map = np.zeros(segmentation.shape, dtype=np.float32)
    chi_map = np.zeros(segmentation.shape, dtype=np.float32)
    
    for tissue_id in tissues_ids:
        if str(tissue_id) in tissues_parameters:
            tissue_params = tissues_parameters[str(tissue_id)]
            for t_idx in range(time_frames):
                t1_map[t_idx][segmentation == tissue_id] = T1[str(tissue_id)][t_idx]
                t2_map[t_idx][segmentation == tissue_id] = T2[str(tissue_id)][t_idx]
            rho_map[segmentation == tissue_id] = tissue_params['rho']
            chi_map[segmentation == tissue_id] = tissue_params['chi']
            if 'T2dash' in tissue_params:
                t2dash_map[segmentation == tissue_id] = tissue_params['T2dash']
            else:
                t2dash_map[segmentation == tissue_id] = default_values['T2dash'] # Default value if not specified in tissue parameters
        # if tissue_id not in tissues_parameters assign default values
        else:
            t1_map[:,segmentation == tissue_id] = default_values['T1']
            t2_map[:, segmentation == tissue_id] = default_values['T2']
            rho_map[segmentation == tissue_id] = default_values['rho']
            chi_map[segmentation == tissue_id] = default_values['chi']
            t2dash_map[segmentation == tissue_id] = default_values['T2dash']

    return t1_map, t2_map, t2dash_map, rho_map, chi_map, times_post


def get_T1_T2_over_time( 
    concentrations_file,
    tissues_parameters,
    r1=4.1e-3,
    r2=4.6e-3,
    ):
    """Get T1 and T2 relaxation times by label over time.

    Args:
        concentrations_file (str): file containing contrast agent concentrations for injected tissue over time [mM].
        tissues_parameters (dict): dictionary of parameters for each tissue. 
        r1 (float, dict, optional): T1 relaxation time. Defaults to 4.1e-3 [s^-1.mM^-1].
        r2 (float, dict, optional): T2 relaxation time. Defaults to 4.6e-3 [s^-1.mM^-1].

    Raises:
        ValueError: if field strength is different than 1.5 or 3

    Returns:
        dict: map each tissue ID to the T1 relaxation times over time [ms].
        dict: map each tissue ID to the T2 relaxation times over time [ms].
        np.ndarray: shape (time_frames): timing of each concentration after the injection [s].
    """
    tissues_ID = tissues_parameters.keys()
    tissues_ID_injected=[ tissue_ID for tissue_ID, tissue_param in tissues_parameters.items() 
                         if any(sub in tissue_param['description'] for sub in ['myo', 'bldp', 'infarct']) ]
    
    # Initialize relaxation times 
    concentrations, times_post = get_concentration_dict(concentrations_file, tissues_ID_injected, tissues_parameters)
    time_frames = len(concentrations['1'])
    nb_labels = len(tissues_ID)
    T1_over_time = {tissue_ID: np.zeros(time_frames) for tissue_ID in tissues_ID}
    T2_over_time = {tissue_ID: np.zeros(time_frames) for tissue_ID in tissues_ID}

    
    # Convert Inputs
    if isinstance(r1, float):
        r1 = {tissue_ID:r1 for tissue_ID in tissues_ID}
    if isinstance(r2, float):
        r2 = {tissue_ID:r2 for tissue_ID in tissues_ID}    
    
    for tissue_ID in tissues_ID: 
        # tissues with contrast agent
        if tissue_ID in tissues_ID_injected:
            for t_idx, c in enumerate(concentrations[tissue_ID]):
                T1_over_time[tissue_ID][t_idx] = compute_current_relaxation(tissues_parameters[tissue_ID]['T1'], r1[tissue_ID], c)
                T2_over_time[tissue_ID][t_idx] = compute_current_relaxation(tissues_parameters[tissue_ID]['T2'], r2[tissue_ID], c)
        # native tissues
        else:
            T1_over_time[tissue_ID] = np.repeat( np.expand_dims(tissues_parameters[tissue_ID]['T1'], axis=0), time_frames, axis=0)
            T2_over_time[tissue_ID] = np.repeat( np.expand_dims(tissues_parameters[tissue_ID]['T2'], axis=0), time_frames, axis=0)
    
    return T1_over_time, T2_over_time, times_post            


def get_concentration_dict(concentrations_file, tissues_ID_injected, tissues_data):
    concentration_tissues = {}
    c_art, c_myo, c_inf, times_post = np.load(concentrations_file).values()
    for tissue_ID in tissues_ID_injected:
        if 'bldp' in tissues_data[tissue_ID]['description'].lower():
            concentration_tissues[tissue_ID] = c_art
        elif 'infarct' in tissues_data[tissue_ID]['description'].lower():
            concentration_tissues[tissue_ID] = c_inf
        elif 'myo' in tissues_data[tissue_ID]['description'].lower():
            concentration_tissues[tissue_ID] = c_myo
    return concentration_tissues, times_post


def compute_current_relaxation(T_native, relaxivity, concentration, time_unit='ms'):
    """Compute the current relaxation from a native with the following formula:
        1/T_post = 1/T_native + relaxivity * concentration

    Args:
        T_native (float): native relaxation time.
        relaxivity (float): relaxivity of the contrast agent [s^-1.mM^-1].
        concentration (float):  contrast agent concentration [mM].
        relaxivity (float): relaxivity of the contrast agent [s^-1.mM^-1].

    Returns:
        float: relaxation time after injection.
    """
    assert time_unit in ['ms', 's'], "time_unit should be either 'ms' or 's'"
    time_factor = 1e-3 if time_unit=='ms' else 1
    T_native *= time_factor
    T_post = 1 / (1 / T_native + relaxivity * concentration)
    T_post /= time_factor
    return T_post
