import re
import numpy as np


def search_key_log(log_file, key):
    """
    Search for a key in the log file and return its value.
    
    Arguments:
        log_file (str): Path to the log file.
        key (str): key that will be search in log file.
        
    Returns:
        value (str, int, float) matching the key in the log file.
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


def get_segmentation(bin_file, log_file, flip_horizontal=True, swap_xy=False):
    """
    Load and reshape a segmentation from a binary file.

    This function reads a binary segmentation file and reshapes it into a 
    3D NumPy array using the array size specified in the log file. It can also 
    flip the segmentation horizontally and optionally swap the first two axes 
    (X and Y).

    Arguments:
        bin_file (str): Path to the binary segmentation file.
        log_file (str): Path to the log file containing the 'array_size' key.
        flip_horizontal (bool, optional): If True, flips the segmentation 
            horizontally (axis=1). Defaults to True.
        swap_xy (bool, optional): If True, swaps the first two axes 
            (axis 0 and 1). Useful if the image appears rotated. Defaults to False.

    Returns:
        np.ndarray: A 3D NumPy array of shape (H, W, Z) or (W, H, Z) depending 
                    on `swap_xy`, with dtype int32.
    """
    matrix = search_key_log(log_file, 'array_size')
    seg = np.fromfile(bin_file, dtype=np.float32).astype(np.int32)
    seg = seg.reshape(-1, matrix, matrix).transpose(1, 2, 0)  # shape (H, W, Z)
    
    if flip_horizontal:
        seg = np.flip(seg, axis=1)  # flip left-right (axis=1)
    if swap_xy:
        seg = seg.swapaxes(0, 1)  # swap H and W axes

    return seg


def save_segmentation(seg, bin_file, log_file, flip_horizontal=True, swap_xy=False):
    """
    Save a 3D segmentation array to a binary file, with optional axis flipping or swapping.

    This function performs the inverse of `get_segmentation()`. It reshapes and optionally
    flips and/or swaps axes of a 3D segmentation array before saving it as a binary file
    compatible with MRXCAT-style input.

    Args:
        seg (np.ndarray): 3D NumPy array of shape (H, W, Z) or (W, H, Z) with int32 or uint8 type.
        bin_file (str): Path where the output binary file will be saved.
        log_file (str): Path to the log file containing the 'array_size' key.
        flip_horizontal (bool, optional): If True, flips the array horizontally (axis=1). Defaults to True.
        swap_xy (bool, optional): If True, swaps axis 0 and 1 before saving. Defaults to False.

    Raises:
        ValueError: If the array dimensions do not match the expected 'array_size' in the log file.
    """
    matrix = search_key_log(log_file, 'array_size')

    data = seg.copy()

    if swap_xy:
        data = data.swapaxes(0, 1)  # reverse swap from get_segmentation

    if flip_horizontal:
        data = np.flip(data, axis=1)

    if data.shape[0] != matrix or data.shape[1] != matrix:
        raise ValueError(
            f"Segmentation shape {data.shape[:2]} does not match expected array_size={matrix} from log."
        )

    data = data.transpose(2, 0, 1)  # back to (Z, H, W)
    data.astype(np.float32).tofile(bin_file)
    print(f"✅ Segmentation saved to: {bin_file}")



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


def get_resolution(log_file):
    """
    Get the resolution from the log file.
    
    Arguments:
        log_file (str): Path to the log file.
    
    Returns
        (np.array) resolution
    """
    rx_cm = search_key_log(log_file, "pixel width")
    rz_cm = search_key_log(log_file, "slice width")
    return np.array([rx_cm, rx_cm, rz_cm])*10  # Convert to mm