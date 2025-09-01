from ast import literal_eval
import argparse
import os
import json


# --- Constants for fallback defaults ---
DEFAULT_FOV = (300., 300., 50.)
DEFAULT_RESOLUTION = (2., 2., 5.)
DEFAULT_MATRIX = (150, 150, 10)

# --- Helpers ---
def is_close(a, b, tol=1e-3):
    return all(abs(x - y) < tol for x, y in zip(a, b))

def compute_fov(res, mat):
    return tuple(r * m for r, m in zip(res, mat))

def compute_resolution(fov, mat):
    return tuple(f / m for f, m in zip(fov, mat))

def compute_matrix(fov, res):
    return tuple(int(round(f / r)) for f, r in zip(fov, res))

def complete_imaging_args(args):
    FOV, res, mat = args.FOV, args.resolution, args.matrix

    # Try to compute missing values
    if res and mat and not FOV:
        FOV = compute_fov(res, mat)
    elif FOV and mat and not res:
        res = compute_resolution(FOV, mat)
    elif FOV and res and not mat:
        mat = compute_matrix(FOV, res)

    # Fill in defaults if still missing
    FOV = FOV or DEFAULT_FOV
    res = res or DEFAULT_RESOLUTION
    mat = mat or DEFAULT_MATRIX

    # Final consistency check (only warn/error if all 3 are set)
    expected_fov = compute_fov(res, mat)
    if not is_close(FOV, expected_fov):
        raise ValueError(f"Inconsistent FOV. Expected {expected_fov}, got {FOV}")

    # Update args
    args.FOV = FOV
    args.resolution = res
    args.matrix = mat
    return args

def parse_key_value_or_json_file(arg):
    # Handle single argument that is a potential JSON file
    print(os.path.isfile(arg))
    print(type(arg))
    if os.path.isfile(arg):
        try:
            with open(arg, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Could not parse JSON file: {e}")
    
    # Otherwise parse key=value pairs
    try:
        key, value = arg.split('=', 1)
        return {key: literal_eval(value)}
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Each argument must be key=value or a valid JSON file path. Got '{arg}'"
        )


def str_to_seconds(time_str: str) -> float:
    """Convert a time string (M or M:S) to total seconds."""
    if ':' in time_str:
        try:
            minutes, seconds = map(float, time_str.split(':'))
            return minutes * 60 + seconds
        except ValueError:
            raise ValueError(f"Invalid time format: '{time_str}'. Use M or M:S")
    else:
        try:
            return float(time_str) * 60
        except ValueError:
            raise ValueError(f"Invalid time format: '{time_str}'. Use M or M:S")