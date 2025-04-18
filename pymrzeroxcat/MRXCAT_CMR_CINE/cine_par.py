import os
import math
import numpy as np
import tkinter as tk
from tkinter import filedialog

def CINEpar(MRX, filename=None):
    """
    This function is the parameter file for MRXCAT_CMR_CINE.
    Change parameters in section "MRXCAT settings" only.
    Note: Not all combinations of any parameter values are possible.
          Some parameter changes require changes in the XCAT files.
    E.g., if you want to increase the number of segments, you need
          more XCAT heart phases for the additional segments, i.e.,
          additional *.bin files.
    """

    # --------------------------------------------------------------------
    #   MRXCAT settings
    # --------------------------------------------------------------------
    RhoMuscle = 80.0       # Proton density muscle [%]
    RhoFat = 70.0          # Proton density fat [%]
    RhoBlood = 95.0        # Proton density blood [%]
    RhoLiver = 90.0        # Proton density liver [%]
    RhoBone = 12.0         # Proton density bone [%]

    T1Muscle = 900.0       # T1 muscle [ms]
    T1Fat = 350.0          # T1 fat [ms]
    T1Blood = 1200.0       # T1 blood [ms]
    T1Liver = 800.0        # T1 liver [ms]
    T1Bone = 250.0         # T1 bone [ms]

    T2Muscle = 50.0        # T2 muscle [ms]
    T2Fat = 30.0           # T2 fat [ms]
    T2Blood = 100.0        # T2 blood [ms]
    T2Liver = 50.0         # T2 liver [ms]
    T2Bone = 20.0          # T2 bone [ms]

    TR = 3.0               # Repetition time [ms]
    TE = 1.5               # Echo time [ms]
    Flip = 60.0            # Flip [deg]
    Frames = 24            # Number of heart phases (default: 24)
    Segments = 1           # Number of segments

    BoundingBox = np.array([[0.2, 0.6], [0.3, 0.7], [0.0, 1.0]])  # BoundingBox in relative units
    RotationXYZ = np.array([115.0, 35.0, 240.0])                   # Rotations about x,y,z [deg] (default: 115/35/240)

    LowPassFilt = 1        # Low-pass filter images
    FilterStr = 1.2        # Low-pass filter strength (default: 1.2)

    SNR = 20               # Signal-to-noise ratio
    Coils = 4              # Number of coils (Biot-Savart)
    CoilDist = 450         # Body radius [mm] = distance of coil centers from origin
    CoilsPerRow = 8        # Number of coils on 1 "ring" or row of coil array (default: 8)

    Trajectory = 'Cartesian'  # K-space trajectory (Cartesian, Radial, GoldenAngle)
    Undersample = 1          # Undersampling factor (only for Radial/GoldenAngle right now)

    # Display title
    print('------------------------------------------')
    print(f'        MRXCAT-CMR-CINE (VER {MRX.Version})      ')
    print('------------------------------------------')

    # Open window, select file
    if filename is None or not os.path.exists(filename):
        root = tk.Tk()
        root.withdraw()

        filename = filedialog.askopenfilename()
        print(filename)

    # Generate XCAT2 *.bin files if needed
    if filename.endswith('par'):
        print('Generating XCAT2 bin files...')
        fname = 'cine'
        exe = 'dxcat2 '
        if os.name == 'nt':
            exe = 'dxcat2.exe '
        elif os.name == 'posix':
            exe = 'dxcat2mac '
        x, y, z = map(str, RotationXYZ)
        s = f"{exe}{filename} --phan_rotx {x} --phan_roty {y} --phan_rotz {z} {fname}"
        os.system(s)
        filename = f"{fname}_act_1.bin"
        print('Done')

    # Read log file
    MRX.Filename = os.path.join(os.getcwd(), filename)
    MRX.read_log_file()

    # Store tissue, contrast, and sequence parameters
    MRX.Par['tissue'] = {
    'rhomuscle': RhoMuscle,
    'rhofat': RhoFat,
    'rhoblood': RhoBlood,
    'rholiver': RhoLiver,
    'rhobone': RhoBone,
    't1muscle': T1Muscle,
    't1fat': T1Fat,
    't1blood': T1Blood,
    't1liver': T1Liver,
    't1bone': T1Bone,
    't2muscle': T2Muscle,
    't2fat': T2Fat,
    't2blood': T2Blood,
    't2liver': T2Liver,
    't2bone': T2Bone,
}

    # Scan parameters
    MRX.Par['scan'].update({
        'tr': TR,
        'te': TE,
        'flip': math.pi * Flip / 180,
        'segments': Segments,
        'bbox': BoundingBox,
        'lowpass': LowPassFilt,
        'lowpass_str': FilterStr,
        'snr': SNR,
        'coils': Coils,
        'coildist': CoilDist,
        'coilsperrow': CoilsPerRow,
        'rotation': np.pi * np.array(RotationXYZ) / 180,
    })

    if Frames > 0:  # Only overwrite Par.scan.frames if Frames != 0
        MRX.Par['scan']['frames'] = Frames
        xcat_segments = round(MRX.Par['scan']['scan_dur'] / MRX.Par['scan']['heartbeat_length'])
        frames_max = MRX.Par['scan']['frames_xcat'] / xcat_segments
        MRX.Par['scan']['phases'] = list(np.linspace(1, frames_max, Frames, dtype=int))
    else:
        MRX.Par['scan']['frames'] = MRX.Par['scan']['frames_xcat'] / MRX.Par['scan']['segments']
        MRX.Par['scan']['phases'] = list(range(1, MRX.Par['scan']['frames'] + 1))

    MRX.Par['scan']['trajectory'] = Trajectory
    MRX.Par['scan']['undersample'] = Undersample    # Error checks
    if MRX.Par['scan']['frames'] % 1 != 0:  # Check if #frames is a whole number
        raise ValueError("Number of frames must be an integer value. Check number of segments in CINEpar.py and number of XCAT .bin files!")
    if 'frames_max' in locals() and frames_max < MRX.Par['scan']['frames']:
        raise ValueError(f"Number of XCAT phases < desired number of phases. Set Frames <= {frames_max} in CINEpar.py")
