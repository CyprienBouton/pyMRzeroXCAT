import os
import math
import numpy as np
import tkinter as tk
from tkinter import filedialog

def LGEpar(MRX, filename=None):
    """
    This function is the parameter file for MRXCAT_PHANTOM_LGE.
    Change parameters in section "MRXCAT settings" only.
    Note: Not all combinations of any parameter values are possible.
          Some parameter changes require changes in the XCAT files.
          E.g., if you want to change orientation of your phantom, you
          need to change RotationXYZ parameter, but you also need to
          create new XCAT masks (*.bin files).
    """

    # --------------------------------------------------------------------
    #   MRXCAT settings
    # Values from  
    #  - (1) 10.1016/j.mri.2016.08.021
    #  - (2) 10.1002/mrm.20605
    #  - (3) LGE_CMRI_Simulation/@MRXCAT_CMR_LGE repo
    # --------------------------------------------------------------------
    RhoMuscle = 80.0       # Proton density muscle [%]
    RhoFat = 70.0          # Proton density fat [%]
    RhoBlood = 95.0        # Proton density blood [%]
    RhoLiver = 90.0        # Proton density liver [%]
    RhoBone = 12.0         # Proton density bone [%]

    T1Muscle = 1100.0       # T1 muscle [ms] modified with (1)
    T1Myocardium = 1160.0  # T1 myocardium [ms] modified with (1)
    T1Fat = 350.0          # T1 fat [ms] modified with (1)
    T1Blood = 1550.0       # T1 blood [ms] modified with (2)
    T1Liver = 750.0        # T1 liver [ms] modified with (1)
    T1Bone = 250.0         # T1 bone [ms]

    T2Muscle = 40.0       # T2 muscle [ms] modified with (1)
    T2Myocardium = 45.0  # T2 myocardium [ms] modified with (1)
    T2Fat = 125.0          # T2 fat [ms] modified with (1)
    T2Blood = 275.0       # T2 blood [ms] modified with (2)
    T2Liver = 30.0        # T2 liver [ms] modified with (1)
    T2Bone = 60.0         # T2 bone [ms] modified with (3)

    T1Relaxivity = 4.1 / 1000  # Gd T1 relaxivity [l/(mmol*ms)] modified with (3)
    T2Relaxivity = 4.6 / 1000  # Gd T2 relaxivity [l/(mmol*ms)] modified with (3)
    MBFrest = 1.0 / 60      # Rest MBF [ml/g/s]
    MBFstress = 3.5 / 60    # Stress MBF [ml/g/s]
    Fermi_alpha = 0.25      # Fermi model parameter alpha
    Fermi_beta = 0.25       # Fermi model parameter beta
    Tshift = 3.0            # Temporal LV-myo shift [s]

    TR = 2.0               # Repetition time [ms]
    Trrc = 1.0             # Duration r-r cycle [s]
    Flip = 15.0            # Flip [deg]
    Tsat = 150.0           # Saturation delay [ms]
    Nky0 = max(np.floor((Tsat-70)/TR),1)  # Number excitations to ky0 (70 ms: eff. prepulse delay)
    Frames = 32            # Number of dynamics (default: 32)
    
    BoundingBox = np.array([[0.0, 0.7], [0.2, 0.8], [0.0, 1.0]])  # BoundingBox in relative units
    RotationXYZ = np.array([115.0, 35.0, 240.0])  # Rotation angles around x, y, z [deg]

    CropProfs = 0          # Crop profiles along x around heart (Recon time)
    LowPassFilt = 0        # Low-pass filter images
    FilterStr = 1.2        # Low-pass filter strength (default: 1.2)

    RespMotion = 0         # 0=no motion; 1=resp motion
    RestStress = 2         # 1=rest; 2=stress
    Dose = 0.075           # [mmol/kg b.w.]
    SNR = 30               # Signal-to-noise ratio (CNR in LGE case!)
    Coils = 4              # Number of coils (Biot-Savart)
    CoilDist = 350         # Body radius [mm] = distance of coil centers from origin
    CoilsPerRow = 8        # Number of coils on 1 "ring" or row of coil array (default: 8)

    Trajectory = 'Cartesian'  # K-space trajectory (Cartesian)
    Undersample = 1           # Undersampling factor (not for Cartesian right now)

    # Display title
    print('------------------------------------------')
    print(f'        MRXCAT-CMR-LGE (VER {MRX.Version})      ')
    print('------------------------------------------')

    # Open window, select file
    if filename is None or not os.path.exists(filename):
        root = tk.Tk()
        root.withdraw()

        filename = filedialog.askopenfilename()
        print(filename)

    # Generate XCAT2 *.bin files if needed
    if filename.endswith('par'):
        raise AssertionError('you must provide bin files for LGE mode')
    
    # Read log file
    MRX.Filename = os.path.join(os.getcwd(), filename)
    MRX.read_log_file()
    aif005 = np.array(
        [0, 0, 0, 0, 0.0009, 0.0178, 0.1343, 0.5466, 1.4530, 2.8276, 
         4.3365, 5.5112, 6.0165, 5.7934, 5.0205, 3.9772, 2.9161, 1.9988, 
         1.2913, 0.7916, 0.4632, 0.2598, 0.1404, 0.0733, 0.0371, 0.0182, 
         0.0087, 0.0041, 0.0019, 0.0008, 0.0004, 0.0002, 0.0001, 0.0000, 
         0.0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
    )

    # Store tissue, contrast, and sequence parameters
    MRX.Par['tissue'] = {
        'rhomuscle': RhoMuscle,
        'rhofat': RhoFat,
        'rhoblood': RhoBlood,
        'rholiver': RhoLiver,
        'rhobone': RhoBone,
        't1muscle': T1Muscle,
        't1myocardium': T1Myocardium,
        't1fat': T1Fat,
        't1blood': T1Blood,
        't1liver': T1Liver,
        't1bone': T1Bone,
        't2muscle': T2Muscle,
        't2myocardium': T2Myocardium,
        't2fat': T2Fat,
        't2blood': T2Blood,
        't2liver': T2Liver,
        't2bone': T2Bone,
    }

    MRX.Par['contrast'] = {
        'r1': T1Relaxivity,
        'r2': T2Relaxivity,
        'qr': MBFrest,
        'qs': MBFstress,
        'rs': RestStress,
        'falpha': Fermi_alpha,
        'fbeta': Fermi_beta,
        'tshift': Tshift,
        'dose': Dose,
        'aif': aif005
    }

    MRX.Par['scan'].update({
        'trep': TR,
        'trrc': Trrc,
        'flip': math.pi * Flip / 180,
        'tsat': Tsat,
        'nky0': Nky0,
        'bbox': BoundingBox,
        'crop': CropProfs,
        'lowpass': LowPassFilt,
        'lowpass_str': FilterStr,
        'resp': RespMotion,
        'snr': SNR,
        'coils': Coils,
        'coildist': CoilDist,
        'coilsperrow': CoilsPerRow,
        'trajectory': Trajectory,
        'undersample': Undersample,
        'rotation': np.pi * np.array(RotationXYZ) / 180,
    })
    
    if Frames > 0:  # Only overwrite Par.scan.frames if Frames != 0
        MRX.Par['scan']['frames'] = Frames
    else:
        MRX.Par['scan']['frames'] = MRX.Par['scan']['frames_xcat']

    MRX.Par['scan']['trajectory'] = Trajectory
    MRX.Par['scan']['undersample'] = Undersample

    # Error checks
    if MRX.Par['scan']['frames'] % 1 != 0:  # Check if #frames is a whole number
        raise ValueError("Number of frames must be an integer value!")

    print("LGEpar function completed successfully.")
