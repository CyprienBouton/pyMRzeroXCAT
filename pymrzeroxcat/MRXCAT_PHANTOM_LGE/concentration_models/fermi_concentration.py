######################
# IMPORTS
######################


import argparse
import numpy as np
from scipy.signal import convolve
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt


######################
# CONSTANTS
######################


# arterial input function with half-dose (0.05 mmol/kg) contrast agent on 6 volunteers, 
# modeled using gamma-variate fit for first-pass dynamics. From MRXCAT
AIF005 = np.array(
    [0, 0, 0, 0, 0.0009, 0.0178, 0.1343, 0.5466, 1.4530, 2.8276, 
        4.3365, 5.5112, 6.0165, 5.7934, 5.0205, 3.9772, 2.9161, 1.9988, 
        1.2913, 0.7916, 0.4632, 0.2598, 0.1404, 0.0733, 0.0371, 0.0182, 
        0.0087, 0.0041, 0.0019, 0.0008, 0.0004, 0.0002, 0.0001, 0.0000, 
        0.0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
)

MBF_REST = 1.0 / 60 # myocardial blood flow [ml/g/s]
MBF_STRESS = 3.5 / 60 # myocardial blood flow [ml/g/s]


######################
# USEFUL FUNCTIONS
######################


def fermi_function(t, tau, k):
    """
    Calculate the Fermi function based on tau, k, and time info.

    Args:
        t (numpy.ndarray): Time input array.
        tau (float, optional): Fermi model parameter tau.
        k (float, optional): Fermi model parameter k.

    Returns:
            (numpy.ndarray) impulse residue function
    """
    return 1 / ( 1 + np.exp((t-tau)*k) )


def convolve_concentration_impulse_residue_function(
    t_upsample, 
    arterial_concentration, 
    impulse_residue_function, 
    myocardial_blood_flow,
):
    """Convolve concentration with impulse residue function

    Args:
        t_upsample (np.ndarray): time input
        arterial_concentration (np.ndarray): arterial concentration [mM].
        impulse_residue_function (np.ndarray): impulse residue function.
        myocardial_blood_flow (float): myocardial blood flow [ml/g/s].

    Returns:
        (np.ndarray): myocardial concentration [mM].
    """
    dt = (t_upsample[-1] - t_upsample[0]) / (len(t_upsample) - 1)
    c = arterial_concentration
    q = myocardial_blood_flow

    # Create convolution matrix (like a Toeplitz matrix)
    c_ = np.zeros((len(c), len(c)))
    for i in range(len(c)):
        for j in range(len(c)):
            if i - j + 1 > 0:
                c_[i, j] = c[i - j]

    # Ensure h is a column vector
    h = np.array(impulse_residue_function).reshape(-1, 1)
    
    # Perform matrix multiplication and scale
    result = q * c_ @ h * dt
    return result.flatten()


def shift_concentration(t, concentration, t_shift):
    """Shift concentration

    Args:
        t (np.ndarray): time inputs.
        concentration (np.ndarray): concentration [mM]
        t_shift (float): time shift (positive).

    Returns:
        np.ndarray: shifted concentration.
    """
    # Create the interpolator for the concentration with t_shift
    interpolator = PchipInterpolator(t + t_shift, concentration, extrapolate=True)
    # Interpolate AIF at the original time points
    y = interpolator(t)
    return y


def get_initial_concentration(T_post, T_native, relaxivity, time_unit='ms'):
    """Get initial concentration from two relaxation times and a given relaxivity.

    Args:
        T_post (float): relaxation time after injection.
        T_native (float): native relaxation time.
        relaxivity (float): relaxivity of the contrast agent [s^-1.mM^-1].
        time_unit (str, optional): time unit used for relaxation times. Defaults to 'ms'.

    Returns:
        float: contrast agent concentration [mM].
    """
    assert T_post < T_native, "You must have T_post < T_native"
    assert time_unit in ['ms', 's'], "time_unit should be either 'ms' or 's'"
    time_factor = 1e-3 if time_unit=='ms' else 1
    T_native *= time_factor
    T_post *= time_factor
    return 1/relaxivity * (1/T_post-1/T_native)


######################
# MAIN FUNCTIONS
######################


def get_concentration_fermi(times_post, tau, k, contrast_dose=0.075, is_stress=False, time_step=0.2, lv_myo_shift=3):
    """Get contrast agent tissue concentration from fermi model.

    Args:
        times_post (np.ndarray): timing of each concentration after the injection [s].
        tau (float): Fermi model parameter tau.
        k (float): Fermi model parameter k.
        contrast_dose (float, optional): contrast dose input Defaults to 0.075 [mmol/kg].
        is_stress (bool, optional): Whether the patient is on stress. Defaults to False.
        time_step (float): time_step. Default to 0.2 [s].
        lv_myo_shift (float): time shift between left ventricle and myocardium. Default to 3 [s].

    Returns:
        np.ndarray: arterial concentration over time [mM]
        np.ndarray: tissue concentration over time [mM]
        
    Estimate max blood concentration of contrast agent (Gadovist):
    - Pure Gadovist: c_Gd = 1.0 mmol/ml = 1000 mmol/l
    - Concentration c = c_Gd * d, where d = dilution factor
    - d estimated from:
        * Stroke volume ≈ 80–100 ml, ejection fraction ≈ 80% → ~75 ml effective
        * AIF normalized peak ≈ 12% → d ≈ 0.12
    - Dose example: 75 kg × 0.1 mmol/kg → 7.5 ml Gd injected
    - Per beat: 12% of 7.5 ml = 0.9 ml Gd in 75 ml blood → d = 0.012
    - Final concentration: c = 1000 × 0.012 = 12 mmol/l
    """
    # Inputs
    t = np.arange(0, len(AIF005)) # aif is given every seconds
    myocardial_blood_flow = MBF_STRESS if is_stress else MBF_REST
    
    # Get arterial concentration
    aif01 = AIF005 * 12 / np.max(AIF005) # Final maximum concentration is 12 mM for a dose of 0.1 mmol/kg
    arterial_concentration = aif01*contrast_dose/0.1;   # scale to desired dose
    
    # Upsample AIF by a factor 100
    t_upsample = np.arange(0, times_post[-1], time_step)
    arterial_concentration_upsample = np.interp(t_upsample, t, arterial_concentration)
    
    # Compute impulse_residue_function with fermi function
    impulse_residue_function = fermi_function(t_upsample, tau, k)
    
    # Convolve with the arterial concentration 
    myocardial_concentration_upsample = convolve_concentration_impulse_residue_function(
        t_upsample, 
        arterial_concentration_upsample, 
        impulse_residue_function, 
        myocardial_blood_flow,
    )
    
    # Shift myocardial concentration
    myocardial_concentration_upsample = shift_concentration(t_upsample, myocardial_concentration_upsample, lv_myo_shift)
    
    # Downscale
    arterial_concentration = np.interp(times_post, t_upsample, arterial_concentration_upsample)
    myocardial_concentration = np.interp(times_post, t_upsample, myocardial_concentration_upsample)
    
    return arterial_concentration, myocardial_concentration
    

def get_concentrations(
    times_post, 
    tau_myo, 
    k_myo, 
    tau_inf, 
    k_inf,
    lv_myo_shift=3,
    lv_inf_shift=3, 
    contrast_dose=0.075, 
    is_stress=False,
    time_step=.2,
    plot=True,
):
    """Get concentration of blood, myocardium and infarction over times  

    Args:
        times_post (np.ndarray): timing of each concentration after the injection [s].
        tau_myo (float, optional): Fermi model parameter tau for myocardium.
        k_myo (float, optional): Fermi model parameter k for myocardium.
        tau_inf (float, optional): Fermi model parameter tau for infarction.
        k_inf (float, optional): Fermi model parameter k for infarction.
        lv_myo_shift (float): time shift between left ventricle and myocardium. Default to 3 [s].
        lv_inf_shift (float): time shift between left ventricle and infarction. Default to 3 [s].
        contrast_dose (float, optional): contrast dose input Defaults to 0.075 [mmol/kg].
        is_stress (bool, optional): Whether the patient is on stress. Defaults to False.
        time_step (float, optional): time_step used to compute tissue concentration. Default to 0.2 [s].
        plot (bool, optional): Whether to plot resulting concentration. Default to True.
    
    Returns:
        (np.ndarray) contrast agent concentration in the arterial for each time step [mM].
        (np.ndarray) contrast agent concentration in the myocardium for each time step [mM].
        (np.ndarray) contrast agent concentration in the infarction for each time step [mM].
    """
    _, c_myo = get_concentration_fermi(
        times_post, tau_myo, k_myo, contrast_dose, 
        is_stress, time_step, lv_myo_shift,
    )
    c_art, c_inf = get_concentration_fermi(
        times_post, tau_inf, k_inf, contrast_dose, 
        is_stress, time_step, lv_inf_shift,
    )
    
    if plot:
        print(c_art.shape)
        plt.plot(times_post/60, c_art)
        plt.plot(times_post/60, c_myo)
        plt.plot(times_post/60, c_inf)
        plt.legend(['Blood', 'Myocardium', 'Infarction'])
        plt.xlabel('Time post-injection (minutes)', fontsize='large')
        plt.ylabel('Gadolinium concentration (mM)', fontsize='large')
        plt.show()
    return c_art, c_myo, c_inf

    
def main():
    parser = argparse.ArgumentParser(description="Compute concentrations for blood, myocardium, and infarction using Fermi model.")
    parser.add_argument('-t_start','--time_start', type=float, default=0., help='Start time after injection [s]')
    parser.add_argument('-d','--duration', type=float, default=60.*5, help='Duration of the scan [s]. Default to 5 minutes,')
    parser.add_argument('-s', '--nb_sampled', type=int, default=300 ,help='Number of sampled. Default to 100')
    parser.add_argument('--tau_myo', type=float, default=0.1, help='Fermi model tau parameter for myocardium')
    parser.add_argument('--k_myo', type=float, default=0.1, help='Fermi model k parameter for myocardium')
    parser.add_argument('--tau_inf', type=float, default=0.1, help='Fermi model tau parameter for infarction')
    parser.add_argument('--k_inf', type=float, default=0.05, help='Fermi model k parameter for infarction')
    parser.add_argument('--lv_myo_shift', type=float, default=30, help='Time shift between LV and myocardium [s].')
    parser.add_argument('--lv_inf_shift', type=float, default=40, help='Time shift between LV and infarction [s].')
    parser.add_argument('--contrast_dose', type=float, default=0.2, help='Contrast dose [mmol/kg]. Default is 0.02.')
    parser.add_argument('--is_stress', action=argparse.BooleanOptionalAction, default=True, help='Flag to indicate stress condition.')
    parser.add_argument('--time_step', type=float, default=0.2, help='Time step used to compute concentration [s]. Default is 0.2.')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=True, help='Flag to plot resulting concentrations.')
    parser.add_argument('-o', '--output_file', type=str, default='concentration_fermi_default.npz', 
                        help='Output file to save concentration 9 (.npz). Default concentration_fermi_default.npz')

    args = parser.parse_args()
    
    times_post = np.linspace(args.time_start, args.time_start+args.duration, args.nb_sampled)
    
    c_art, c_myo, c_inf = get_concentrations(
        times_post=times_post,
        tau_myo=args.tau_myo,
        k_myo=args.k_myo,
        tau_inf=args.tau_inf,
        k_inf=args.k_inf,
        lv_myo_shift=args.lv_myo_shift,
        lv_inf_shift=args.lv_inf_shift,
        contrast_dose=args.contrast_dose,
        is_stress=args.is_stress,
        time_step=args.time_step,
        plot=args.plot
    )
    assert args.output_file.endswith('.npz'), 'Output file must end with (.npz)'
    np.savez_compressed(args.output_file, c_art=c_art, c_myo=c_myo, c_inf=c_inf, times_post=times_post)


if __name__ == "__main__":
    main()