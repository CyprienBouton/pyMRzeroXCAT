
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import convolve

from pymrxcat.mrxcat import MRXCAT
from pymrxcat.MRXCAT_PHANTOM_LGE.lge_par import LGEpar


def img_to_matlab_style(img: np.ndarray, channel_axis=0):
    """Convert image from NumPy format to MATLAB-style format.
    Handles both 3D and 4D images, with flexible channel axis.
    
    Args:
    - img (np.ndarray): The input image (3D or 4D NumPy array).
    - channel_axis (int): The axis corresponding to channels in a 4D image (default=0).
    
    Returns:
    - np.ndarray: The converted image in MATLAB-style format (with shape adjusted).
    """
    if img.ndim==3:
        return np.flip(img,0).transpose(1,0,2)
    else:
        img = np.moveaxis(img, channel_axis, 0)
        return np.flip(img,1).transpose(0,2,1,3)


class MRXCAT_PHANTOM_LGE(MRXCAT):
    def __init__(self, filename='', *args):
        super().__init__()

        LGEpar(self, filename)

        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg, str):
                if arg.lower() == 'snr':
                    self.Par['scan']['snr'] = args[i + 1]
                elif arg.lower() == 'dose':
                    self.Par['contrast']['dose'] = args[i + 1]
                elif arg.lower() == 'frames':
                    self.Par['scan']['frames'] = args[i + 1]
                elif arg.lower() == 'coils':
                    self.Par['scan']['coils'] = args[i + 1]
                elif arg.lower() == 'tshift':
                    self.Par['contrast']['tshift'] = args[i + 1]
                elif arg.lower() == 'flip':
                    self.Par['scan']['flip'] = np.pi / 180 * args[i + 1]
                elif arg.lower() == 'crop':
                    self.Par['scan']['crop'] = args[i + 1]
                elif arg.lower() == 'demo_gui':
                    par = args[i + 1]
                    for k, v in par['contrast'].items():
                        self.Par['contrast'][k] = v
                    for k, v in par['scan'].items():
                        self.Par['scan'][k] = v

        ca, cm = self.compute_dynamic_conc()
        sen = self.calculate_coil_maps()

        st = 0
        t1_maps, t2_maps = [], []
        for t in range(self.Par['scan']['frames']):
            data = self.read_img_data(t).astype(np.float32)
            if t==0:
                rho_map, msk = self.get_rho_map(data)
            self.update_contrast_conc(t, ca, cm)
            t1_map, t2_map = self.get_t1_t2_map(data)
            t1_maps.append(t1_map)
            t2_maps.append(t2_map)
        
        self.save_phantom_data(msk, rho_map, t1_maps, t2_maps, sen)
        self.save_pars_for_recon(data)

    
    def compute_dynamic_conc(self):
        """
        Compute Gd concentration at input and ROI as a function of time.

        Args:
            MRX: MRXCAT object containing scan parameters and contrast information.

        Returns:
            tuple: 
                - ca (numpy.ndarray): Arterial input concentration at the dose specified in Par.contrast.dose [mmol/l].
                - cm (numpy.ndarray): Myocardial tissue concentration [mmol/l].

        Notes:
            Tissue density is not used here because the conversion from dose [mmol/kg] 
            to c_Gd [mmol/ml] is done implicitly. We inject dose*b.w./c_Gd [ml] 
            of contrast medium, i.e., b.w. drops out in the calculation.

            The calculation steps involve:
                - Estimating the dilution factor based on stroke volume and ejection fraction.
                - Normalizing the AIF (arterial input function).
                - Calculating the myocardial concentration by convolving the AIF with the impulse residue function (IRF).
        """

        # Crop population average AIF
        aif005 = self.Par["contrast"]["aif"][:self.Par["scan"]["frames"]]

        # Convert to absolute contrast agent concentration using assumptions:
        # - Pure Gadovist: c_Gd = 1.0 mmol/ml = 1000 mmol/l
        # - c = c_Gd * dilution factor d
        # - Estimation of dilution factor d ~ 12% (approximation)
        aif01 = aif005 * 12 / np.max(aif005)
        ca = aif01 * self.Par["contrast"]["dose"] / 0.1  # Scale to desired dose

        # Upsample AIF to "pseudo-continuous" AIF
        t = np.linspace(0, (self.Par["scan"]["frames"] - 1) * self.Par["scan"]["trrc"], len(ca))
        tinf = np.arange(0, t[-1], 1 / 100)  # Upsample by a factor of 100
        cainf = np.interp(tinf, t, ca, left=None, right=None)

        # Scale flow and calculate impulse residue function
        qfl = self.Par["contrast"]["qr"]  # Rest flow
        if self.Par["contrast"]["rs"] > 1:
            qfl = self.Par["contrast"]["qs"]  # Stress flow
        qfl *= len(ca) / round(self.Par["scan"]["frames"] * self.Par["scan"]["trrc"])
        irf = self.fermi_function(tinf)  # Impulse residue function

        # Calculate myocardial concentration, apply Tshift and downsample
        cm = MRXCAT_PHANTOM_LGE.convolve(tinf, cainf, irf, qfl)  # Myocardial concentration
        cm = self.shift_aif(tinf, cm)  # Apply Tshift
        ca = np.interp(t, tinf, cainf)  # Downsample AIF (discrete measurement)
        cm = np.interp(t, tinf, cm)    # Downsample MYO concentration

        return ca, cm
    
        
    def update_contrast_conc(self, t, ca, cm):
        """
        Update contrast concentration in the right atrium (ra), right ventricle (rv),
        left atrium (la), and left ventricle (lv) at specific time t for blood 
        pool and myocardium.

        Args:
            t (int): Time step
            ca (numpy.ndarray): Contrast agent concentrations for arterial input function (AIF)
            cm (numpy.ndarray): Contrast agent concentrations for myocardial tissue

        This method is used during the loop over time frames when creating the phantom.
        """

        # Calculate indices for different compartments based on time step `t`
        ra = min(round((t + 3) * len(ca) / self.Par["scan"]["frames"]), len(ca)-1)
        rv = min(round((t + 2) * len(ca) / self.Par["scan"]["frames"]), len(ca)-1)
        la = min(round((t + 1) * len(ca) / self.Par["scan"]["frames"]), len(ca)-1)
        lv = min(round(t * len(ca) / self.Par["scan"]["frames"]), len(ca)-1)

        # Update the contrast concentrations for the various blood pools and myocardium
        self.Par["contrast"]["ca"]['ra'] = ca[ra]  # RA blood pool
        self.Par["contrast"]["ca"]['rv'] = ca[rv]  # RV blood pool
        self.Par["contrast"]["ca"]['la'] = ca[la]  # LA blood pool
        self.Par["contrast"]["ca"]['lv'] = ca[lv]  # LV blood pool

        self.Par["contrast"]["cm"]['ra'] = cm[ra]  # RA myocardium
        self.Par["contrast"]["cm"]['rv'] = cm[rv]  # RV myocardium
        self.Par["contrast"]["cm"]['la'] = cm[la]  # LA myocardium
        self.Par["contrast"]["cm"]['lv'] = cm[lv]  # LV myocardium


    def fermi_function(self, t):
        """
        Calculate the Fermi function based on alpha, beta, and time info.

        Args:
            t (numpy.ndarray): Time array or a single time value

        Returns:
            numpy.ndarray: Computed Fermi function for each time value

        Notes:
            This function is based on the formula:
                h(t) = (1 + beta) / (1 + beta * exp(alpha * t))
        """
        alpha = self.Par["contrast"]["falpha"]
        beta = self.Par["contrast"]["fbeta"]
        h = (1 + beta) / (1 + beta * np.exp(alpha * t))
        return h


    def shift_aif(MRX, t, aif):
        # Extract the time shift value dT from MRX
        dT = MRX.Par['contrast']['tshift']
        # Create the interpolator for the AIF with time shift dT
        interpolator = PchipInterpolator(t + dT, aif, extrapolate=True)
        # Interpolate AIF at the original time points
        y = interpolator(t)
        return y

    def get_rho_map(self, data):
        """Calculate proton density map using tissue parameters.

        Args:
            data (numpy.ndarray): Data array containing tissue types represented by integers.

        Returns:
            np.ndarray: rho density map.
        """
        act = list(self.Par["act"].values())  # Tissue activity values
        tis = list(self.Par["act"].keys())    # Tissue type names
        rho_map = np.zeros_like(data, dtype=np.float32)  # Initialize the proton density map
        msk = np.zeros_like(data, dtype=np.uint8)    # Initialize the mask
        
        for i in range(len(act)):
            # Select tissue type
            tissue_type = tis[i]

            # Tissue property assignment based on tissue type
            if tissue_type == 'myoLV_act':
                rho = self.Par["tissue"]["rhomuscle"]
            elif tissue_type == 'myoRV_act':
                rho = self.Par["tissue"]["rhomuscle"]
            elif tissue_type == 'myoLA_act':
                rho = self.Par["tissue"]["rhomuscle"]
            elif tissue_type == 'myoRA_act':
                rho = self.Par["tissue"]["rhomuscle"]
            elif tissue_type == 'bldplLV_act':
                rho = self.Par["tissue"]["rhoblood"]
            elif tissue_type in ['bldplRV_act', 'art_activity', 'vein_activity']:
                rho = self.Par["tissue"]["rhoblood"]
            elif tissue_type == 'bldplLA_act':
                rho = self.Par["tissue"]["rhoblood"]
            elif tissue_type == 'bldplRA_act':
                rho = self.Par["tissue"]["rhoblood"]
            elif tissue_type in ['body_activity', 'pericardium_activity']:
                rho = self.Par["tissue"]["rhofat"]
            elif tissue_type == 'muscle_activity':
                rho = self.Par["tissue"]["rhomuscle"]
            elif tissue_type == 'liver_activity':
                rho = self.Par["tissue"]["rholiver"]
            elif tissue_type in ['rib_activity', 'cortical_bone_activity', 'spine_activity', 'bone_marrow_activity']:
                rho = self.Par["tissue"]["rhobone"]
            else:
                rho = 0
                
            rho_map[data == act[i]] = rho
            msk[data == act[i]] = act[i]
        
        return rho_map, msk
    
        
    def get_t1_t2_map(self, data):
        """ Calculate t1 map and t2 map using tissue and contrast parameters.

        Args:
            data (numpy.ndarray): Data array containing tissue types represented by integers.

        Returns:
            tuple:
            - t1_map (numpy.ndarray): T1 relaxation times map.
            - t2_map (numpy.ndarray): T2 relaxation times map.
        """
        
        act = list(self.Par["act"].values())  # Tissue activity values
        tis = list(self.Par["act"].keys())    # Tissue type names
        t1_map = np.zeros_like(data, dtype=np.float32)  # Initialize the T1 map
        t2_map = np.zeros_like(data, dtype=np.float32)  # Initialize the T2 map

        for i in range(len(act)):
            # Select tissue type
            tissue_type = tis[i]
            
            if tissue_type == 'myoLV_act':
                r1 = 1 / self.Par["tissue"]["t1myocardium"] + self.Par["contrast"]["cm"]["lv"] * self.Par["contrast"]["r1"]
                r2 = 1 / self.Par["tissue"]["t2myocardium"] + self.Par["contrast"]["cm"]["lv"] * self.Par["contrast"]["r2"]
            elif tissue_type == 'myoRV_act':
                r1 = 1 / self.Par["tissue"]["t1myocardium"] + self.Par["contrast"]["cm"]["rv"] * self.Par["contrast"]["r1"]
                r2 = 1 / self.Par["tissue"]["t2myocardium"] + self.Par["contrast"]["cm"]["rv"] * self.Par["contrast"]["r2"]
            elif tissue_type == 'myoLA_act':
                r1 = 1 / self.Par["tissue"]["t1myocardium"] + self.Par["contrast"]["cm"]["la"] * self.Par["contrast"]["r1"]
                r2 = 1 / self.Par["tissue"]["t2myocardium"] + self.Par["contrast"]["cm"]["la"] * self.Par["contrast"]["r2"]
            elif tissue_type == 'myoRA_act':
                r1 = 1 / self.Par["tissue"]["t1myocardium"] + self.Par["contrast"]["cm"]["ra"] * self.Par["contrast"]["r1"]
                r2 = 1 / self.Par["tissue"]["t2myocardium"] + self.Par["contrast"]["cm"]["ra"] * self.Par["contrast"]["r2"]
            elif tissue_type == 'bldplLV_act':
                r1 = 1 / self.Par["tissue"]["t1blood"] + self.Par["contrast"]["ca"]["lv"] * self.Par["contrast"]["r1"]
                r2 = 1 / self.Par["tissue"]["t2blood"] + self.Par["contrast"]["ca"]["lv"] * self.Par["contrast"]["r2"]
            elif tissue_type in ['bldplRV_act', 'art_activity', 'vein_activity']:
                r1 = 1 / self.Par["tissue"]["t1blood"] + self.Par["contrast"]["ca"]["rv"] * self.Par["contrast"]["r1"]
                r2 = 1 / self.Par["tissue"]["t2blood"] + self.Par["contrast"]["ca"]["rv"] * self.Par["contrast"]["r2"]
            elif tissue_type == 'bldplLA_act':
                r1 = 1 / self.Par["tissue"]["t1blood"] + self.Par["contrast"]["ca"]["la"] * self.Par["contrast"]["r1"]
                r2 = 1 / self.Par["tissue"]["t2blood"] + self.Par["contrast"]["ca"]["la"] * self.Par["contrast"]["r2"]
            elif tissue_type == 'bldplRA_act':
                r1 = 1 / self.Par["tissue"]["t1blood"] + self.Par["contrast"]["ca"]["ra"] * self.Par["contrast"]["r1"]
                r2 = 1 / self.Par["tissue"]["t2blood"] + self.Par["contrast"]["ca"]["ra"] * self.Par["contrast"]["r2"]
            elif tissue_type in ['body_activity', 'pericardium_activity']:
                r1 = 1 / self.Par["tissue"]["t1fat"]
                r2 = 1 / self.Par["tissue"]["t2fat"]
            elif tissue_type == 'muscle_activity':
                r1 = 1 / self.Par["tissue"]["t1muscle"]
                r2 = 1 / self.Par["tissue"]["t2muscle"]
            elif tissue_type == 'liver_activity':
                r1 = 1 / self.Par["tissue"]["t1liver"]
                r2 = 1 / self.Par["tissue"]["t2liver"]
            elif tissue_type in ['rib_activity', 'cortical_bone_activity', 'spine_activity', 'bone_marrow_activity']:
                r1 = 1 / self.Par["tissue"]["t1bone"]
                r2 = 1 / self.Par["tissue"]["t2bone"]
            else:
                # Or maybe np.nan if you want to mark unknowns
                r1 = 0  
                r2 = 0 

            # Update t1 map
            t1_map[data == act[i]] = 1 / r1 if r1 > 0 else 0  # avoid division by zero

            # Update t2 map
            t2_map[data == act[i]] = 1 / r2 if r2 > 0 else 0  # avoid division by zero

        return t1_map, t2_map

    def save_phantom_data(self, msk, rho_map, t1_maps, t2_maps, sen):
        """Save phantom data in a .npz file

        Args:
            msk (np.ndarray): tissues mask
            rho_map (np.ndarray): proton density map
            t1_maps (list): t1_maps for all frames
            t2_maps (list): t2_maps for all frames
            sen (np.array): coil sensitivity maps
        """
        fname = self.generate_filename(msk)
        fphantom = f"{fname}_phantom.npz"
        # one frame for each RR cycle
        time_points = np.arange(self.Par['scan']['frames'])*self.Par['scan']['trrc']
        # get tissues mask
        tissues = {}
        for tis, act in self.Par['act'].items():
            tissues['tissue_'+tis] = img_to_matlab_style(msk==act)
        T1_map = np.stack(t1_maps)*1e-3 # convert values to seconds
        T2_map = np.stack(t2_maps)*1e-3 # convert values to seconds
        np.savez_compressed(
            fphantom,
            PD_map=img_to_matlab_style(rho_map),
            T1_map=img_to_matlab_style(T1_map),
            T2_map=img_to_matlab_style(T2_map),
            coil_sens=img_to_matlab_style(sen,-1),
            time_points=time_points,
            **tissues
        )

    @staticmethod
    def convolve(t, c, h, q):
        dt = (t[-1] - t[0]) / (len(t) - 1)
        c_result = convolve(c, h, mode='same') * dt
        
        # Scale the result by factor q
        c_result = q * c_result
        return c_result
    
    
def main():
    MRXCAT_PHANTOM_LGE()


if __name__ == "__main__":
    main()