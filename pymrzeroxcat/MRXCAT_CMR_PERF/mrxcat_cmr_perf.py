
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import convolve

from pymrzeroxcat.mrxcat import MRXCAT
from pymrzeroxcat.MRXCAT_CMR_PERF.perf_par import PERFpar


class MRXCAT_CMR_PERF(MRXCAT):
    def __init__(self, filename='', *args):
        super().__init__()

        PERFpar(self, filename)

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
        self.Par['scan']['noisestd'] = self.compute_noise_std_dev(sen)

        st = 0
        for t in range(self.Par['scan']['frames']):
            data = self.read_img_data(t).astype(np.float32)
            self.update_contrast_conc(t, ca, cm)
            img, msk = self.map_tissue_props(data)
            img, msk = self.low_pass_filter(img, msk)
            img = self.multiply_coil_maps(img, sen)
            img, nois = self.add_noise(img)

            if self.Par['scan']['crop'] and not self.Par['scan']['resp']:
                xi = np.where((data >= 1) & (data <= 8))[0]
                ranx = np.arange(np.min(xi) - 5, np.max(xi) + 6)
                img = img[ranx]
                msk = msk[ranx]
                nois = nois[ranx]
            else:
                ranx = np.arange(data.shape[0])

            self.save_img_data(img, msk, nois, sen[ranx], t)

        self.Par['scan']['crop_xprofs'] = ranx
        self.save_pars_for_recon(img)
    
    
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
        cm = MRXCAT_CMR_PERF.convolve(tinf, cainf, irf, qfl)  # Myocardial concentration
        cm = self.shift_aif(tinf, cm)  # Apply Tshift
        ca = np.interp(t, tinf, cainf)  # Downsample AIF (discrete measurement)
        cm = np.interp(t, tinf, cm)    # Downsample MYO concentration

        return ca, cm
    
    def compute_noise_std_dev(self, sen):
        """
        Calculate the noise standard deviation based on the desired CNR.

        Args:
            sen: Sensitivity maps (4D array with shape [x, y, z, coils]).

        Returns:
            float: Standard deviation of noise based on the calculated CNR.

        Notes:
            This function calculates the standard deviation of noise by:
            - Setting a reference dose for the contrast agent (0.075) and computing the arterial input function (AIF) and myocardial concentration (MYO).
            - Identifying the maximum and minimum myocardial enhancement signals and computing their corresponding means.
            - The noise standard deviation is then calculated based on the difference in signal at max and min enhancement, normalized by the specified SNR.

        """

        # Save the original dose and set the reference dose
        dose_bkp = self.Par["contrast"]["dose"]
        self.Par["contrast"]["dose"] = 0.075  # Reference dose for SNR

        # Get concentration of AIF and MYO for reference dose
        ca, cm = self.compute_dynamic_conc()

        # Restore the original dose
        self.Par["contrast"]["dose"] = dose_bkp

        # Maximum myocardial enhancement mean signal
        tmax = np.argmax(cm)
        data = self.read_img_data(tmax)
        data = data.astype(np.float32)
        self.update_contrast_conc(tmax, ca, cm)
        img, msk = self.map_tissue_props(data)
        img = img.astype(np.float32)
        msk = msk.astype(np.float32)
        img = self.multiply_coil_maps(img, sen)

        # Define region of interest (ROI)
        roi = np.logical_or(msk == self.Par["act"]["myoLA_act"], msk == self.Par["act"]["myoLV_act"])
        roi = np.repeat(roi[..., np.newaxis], sen.shape[3], axis=-1) * img  # Expand ROI for each coil
        smax = [np.mean(roi[..., k][roi[..., k] != 0]) for k in range(self.Par["scan"]["coils"])]
        sumsqmax = np.sqrt(np.sum(np.abs(smax) ** 2))

        # Minimum myocardial enhancement mean signal
        tmin = np.argmin(cm)
        data = self.read_img_data(tmin)
        data = data.astype(np.float32)
        self.update_contrast_conc(tmin, ca, cm)
        img, msk = self.map_tissue_props(data)
        img = img.astype(np.float32)
        msk = msk.astype(np.float32)
        img = self.multiply_coil_maps(img, sen)

        roi = np.logical_or(msk == self.Par["act"]["myoLA_act"], msk == self.Par["act"]["myoLV_act"])
        roi = np.repeat(roi[..., np.newaxis], sen.shape[3], axis=-1) * img  # Expand ROI for each coil
        smin = [np.mean(roi[..., k][roi[..., k] != 0]) for k in range(self.Par["scan"]["coils"])]
        sumsqmin = np.sqrt(np.sum(np.abs(smin) ** 2))

        # Calculate standard deviation based on CNR and contrast
        stdev = 1 / self.Par["scan"]["snr"] * (sumsqmax - sumsqmin)
        
        print(f'Adding noise with standard deviation: {stdev}')
        
        return stdev
    
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

    def map_tissue_props(self, data):
        """
        Calculate signal intensities using tissue and sequence parameters.

        Args:
            data (numpy.ndarray): Data array containing tissue types represented by integers.

        Returns:
            tuple:
                - img (numpy.ndarray): Signal intensity image for each tissue type.
                - msk (numpy.ndarray): Mask image indicating tissue compartment labels.
        """
        
        act = list(self.Par["act"].values())  # Tissue activity values
        tis = list(self.Par["act"].keys())    # Tissue type names
        img = np.zeros_like(data, dtype=np.float32)  # Initialize the image
        msk = np.zeros_like(data, dtype=np.uint8)    # Initialize the mask

        for i in range(len(act)):
            # Select tissue type
            tissue_type = tis[i]

            # Tissue property assignment based on tissue type
            if tissue_type == 'myoLV_act':
                rho = self.Par["tissue"]["rhomuscle"]
                r1 = 1 / self.Par["tissue"]["t1muscle"] + self.Par["contrast"]["cm"]["lv"] * self.Par["contrast"]["ry"]
            elif tissue_type == 'myoRV_act':
                rho = self.Par["tissue"]["rhomuscle"]
                r1 = 1 / self.Par["tissue"]["t1muscle"] + self.Par["contrast"]["cm"]["rv"] * self.Par["contrast"]["ry"]
            elif tissue_type == 'myoLA_act':
                rho = self.Par["tissue"]["rhomuscle"]
                r1 = 1 / self.Par["tissue"]["t1muscle"] + self.Par["contrast"]["cm"]["la"] * self.Par["contrast"]["ry"]
            elif tissue_type == 'myoRA_act':
                rho = self.Par["tissue"]["rhomuscle"]
                r1 = 1 / self.Par["tissue"]["t1muscle"] + self.Par["contrast"]["cm"]["ra"] * self.Par["contrast"]["ry"]
            elif tissue_type == 'bldplLV_act':
                rho = self.Par["tissue"]["rhoblood"]
                r1 = 1 / self.Par["tissue"]["t1blood"] + self.Par["contrast"]["ca"]["lv"] * self.Par["contrast"]["ry"]
            elif tissue_type in ['bldplRV_act', 'art_activity', 'vein_activity']:
                rho = self.Par["tissue"]["rhoblood"]
                r1 = 1 / self.Par["tissue"]["t1blood"] + self.Par["contrast"]["ca"]["rv"] * self.Par["contrast"]["ry"]
            elif tissue_type == 'bldplLA_act':
                rho = self.Par["tissue"]["rhoblood"]
                r1 = 1 / self.Par["tissue"]["t1blood"] + self.Par["contrast"]["ca"]["la"] * self.Par["contrast"]["ry"]
            elif tissue_type == 'bldplRA_act':
                rho = self.Par["tissue"]["rhoblood"]
                r1 = 1 / self.Par["tissue"]["t1blood"] + self.Par["contrast"]["ca"]["ra"] * self.Par["contrast"]["ry"]
            elif tissue_type in ['body_activity', 'pericardium_activity']:
                rho = self.Par["tissue"]["rhofat"]
                r1 = 1 / self.Par["tissue"]["t1fat"]
            elif tissue_type == 'muscle_activity':
                rho = self.Par["tissue"]["rhomuscle"]
                r1 = 1 / self.Par["tissue"]["t1muscle"]
            elif tissue_type == 'liver_activity':
                rho = self.Par["tissue"]["rholiver"]
                r1 = 1 / self.Par["tissue"]["t1liver"]
            elif tissue_type in ['rib_activity', 'cortical_bone_activity', 'spine_activity', 'bone_marrow_activity']:
                rho = self.Par["tissue"]["rhobone"]
                r1 = 1 / self.Par["tissue"]["t1bone"]
            else:
                rho = 0

            # Signal model (Spoiled GRE)
            a = np.cos(self.Par["scan"]["flip"]) * np.exp(-self.Par["scan"]["trep"] * r1)
            b = 1 - np.exp(-self.Par["scan"]["trep"] * r1)
            n = self.Par["scan"]["nky0"]
            TD = self.Par["scan"]["tsat"]
            
            # Signal equation
            sig = rho * ((1 - np.exp(-TD * r1)) * a ** (n - 1) + b * (1 - a ** (n - 1)) / (1 - a))

            # Update tissue compartment
            img[data == act[i]] = sig

            # Update tissue masks
            msk[data == act[i]] = act[i]

        return img, msk

    @staticmethod
    def convolve(t, c, h, q):
        dt = (t[-1] - t[0]) / (len(t) - 1)
        c_result = convolve(c, h, mode='same') * dt
        
        # Scale the result by factor q
        c_result = q * c_result
        return c_result
    
    
def main():
    MRXCAT_CMR_PERF()

    
if __name__ == '__main__':
    main()