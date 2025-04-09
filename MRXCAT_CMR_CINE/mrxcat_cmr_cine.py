import numpy as np
from skimage.measure import label, regionprops
from pynufft import NUFFT

from mrxcat import MRXCAT
from MRXCAT_CMR_CINE.cine_par import CINEpar

class MRXCAT_CMR_CINE(MRXCAT):

    def __init__(self, filename='', *args):
        super().__init__()

        # ------------------------------------------------------------
        # Load and assign parameters
        # ------------------------------------------------------------
        CINEpar(self, filename)

        # ------------------------------------------------------------
        # Check for additional inputs
        # ------------------------------------------------------------
        for i in range(len(args)):
            if str(args[i]).lower() == 'snr':
                self.Par['scan']['snr'] = args[i+1]
            elif str(args[i]).lower() == 'frames':
                self.Par['scan']['frames'] = args[i+1]
            elif str(args[i]).lower() == 'coils':
                self.Par['scan']['coils'] = args[i+1]
            elif str(args[i]).lower() == 'flip':
                self.Par['scan']['flip'] = np.pi / 180 * args[i+1]
            elif str(args[i]).lower() == 'demo_gui':
                par = args[i+1]
                for name in par['contrast']:
                    if name in self.Par['contrast']:
                        self.Par['contrast'][name] = par['contrast'][name]
                for name in par['scan']:
                    if name in self.Par['scan']:
                        self.Par['scan'][name] = par['scan'][name]

        # --------------------------------------------------------------------
        #   Calculate coil sensitivities (Biot-Savart)
        # --------------------------------------------------------------------
        sen = self.calculate_coil_maps()

        # --------------------------------------------------------------------
        # Compute standard deviation factor for noise addition (SNR way)
        # --------------------------------------------------------------------
        self.Par['scan']['noisestd'] = self.compute_noise_std_dev(sen)

        st = 0  # initialize waitbar

        # --------------------------------------------------------------------
        #   Produce MRXCAT phantom loops over heart phases & k-space segments
        # --------------------------------------------------------------------
        for t in self.Par['scan']['phases']:
            for s in range(1, self.Par['scan']['segments'] + 1):

                # --------------------------------------------------------------------
                #   Read data
                # --------------------------------------------------------------------
                xcat_no = t + ((s - 1) * self.Par['scan']['frames'])
                data = self.read_img_data(xcat_no).astype(np.float32)

                # ----------------------------------------------------------------
                #   Map MR tissue properties
                # ----------------------------------------------------------------
                img, msk = self.map_tissue_props(data)
                img = img.astype(np.float32)
                msk = msk.astype(np.float32)

                # ----------------------------------------------------------------
                #   Low Pass Filter (Blur)
                # ----------------------------------------------------------------
                img, msk = self.low_pass_filter(img, msk)

                # ----------------------------------------------------------------
                #   Add coils
                # ----------------------------------------------------------------
                img = self.multiply_coil_maps(img, sen)

                # ----------------------------------------------------------------
                #   Add noise
                # ----------------------------------------------------------------
                img, noi = self.add_noise(img)

                # ----------------------------------------------------------------
                #   Extract needed k-space segment
                # ----------------------------------------------------------------
                if s == 1:
                    self.Ksp = np.zeros_like(img)
                self.extract_segment(s, img)
                if s == self.Par['scan']['segments']:
                    img = MRXCAT.k2i(self.Ksp, [0, 1, 2])

            self.Ksp = []

            # -----------------------------------------------------------------
            #   Regrid data for radial trajectory
            # -----------------------------------------------------------------
            if self.Par['scan']['trajectory'].lower() in ['radial', 'goldenangle']:
                img = self.radial_resample(img)

            # -----------------------------------------------------------------
            #   Save data to .mat file
            # -----------------------------------------------------------------
            self.save_img_data(img, msk, noi, sen, t)
            # st = self.textwaitbar(t, st, 'Writing MRXCAT output data')

        # --------------------------------------------------------------------
        #   Save Parameters for Recon
        # --------------------------------------------------------------------
        self.save_pars_for_recon(img)

    def compute_noise_std_dev(self, sen: np.ndarray) -> float:
        """
        Compute the noise standard deviation based on the desired SNR and signal
        intensity in the heart region.

        Args:
            sen (np.ndarray): Coil sensitivity maps of shape [..., coils].

        Returns:
            float: Standard deviation of noise.
        """
        phases = self.Par["scan"]["phases"]
        ncoils = self.Par["scan"]["coils"]
        snr_target = self.Par["scan"]["snr"]
        heart_labels = [
            self.Par["act"]["myoLA_act"],
            self.Par["act"]["myoLV_act"],
            self.Par["act"]["myoRA_act"],
            self.Par["act"]["myoRV_act"]
        ]

        roiall = []

        for t in phases:
            data = self.read_img_data(t).astype(np.float32)
            img, msk = self.map_tissue_props(data)
            img, msk = img.astype(np.float32), msk.astype(np.float32)

            img = self.multiply_coil_maps(img, sen)

            roi = np.isin(msk, heart_labels)

            # Identify the largest connected region (heart mask)
            labeled = label(roi)
            props = regionprops(labeled)
            if len(props) > 1:
                max_region = max(props, key=lambda x: x.area)
                roi = np.zeros_like(roi, dtype=np.uint8)
                roi[max_region.coords[:, 0], max_region.coords[:, 1], max_region.coords[:, 2]] = 1

            # Broadcast roi across coil dimensions
            roi_broadcast = np.repeat(roi[..., np.newaxis], sen.shape[3], axis=3)
            roiall.append(roi_broadcast * img)

        roiall = np.stack(roiall, axis=-1)  # shape: [x, y, z, coils, phases]

        smean = []
        for k in range(ncoils):
            roik = roiall[:, :, :, k, :]
            nonzero_values = roik[roik != 0]
            smean.append(nonzero_values.mean() if nonzero_values.size > 0 else 0)

        smean = np.array(smean)
        sosmean = np.sqrt(np.sum(np.abs(smean) ** 2))
        stdev = sosmean / snr_target

        print(f"adding noise w/ std dev : {stdev:.6f}")
        return stdev

    def map_tissue_props(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate signal intensities and masks using tissue and sequence parameters.

        Args:
            data (np.ndarray): 3D or 4D array containing anatomical labels.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - img: Signal intensity image (same shape as `data`, dtype float32).
                - msk: Tissue label mask (same shape as `data`, dtype uint8).
        """
        act_values = np.array(list(self.Par["act"].values()))
        tis_keys = list(self.Par["act"].keys())

        img = np.zeros_like(data, dtype=np.float32)
        msk = np.zeros_like(data, dtype=np.uint8)

        for i in range(len(act_values)):
            label = act_values[i]
            tissue_name = tis_keys[i]

            # Initialize defaults
            rho, t1, t2 = 0, 0, 0

            # Assign tissue properties
            if tissue_name in ['myoLV_act', 'myoRV_act', 'myoLA_act', 'myoRA_act']:
                rho = self.Par["tissue"]["rhomuscle"]
                t1 = self.Par["tissue"]["t1muscle"]
                t2 = self.Par["tissue"]["t2muscle"]
            elif tissue_name in ['bldplLV_act', 'bldplRV_act', 'art_activity', 'vein_activity', 'bldplLA_act', 'bldplRA_act']:
                rho = self.Par["tissue"]["rhoblood"]
                t1 = self.Par["tissue"]["t1blood"]
                t2 = self.Par["tissue"]["t2blood"]
            elif tissue_name in ['body_activity', 'pericardium_activity']:
                rho = self.Par["tissue"]["rhofat"]
                t1 = self.Par["tissue"]["t1fat"]
                t2 = self.Par["tissue"]["t2fat"]
            elif tissue_name == 'muscle_activity':
                rho = self.Par["tissue"]["rhomuscle"]
                t1 = self.Par["tissue"]["t1muscle"]
                t2 = self.Par["tissue"]["t2muscle"]
            elif tissue_name == 'liver_activity':
                rho = self.Par["tissue"]["rholiver"]
                t1 = self.Par["tissue"]["t1liver"]
                t2 = self.Par["tissue"]["t2liver"]
            elif tissue_name in ['rib_activity', 'cortical_bone_activity', 'spine_activity', 'bone_marrow_activity']:
                rho = self.Par["tissue"]["rhobone"]
                t1 = self.Par["tissue"]["t1bone"]
                t2 = self.Par["tissue"]["t2bone"]

            if rho != 0:
                a = self.Par["scan"]["flip"]
                te = self.Par["scan"]["te"]
                sig = rho * np.sin(a) / ((t1 / t2 + 1) - np.cos(a) * (t1 / t2 - 1)) * np.exp(-te / t2)

                mask = (data == label)
                img[mask] = sig
                msk[mask] = label

        return img, msk

    def extract_segment(self, segm, img):
        """
        Extract a k-space segment from the fully sampled k-space.

        Args:
            segm (int): The segment number to extract.
            img (numpy.ndarray): The image data to be converted to k-space.

        Updates:
            self.Ksp: The k-space data (global k-space array).
        """
        nsegm = self.Par['scan']['segments']  # number of segments
        # Define k-space slice range
        kyrange = range(
            int(np.ceil((segm - 1) * img.shape[1] / nsegm)),
            min(int(np.ceil(segm * img.shape[1] / nsegm)), img.shape[1])
        )
        temp = MRXCAT.i2k(img, dims=(0, 1, 2))  # Assuming image is 3D
        # Extract the segment from k-space
        self.Ksp[:, kyrange, ...] = temp[:, kyrange, ...]
        
    @staticmethod
    def radial_resample(MRX, img):
        """
        Radial Trajectory Resampling and Calculation.

        Args:
            MRX (MRXCAT_CMR_CINE): MRXCAT_CMR_CINE instance containing scan parameters.
            img (numpy.ndarray): Image data to resample, typically in Cartesian coordinates.

        Returns:
            tuple: (img_rad, ksp) where:
                - img_rad is the resampled image data in radial coordinates.
                - ksp is the k-space data corresponding to the resampled image.
        """
        # Determine trajectory type
        if MRX.Par['scan']['trajectory'].lower() == 'goldenangle':
            ga = 1  # Golden angle
        else:
            ga = 0  # Standard radial

        # Calculate the radial trajectory
        samp = img.shape[0]
        prof = round(samp * 2 / np.pi * 1 / MRX.Par['scan']['undersample'])  # Number of profiles
        w = MRXCAT_CMR_CINE.get_rad_weights_2d(samp, prof, 0, ga, 1)
        k = MRXCAT_CMR_CINE.build_rad_traj_2d(samp, prof, 0, ga, 1)

        # Check for the existence of a NUFFT library
        try:
            # Assuming NUFFT is available as a Python class or function
            e = NUFFT(k, w, 1, [0, 0], [samp, samp], 1)
        except ImportError:
            raise ImportError("NUFFT library not found. Please install a NUFFT package.")

        # Radially resample for all coil elements
        ksp = np.zeros_like(img, dtype=np.complex64)  # K-space data
        img_rad = np.zeros_like(img, dtype=np.complex64)  # Resampled image

        for coil_idx in range(MRX.Par['scan']['coils']):
            # Perform NUFFT for each coil element
            ksp[:, :, :, coil_idx] = e @ img[:, :, :, coil_idx].astype(np.float64)
            img_rad[:, :, :, coil_idx] = np.conj(e) @ ksp[:, :, :, coil_idx]

        return img_rad, ksp

    @staticmethod
    def get_rad_weights_2d(samp, prof, start_angle, golden_angle, flag):
        # Placeholder for function to compute radial weights
        # Implement based on specific formula for radial weighting
        # For simplicity, return a mock array with the same shape as 'img'
        return np.ones((samp, prof))

    @staticmethod
    def build_rad_traj_2d(samp, prof, start_angle, golden_angle, flag):
        # Placeholder for function to compute radial trajectory
        # Implement based on specific formula for radial trajectory
        # For simplicity, return a mock array of the trajectory
        return np.ones((prof, samp))  # Mock radial trajectory