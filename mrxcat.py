import os
import numpy as np
import time
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter
from scipy.io import savemat

class MRXCAT:
    def __init__(self):
        self.Data = None
        self.Sensitivities = None
        self.Mask = None
        self.Par = {}
        self.Ksp = None
        self.Filename = None
        self.Version = 1.4
    
    def read_log_file(self):
        fname = self.Filename
        
        # Update number of frames
        self.Par.setdefault("scan", {})
        prefix = '_'.join(os.path.basename(fname).split('_')[:-1])
        suffix = ".bin"
        self.Par["scan"]["frames_xcat"] = len([f for f in os.listdir(os.path.dirname(fname)) if f.startswith(prefix) and f.endswith(suffix)])
        
        # Trim file name for log
        fname = fname[:-9] + "log"
        
        try:
            with open(fname, "r") as fid:
                for line in fid:
                    tokens = line.split()
                    if not tokens:
                        continue
                    key = tokens[0]
                    
                    # Mapping keys to attributes
                    param_map = {
                        "array_size": ("scan", "matrix", int),
                        "myoLV_act": ("act", "myoLV_act", float),
                        "myoRV_act": ("act", "myoRV_act", float),
                        "myoLA_act": ("act", "myoLA_act", float),
                        "myoRA_act": ("act", "myoRA_act", float),
                        "bldplLV_act": ("act", "bldplLV_act", float),
                        "bldplRV_act": ("act", "bldplRV_act", float),
                        "bldplLA_act": ("act", "bldplLA_act", float),
                        "bldplRA_act": ("act", "bldplRA_act", float),
                        "body_activity": ("act", "body_activity", float),
                        "muscle_activity": ("act", "muscle_activity", float),
                        "liver_activity": ("act", "liver_activity", float),
                        "rib_activity": ("act", "rib_activity", float),
                        "cortical_bone_activity": ("act", "cortical_bone_activity", float),
                        "spine_activity": ("act", "spine_activity", float),
                        "bone_marrow_activity": ("act", "bone_marrow_activity", float),
                        "art_activity": ("act", "art_activity", float),
                        "vein_activity": ("act", "vein_activity", float),
                        "pericardium_activity": ("act", "pericardium_activity", float),
                        "pixel": ("scan", "rx_cm", float) if "width" in tokens else None,
                        "slice": ("scan", "rz_cm", float) if "width" in tokens else None,
                        "==>Total": ("scan", "scan_dur", float) if "Output" in tokens else None,
                        "beating": ("scan", "heartbeat_length", float) if "heart" in tokens else None,
                    }
                    
                    if key in param_map and param_map[key]:
                        print(f"Processing {key}...")
                        category, param, dtype = param_map[key]
                        print(category, param, dtype)
                        print(tokens)
                        self.Par.setdefault(category, {})
                        idx_value = next((i+1 for i, x in enumerate(tokens) if x == '='), -1)
                        self.Par[category][param] = dtype(tokens[idx_value])
                    
                    elif "Respiration motion and beating heart motions included" in line:
                        self.Par["scan"]["resp"] = 1
                    elif "Beating heart motion included only" in line:
                        self.Par["scan"]["resp"] = 0
            
                    # Check if slice start and end are defined and calculate the number of slices
                    elif "starting" in tokens and "slice" in tokens and "number" in tokens:
                        sl_start = float(tokens[-1])  # Capture slice start number
                    elif "ending" in tokens and "slice" in tokens and "number" in tokens:
                        sl_end = float(tokens[-1])  # Capture slice end number

            # If both sl_start and sl_end are defined, calculate the number of slices
            if sl_start is not None and sl_end is not None:
                self.Par['scan']['slices'] = int(sl_end - sl_start + 1)
            print("\nPhantom information:")
            print(f"  matrix     : {self.Par['scan']['matrix']}")
            print(f"  frames     : {self.Par['scan']['frames_xcat']}")
            print(f"  resolution : {self.Par['scan']['rx_cm']} x {self.Par['scan']['rx_cm']} x {self.Par['scan']['rz_cm']} cm3\n")
            
        except FileNotFoundError:
            raise ValueError("Cannot read XCAT log file. Aborting ...")
    
    def read_img_data(self, t):
        fname = self.Filename
        
        # Adjust frame number if necessary
        if t > self.Par["scan"]["frames_xcat"]:
            t = t % self.Par["scan"]["frames_xcat"]
        
        if self.Par["scan"].get("resp", False):
            fname = f"{fname[:-5]}{t}.bin"
        
        try:
            with open(fname, "rb") as fid:
                img = np.fromfile(fid, dtype=np.float32)
                matrix_size = self.Par["scan"]["matrix"]
                img = img.reshape((matrix_size, matrix_size, -1), order='F').copy()
                
                return img  # Could apply bounding box cropping here if necessary
        except FileNotFoundError:
            print(f"{fname} cannot be read")
            return None

    def calculate_coil_maps(self):
        # -----------------------------------------------------------------
        # Bounding box indices
        # -----------------------------------------------------------------
        xdim, ydim, zdim = self.compute_bounding_box()

        if self.Par['scan']['coils'] > 1:
            nc = self.Par['scan']['coils']
            rx = self.Par['scan']['rx_cm'] * 10  # [mm] voxel size
            rz = self.Par['scan']['rz_cm'] * 10  # [mm] slice width

            # -----------------------------------------------------------------
            # Coil centre locations, coil radius
            # -----------------------------------------------------------------
            cc, R = self.coil_centres()  # nc coils, 450mm radius (??), 600 mm coil array length => replace/remove

            # -----------------------------------------------------------------
            # Define rotation
            # -----------------------------------------------------------------
            a, b, c = 0, 0, 0  # self.Par.scan.rotation (assuming rotation angles are 0 for now)
            rotx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
            roty = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
            rotz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
            invr = np.linalg.inv(rotx @ rotz @ roty)

            # -----------------------------------------------------------------
            # Angles for integration
            # -----------------------------------------------------------------
            angles = 60
            dtheta = 2 * np.pi / angles
            theta = np.arange(-np.pi, np.pi, dtheta)

            # -----------------------------------------------------------------
            # Voxel coordinates with origin in image centre
            # -----------------------------------------------------------------
            x = np.arange(0, len(xdim), dtype=float)
            x -= x[-1] / 2
            y = np.arange(0, len(ydim), dtype=float)
            y -= y[-1] / 2
            z = np.arange(0, len(zdim), dtype=float)
            z -= z[-1] / 2

            # -----------------------------------------------------------------
            # Convert voxel coordinates to mm
            # -----------------------------------------------------------------
            x = x * rx
            y = y * rx
            z = z * rz

            # -----------------------------------------------------------------
            # Calculate sensitivity for each coil
            # -----------------------------------------------------------------
            sen = np.zeros((len(x), len(y), len(z), nc), dtype=complex)
            T = np.meshgrid(x, y, z, theta, indexing='ij')[-1]  # T is the same for all coils
            t = 0  # tic-toc counter
            sinT = np.sin(T)
            cosT = np.cos(T)

            for i in range(nc):
                tic = time.time()  # Starting time for tic
                print(f'Calculating coil sensitivities for coil {i+1} / {nc} (time elapsed: {t:.1f} s)')

                ci = cc[i, :]
                ang = np.arctan2(cc[i, 0], cc[i, 1])

                # Vector from all voxels to coil centre
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                imrot = np.dot(invr, np.vstack([X.ravel(order='F'), Y.ravel(order='F'), Z.ravel(order='F')]))
                X = np.reshape(imrot[0, :] - ci[1], (len(x), len(y), len(z)), order='F').copy()
                Y = np.reshape(imrot[1, :] - ci[0], (len(x), len(y), len(z)), order='F').copy()
                Z = np.reshape(imrot[2, :] - ci[2], (len(x), len(y), len(z)), order='F').copy()
                X = np.repeat(X[..., np.newaxis], angles, axis=3)
                Y = np.repeat(Y[..., np.newaxis], angles, axis=3)
                Z = np.repeat(Z[..., np.newaxis], angles, axis=3)

                # Calculate x, y, z components and integrate them
                sina = np.sin(ang)
                cosa = np.cos(ang)

                # Denominator
                denom = (R**2) + X**2 + Y**2 + Z**2
                denom -= 2 * R * (-X * cosT * sina + Y * cosT * cosa + Z * sinT)
                denom = np.abs(denom)**(3/2)

                # Nominators (x, y, z)
                nomx = R * (Y * cosT + Z * sinT * cosa - R * cosa)
                nomy = R * (-X * cosT + Z * sinT * sina - R * sina)
                nomz = R * (-Y * sinT * sina - X * sinT * cosa)

                # Sensitivity components (x, y, z)
                sx = nomx / denom
                sx = dtheta * np.sum(sx, axis=3)  # Integration over theta
                sy = nomy / denom
                sy = dtheta * np.sum(sy, axis=3)
                sz = nomz / denom
                sz = dtheta * np.sum(sz, axis=3)

                # Angle in yz plane
                angy = sina * (X[..., 0] + ci[1]) - cosa * (Y[..., 0] + ci[0])
                angz = Z[..., 0]
                yz = np.angle(angy + 1j * angz)

                # Calculate sensitivity
                sen[..., :, i] = cosa * sx + sina * sy + 1j * ((-sina * sx + cosa * sy) * np.cos(yz) + sz * np.sin(yz))

                if i < nc - 1:
                    for k in range(73):
                        print('\b', end='')  # Progress bar behavior
                t += time.time() - tic  # Update elapsed time

            # Reset contrast concentrations
            if 'contrast' not in self.Par.keys():
                self.Par['contrast'] = {'cm': {}, 'ca': {}}
            else:
                self.Par['contrast']['cm'] = {}
                self.Par['contrast']['ca'] = {}
                
            self.Par['contrast']['cm']['lv'] = 0
            self.Par['contrast']['cm']['rv'] = 0
            self.Par['contrast']['cm']['la'] = 0
            self.Par['contrast']['cm']['ra'] = 0
            self.Par['contrast']['ca']['lv'] = 0
            self.Par['contrast']['ca']['rv'] = 0
            self.Par['contrast']['ca']['la'] = 0
            self.Par['contrast']['ca']['ra'] = 0

            # Normalize sensitivity
            msen = np.mean(sen[sen != 0])
            sen = 1 / msen * (np.real(sen) + 1j * np.imag(sen))
        else:
            sen = np.ones((len(xdim), len(ydim), len(zdim), 1))

        return sen

    def compute_bounding_box(self):
        """
        Computes the bounding box based on the scan parameters.

        Returns:
            tuple: Three arrays (xdim, ydim, zdim) representing the dimensions of the bounding box.
        """
        bbox = self.Par['scan']['bbox']
        matrix = self.Par['scan']['matrix']
        slices = self.Par['scan']['slices']
        eps= 1e-5

        # Adjust the bounding box dimensions
        xdim = np.fix(np.arange(bbox[0, 0] * matrix, bbox[0, 1] * matrix - 1 + eps)).astype(int)
        ydim = np.fix(np.arange(bbox[1, 0] * matrix, bbox[1, 1] * matrix - 1 + eps)).astype(int)
        zdim = np.fix(np.arange(bbox[2, 0] * slices, bbox[2, 1] * slices - 1 + eps)).astype(int)


        if xdim.size == 0 or ydim.size == 0 or zdim.size == 0:
            raise ValueError('Bounding box dimension empty! Increase BoundingBox size or XCAT size')

        # Adapt ydim to multiple of possible k-t factors
        f = factor(self.Par['scan']['frames'])
        if len(f) < 2 or f[0] != 2:
            print('Warning: Number of time points should be multiple of 2 and no prime number for k-t studies!')

        if 'adaptPhaseEncDim' in self.Par['scan'] and self.Par['scan']['adaptPhaseEncDim']:
            rdy = len(ydim) % self.Par['scan']['frames']
            ydim = np.arange(ydim[0] + np.ceil(rdy / 2), ydim[-1] - np.floor(rdy / 2))

        if len(xdim) != len(ydim) and self.Par['scan']['trajectory'].lower() != 'cartesian':
            # Square FOV for radial resampling
            dly = len(ydim) - len(xdim)
            xdim = np.arange(xdim[0] - np.floor(dly / 2), xdim[-1] + np.ceil(dly / 2))
            # Adjust FOV for radial (radial Nyquist requirement)
            samp = round(len(xdim) * np.pi / 2)
            xdimplus = (samp - len(xdim)) / 2
            xdim = np.arange(xdim[0] - np.floor(xdimplus), xdim[-1] + np.floor(xdimplus))
            ydim = np.arange(ydim[0] - np.floor(xdimplus), ydim[-1] + np.floor(xdimplus))
            # !!! add check for negative indices here !!!

        return xdim, ydim, zdim
    
    def coil_centres(self):
        """
        Generate coil centers on a cylinder surface for MR sensitivity simulation.

        Returns:
            Tuple[np.ndarray, float]: 
                - cc: A (n_coils x 3) numpy array of coil center positions [x, y, z] in mm.
                - rcoil: Coil element radius in mm.
        
        Uses:
            self.Par['scan']['coils'] (int): Number of coils.
            self.Par['scan']['coildist'] (float): Distance from image center (body radius).
            self.Par['scan']['coilsperrow'] (int): Max number of coils per row (ring).
        """
        nc = self.Par['scan']['coils']
        rbody_mm = self.Par['scan']['coildist']
        c_per_row = self.Par['scan']['coilsperrow']

        angles2 = np.array([150, 210]) * np.pi / 180
        angles3 = np.array([130, 180, 230]) * np.pi / 180
        angles4 = np.array([150, 210, 330, 30]) * np.pi / 180
        angles5 = np.array([130, 180, 230, 330, 30]) * np.pi / 180
        angles6 = np.array([130, 180, 230, 310, 0, 50]) * np.pi / 180

        if nc % c_per_row != 0:
            compl_rows = max((nc - c_per_row) // c_per_row, 0)
            remc = nc - compl_rows * c_per_row
            if remc > c_per_row:
                remc = [remc // 2, remc - remc // 2]
            else:
                remc = [remc]
        else:
            compl_rows = nc // c_per_row
            remc = []

        rings = compl_rows + len(remc)
        nc_ring = [c_per_row] * rings
        if len(remc) == 1:
            nc_ring[0] = remc[0]
        elif len(remc) == 2:
            nc_ring[0] = remc[0]
            nc_ring[-1] = remc[1]

        if any(x > 6 for x in nc_ring):
            minang = 2 * np.pi / max(nc_ring)
        else:
            minang = 50 * np.pi / 180
        rcoil = 0.5 * rbody_mm * minang

        z = np.arange(-(rings - 1) * rcoil, (rings - 1) * rcoil + 2 * rcoil, 2 * rcoil)
        cc = np.zeros((nc, 3))
        lctr = 0

        for k in range(rings):
            if nc_ring[k] == 1:
                x, y = pol2cart(0, rbody_mm)
            elif nc_ring[k] == 2:
                x, y = pol2cart(angles2, rbody_mm)
            elif nc_ring[k] == 3:
                x, y = pol2cart(angles3, rbody_mm)
            elif nc_ring[k] == 4:
                x, y = pol2cart(angles4, rbody_mm)
            elif nc_ring[k] == 5:
                x, y = pol2cart(angles5, rbody_mm)
            elif nc_ring[k] == 6:
                x, y = pol2cart(angles6, rbody_mm)
            else:
                angles = np.linspace(0, 2 * np.pi * (nc_ring[k] - 1) / nc_ring[k], nc_ring[k])
                x, y = pol2cart(angles, rbody_mm)

            cc[lctr:lctr + len(x), :] = np.column_stack([x, y, np.full(len(x), z[k])])
            lctr += len(x)

        return cc, rcoil
    
    def multiply_coil_maps(self, img: np.ndarray, sen: np.ndarray) -> np.ndarray:
        """
        Multiply the image volume with coil sensitivity maps.

        Args:
            img (np.ndarray): The input image volume with shape (X, Y, Z).
            sen (np.ndarray): The coil sensitivity maps, expected to be reshaped to (X, Y, Z, coils).

        Returns:
            np.ndarray: The coil-encoded image volume with shape (X, Y, Z, coils).
        """
        coils = self.Par["scan"]["coils"]
        sen = sen.reshape(img.shape[0], img.shape[1], img.shape[2], coils, order='F').copy()
        img = np.repeat(img[..., np.newaxis], coils, axis=3)
        img = img * sen
        return img
    
    def read_img_data(self, t: int) -> np.ndarray:
        """
        Read image data from an XCAT .bin file.

        Args:
            t (int): The time/frame index to read.

        Returns:
            np.ndarray: The image data as a 3D numpy array (x, y, z).
        """
        fname = self.Filename

        # If more output than XCAT frames, wrap around (only in perfusion mode)
        if (
            t > self.Par["scan"]["frames_xcat"]
            and self.__class__.__name__.lower() == "mrxcat_cmr_perf"
        ):
            t = t % self.Par["scan"]["frames_xcat"]

        # Construct correct filename for cine or respiratory scan
        if self.Par["scan"]["resp"] or self.__class__.__name__.lower() == "mrxcat_cmr_cine":
            fname = f"{fname[:-5]}{t}.bin"

        # Try to read the binary file
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                img_flat = np.fromfile(f, dtype=np.float32)

            # Reshape to 3D
            matrix_size = self.Par["scan"]["matrix"]
            img = img_flat.reshape((matrix_size, matrix_size, -1), order='F').copy()

            # Crop using bounding box
            xdim, ydim, zdim = self.compute_bounding_box()
            img = img[np.ix_(xdim, ydim, zdim)]
        else:
            print(f"{fname} cannot be read")
            img = None

        return img
    
    def low_pass_filter(self, img, msk):
        """
        Apply a low-pass filter to the image and mask.

        Args:
            img (ndarray): The input image to be filtered.
            msk (ndarray): The mask to be filtered.

        Returns:
            img (ndarray): The filtered image.
            msk (ndarray): The filtered mask.
        """
        if self.Par['scan']['lowpass']:
            # Create the low-pass filter (disk filter)
            radius = self.Par['scan']['lowpass_str']
            H = fspecial_disk(radius)
            H = np.expand_dims(H, -1) # make H 3D
            # Filter image and mask
            img = convolve(img, H, mode='nearest')

            msk1 = convolve(msk, H, mode='nearest')
            msk2 = np.zeros_like(msk)

            # Loop through the indices for the relevant regions (1, 5, 6, 7, 8)
            for j in [1, 5, 6, 7, 8]:  # myo=1, lv, la, rv, ra=5:8
                idxr1 = np.where(np.round(msk1) == j)
                x, y, z = idxr1

                for k in range(len(x)):
                    if msk[x[k], y[k], z[k]] == j: 
                        msk2[x[k], y[k], z[k]] = j
            
            msk = msk2
        
        return img, msk

    def add_noise(self, img):
        """
        Add noise to the input image based on the noise standard deviation.

        Args:
            img (numpy.ndarray): The input image data.

        Returns:
            tuple: The noisy image and the generated noise.
        """
        stdev = self.Par['scan']['noisestd']  # Get noise standard deviation
        # Generate complex noise using normal distribution (real and imaginary parts)
        nois = stdev * (np.random.randn(*img.shape) + 1j * np.random.randn(*img.shape))
        # Add the noise to the image
        img_noisy = img + nois
        return img_noisy, nois

    @staticmethod
    def i2k(img, dims=None):
        """
        Perform FFT (Image to K-space transformation).

        Args:
            img (numpy.ndarray): Image data (spatial domain).
            dims (list, optional): Dimensions along which FFT is performed. Defaults to None.

        Returns:
            numpy.ndarray: K-space data.
        """
        dim_img = img.shape
        
        if dims is None:
            factor = np.prod(dim_img)
            img = (1 / np.sqrt(factor)) * np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img)))
        else:
            for dim in dims:
                if img.shape[dim] > 1:
                    img = (1 / np.sqrt(dim_img[dim])) * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img, axes=[dim]), axis=dim), axes=[dim])
        return img

    @staticmethod
    def k2i(img, dims=None):
        """
        Perform Inverse FFT (K-space to Image transformation).

        Args:
            img (numpy.ndarray): K-space data.
            dims (list, optional): Dimensions along which inverse FFT is performed. Defaults to None.

        Returns:
            numpy.ndarray: Image data.
        """
        dim_img = img.shape
        
        if dims is None:
            factor = np.prod(dim_img)
            img = np.sqrt(factor) * np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(img)))
        else:
            for dim in dims:
                if img.shape[dim] > 1:
                    img = np.sqrt(dim_img[dim]) * np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(img, axes=[dim]), axis=dim), axes=[dim])
        return img
    
    def save_img_data(self, img, msk, nois, sen, t):
        """
        Save image data (complex), mask, noise, and sensitivities to files.

        Args:
            img (numpy.ndarray): Image data (complex).
            msk (numpy.ndarray): Mask data (binary).
            nois (numpy.ndarray): Noise data (complex).
            sen (numpy.ndarray): Sensitivities data (complex).
            t (int): Time/frame index (1 for first time point, for appending otherwise).
        """
        # Generate filename and append appropriate extensions
        fname = self.generate_filename(img)
        fimg = f"{fname}.cpx"
        fmsk = f"{fname}.msk"
        fnoi = f"{fname}.noi"
        fsen = f"{fname}.sen"

        # Write/append data files
        if t == 1:
            print("\nOutput file information:")
            print(f"  matrix :{img.shape[0]:4d} x{img.shape[1]:4d} x{img.shape[2]:4d}")
            print(f"  frames :{self.Par['scan']['frames']:4d}")
            print(f"  coils  :{self.Par['scan']['coils']:4d}")

            fidimg = open(fimg, 'wb')
            fidmsk = open(fmsk, 'wb')
            fidnoi = open(fnoi, 'wb')
            fidsen = open(fsen, 'wb')
        else:
            fidimg = open(fimg, 'ab')
            fidmsk = open(fmsk, 'ab')
            fidnoi = open(fnoi, 'ab')
            fidsen = open(fsen, 'ab')

        try:
            # Convert complex numbers by splitting them into real and imaginary parts
            dim = img.shape
            if len(dim) < 3:
                dim = (dim[0], dim[1], 1) + dim[2:]

            # Reshape and permute (real and imaginary parts)
            img = np.stack([np.real(img), np.imag(img)], axis=2)
            img = np.transpose(img, (2, 0, 1, 3, 4))

            # Do the same for noise and sensitivities
            nois = np.stack([np.real(nois), np.imag(nois)], axis=2)
            nois = np.transpose(nois, (2, 0, 1, 3, 4))

            sen = np.stack([np.real(sen), np.imag(sen)], axis=2)
            sen = np.transpose(sen, (2, 0, 1, 3, 4))

            # Write the data to the respective files
            fidimg.write(np.array(img, dtype=np.float32).flatten(order='F').tobytes())
            fidmsk.write(np.array(msk, dtype=np.uint8).flatten(order='F').tobytes())
            fidnoi.write(np.array(nois, dtype=np.float32).flatten(order='F').tobytes())
            fidsen.write(np.array(sen, dtype=np.float32).flatten(order='F').tobytes())

        except Exception as e:
            print(f"Error writing to files: {e}")
        finally:
            fidimg.close()
            fidmsk.close()
            fidnoi.close()
            fidsen.close()

    def generate_filename(self, img):
        """
        Generate filename for saving MRXCAT phantom data.

        Args:
            img (numpy.ndarray): Image data (not used directly in filename generation, but its size is).
        
        Returns:
            str: Generated filename.
        """
        # Extract path and filename (equivalent to MATLAB's fileparts)
        p, f = os.path.split(self.Filename)
        fname = os.path.join(p, f[:4])  # Take the first 4 characters of the filename

        # Generate different strings to be concatenated into the filename
        bhstr = '_bh' if self.Par['scan']['resp'] == 0 else '_fb'
        
        resolstr = f"_{int(self.Par['scan']['rx_cm'] * 10)}x{int(self.Par['scan']['rx_cm'] * 10)}x{int(self.Par['scan']['rz_cm'] * 10)}mm"
        
        matstr = f"_{img.shape[0]}x{img.shape[1]}x{img.shape[2]}x{self.Par['scan']['frames']}"
        if self.Par['scan']['coils'] > 1:
            matstr += f"x{self.Par['scan']['coils']}"
        
        snrstr = f"_snr{self.Par['scan']['snr']}"
        fastr = f"_fa{round(self.Par['scan']['flip'] * 180 / 3.14159265359)}"
        
        # Handle case separation between perfusion and other phantom types
        if hasattr(self.Par, 'contrast') and hasattr(self.Par.contrast, 'dose'):  # perfusion-specific case
            dosestr = f"_dose{self.Par['contrast']['dose']}"
            tshiftstr = f"_tshift{self.Par['contrast']['tshift']}"
            restStress = ''
            if self.Par['contrast']['rs'] == 2:
                restStress = 'Stress'
            elif self.Par['contrast']['rs'] == 1:
                restStress = 'Rest'
            
            filename = f"{fname}{restStress}{resolstr}{matstr}{snrstr}{dosestr}{fastr}{tshiftstr}{bhstr}"
        else:  # cine or other phantom types
            filename = f"{fname}{resolstr}{matstr}{snrstr}{fastr}{bhstr}"

        return filename
    
    def save_pars_for_recon(self, img):
        """
        Save MRXCAT parameters for reconstruction into a .mat file.

        Args:
            img (numpy.ndarray): The image data.
        """
        # Generate filename
        fname = self.generate_filename(img)
        fpar = f"{fname}_par.mat"
        
        # Get image dimensions
        xdim, ydim, zdim = img.shape[:3]
        
        # Set MRX parameters for reconstruction
        self.Par['acq_ovs'] = np.ones(4)
        self.Par['acq_matrix'] = [xdim, ydim, zdim, 1]
        self.Par['acq_ne0'] = xdim
        self.Par['acq_ne1'] = ydim
        self.Par['acq_ne2'] = zdim
        self.Par['rec_matrix'] = self.Par['acq_matrix'][:3]
        self.Par['rec_file_name'] = f"{fname}.cpx"
        self.Par['proto_array'] = [10, 11, 7, 1, 0, 0, 100, 0]  # Example values for the protocol
        self.Par['ncoils'] = self.Par['scan']['coils']  # Assuming this is how coils are represented
        
        # Save parameters to .mat file
        try:
            savemat(fpar, {'Par': self.Par})
            print(f"Parameters saved to {fpar}")
        except Exception as e:
            print(f"Error saving parameters: {e}")

def fspecial_disk(radius):
    """
    Approximate MATLAB's fspecial('disk', radius) with a smooth, normalized disk filter.
    """
    if radius==1.2:
        disk = np.array([
            [0.0479,    0.1468,    0.0479],
            [0.1468,    0.2210,    0.1468],
            [0.0479,    0.1468,    0.0479],
        ])
        return disk
    size = int(radius* 2 + 1)
    center = size // 2
    y, x = np.ogrid[:size, :size]
    distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)

    # Soft circular mask
    disk = (distance <= radius).astype(float)

    # Apply Gaussian blur to smooth edges (mimicking MATLAB's antialiasing)
    disk = gaussian_filter(disk, sigma=0.5)

    # Normalize
    disk /= disk.sum()

    return disk


def pol2cart(angles, radius):
    """
    Convert polar coordinates to Cartesian coordinates.

    Args:
        angles (float or np.ndarray): Angles in radians.
        radius (float): Radius in mm.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - x: X-coordinates.
            - y: Y-coordinates.
    """
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return x, y


def factor(n):
    # Function to find the factors of a number (equivalent to MATLAB's `factor`)
    factors = []
    i = 2
    while i * i <= n:
        while (n % i) == 0:
            factors.append(i)
            n //= i
        i += 1
    if n > 1:
        factors.append(n)
    return factors

