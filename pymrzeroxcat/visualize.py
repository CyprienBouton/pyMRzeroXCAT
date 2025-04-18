import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import io
from PIL import Image

def display_mrxcat(fname=None, plot_result=True):
    if fname is None or not os.path.isfile(fname):
        # get MRXCAT output (.cpx file)
        from tkinter import filedialog
        from tkinter import Tk
        root = Tk()
        root.withdraw()  # Hide the root window
        fname = filedialog.askopenfilename(filetypes=[("MRXCAT Files", "*.cpx")])
    
    if fname is None or not os.path.isfile(fname):
        print("No file selected")
        return None, None
    
    # Load MRXCAT parameter file (acquisition matrix, coils, ...)
    par_filename = fname[:-4] + '_par.mat'
    par_data = loadmat(par_filename)
    Par = par_data['Par']
    Par = {field: Par[field][0][0][0] for field in Par.dtype.names}
    if 'contrast' in Par:
        Par['contrast'] = {field: Par['contrast'][field][0][0][0] for field in Par['contrast'].dtype.names}
    
    # Load MRXCAT data
    with open(fname, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    
    # Load sensitivity maps
    sen_filename = fname[:-4] + '.sen'
    with open(sen_filename, 'rb') as f:
        sen = np.fromfile(f, dtype=np.float32)
    
    # Reformat to complex data (image and sensitivities)
    data = data.reshape(2, -1, order='F')
    data = data[0, :] + 1j * data[1, :]
    data = data.reshape(Par['acq_matrix'][0], Par['acq_matrix'][1], Par['acq_matrix'][2], Par['ncoils'][0], -1, order='F')
    data = np.transpose(data, (0, 1, 2, 4, 3))

    sen = sen.reshape(2, -1, order='F')
    sen = sen[0, :] + 1j * sen[1, :]
    sen = sen.reshape(Par['acq_matrix'][0], Par['acq_matrix'][1], Par['acq_matrix'][2], Par['ncoils'][0], -1, order='F')
    sen = np.transpose(sen, (0, 1, 2, 4, 3))
    
    # "coil-combine" data
    sos = np.sum(data / sen, axis=4)

    # Display 4-panel figure showing time frame, slices, coil images, and (for perf) example signal-time curves
    if plot_result:
        plt.figure(figsize=(10, 8))
        
        # Display time frames
        plt.subplot(2, 2, 1)
        display_movie(np.abs(sos[:, :, sos.shape[2]//2, :]), [2, 2, 1], 1, 0.3)
        plt.title('Time frames')

        # Display slices
        plt.subplot(2, 2, 2)
        display_movie(np.abs(sos[:, :, :, sos.shape[3]//2]), [2, 2, 2], 1, 0.3)
        plt.title('Slices')

        # Display coil maps
        plt.subplot(2, 2, 3)
        display_movie(np.abs(data[:, :, data.shape[2]//2, data.shape[3]//2, :]), [2, 2, 3], 1, 1)
        plt.title('Coil maps')

        # If it's a perfusion scan, display AIF and MYO signal
        if 'contrast' in Par and 'aif' in Par['contrast']:
            sa, sm, sa_ind, sm_ind = extract_signal_time_curves(sos, fname)
            plt.subplot(2, 2, 4)
            plt.plot(np.arange(1, len(sa)+1), np.abs(sa), label="AIF")
            plt.plot(np.arange(1, len(sm)+1), np.abs(sm), label="MYO")
            plt.plot(np.arange(1, len(sa_ind)+1), np.abs(sa_ind), label="AIF Ind")
            plt.plot(np.arange(1, len(sm_ind)+1), np.abs(sm_ind), label="MYO Ind")
            plt.title('Mean and Single-Voxel AIF and MYO Signal')
            plt.xlabel('Time frame [heart beats]')
            plt.ylabel('Signal intensity [a.u.]')
            plt.legend()
            plt.tight_layout()

        plt.show()
    
    return data, fname

def display_movie(data, subplotno, loops=3, dt=0.1, scale=(None, None), gif_filename=None):
    number_images = data.shape[2]
    image_pos_no = 1

    if len(subplotno) >= 2:
        gridsizex = subplotno[0]
        gridsizey = subplotno[1]
        if len(subplotno) >= 3:
            image_pos_no = subplotno[2]
    else:
        gridsizex = 1
        gridsizey = 1

    min_scale, max_scale = scale
    if min_scale is None or max_scale is None:
        minint = np.min(data)
        maxint = np.max(data)
    else:
        minint = min_scale
        maxint = max_scale

    if maxint == minint:
        maxint += 1
        minint -= 1
    
    fig = plt.gcf()
    frames = []
    
    for loop in range(loops):
        for imageno in range(number_images):
            plt.subplot(gridsizex, gridsizey, image_pos_no)
            plt.imshow(data[:, :, imageno], cmap='gray', vmin=minint, vmax=maxint)
            plt.title(f'{imageno + 1}/{number_images}')
            plt.xticks([])
            plt.yticks([])
            plt.pause(dt)
            
            if gif_filename:
                # Get current subplot axis and its extent
                ax = plt.gca()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches=ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()))
                buf.seek(0)
                frames.append(Image.open(buf).convert('RGB'))
                buf.close()
    
    # Save as GIF
    if gif_filename and frames:
        frames[0].save(
            gif_filename,
            save_all=True,
            append_images=frames[1:],
            duration=int(dt * 1000),
            loop=0
        )
        print(f"GIF saved to {gif_filename}")
        
    imageno = number_images // 2
    plt.subplot(gridsizex, gridsizey, image_pos_no)
    plt.imshow(data[:, :, imageno], cmap='gray', vmin=minint, vmax=maxint)
    plt.title(f'{imageno + 1}/{number_images}')
    plt.xticks([])
    plt.yticks([])

def extract_signal_time_curves(data, filename):
    msk_filename = filename.replace('.cpx', '.msk')
    
    with open(msk_filename, 'rb') as f:
        msk = np.fromfile(f, dtype=np.uint8)
    
    msk = msk.reshape(data.shape[:4], order='F')

    mym = (msk == 1)
    lvm = (msk == 5)

    if data.ndim>4:
        datacc = np.sqrt(np.sum(np.abs(data) ** 2, axis=4)) / np.sqrt(data.shape[4])
    else:
        datacc = np.abs(data)
    mid_slice = datacc.shape[2]//2
    mym = datacc[:, :, mid_slice-1:mid_slice, :] * mym[:, :, mid_slice-1:mid_slice, :]
    lvm = datacc * lvm

    ind = np.nonzero(mym[:, :, 0, 0] > 0)
    indl = np.nonzero(lvm[:, :, :, 0] > 0)

    sa, sm, sa_ind, sm_ind = [], [], [], []

    for k in range(mym.shape[3]):
        sm.append(np.mean(mym[:, :, 0, k][ind]))
        sm_ind.append(mym[:, :, 0, k][ind][0])
        sa.append(np.mean(lvm[:, :, :, k][indl]))
        sa_ind.append(lvm[:, :, :, k][indl][0])

    return sa, sm, sa_ind, sm_ind


def main():
    display_mrxcat()


if __name__ == "__main__":
    main()