import numpy as np
import matplotlib.pyplot as plt
import os

from pymrxcat.visualize import display_movie

def display_phantom_data(fname=None, plot_result=True):
    if fname is None or not os.path.isfile(fname):
        # Get the phantom output (.npz file)
        from tkinter import filedialog
        from tkinter import Tk
        root = Tk()
        root.withdraw()  # Hide the root window
        fname = filedialog.askopenfilename(filetypes=[("Phantom Files", "*.npz")])
    
    if fname is None or not os.path.isfile(fname):
        print("No file selected")
        return None, None
    
    # Load phantom data from the .npz file
    phantom_data = np.load(fname)
    
    # Extract the data from the .npz file
    rho_map = phantom_data['rho_map']
    t1_maps = phantom_data['t1_maps']
    t2_maps = phantom_data['t2_maps']
    sen = phantom_data['sen']
    
    
    # Display 4-panel figure showing time frame, slices, coil images, and (for perf) example signal-time curves
    if plot_result:
        plt.figure(figsize=(10, 8))
        mid_slice = rho_map.shape[2]//2
        # Display rho_map (proton density map)
        plt.subplot(2, 2, 1)
        # plt.imshow(rho_map[..., mid_slice], cmap='gray')
        display_movie(rho_map, [2,2,1], loops=1, dt=0.2)
        plt.title('Rho Map')
        plt.axis('off')
        
        # Display T1 maps (showing one slice)
        plt.subplot(2, 2, 2)
        # plt.imshow(t1_maps[..., mid_slice], cmap='hot')  # Displaying the first T1 map
        display_movie(t1_maps[..., mid_slice,:], [2,2,2], loops=1, dt=0.2)
        plt.title('T1 Map')
        plt.axis('off')
        
        # Display T2 maps (showing one slice)
        plt.subplot(2, 2, 3)
        # plt.imshow(t2_maps[..., mid_slice], cmap='hot')  # Displaying the first T2 map
        display_movie(t2_maps[..., mid_slice,:], [2,2,3], loops=1, dt=0.2)
        plt.title('T2 Map')
        plt.axis('off')
        
        # Display sensitivity map (coil-combined image)
        plt.subplot(2, 2, 4)
        display_movie(np.abs(sen[..., mid_slice, :]), [2,2,4], loops=1, dt=1)
        plt.title('Sensitivity Map (Coil Combined)')
        plt.axis('off')

        plt.show()

    return phantom_data, fname

def display_movie(data, subplotno, loops=3, dt=0.1, scale=(None, None)):
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

    for _ in range(loops):
        for imageno in range(number_images):
            plt.subplot(gridsizex, gridsizey, image_pos_no)
            plt.imshow(data[:, :, imageno], cmap='gray', vmin=minint, vmax=maxint)
            plt.title(f'{imageno + 1}/{number_images}')
            plt.axis('off')
            plt.pause(dt)

    imageno = number_images // 2
    plt.subplot(gridsizex, gridsizey, image_pos_no)
    plt.imshow(data[:, :, imageno], cmap='gray', vmin=minint, vmax=maxint)
    plt.title(f'{imageno + 1}/{number_images}')
    plt.axis('off')


def main():
    display_phantom_data()

if __name__ == "__main__":
    main()