import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from pymrzeroxcat.MRXCAT_PHANTOM_LGE.build_static_phantom import get_segmentation


def load_segmentation(bin_path, shape):
    data = np.fromfile(bin_path, dtype=np.uint8).reshape(shape)
    return data


def visualize_infarct_and_myocardium(seg, infarct_label):
    shape = seg.shape
    current_slice = [0]

    myocardium_label = 1
    print(f'User label {myocardium_label} for left myocardium and label {infarct_label} for infarct.')

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)
    img = ax.imshow(np.zeros_like(seg[..., 0]), cmap='gray')

    def update_display():
        slice_idx = current_slice[0]
        base = seg[..., slice_idx]

        rgb = np.stack([base * 0.2, base * 0.2, base * 0.2], axis=-1)  # grayscale base
        myocardium_mask = (base == myocardium_label)
        infarct_mask_rgb = (base == infarct_label)

        rgb[myocardium_mask] = [0.6, 1.0, 0.6]    # light green myocardium
        rgb[infarct_mask_rgb] = [1.0, 0.0, 0.0]   # red infarct (overrides myocardium)

        img.set_data(rgb)
        ax.set_title(f"Slice {slice_idx + 1} / {shape[2]}")
        fig.canvas.draw_idle()

    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, shape[2] - 1, valinit=0, valstep=1)

    def on_slider(val):
        current_slice[0] = int(val)
        update_display()
        
    def on_key(event):
        if event.key == 'right':
            if current_slice[0] < shape[2] - 1:
                current_slice[0] += 1
                update_display()
        elif event.key == 'left':
            if current_slice[0] > 0:
                current_slice[0] -= 1
                update_display()
            
    slider.on_changed(on_slider)
    fig.canvas.mpl_connect('key_press_event', on_key)
    update_display()
    plt.show()



def main():
    parser = argparse.ArgumentParser('Add Infarct Mask to a MRXCAT segmentation')
    parser.add_argument('bin_file', help='Input binary (.bin) file for phantom generation')
    parser.add_argument('--log_file', help='Input log (_log) file', default=None)
    parser.add_argument('--infarct_label', help='Infarct label', default=None)
    
    args = parser.parse_args()
    
    if args.log_file is None:
        if args.bin_file.endswith('_with_inf.bin'): # default naming suffix for mask with infarct
            log_file = log_file = '_'.join(args.bin_file.split('_')[:-4]) + '_log'
        else:
            log_file = '_'.join(args.bin_file.split('_')[:-2]) + '_log'
        if not os.path.isfile(log_file):
            raise FileNotFoundError(f"Auto-generated log file '{log_file}' does not exist. Please provide one using --log_file.")
    else:
        log_file = args.log_file
    
    seg = get_segmentation(args.bin_file, log_file, flip_horizontal=False, swap_xy=True)
    
    if args.infarct_label is None:
        infarct_label = seg.max()
    else:
        infarct_label = args.infarct_label
        
    visualize_infarct_and_myocardium(seg, infarct_label)
    
# === Example usage ===
if __name__ == "__main__":
    main()
