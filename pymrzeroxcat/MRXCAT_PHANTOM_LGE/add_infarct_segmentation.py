import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RectangleSelector, Button
from skimage.measure import find_contours

from pymrzeroxcat.read_mrxcat_raw_data import get_segmentation, save_segmentation, resolve_log_file

def load_segmentation(bin_path, shape):
    seg = np.fromfile(bin_path, dtype=np.float32).astype(np.uint8)
    seg = seg.reshape(shape)
    return seg

def annotate_infarct(seg, initial_infarct_radius=1):
    shape = seg.shape
    infarct_mask = np.zeros_like(seg, dtype=np.uint8)
    current_slice = [0]
    drawing = [False]
    infarct_radius = [initial_infarct_radius]
    zoom_bounds = {}
    erase_mode = [False]

    # === Initial zoom selection ===
    def onselect(eclick, erelease):
        x0, y0 = int(eclick.xdata), int(eclick.ydata)
        x1, y1 = int(erelease.xdata), int(erelease.ydata)
        zoom_bounds['x0'], zoom_bounds['x1'] = sorted([x0, x1])
        zoom_bounds['y0'], zoom_bounds['y1'] = sorted([y0, y1])
        plt.close()

    fig_zoom, ax_zoom = plt.subplots()
    ax_zoom.set_title("Select ROI to zoom — drag rectangle and release")
    ax_zoom.imshow(seg[..., 0], cmap='gray')
    toggle_selector = RectangleSelector(ax_zoom, onselect, useblit=True,
                                        button=[1], minspanx=5, minspany=5,
                                        spancoords='pixels', interactive=True)
    plt.show()

    if not zoom_bounds:
        print("❌ No region selected. Exiting.")
        return

    x0, x1 = zoom_bounds['x0'], zoom_bounds['x1']
    y0, y1 = zoom_bounds['y0'], zoom_bounds['y1']

    # === Main annotation interface ===
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    img_ax = ax.imshow(seg[y0:y1, x0:x1, current_slice[0]], cmap='gray')
    overlay_img = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.float32)
    overlay = ax.imshow(overlay_img)

    # Myocardium contour overlay (green)
    myocardium_contour = []

    def draw_myocardium_contour():
        nonlocal myocardium_contour
        for line in myocardium_contour:
            line.remove()
        myocardium_contour.clear()
        mask = (seg[y0:y1, x0:x1, current_slice[0]] == 1).astype(np.uint8)
        contours = find_contours(mask, 0.5)
        for contour in contours:
            line, = ax.plot(contour[:, 1], contour[:, 0], color='lightgreen', linewidth=0.7)
            myocardium_contour.append(line)

    # === Slider ===
    ax_slider = plt.axes([0.25, 0.08, 0.5, 0.03])
    slider = Slider(ax_slider, 'Infarct Radius', 0, 5, valinit=initial_infarct_radius, valstep=1)
    slider.on_changed(lambda val: infarct_radius.__setitem__(0, int(val)))

    # === Copy Prev Slice Button ===
    ax_button = plt.axes([0.8, 0.02, 0.15, 0.05])
    btn = Button(ax_button, "Copy Prev Slice")

    def copy_prev_slice(event):
        z = current_slice[0]
        if z > 0:
            myocardium_mask = (seg[..., z] == 1)
            infarct_mask[..., z] = np.where(myocardium_mask,
                                            infarct_mask[..., z - 1],
                                            0)
            print(f"📋 Copied infarct mask from slice {z} inside myocardium.")
            update_overlay()

    btn.on_clicked(copy_prev_slice)

    # === Update overlay ===
    def update_overlay():
        mask_slice = infarct_mask[y0:y1, x0:x1, current_slice[0]]
        overlay_img[..., 0] = mask_slice
        overlay_img[..., 1] = 0
        overlay_img[..., 2] = 0
        overlay_img[..., 3] = mask_slice * 0.6
        overlay.set_data(overlay_img)
        draw_myocardium_contour()
        fig.canvas.draw_idle()

    # === Painting function ===
    def paint(event):
        if not drawing[0] or event.xdata is None or event.ydata is None:
            return
        x = int(event.xdata) + x0
        y = int(event.ydata) + y0
        radius = infarct_radius[0]
        edited_pixels = 0

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                yy = y + dy
                xx = x + dx
                if 0 <= yy < shape[0] and 0 <= xx < shape[1]:
                    if erase_mode[0]:
                        if infarct_mask[yy, xx, current_slice[0]] == 1:
                            infarct_mask[yy, xx, current_slice[0]] = 0
                            edited_pixels += 1
                    else:
                        if seg[yy, xx, current_slice[0]] == 1 and infarct_mask[yy, xx, current_slice[0]] == 0:
                            infarct_mask[yy, xx, current_slice[0]] = 1
                            edited_pixels += 1

        if edited_pixels > 0:
            action = "Erased" if erase_mode[0] else "Painted"
            print(f"{action} {edited_pixels} pixels at slice {current_slice[0]+1}, position ({x},{y})")
            update_overlay()

    # === Event handlers ===
    def on_press(event):
        drawing[0] = True
        erase_mode[0] = (event.key == 'shift')
        paint(event)

    def on_release(event): drawing[0] = False
    def on_motion(event): paint(event)

    def on_key(event):
        if event.key == 'right' and current_slice[0] < shape[2] - 1:
            current_slice[0] += 1
            update_slice()
        elif event.key == 'left' and current_slice[0] > 0:
            current_slice[0] -= 1
            update_slice()
        elif event.key == 'escape':
            print("Exiting without saving.")
            plt.close(fig)

    def update_slice():
        ax.set_title(f"Slice {current_slice[0]+1} / {shape[2]} — Radius: {infarct_radius[0]}")
        img_ax.set_data(seg[y0:y1, x0:x1, current_slice[0]])
        update_overlay()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('key_press_event', on_key)

    update_overlay()
    plt.show()
    return infarct_mask


def main():
    parser = argparse.ArgumentParser('Add Infarct Mask to a MRXCAT segmentation')
    parser.add_argument('bin_file', help='Input binary (.bin) file for phantom generation')
    parser.add_argument('--log_file', help='Input log (_log) file', default=None)
    parser.add_argument('--output_bin_file', help='Output bin (.bin) file', default=None)
    parser.add_argument('--infarct_label', help='Infarct label', default=None)
    
    args = parser.parse_args()
    
    if args.log_file is None:
        log_file = resolve_log_file(args.bin_file)
    else:
        log_file = args.log_file
    
    if args.output_bin_file is None:
        output_bin_file = args.bin_file[:-4] + '_with_inf.bin'
    else:
        output_bin_file = args.output_bin_file
            
    seg = get_segmentation(args.bin_file, log_file, flip_horizontal=False, swap_xy=True)
    infarct_mask = annotate_infarct(seg)
    
    if args.infarct_label is None:
        # Find max label in original seg to assign new infarct label
        max_label = seg.max()
        infarct_label = max_label + 1
    else:
        infarct_label = args.infarct_label

    # Create new segmentation with infarct label added
    new_seg = seg.copy()
    new_seg[infarct_mask == 1] = infarct_label

    # Save new segmentation with infarct included
    save_segmentation(new_seg, output_bin_file, log_file, flip_horizontal=False, swap_xy=True)


# === Entry point ===
if __name__ == "__main__":
    main()