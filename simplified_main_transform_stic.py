import argparse
import yaml
from vtk.util.numpy_support import numpy_to_vtk
from src.interactive import *

CROSSHAIR_LENGTH = 110

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config file (.yaml) containing directories")
    args = parser.parse_args()
    with open(args.config) as cf_file:
        CONFIG = yaml.safe_load(cf_file.read())
        print(CONFIG)

    volume_path = CONFIG['2D_volume_path']
    volume_path_full = CONFIG['STIC_path']
    mask_path = CONFIG['mask_path']
    transforms_path = CONFIG['transforms_path']
    convert_transform = CONFIG['convert_transform']

    print("Volume_path", volume_path)

    # Load volumes and masks
    volume, volume_np = load_volume(volume_path)
    volume_full, volume_np_full = load_volume(volume_path_full)
    if mask_path is not None:
        mask, mask_np = load_volume(mask_path)
        vol = mask_np * volume_np

    else:
        vol = volume_np

    # Load transforms
    transforms_gt, transforms_pred = load_transforms(transforms_path)
    transforms_gt = [t for t in transforms_gt]
    transforms_pred = [t for t in transforms_pred]
    pred_inv_transforms = [inv(t) for t in transforms_pred]
    gt_inv_transforms = [inv(t) for t in transforms_gt]


    # Identify useful slices
    useful_slices = get_useful_slices(vol)

    # Convert and save transform and slices to be read by other toolkits
    if convert_transform:
        save_for_slicer(useful_slices, volume_np, volume, transforms_path, transforms_gt, pred_inv_transforms)

    # Create a PyVista plotter with two side-by-side viewports
    plotter = pv.Plotter(shape=(2, 3))

    # GT (2D)
    plotter.subplot(0, 0)
    plotter.add_text("GT (2D) - right-click to highlight slice", font_size=10)
    slice_actors1, _, _, _ = add_slices_to_plotter(plotter, volume_np, useful_slices, volume,
                                                   None, None, 0, 0, "gray", 0.5)

    # Pred (2D)
    plotter.subplot(0, 1)
    plotter.add_text("Pred (2D)", font_size=10)

    slice_actors2, coords_pred, grid_spacing, grid_origin = add_slices_to_plotter(plotter, volume_np, useful_slices, volume,
                                                                                  transforms_gt, pred_inv_transforms, 0,
                                                                                  1,
                                                                                  "gray", 0.5)

    # Pred (2D) - Overlap on GT (2D)
    plotter.subplot(0, 2)
    plotter.add_text("Pred (2D) - Overlap on GT (2D)", font_size=10)
    slice_actors3, _, _, _ = add_slices_to_plotter(plotter, volume_np, useful_slices,
                                                   volume, transforms_gt,
                                                   pred_inv_transforms, 0, 2,
                                                   "gray", 0.1, overlay=True)

    # STIC (axial slices)
    plotter.subplot(1, 0)
    plotter.add_text("STIC (axial slices)", font_size=10)
    slices_3d = [i for i in range(volume_np_full.shape[0]) if np.sum(volume_np_full[i]) > 0]
    slice_actors4, _, _, _ = add_slices_to_plotter(plotter, volume_np_full, slices_3d,
                                                   volume, None, None, 1, 0, "gray", 0.0)

    # STIC (full)
    plotter.subplot(1, 1)
    plotter.add_text("STIC (full) - j=up, k=down", font_size=10)
    slice_actors5, _, _, _ = add_slices_to_plotter(plotter, volume_np_full, slices_3d,
                                                   volume, None, None, 1, 1, "gray", 0.1)

    # STIC (predicted)
    plotter.subplot(1, 2)
    plotter.add_text("STIC (predicted) - j=up, k=down", font_size=10)

    slice_actors6 = compute_transformed_stic(plotter, volume, volume_np_full, coords_pred,
                             useful_slices, transforms_gt, pred_inv_transforms,
                             1, 2, "gray", 0.1)


    # Create and add the crosshairs actors
    crosshair_actors = create_crosshair_actors(CROSSHAIR_LENGTH)
    for _, actor in crosshair_actors.values():
        for subplot_idx in range(6):
            plotter.subplot(subplot_idx // 3, subplot_idx % 3)
            plotter.renderer.add_actor(actor)

    current_slice_idx = [min(slice_actors5.keys())]

    # Add mouse and keyboard observers
    plotter.iren.add_observer('RightButtonPressEvent',
                              lambda w, e: click_callback(w, e, plotter, volume, slice_actors1, slice_actors5,
                                                          [slice_actors1, slice_actors2, slice_actors3, slice_actors6],
                                                          crosshair_actors, CROSSHAIR_LENGTH, current_slice_idx, slice_actors4))

    plotter.iren.add_observer('KeyPressEvent',
                              lambda w, e: keyboard_callback(w, e, plotter, volume, slice_actors4,
                                                             [slice_actors1, slice_actors2, slice_actors3, slice_actors6],
                                                             crosshair_actors, CROSSHAIR_LENGTH, current_slice_idx, slice_actors4))

    # Link views and show
    plotter.link_views()
    plotter.show()