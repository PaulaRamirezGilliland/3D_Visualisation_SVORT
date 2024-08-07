import numpy as np
import argparse
import yaml
from vtkmodules.vtkRenderingCore import vtkPointPicker
from src.utils import *
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

CROSSHAIR_LENGTH = 110

def set_opacity(slice_actors1, slice_actors2, slice_actors3, slice_actors4, slice_actors6, clicked_slice_idx):
    for slice_idx, actor in slice_actors5.items():
        opacity = 1.0 if slice_idx == clicked_slice_idx else 0.1
        actor.GetProperty().SetOpacity(opacity)
        if slice_idx in slice_actors1:
            slice_actors1[slice_idx].GetProperty().SetOpacity(opacity)
        if slice_idx in slice_actors2:
            slice_actors2[slice_idx].GetProperty().SetOpacity(opacity)
        if slice_idx in slice_actors3:
            for actor in slice_actors3[slice_idx]:
                actor.GetProperty().SetOpacity(opacity)

        if slice_idx in slice_actors4:
            if slice_idx != clicked_slice_idx:
                slice_actors4[slice_idx].GetProperty().SetOpacity(0)
            else:
                slice_actors4[slice_idx].GetProperty().SetOpacity(opacity)

        if slice_idx in slice_actors6:
            slice_actors6[slice_idx].GetProperty().SetOpacity(opacity)

    plotter.update()

def click_callback(widget, event_name):
    x, y = plotter.iren.interactor.GetEventPosition()
    picker.Pick(x, y, 0, plotter.renderer)
    picked_position = picker.GetPickPosition()
    closest_slice_idx, min_distance = None, float('inf')

    for slice_idx, actor in slice_actors1.items():
        slice_center = actor.GetCenter()
        distance = np.linalg.norm(np.array(slice_center) - np.array(picked_position))
        if distance < min_distance:
            min_distance = distance
            closest_slice_idx = slice_idx

    if closest_slice_idx is not None:
        set_opacity(slice_actors1, slice_actors2, slice_actors3, slice_actors4, slice_actors6, closest_slice_idx)
        update_crosshairs([0, 0, closest_slice_idx * volume.GetSpacing()[0]])

def nearest_neighbor_interpolate(volume, x, y, z):
    x_idx = np.clip(int(np.round(x)), 0, volume.shape[0] - 1)
    y_idx = np.clip(int(np.round(y)), 0, volume.shape[1] - 1)
    z_idx = np.clip(int(np.round(z)), 0, volume.shape[2] - 1)
    return volume[x_idx, y_idx, z_idx]


def add_slices_to_plotter(plotter, volume_np, useful_slices, transforms, row_index, subplot_index, cmap, opacity,
                          overlay=False, stic_data=None):
    """
    :param plotter: Pyvista plotter
    :param volume_np: (numpy array) containing volume to plot
    :param useful_slices: (list) containing indices of foreground slices
    :param transforms: (list) containing transformation matrices (in list form, for each slice) in order of being applied
    :param row_index: (int) subplot row index
    :param subplot_index: (int) subplot column index
    :param cmap: (str) cmap type for plotting
    :param opacity: (float) baseline opacity
    :param overlay: (bool) set to True if overlaying GT + Pred
    :param stic_data: Numpy array or None - STIC data to be plotted in predicted slice location
    :return: updated Pyvista slice actors
    """
    plotter.subplot(row_index, subplot_index)
    slice_actors = {} if not overlay else {idx: [] for idx in useful_slices}
    transformed_slices = []

    for ind_tr, slice_idx in enumerate(useful_slices):
        slice_data = volume_np[slice_idx, :, :]
        if np.any(slice_data):
            slice_grid = get_slice_grid(slice_data, volume, slice_idx)
            if transforms is None:
                plotted_slice = slice_grid
            else:
                if len(transforms) == 1:
                    final_transforms = transforms[ind_tr]
                else:
                    final_transforms = compose_transform(transforms[0][ind_tr], transforms[1][ind_tr])
                transformed_slice = get_transform_slice(final_transforms, slice_grid)
                plotted_slice = transformed_slice
                if stic_data is not None:
                    transformed_slices.append(plotted_slice)
        slice_actors = add_mesh_to_actor(plotter, overlay, slice_idx, cmap, plotted_slice, opacity, slice_actors,
                                         slice_grid)

    if stic_data is not None:
        # Combine transformed slices into a 3D volume
        grid_shape = volume_np.shape
        combined_volume = np.zeros(grid_shape)

        for transformed_slice in transformed_slices:
            # Get the transformed slice data back as a numpy array
            transformed_slice_data = vtk_to_numpy(transformed_slice.GetPointData().GetScalars())
            transformed_slice_data = transformed_slice_data.reshape((94, 94), order='F')
            # Get the transformed coordinates
            transformed_points = np.array(transformed_slice.points)
            # Convert the transformed coordinates to the corresponding indices in the 3D volume
            origin = np.array(volume.GetOrigin())
            spacing = np.array(volume.GetSpacing())
            indices = (transformed_points - origin) / spacing
            # Ensure indices are within bounds and insert data into the 3D grid
            for i, (x, y, z) in enumerate(indices):
                x_idx = np.clip(int(np.round(x)), 0, grid_shape[0] - 1)
                y_idx = np.clip(int(np.round(y)), 0, grid_shape[1] - 1)
                z_idx = np.clip(int(np.round(z)), 0, grid_shape[2] - 1)
                combined_volume[x_idx, y_idx, z_idx] = transformed_slice_data.flatten()[i]

        # Multiply the populated 3D grid with the 3D volume
        result_volume = combined_volume * stic_data
        # Convert the result back to a vtkVolume
        result_vtk_volume = pv.UniformGrid()
        result_vtk_volume.dimensions = result_volume.shape[::-1]
        result_vtk_volume.spacing = volume.GetSpacing()
        result_vtk_volume.origin = volume.GetOrigin()
        result_vtk_data_array = numpy_to_vtk(result_volume.ravel(order='F'), deep=True)
        result_vtk_data_array.SetName("values")
        result_vtk_volume.GetPointData().SetScalars(result_vtk_data_array)
        # Add the result volume to the plotter
        plotter.add_mesh(result_vtk_volume, cmap=cmap, opacity=opacity)

    return slice_actors

def keyboard_callback(widget, event_name):
    global current_slice_idx
    key = plotter.iren.interactor.GetKeySym()
    if key == "j":
        current_slice_idx = min(current_slice_idx + 1, max(slice_actors4.keys()))
        set_opacity(slice_actors1, slice_actors2, slice_actors3, slice_actors4, slice_actors6, current_slice_idx)
        update_crosshairs([0, 0, current_slice_idx * volume.GetSpacing()[0]])
    elif key == "k":
        current_slice_idx = max(current_slice_idx - 1, min(slice_actors4.keys()))
        set_opacity(slice_actors1, slice_actors2, slice_actors3, slice_actors4, slice_actors6, current_slice_idx)
        update_crosshairs([0, 0, current_slice_idx * volume.GetSpacing()[0]])

def create_crosshair_actors():
    actors = {}
    for axis in ['x', 'y', 'z']:
        line_source = vtk.vtkLineSource()
        if axis == 'x':
            line_source.SetPoint1([-CROSSHAIR_LENGTH / 2, 0, 0])
            line_source.SetPoint2([CROSSHAIR_LENGTH / 2, 0, 0])
        elif axis == 'y':
            line_source.SetPoint1([0, -CROSSHAIR_LENGTH / 2, 0])
            line_source.SetPoint2([0, CROSSHAIR_LENGTH / 2, 0])
        elif axis == 'z':
            line_source.SetPoint1([0, 0, -CROSSHAIR_LENGTH / 2])
            line_source.SetPoint2([0, 0, CROSSHAIR_LENGTH / 2])

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(line_source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 1, 0)  # Yellow color
        actors[axis] = (line_source, actor)

    return actors

def update_crosshairs(position):
    for axis, (line_source, actor) in crosshair_actors.items():
        if axis == 'x':
            line_source.SetPoint1([position[0] - CROSSHAIR_LENGTH / 2, position[1], position[2]])
            line_source.SetPoint2([position[0] + CROSSHAIR_LENGTH / 2, position[1], position[2]])
        elif axis == 'y':
            line_source.SetPoint1([position[0], position[1] - CROSSHAIR_LENGTH / 2, position[2]])
            line_source.SetPoint2([position[0], position[1] + CROSSHAIR_LENGTH / 2, position[2]])
        elif axis == 'z':
            line_source.SetPoint1([position[0], position[1], position[2] - CROSSHAIR_LENGTH / 2])
            line_source.SetPoint2([position[0], position[1], position[2] + CROSSHAIR_LENGTH / 2])

    plotter.update()

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

    volume, volume_np = load_volume(volume_path)
    volume_full, volume_np_full = load_volume(volume_path_full)
    if mask_path is not None:
        mask, mask_np = load_volume(mask_path)
        vol = mask_np * volume_np
    else:
        vol = volume_np

    transforms_gt, transforms_pred = load_transforms(transforms_path)
    pred_inv_transforms = [inv(t) for t in transforms_pred]
    transforms_gt = [t for t in transforms_gt]
    transform_pred = [t for t in transforms_pred]

    useful_slices = get_useful_slices(vol)

    plotter = pv.Plotter(shape=(2, 3))

    plotter.subplot(0, 0)
    plotter.add_text("GT (2D) - right-click to highlight slice", font_size=10)
    slice_actors1 = add_slices_to_plotter(plotter, volume_np, useful_slices, None, 0, 0, "gray", 0.5)

    plotter.subplot(0, 1)
    plotter.add_text("Pred (2D)", font_size=10)
    slice_actors2 = add_slices_to_plotter(plotter, volume_np, useful_slices, [transforms_gt, pred_inv_transforms], 0, 1,
                                          "gray", 0.5)

    plotter.subplot(0, 2)
    plotter.add_text("Pred (2D) - Overlap on GT (2D)", font_size=10)
    slice_actors3 = add_slices_to_plotter(plotter, volume_np, useful_slices, [transforms_gt, pred_inv_transforms], 0, 2,
                                          "gray", 0.0, overlay=True)

    plotter.subplot(1, 0)
    plotter.add_text("STIC (axial slices)", font_size=10)
    slices_3d = [i for i in range(volume_np_full.shape[0]) if np.sum(volume_np_full[i]) > 0]
    slice_actors4 = add_slices_to_plotter(plotter, volume_np_full, slices_3d, None, 1, 0, "gray", 0.0)

    plotter.subplot(1, 1)
    plotter.add_text("STIC (full) - j=up, k=down", font_size=10)
    slice_actors5 = add_slices_to_plotter(plotter, volume_np_full, slices_3d, None, 1, 1, "gray", 0.1)

    plotter.subplot(1, 2)
    plotter.add_text("Binarized Mask * Volume", font_size=10)
    slice_actors6 = add_slices_to_plotter(plotter, volume_np, useful_slices, [transforms_gt, pred_inv_transforms], 1, 2, "gray", 0.1, stic_data=volume_np_full)

    current_slice_idx = min(slice_actors4.keys())

    picker = vtkPointPicker()

    crosshair_actors = create_crosshair_actors()
    for _, actor in crosshair_actors.values():
        for subplot_idx in range(6):
            plotter.subplot(subplot_idx // 3, subplot_idx % 3)
            plotter.renderer.add_actor(actor)

    plotter.iren.add_observer('RightButtonPressEvent', click_callback)
    plotter.iren.add_observer('KeyPressEvent', keyboard_callback)

    plotter.link_views()
    plotter.show()
