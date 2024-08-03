import os
import SimpleITK as sitk
import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkRenderingCore import vtkPointPicker


def load_volume(path):
    """
    Loads 3D image from path
    :param path: (str) path to image file
    :return: simple ITK image, numpy array
    """
    volume = sitk.ReadImage(path)
    volume_np = sitk.GetArrayFromImage(volume)
    return volume, volume_np


def load_transforms(transforms_path):
    """
    Load the predicted and GT transforms given the path
    :param transforms_path: (str) path containing transforms_gt.npy and transforms.npy files
    :return: numpy array (transforms_gt), numpy array (transforms_pred)
    """
    transforms_gt = np.load(os.path.join(transforms_path, 'transforms_gt.npy'))
    transforms_pred = np.load(os.path.join(transforms_path, 'transforms.npy'))
    return transforms_gt, transforms_pred


def get_useful_slices(volume_masked):
    """
    Determines which slices contain foreground information
    :param volume_masked: (numpy array) volume to check foreground
    :return: (list) containing indices of foreground slices
    """
    return [i for i in range(volume_masked.shape[0]) if np.sum(volume_masked[i]) > 0]


def set_opacity(slice_actors1, slice_actors2, slice_actors3, slice_actors4, clicked_slice_idx):
    """
    Function to set the correct opacity, to highlight only relevant slices in all windows

    :param slice_actors1: (Pyvista actor) GT 2D
    :param slice_actors2: (Pyvista actor) Pred 2D
    :param slice_actors3: (list of two Pyvista actors) GT and Pred 2D to overlay
    :param slice_actors4: (Pyvista actor) STIC
    :param clicked_slice_idx: index of the slice to highlight
    """
    for slice_idx, actor in slice_actors4.items():
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

    plotter.update()


def click_callback(widget, event_name):
    """
    Finds the closest slices to the interaction (click event)
    """
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
        set_opacity(slice_actors1, slice_actors2, slice_actors3, slice_actors4, closest_slice_idx)
        update_crosshairs(picked_position)


def inv(mat):
    """
    Computes the inverse of the transformation 
    :param mat: (numpy array) transformation matrix  
    :return: (numpy array) inverse transformation matrix
    """
    R = mat[:, :3]
    t = mat[:, 3:]
    mat_inv = np.concatenate((R.transpose(), -np.matmul(R.transpose(), t)), axis=-1)
    return mat_inv


def add_slices_to_plotter(plotter, volume_np, useful_slices,
                          transforms, inv_transforms, row_index,
                          subplot_index, cmap,  opacity,
                          overlay=False):
    """

    :param plotter: Pyvista plotter
    :param volume_np: (numpy array) containing volume to plot
    :param useful_slices: (list) containing indices of foreground slices
    :param transforms: (numpy array) containing transformation matrix - set to None for GT
    :param inv_transforms: (numpy array) containing inverse of transforms - set to None for GT
    :param row_index: (int) subplot row index
    :param subplot_index: (int) subplot column index
    :param cmap: (str) cmap type for plotting
    :param opacity: (float) baseline opacity
    :param overlay: (bool) set to True if overlaying GT + Pred
    :return: updated Pyvista slice actors
    """
    plotter.subplot(row_index, subplot_index)
    slice_actors = {} if not overlay else {idx: [] for idx in useful_slices}
    for ind_tr, slice_idx in enumerate(useful_slices):
        slice_data = volume_np[slice_idx, :, :]
        if np.any(slice_data):
            slice_grid = pv.UniformGrid()
            slice_grid.dimensions = np.array(slice_data.shape[::-1] + (1,))
            slice_grid.spacing = (volume.GetSpacing()[1], volume.GetSpacing()[2], 1)
            slice_grid.origin = (volume.GetOrigin()[1], volume.GetOrigin()[2], slice_idx * volume.GetSpacing()[0])
            vtk_data_array = numpy_to_vtk(slice_data.ravel(order='F'), deep=True)
            vtk_data_array.SetName("values")
            slice_grid.GetPointData().SetScalars(vtk_data_array)

            if transforms is not None:
                transform = vtk.vtkTransform()
                transform.SetMatrix(np.append(transforms[ind_tr], [0, 0, 0, 1]).flatten())
                transform_filter = vtk.vtkTransformFilter()
                transform_filter.SetInputData(slice_grid)
                transform_filter.SetTransform(transform)
                transform_filter.Update()
                transformed_slice = pv.wrap(transform_filter.GetOutput())

                if inv_transforms is not None:
                    inv_transform = vtk.vtkTransform()
                    inv_transform.SetMatrix(np.append(inv_transforms[ind_tr], [0, 0, 0, 1]).flatten())
                    transform_filter.SetInputData(transformed_slice)
                    transform_filter.SetTransform(inv_transform)
                    transform_filter.Update()
                    transformed_slice = pv.wrap(transform_filter.GetOutput())
            else:
                transformed_slice = slice_grid

            if overlay:
                actor1 = plotter.add_mesh(transformed_slice, cmap='viridis', show_edges=False,
                                          name=f"Pred_Slice_{slice_idx}", opacity=0.0)
                actor2 = plotter.add_mesh(slice_grid, cmap=cmap, show_edges=False, name=f"GT_Slice_{slice_idx}")
                plotter.add_actor(actor1)
                plotter.add_actor(actor2)
                actor1.GetProperty().SetOpacity(opacity)
                actor2.GetProperty().SetOpacity(opacity)

                slice_actors[slice_idx].append(actor1)
                slice_actors[slice_idx].append(actor2)

            else:
                actor = plotter.add_mesh(transformed_slice, cmap=cmap, show_edges=False, name=f"Slice_{slice_idx}")
                actor.GetProperty().SetOpacity(opacity)
                slice_actors[slice_idx] = actor

    return slice_actors


def keyboard_callback(widget, event_name):
    """
    Set correct opacity for keyboard events

    """
    global current_slice_idx
    key = plotter.iren.interactor.GetKeySym()
    if key == "j":
        current_slice_idx = min(current_slice_idx + 1, max(slice_actors4.keys()))
        set_opacity(slice_actors1, slice_actors2, slice_actors3, slice_actors4, current_slice_idx)
        update_crosshairs([0, 0, current_slice_idx * volume.GetSpacing()[0]])
    elif key == "k":
        current_slice_idx = max(current_slice_idx - 1, min(slice_actors4.keys()))
        set_opacity(slice_actors1, slice_actors2, slice_actors3, slice_actors4, current_slice_idx)
        update_crosshairs([0, 0, current_slice_idx * volume.GetSpacing()[0]])


def create_crosshair_actors():
    """
    Creates crosshairs across all actors for selected slices
    """
    actors = {}
    for axis in ['x', 'y', 'z']:
        line_source = vtk.vtkLineSource()
        line_source.SetPoint1([0, 0, 0])
        line_source.SetPoint2([1, 0, 0])  # Arbitrary length, will be updated

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(line_source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 1, 0)  # Yellow color
        actors[axis] = (line_source, actor)  # Store both line_source and actor

    return actors


def update_crosshairs(position):
    """
    Update the position of the crosshairs for a keyboard event

    """
    for axis, (line_source, actor) in crosshair_actors.items():
        if axis == 'x':
            line_source.SetPoint1([position[0] - 1000, position[1], position[2]])
            line_source.SetPoint2([position[0] + 1000, position[1], position[2]])
        elif axis == 'y':
            line_source.SetPoint1([position[0], position[1] - 1000, position[2]])
            line_source.SetPoint2([position[0], position[1] + 1000, position[2]])
        elif axis == 'z':
            line_source.SetPoint1([position[0], position[1], position[2] - 1000])
            line_source.SetPoint2([position[0], position[1], position[2] + 1000])

    plotter.update()


# Paths
volume_path = "C:\\Users\\prg20local\\OneDrive - King's College London\\Research Project\\PhD\\US_recon\\Data\\redo_preproc_STIC_Jan24\\reg-mvp-FEB24-all-data\\reg-ED-downsampled\\iFIND00212_27Jan2017\\r2-reg-2d-3d-masked-1.nii.gz"
volume_path_full = "C:\\Users\\prg20local\\OneDrive - King's College London\\Research Project\\PhD\\US_recon\\Data\\redo_preproc_STIC_Jan24\\reg-mvp-FEB24-all-data\\reg-ED-downsampled\\iFIND00212_27Jan2017\\reg-IM_0170-ED-iso-1.nii.gz"
mask_path = "C:\\Users\\prg20local\\OneDrive - King's College London\\Research Project\\PhD\\US_recon\\Data\\redo_preproc_STIC_Jan24\\reg-mvp-FEB24-all-data\\reg-ED-downsampled\\iFIND00483_22Feb2019\\Repeat_mask-ED-iso-1.nii.gz"
transforms_path = "C:\\Users\\prg20local\\OneDrive - King's College London\\Research Project\\PhD\\US_recon\\Experiments\\3D_2D_Jun24\\ablation_studies\\700000+iters\\SVoRT_3D_axialview_jitter45normal_ptsliceloss_res1_b8_sparsein_noPE\\test2D\\test_unseen_overlap\\maskheart\\r1\\1\\"

# Load volumes and masks
volume, volume_np = load_volume(volume_path)
volume_full, volume_np_full = load_volume(volume_path_full)
mask, mask_np = load_volume(mask_path)
volume_masked = mask_np * volume_np

# Load transforms
transforms_gt, transforms_pred = load_transforms(transforms_path)

# Identify useful slices
useful_slices = get_useful_slices(volume_masked)

# Create a PyVista plotter with two side-by-side viewports
plotter = pv.Plotter(shape=(2, 3))

# GT (2D)
plotter.subplot(0, 0)
plotter.add_text("GT (2D) - right-click to highlight slice", font_size=10)
slice_actors1 = add_slices_to_plotter(plotter, volume_np, useful_slices, None, None, 0, 0, "gray", 0.5)

# Pred (2D)
plotter.subplot(0, 1)
plotter.add_text("Pred (2D)", font_size=10)
pred_inv_transforms = [inv(t) for t in transforms_pred]
slice_actors2 = add_slices_to_plotter(plotter, volume_np, useful_slices, transforms_gt, pred_inv_transforms, 0, 1,
                                      "gray", 0.5)

# Pred (2D) - Overlap on GT (2D)
plotter.subplot(0, 2)
plotter.add_text("Pred (2D) - Overlap on GT (2D)", font_size=10)
slice_actors3 = add_slices_to_plotter(plotter, volume_np, useful_slices, transforms_gt, pred_inv_transforms, 0, 2,
                                      "gray", 0.0, overlay=True)

# STIC (axial slices)
plotter.subplot(1, 0)
plotter.add_text("STIC (axial slices)", font_size=10)
slices_3d = [i for i in range(volume_np_full.shape[0]) if np.sum(volume_np_full[i]) > 0]
slice_actors4 = add_slices_to_plotter(plotter, volume_np_full, slices_3d, None, None, 1, 0, "gray", 0.0)

# STIC (full)
plotter.subplot(1, 1)
plotter.add_text("STIC (full) - j=up, k=down", font_size=10)
slice_actors5 = add_slices_to_plotter(plotter, volume_np_full, slices_3d, None, None, 1, 1, "gray", 0.1)

# Initialize the current slice index
current_slice_idx = min(slice_actors4.keys())

# Set up the point picker and click callback
picker = vtkPointPicker()

# Create and add the crosshairs actors
crosshair_actors = create_crosshair_actors()
for _, actor in crosshair_actors.values():
    for subplot_idx in range(6):
        plotter.subplot(subplot_idx // 3, subplot_idx % 3)
        plotter.renderer.add_actor(actor)

# Add mouse and keyboard observers
plotter.iren.add_observer('RightButtonPressEvent', click_callback)
plotter.iren.add_observer('KeyPressEvent', keyboard_callback)

# Link views and show
plotter.link_views()
plotter.show()
