import os
import SimpleITK as sitk
import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from scipy.interpolate import RegularGridInterpolator


def transform_slice(transforms, inv_transforms, ind_tr, slice_grid, grid_points=None):
    """
    Transforms a  slice grid by applying transforms
    :param transforms: (numpy array) First transformation matrix to apply
    :param inv_transforms: (numpy array) Second transformation matrix to apply (slice to volume)
    :param ind_tr: (int) index of the transformation
    :param slice_grid: Pyvista uniform grid containing slice data to plot
    :param grid_points: original grid points (before transformation)

    :return: (transformed slice, transformed coordinate points)
    """
    if transforms is not None:
        composed_transform = transforms[ind_tr]
        if inv_transforms is not None:
            composed_transform = compose_transform(transforms[ind_tr], inv_transforms[ind_tr])
        transform = vtk.vtkTransform()
        transform.SetMatrix(np.append(composed_transform, [0, 0, 0, 1]).flatten())
        transform_filter = vtk.vtkTransformFilter()
        transform_filter.SetInputData(slice_grid)
        transform_filter.SetTransform(transform)
        transform_filter.Update()
        transformed_slice = pv.wrap(transform_filter.GetOutput())

        # Extract transformed points for the slice
        transformed_points = transformed_slice.points

    else:
        transformed_slice = slice_grid
        transformed_points = grid_points

    return transformed_slice, transformed_points


def add_slices_to_plotter(plotter, volume_np, useful_slices, volume,
                          transforms, inv_transforms, row_index,
                          subplot_index, cmap, opacity,
                          overlay=False):
    """
    :param plotter: Pyvista plotter
    :param volume_np: (numpy array) containing volume to plot
    :param useful_slices: (list) containing indices of foreground slices
    :param volume: SimpleITK volume containing spacing and origin information
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
    slice_coordinates = {}  # To store the coordinates of each slice
    slice_spacing = {}
    slice_origin = {}

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
            # Extract the original grid points
            grid_points = slice_grid.points
            slice_spacing[slice_idx] = slice_grid.spacing
            slice_origin[slice_idx] = slice_grid.origin

            transformed_slice, transformed_points = transform_slice(transforms, inv_transforms,
                                                                    ind_tr, slice_grid, grid_points)

            # Store the coordinates of the transformed slice
            slice_coordinates[slice_idx] = transformed_points
            add_mesh_to_actor(plotter, overlay, slice_idx, cmap, transformed_slice,
                                             opacity, slice_actors, slice_grid)

    return slice_actors, slice_coordinates, slice_spacing, slice_origin


def compute_transformed_stic(plotter, volume, volume_np_full, coords_pred,
                               useful_slices, transforms, inv_transforms,
                               row_index, subplot_index, cmap, opacity):
    """
    :param plotter: Pyvista plotter
    :param volume: (SimpleITK image) with spacing and origin properties
    :param volume_np_full: (Numpy array) containing data for full 3D acquisition
    :param coords_pred: (Numpy array) containing 3D coordinates for predicted 2D slice locations
    :param useful_slices: (list) of slice indices with 2D data
    :param transforms: (numpy array) containing transformation matrix
    :param inv_transforms: (numpy array) containing inverse of transforms
    :param row_index: (int) subplot row index
    :param subplot_index: (int) subplot column index
    :param cmap: (str) cmap type for plotting
    :param opacity: (float) baseline opacity

    :return: updated Pyvista slice actors
    """

    plotter.subplot(row_index, subplot_index)
    slice_actors = {}

    origin = np.array(volume.GetOrigin())  # Extract origin from the volume
    spacing = np.array(volume.GetSpacing())  # Extract spacing from the volume

    # Generate the grid for each dimension
    # z = np.arange(volume_np.shape[0]) * spacing[2] + origin[2]
    z = np.arange(volume_np_full.shape[0]) * spacing[0]
    y = np.arange(volume_np_full.shape[2]) * spacing[2] + origin[2]
    x = np.arange(volume_np_full.shape[1]) * spacing[1] + origin[1]

    interpolator = RegularGridInterpolator((z, y, x), volume_np_full, method='cubic', bounds_error=False, fill_value=0)

    #interpolator = RegularGridInterpolator((z, y, x), volume_np_full, method='linear', bounds_error=False,
    #                                       fill_value=None)

    for ind_tr, slice_idx in enumerate(useful_slices):
        x_coords = coords_pred[slice_idx][:, 0]
        y_coords = coords_pred[slice_idx][:, 1]
        z_coords = coords_pred[slice_idx][:, 2]

        # array_coords = physical_to_index(coords_pred[slice_idx], volume.GetOrigin(), volume.GetSpacing())
        data_value = interpolator((z_coords, y_coords, x_coords))
        slice_data = data_value.reshape(94, 94)
        slice_data[slice_data < 0] = 0

        slice_grid = pv.UniformGrid()
        slice_grid.dimensions = np.array(slice_data.shape[::-1] + (1,))
        slice_grid.spacing = (volume.GetSpacing()[1], volume.GetSpacing()[2], 1)
        slice_grid.origin = (volume.GetOrigin()[1], volume.GetOrigin()[2], slice_idx * volume.GetSpacing()[0])

        vtk_data_array = numpy_to_vtk(slice_data.ravel(order='F'), deep=True)
        vtk_data_array.SetName("values")
        slice_grid.GetPointData().SetScalars(vtk_data_array)

        transformed_slice, transformed_points = transform_slice(transforms, inv_transforms, ind_tr, slice_grid)

        add_mesh_to_actor(plotter, False, slice_idx, cmap, transformed_slice,
                                         opacity, slice_actors)

    return slice_actors


def load_volume(file_path, return_aff=False):
    """
    Load a NIfTI volume using SimpleITK.

    :param file_path: Path to the NIfTI file.

    :return: Tuple containing the image data and the affine matrix.
    """
    # Read the image with SimpleITK
    itk_image = sitk.ReadImage(file_path)
    volume_np = sitk.GetArrayFromImage(itk_image).astype(np.float32)  # Ensure data type
    spacing = itk_image.GetSpacing()
    origin = itk_image.GetOrigin()
    direction = np.array(itk_image.GetDirection()).reshape((3, 3))

    # Create an affine transformation matrix
    affine = np.eye(4)
    affine[:3, :3] = direction * np.array(spacing).reshape((3, 1))
    affine[:3, 3] = origin

    if return_aff:
        return itk_image, volume_np, affine
    else:
        return itk_image, volume_np


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


def compose_transform(trans1, trans2):
    """
    Computes the final transformation by applying composing transformations
    :param trans1: (numpy array) first transformation matrix
    :param trans2: (numpy array) subsequent transformation matrix, applied after the first

    :return: (numpy array) final transformation matrix
    """
    final_transform = np.dot(trans1, np.append(trans2, [[0, 0, 0, 1]], axis=0))  # Homogeneous coordinates
    return final_transform


def add_mesh_to_actor(plotter, overlay, slice_idx, cmap, plotted_slice, opacity, slice_actors, slice_grid=None):
    """
    Adds a slice as a mesh to the plotter
    :param plotter: Pyvista plotter
    :param overlay: (bool) set to True if overlaying GT + Pred
    :param slice_idx: (int) slice index
    :param cmap: (str) cmap type for plotting
    :param plotted_slice: vtk object containing slice grid to be plotted
    :param opacity: (float) baseline opacity
    :param slice_actors: Pyvista actor for specific slice being plotted
    :param slice_grid: Pyvista uniform grid, second slice plotted if overlay


    """
    if overlay:
        actor1 = plotter.add_mesh(plotted_slice, cmap='viridis', show_edges=False,
                                  name=f"Pred_Slice_{slice_idx}", opacity=0.0)
        actor2 = plotter.add_mesh(slice_grid, cmap=cmap, show_edges=False, name=f"GT_Slice_{slice_idx}")
        plotter.add_actor(actor1)
        plotter.add_actor(actor2)
        actor1.GetProperty().SetOpacity(opacity)
        actor2.GetProperty().SetOpacity(opacity)

        slice_actors[slice_idx].append(actor1)
        slice_actors[slice_idx].append(actor2)

    else:
        actor = plotter.add_mesh(plotted_slice, cmap=cmap, show_edges=False, name=f"Slice_{slice_idx}")
        actor.GetProperty().SetOpacity(opacity)
        slice_actors[slice_idx] = actor


def save_transform(transform, ind_tr, transforms_path, save_name = "transform-slice-"):
    # TODO- Parametrize save name
    """
    Converts and saves a numpy array transformation simpleitk transform
    :param transform: numpy array transformation matrix
    :param ind_tr: (int) Slice index
    :param transforms_path: (path) to save transforms
    :param save_name: (str) filename for saved transform

    """
    # Convert your numpy matrix to a SimpleITK Transform
    np_transform = np.append(transform, [0, 0, 0, 1]).reshape((4, 4))
    sitk_transform = sitk.AffineTransform(3)  # Assuming 3D affine transform
    sitk_transform.SetMatrix(np_transform[:3, :3].flatten())
    sitk_transform.SetTranslation(np_transform[:3, 3])

    # Save the transform as a .tfm file
    sitk.WriteTransform(sitk_transform, os.path.join(transforms_path, save_name + str(ind_tr) + '.tfm'))


def save_slice(volume_np, volume, index_slice, ind_tr, transforms_path):
    """
    Saves a 3D volume containing only the slice found in index slice
    :param volume_np: (numpy array) containing slices to save
    :param volume: (SimpleITK image) containing origin and spacing information
    :param index_slice: (int) index of the slice to save
    :param ind_tr: (int) index of the transformation slice
    :param transforms_path: (path) to save volume file

    """
    slice_vol = np.zeros_like(volume_np)
    slice_vol[index_slice, ...] = volume_np[index_slice, ...]
    slice_img = sitk.GetImageFromArray(slice_vol)
    slice_img.CopyInformation(volume)
    sitk.WriteImage(slice_img, os.path.join(transforms_path, '2D-slice-' + str(ind_tr) + '.nii.gz'))


def save_for_slicer(useful_slices, volume_np, volume,
                    transforms_path, transforms_gt, transforms_pred,
                    gt_inv_transforms, pred_inv_transforms
                    ):
    """
    Save nifti files for each slice, as well as the composed GT + predicted inverse transforms as .vtk files
    :param useful_slices: (list) of ints, for slice indices to save
    :param volume_np: (numpy array) containing slices to save
    :param volume: (SimpleITK image) containing origin and spacing information
    :param transforms_path: (path) to save volume files and transformations
    :param transforms_gt: (list) or numpy array containing GT transformations
    :param transforms_pred: (list or numpy array containing the predicted transformations
    :param gt_inv_transforms: (list) or numpy array containing inverse of GT transformations
    :param pred_inv_transforms: (list or numpy array containing inverse of predicted transformations


    """
    for ind_tr, index_slice in enumerate(useful_slices):
        save_slice(volume_np, volume, index_slice, ind_tr, transforms_path)
        # Compose the transform
        transform_convert = compose_transform(transforms_gt[ind_tr], pred_inv_transforms[ind_tr])
        # Save into Slicer compatible format
        save_transform(transform_convert, ind_tr, transforms_path)
        # Save transforms to be applied to STIC
        transform_convert_stic = compose_transform(transforms_pred[ind_tr], gt_inv_transforms[ind_tr])
        # Save into Slicer compatible format
        save_transform(transform_convert_stic, ind_tr, transforms_path, save_name='stic-tr-')