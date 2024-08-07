import os
import SimpleITK as sitk
import numpy as np
import vtk
import pyvista as pv
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


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


def add_volume_to_plotter(plotter, volume_np, affine, row_index, subplot_index, cmap, opacity):
    """
    Add a 3D volume to the PyVista plotter.

    :param plotter: Pyvista plotter
    :param volume_np: (numpy array) containing the volume data
    :param affine: (numpy array) affine transformation matrix
    :param row_index: (int) subplot row index
    :param subplot_index: (int) subplot column index
    :param cmap: (str) colormap type for plotting
    :param opacity: (float) opacity for volume rendering
    """
    plotter.subplot(row_index, subplot_index)

    # Dimensions are set based on the data shape
    dimensions = volume_np.shape[::-1]  # z, y, x order to x, y, z order

    # Spacing and origin extracted from affine
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    origin = affine[:3, 3]

    # Set up the grid
    volume_grid = pv.UniformGrid()
    volume_grid.dimensions = dimensions
    volume_grid.spacing = spacing
    volume_grid.origin = origin

    # Attach data
    vtk_data_array = numpy_to_vtk(volume_np.ravel(order='F'), deep=True)
    volume_grid.point_data['values'] = vtk_data_array

    plotter.add_volume(volume_grid, cmap="viridis", opacity=0.5)
    plotter.add_text("Volume Visualization", font_size=10)


# Example of how you might extract slices from the dictionary of actors
def extract_slices(slice_actors, num_slices):
    slices = []
    for slice_idx in range(num_slices):
        if slice_idx in slice_actors:
            actors = slice_actors[slice_idx]
            slice_data = extract_slice_from_actor(actors)
            slices.append(slice_data)
    return slices


# Function to extract numpy array from VTK actor
def extract_slice_from_actor(actor):
    mapper = actor.GetMapper()
    polydata = mapper.GetInput()
    points = polydata.points
    # These are the points of my slice
    # Now I could populate these points with the stic data somehow

    data = vtk_to_numpy(points.GetData())
    dims = polydata.GetDimensions()
    return data.reshape(dims)


# Nearest neighbor interpolation function
def nearest_neighbor_interpolate(volume, x, y, z):
    x_idx = int(np.round(x))
    y_idx = int(np.round(y))
    z_idx = int(np.round(z))
    if x_idx >= volume.shape[0] or y_idx >= volume.shape[1] or z_idx >= volume.shape[2] or x_idx < 0 or y_idx < 0 or z_idx < 0:
        return 0.0
    return volume[x_idx, y_idx, z_idx]

# Trilinear interpolation function
def trilinear_interpolate(volume, x, y, z):
    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1
    z0 = int(np.floor(z))
    z1 = z0 + 1

    if x1 >= volume.shape[0] or y1 >= volume.shape[1] or z1 >= volume.shape[2]:
        return 0.0

    xd = x - x0
    yd = y - y0
    zd = z - z0

    c00 = volume[x0, y0, z0] * (1 - xd) + volume[x1, y0, z0] * xd
    c01 = volume[x0, y0, z1] * (1 - xd) + volume[x1, y0, z1] * xd
    c10 = volume[x0, y1, z0] * (1 - xd) + volume[x1, y1, z0] * xd
    c11 = volume[x0, y1, z1] * (1 - xd) + volume[x1, y1, z1] * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd

    return c



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
    Computes the final transformation by applying the inverse of the predicted transform followed by the GT transform
    :param trans1: (numpy array) first transformation matrix
    :param trans2: (numpy array) subsequent transformation matrix, applied after the first
    :return: (numpy array) final transformation matrix
    """
    final_transform = np.dot(trans1, np.append(trans2, [[0, 0, 0, 1]], axis=0))  # Homogeneous coordinates
    return final_transform


def get_transform_slice(transforms, slice_grid):
    transform = vtk.vtkTransform()
    transform.SetMatrix(np.append(transforms, [0, 0, 0, 1]).flatten())
    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetInputData(slice_grid)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    transformed_slice = pv.wrap(transform_filter.GetOutput())
    return transformed_slice




def get_slice_grid(slice_data, volume, slice_idx):
    slice_grid = pv.UniformGrid()
    slice_grid.dimensions = np.array(slice_data.shape[::-1] + (1,))
    slice_grid.spacing = (volume.GetSpacing()[1], volume.GetSpacing()[2], 1)
    slice_grid.origin = (volume.GetOrigin()[1], volume.GetOrigin()[2], slice_idx * volume.GetSpacing()[0])
    vtk_data_array = numpy_to_vtk(slice_data.ravel(order='F'), deep=True)
    vtk_data_array.SetName("values")
    slice_grid.GetPointData().SetScalars(vtk_data_array)
    return slice_grid


def add_mesh_to_actor(plotter, overlay, slice_idx, cmap, plotted_slice, opacity, slice_actors, slice_grid):
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

    return slice_actors
