import os
import SimpleITK as sitk
import numpy as np


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
