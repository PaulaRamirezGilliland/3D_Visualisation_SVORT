# Visualisation Tool Description 

3D Visualisation tool based on VTK and PyVista for assessing 3D pose prediction. 

# Requirements

Python 3.8
pyvista==0.39.0                                                                                                       
PyYAML==6.0.1
scipy==1.10.1
SimpleITK==2.2.1
vtk==9.2.6

Docker: 


# Config file

The config file should contain paths to: 
- 2D_volume_path: real 2D data in nifti format, prealigned to 3D space (sparse 3D volume, containing 2D slices in standard space)  
- STIC_path: 3D acquisition in nifti format
- mask_path: if using a mask, path to 3D mask (nifti image) 
- transforms_path: path to folder containing both "transforms.npy" and "transforms_gt.npy"

  
# Usage

To use, please update paths to predicted transformations and images (2D and 3D) in the config file. 

GT transformations should be in numpy array format (.npy), with filename "transforms.npy" for predicted transformations and "transforms_gt.npy" for ground truth transformations (both transforms in the same folder, specified by the path). 

Images should be in nifti format.

Command to use: 

python main.py --config path_to_config.yaml

# Interactive Display

An interactive display as depicted below is generated. To navigate thorugh it, keys "j" - up one slice, "k" - down one slice, right click on any 2D slice to highlight it. Zoom in and out by scrolling on mouse/hold right click and zoom. 


![image](https://github.com/user-attachments/assets/4d7f965d-7798-4ce8-aedd-7e956d2bf48d)

