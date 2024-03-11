import uproot
import numpy as np
import glob
import os

def convert_root_directory_to_npy(folder_path, output_folder_path, tree_name, branch_name, prefix):
    """
    Convert all ROOT files in a specified directory to NPY files with a new naming scheme.

    Parameters:
    - folder_path: str, the path to the input directory containing ROOT files.
    - output_folder_path: str, the path to the output directory where NPY files will be saved.
    - tree_name: str, the name of the tree in the ROOT files.
    - branch_name: str, the name of the branch to be converted.
    - prefix: str, the prefix for the output NPY filenames.

    Returns:
    None
    """
    # Ensure output directory exists
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # List all ROOT files in the directory
    root_files = glob.glob(os.path.join(folder_path, "*.root"))
    
    # Convert each ROOT file to NPY
    for index, root_file_path in enumerate(root_files, start=1):
        # Generate the output filename with the specified prefix and index
        output_filename = f"{prefix}_{index}.npy"
        output_npy_path = os.path.join(output_folder_path, output_filename)
        
        # Open the ROOT file
        with uproot.open(root_file_path) as file:
            # Access the tree
            tree = file[tree_name]
            # Access the branch data as a NumPy array
            branch_data = tree.arrays(library="np")
            # Save the array to a .npy file
            np.save(output_npy_path, branch_data)

# Example usage

# type = "pizero"
# folder_path = '/hpcfs/users/a1768536/AGPF/gnn4pions/ML_TREE_DATA/fixed2/pi0'  # Update this to your directory's path
# output_folder_path = '/hpcfs/users/a1768536/AGPF/gnn4pions/ML_TREE_DATA/npy/pizero'  # Update this to your desired output path

prefix = 'n0'  # Specify the prefix for the output filenames
folder_path = '/hpcfs/users/a1768536/AGPF/gnn4pions/ML_TREE_DATA/fixed2/n0/user.mjgreen/'  # Update this to your directory's path
output_folder_path = '/hpcfs/users/a1768536/AGPF/gnn4pions/ML_TREE_DATA/npy/n0'  # Update this to your desired output path

tree_name = 'EventTree'  # Update this with the name of your tree
branch_name = 'your_branch_name'  # Update this with the branch you want to convert

convert_root_directory_to_npy(folder_path, output_folder_path, tree_name, branch_name, prefix)
