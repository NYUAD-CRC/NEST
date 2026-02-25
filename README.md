# NEST
Neuron Enhancement and Spine Segmentation Toolkit


Steps to get started

1. Create a conda environment using the below command
   ```
   conda env create -f environment_portable.yml
   conda activate nest
   ```
2. Download model weights and sample test files.
   ```
   cd NEST/nnUNet_files/nnUNet_results/Dataset101_BrainCells/nnUNetTrainer__nnUNetPlans__3d_fullres_0.125scale/fold_all
   wget https://drive.google.com/file/d/1aFg7qkumKLcXfoXAQFpXf8FX0Al1NiPt/view?usp=sharing
   
   cd NEST/github_files/NEST/nnUNet_files/nnUNet_results/Dataset101_BrainCells/nnUNetTrainer__nnUNetPlans__3d_fullres_SpineDend6xZnorm/fold_all
   wget https://drive.google.com/file/d/1nx1HdGCxrj_BNpA-L4Pksss_utIGNud3/view?usp=drive_link

   cd test_files
   wget https://drive.google.com/file/d/1iy4lTveAvpyrlBYJHRjIgd9M9BK1U7hW/view?usp=drive_link
   ```
3. Run the NEST pipeline on the downloaded samples.
 
