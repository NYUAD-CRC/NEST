# NEST
Neuron Enhancement and Spine Segmentation Toolkit


Steps to get started

1. Create a conda environment using the below command
   ```
   cd NEST
   conda env create -f environment_portable.yml
   conda activate nest
   ```
2. Download model weights and sample test files.
   ```
   gdown 1aFg7qkumKLcXfoXAQFpXf8FX0Al1NiPt -O nnUNet_files/nnUNet_results/Dataset101_BrainCells/nnUNetTrainer__nnUNetPlans__3d_fullres_0.125scale/fold_all/checkpoint_final.pth
   ```
   ``` 
   gdown 1nx1HdGCxrj_BNpA-L4Pksss_utIGNud3 -O nnUNet_files/nnUNet_results/Dataset101_BrainCells/nnUNetTrainer__nnUNetPlans__3d_fullres_SpineDend6xZnorm/fold_all/checkpoint_final.pth
   ```
   ```
   gdown 1iy4lTveAvpyrlBYJHRjIgd9M9BK1U7hW -O test_files/test1.ims
   ```
3. Run the NEST pipeline on the downloaded samples.

   ```
   python pipeline_script.py -i1 test_files/test1.ims -i2 8 -i3 True -i5 '2'
   ```
 
