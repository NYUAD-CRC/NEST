import os
import gc
import subprocess
import shutil
from tifffile import imread, imwrite, TiffFile
import time
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom
import numpy as np
import cv2


def nnUNET(write_path, read_path, foldername):
    """
    Reads a nii.gz file and run through model nnUNET. The result is saved.
    """
    print('Starting nnUNET inference')
    write_path = os.path.join(write_path, 'nnUNET_results', foldername)
    if os.path.exists(os.path.join(write_path, foldername+'.nii.gz')):
        print('prediction file already exists, skipping inference. {}'.format(os.path.join(write_path, foldername+'.nii.gz')))
        return write_path
    os.makedirs(write_path, exist_ok = True)
    read_path = os.path.dirname(read_path)
    #environment variables
    custom_env_vars= "nnUNet_preprocessed=/home/dhanu/dhanu_work/files/nnunet_work/dataset_preprocessed nnUNet_results=/home/dhanu/dhanu_work/files/nnunet_work/nnUNet_results nnUNet_raw=/home/dhanu/dhanu_work/files/nnunet_work/nnUNet_raw"
    
    # cmd = "nnUNetv2_predict -i {} -o {} -d 101 -c 3d_fullres_0.125scale -f all".format(read_path, write_path)
    cmd = "nnUNetv2_predict -i {} -o {} -d 101 -c 3d_fullres_0.125scale -f all".format(read_path, write_path)
    final_cmd = "conda run -n nnunetenv env {} {}".format(custom_env_vars, cmd)
    subprocess.run(final_cmd, shell=True, check=True)
    
    return write_path


def deepD3(path_output, inputfilepath, path_deepd3_model):
    """
    Reads a tiff file and run through model deepd3 with the given deepd3 weight to be used. It saves dendrites, spines and combined result. The result is saved after cleaning using nnunet model mask.
    """
    path_output = os.path.join(path_output, 'deepd3_results')
    os.makedirs(path_output, exist_ok=True)
    print('Starting DeepD3')
    cmd = "python /home/dhanu/dhanu_work/files/deepd3/DeepD3/deepd3/inference/batch.py {} {} -i1 {}".format(inputfilepath, path_deepd3_model, path_output)
    final_cmd = "conda run -n tf-gpu {}".format(cmd)
    subprocess.run(final_cmd, shell=True, check=True)
    return path_output

def collatebackFrmChunksSpineDendSomaPred(pathNNunetSpDendSomaResults, chunk_size):
    ymax,xmax = 0, 0
    for fname in os.listdir(pathNNunetSpDendSomaResults):
        if not fname.endswith('.nii.gz'):continue
        ystart, xstart = int(fname.split('.nii.gz')[0].split('_')[-2]), int(fname.split('.nii.gz')[0].split('_')[-1])
        if ystart+chunk_size>ymax:
            ymax = ystart+chunk_size
        if xstart+chunk_size>xmax:
            xmax = xstart+chunk_size
    for fname in os.listdir(pathNNunetSpDendSomaResults):

        if not fname.endswith('.nii.gz'):continue
        ystart, xstart = int(fname.split('.nii.gz')[0].split('_')[-2]), int(fname.split('.nii.gz')[0].split('_')[-1])
        print(ystart, ymax-chunk_size, xstart, xmax-chunk_size )
        if ystart == ymax-chunk_size and xstart==xmax-chunk_size:
            arr = nib.load(os.path.join(pathNNunetSpDendSomaResults, fname))
            ymax, xmax, z = ystart + arr.shape[0], xstart+arr.shape[1], arr.shape[2]
            break
    collated_arr = np.zeros((ymax,xmax,z), dtype=np.uint8)
    
    counter = 0
    for fname in tqdm(os.listdir(pathNNunetSpDendSomaResults)):
        counter+=1
        if not fname.endswith('.nii.gz'):continue
        ystart, xstart = int(fname.split('.nii.gz')[0].split('_')[-2]), int(fname.split('.nii.gz')[0].split('_')[-1])
        arr = nib.load(os.path.join(pathNNunetSpDendSomaResults, fname))
        arr = arr.get_fdata()
        h,w,_ = arr.shape
        arr = arr.astype(np.uint8)
        collated_arr[ystart:ystart+h, xstart:xstart+w,:] = arr
    write_path = os.path.join(os.path.dirname(pathNNunetSpDendSomaResults), os.path.basename(pathNNunetSpDendSomaResults)+'_collated.nii.gz')
    cv2.imwrite(write_path.replace('.nii.gz', '_spine.jpg'), np.max((collated_arr==1).astype(np.uint8)*255, axis=-1))
    cv2.imwrite(write_path.replace('.nii.gz', '_dendrite.jpg'), np.max((collated_arr==2).astype(np.uint8)*255, axis=-1))
    affine = np.eye(4)  # Identity matrix for affine transformation
    nifti_img = nib.Nifti1Image(collated_arr, affine)
    nifti_img.set_data_dtype(np.uint8)
    nib.save(nifti_img, write_path)
    return

def chunk_frmniigz(arr, factor, model_identifier, filename, write_path_chunks, chunk_size=512):
    y,x,z = arr.shape
    print('File shape is {}'.format(arr.shape))
    print('Resize factor is {} and model variant used is {}'.format(factor, model_identifier))
    print('Splitting the whole file into chunks')
    for i in tqdm(range(y//chunk_size + 1)):
        for j in range(x//chunk_size + 1):
            ystart, xstart = max(0, i*chunk_size), max(0, j*chunk_size)
            yend, xend = min((i+1)*chunk_size, y), min((j+1)*chunk_size, x)
            
            chunkname = filename.split('.nii.gz')[0]+'_'+str(ystart)+'_'+str(xstart)+'.nii.gz'
            if os.path.exists(os.path.join(write_path_chunks, chunkname.replace('.nii.gz', '_0000.nii.gz'))):
                continue
            chunk = arr[ystart:yend, xstart:xend,:]
            chunk = zoom(chunk, (1,1,factor), order=1)
            affine = np.eye(4)  # Identity matrix for affine transformation
            nifti_img = nib.Nifti1Image(chunk, affine)
            nifti_img.set_data_dtype(np.uint16)
            nib.save(nifti_img, os.path.join(write_path_chunks, chunkname.replace('.nii.gz', '_0000.nii.gz')))
    return


def chunk_frmTiff(arr, factor, model_identifier, filename, write_path_chunks, chunk_size=512):
    z,y,x = arr.shape
    print('File shape is {}'.format(arr.shape))
    print('Resize factor is {} and model variant used is {}'.format(factor, model_identifier))
    print('Splitting the whole file into chunks')
    for i in tqdm(range(y//chunk_size + 1)):
        for j in range(x//chunk_size + 1):
            ystart, xstart = max(0, i*chunk_size), max(0, j*chunk_size)
            yend, xend = min((i+1)*chunk_size, y), min((j+1)*chunk_size, x)
            
            chunkname = filename.split('.tif')[0]+'_'+str(ystart)+'_'+str(xstart)+'.tiff'
            if os.path.exists(os.path.join(write_path_chunks, chunkname.replace('.tiff', '_0000.nii.gz'))):
                continue
            chunk = arr[:, ystart:yend, xstart:xend]
            chunk = zoom(chunk, (factor,1,1), order=1)
            
            chunk = chunk.transpose(1,2,0)
            affine = np.eye(4)  # Identity matrix for affine transformation
            nifti_img = nib.Nifti1Image(chunk, affine)
            nifti_img.set_data_dtype(np.uint16)
            nib.save(nifti_img, os.path.join(write_path_chunks, chunkname.replace('.tiff', '_0000.nii.gz')))
    return

def nnUNETSpineDendriteSomaModel(write_path, read_path, modelname):
    """
    Reads chunks, and runs the nnunet model to predict spine, dendrite and soma.
    """
    print('Starting nnUNET inference  for Spine and Dendrite segmentation')
    os.makedirs(write_path, exist_ok = True)
    #environment variables
    custom_env_vars= "nnUNet_preprocessed=/home/dhanu/dhanu_work/files/nnunet_work/dataset_preprocessed nnUNet_results=/home/dhanu/dhanu_work/files/nnunet_work/nnUNet_results nnUNet_raw=/home/dhanu/dhanu_work/files/nnunet_work/nnUNet_raw"
    
    cmd = "nnUNetv2_predict -i {} -o {} -d 101 -c {} -f all".format(read_path, write_path, modelname)
    # cmd = "nnUNetv2_predict -i {} -o {} -d 101 -c {} -f all --save_probabilities".format(read_path, write_path, modelname)
    final_cmd = "conda run -n nnunetenv env {} {}".format(custom_env_vars, cmd)
    subprocess.run(final_cmd, shell=True, check=True)
    return write_path


def nnUNETSpineDendriteSoma(filename, write_path, foldername, model_identifier='3d_fullresSpineDendSoma6x', factor =6, chunk_size=512, read_format='.nii.gz'):
    """
    Reads the cleaned neuron sample and chunks it. Runs the nnunet model to predict
    spine and dendrites. 
    """
    
    write_path = os.path.join(write_path, 'nnUNET_results', foldername)
    write_path_base = os.path.join(write_path, 'spineDendSegModelResult')
    
    
    start_time = time.time()    
    if read_format=='.tiff' or read_format =='.tif':
        if not filename.endswith('.tiff') and not filename.endswith('.tif'):
            raise Exception('Invalid read format passed {}'.format(read_format))
        cleanNeuronfile = imread(filename)
    elif read_format=='.nii.gz':
        if not filename.endswith('.nii.gz'):
            raise Exception('Invalid read format passed {}'.format(read_format))
        cleanNeuronfile = nib.load(filename).get_fdata()
    else:
        raise Exception('Invalid read format passed {}'.format(read_format))
    cleanNeuronfile = cleanNeuronfile.astype(np.uint16)
    
    if read_format=='.tiff' or read_format =='.tif':
        fname = os.path.basename(filename)
        fname = fname.split('.tif')[0] if fname.endswith('.tif') else fname.split('.tiff')[0]
        write_path_chunks = os.path.join(write_path_base, 'raw_{}x_{}'.format(factor, fname))
        if not os.path.exists(write_path_chunks):
            os.makedirs(write_path_chunks, exist_ok=True)
        write_path_nnunetresults = os.path.join(write_path_base, 'modelResults_{}x_{}'.format(factor, fname))
        chunk_frmTiff(cleanNeuronfile, factor, model_identifier, fname, write_path_chunks, chunk_size)
        
    elif read_format=='.nii.gz':
        fname = os.path.basename(filename)
        fname = fname.split('.nii.gz')[0]
        write_path_chunks = os.path.join(write_path_base, 'raw_{}x_{}'.format(factor, fname))
        if not os.path.exists(write_path_chunks):
            os.makedirs(write_path_chunks, exist_ok=True)
        write_path_nnunetresults = os.path.join(write_path_base, 'modelResults_{}x_{}'.format(factor, fname.split('.nii.gz')[0]))
        chunk_frmniigz(cleanNeuronfile, factor, model_identifier, fname, write_path_chunks, chunk_size)
    else:
        raise Exception('Invalid read format passed {}'.format(read_format))
    del cleanNeuronfile
    gc.collect()
    nnUNETSpineDendriteSomaModel(write_path_nnunetresults, write_path_chunks, model_identifier)
    collatebackFrmChunksSpineDendSomaPred(write_path_nnunetresults, chunk_size)
    shutil.rmtree(write_path_chunks)
    end_time = time.time()
    print('total time taken in nnunet is {} mins'.format(round((end_time-start_time)/60),2))
    return write_path_nnunetresults
    
    
    
    # write_path = os.path.join(write_path, 'nnUNET_results', foldername)
    # write_path_base = os.path.join(write_path, 'spineDendSomaSegModelResult')
    
    # start_time = time.time()    
    # inputfilePresent=False
    # for fname in os.listdir(os.path.join(write_path, 'nnunetpostprocessed')):
    #     if 'final' in fname:
    #         inputfilePresent = True
    #         if read_format=='.tiff':
    #             cleanNeuronfile = imread(os.path.join(write_path, 'nnunetpostprocessed', fname))
    #         elif read_format=='.nii.gz':
    #             cleanNeuronfile = nib.load(os.path.join(write_path, 'nnunetpostprocessed', fname)).get_fdata()
    #         else:
    #             raise Exception('Invalid read format passed {}'.format(read_format))
    #         cleanNeuronfile = cleanNeuronfile.astype(np.uint16)
    #         if read_format=='.tiff':
    #             write_path_chunks = os.path.join(write_path_base, 'raw_{}x_{}'.format(factor, fname.split('.tif')[0]))
    #             if not os.path.exists(write_path_chunks):
    #                 os.makedirs(write_path_chunks, exist_ok=True)
    #             write_path_nnunetresults = os.path.join(write_path_base, 'modelResults_{}x_{}'.format(factor, fname.split('.tif')[0]))
                
    #             chunk_frmTiff(cleanNeuronfile, factor, model_identifier, fname, write_path_chunks, chunk_size)
    #         elif read_format=='.nii.gz':
    #             write_path_chunks = os.path.join(write_path_base, 'raw_{}x_{}'.format(factor, fname.split('.nii.gz')[0]))
    #             if not os.path.exists(write_path_chunks):
    #                 os.makedirs(write_path_chunks, exist_ok=True)
    #             write_path_nnunetresults = os.path.join(write_path_base, 'modelResults_{}x_{}'.format(factor, fname.split('.nii.gz')[0]))
    #             chunk_frmniigz(cleanNeuronfile, factor, model_identifier, fname, write_path_chunks, chunk_size)
    #         else:
    #             raise Exception('Invalid read format passed {}'.format(read_format))
    #         del cleanNeuronfile
    #         gc.collect()
    #         nnUNETSpineDendriteSomaModel(write_path_nnunetresults, write_path_chunks, model_identifier)
    #         collatebackFrmChunksSpineDendSomaPred(write_path_nnunetresults, chunk_size)
    #         shutil.rmtree(write_path_chunks)
    # if not inputfilePresent: #'spineDendSomaSegModelResult'
    #     print('The cleaned neuron sample (projected combined file) doesnot exists at {}'.format(os.path.join(write_path, 'nnunetpostprocessed')))
    # else:
    #     end_time = time.time()
    #     print('total time taken in nnunet is {} mins'.format(round((end_time-start_time)/60),2))
    # return write_path_nnunetresults

# if __name__=='__main__':
#     deepD3('./deepd3_results_temp', '/home/dhanu/dhanu_work/data/temp/1.tif', '/home/dhanu/dhanu_work/files/deepd3/DeepD3/deepd3/model_weights/DeepD3_8F_94nm.h5')