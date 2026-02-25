import os
import gc
import h5py
import cv2
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from scipy import ndimage
from trimesh import Trimesh
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage.measure import label, marching_cubes
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tifffile import imread, imwrite, TiffFile
from skimage.morphology import h_maxima, ball
from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt
from imaris_ims_file_reader import ims as imsreader




def data_read_resize_ims(raw_data_path, resize_factor, path_write):
    """
    Expects ims data file. Resizes it and saves it in nii.gz format
    """
    resize_factor_inv = resize_factor
    resize_factor = 1/resize_factor
    fname = os.path.basename(raw_data_path)
    arr = imsreader(raw_data_path)
    shapefile = arr.shape
    if len(shapefile)==3:
        arr = arr
    elif len(shapefile)==4:
        arr = arr[0]
    elif len(shapefile)==5:
        arr =arr[0,0]
    else:
        raise Exception('Invalid shape of the input ims data {}'.format(shapefile))
    print('The shape of the input data is {} and dtype {}'.format(shapefile, arr.dtype))
    if arr.dtype!=np.uint8:
        arr = arr.astype(np.uint8)
    path_write_inp =os.path.join(path_write, 'input_datafiles')
    os.makedirs(path_write_inp,exist_ok=True)
    cv2.imwrite(os.path.join(path_write_inp, fname.replace('.ims', '_maxproj_inputdata.jpg')), np.max(arr, axis=0))
    arr = arr.transpose(1,2,0)
    y,x,_ = arr.shape
    newy, newx = int(y*resize_factor), int(x*resize_factor)
    if resize_factor_inv==8:
        print('Before:', arr.shape)
        arr = cv2.resize(arr, dsize=(max(512, newx), max(512, newy)))
        print('after:', arr.shape)
    else:
        arr = cv2.resize(arr, fx = resize_factor, fy=resize_factor, dsize = None)
    
    path_write_nnunetinp =os.path.join(path_write_inp, 'nnunet_input')
    if os.path.exists(path_write_nnunetinp):
        shutil.rmtree(path_write_nnunetinp)
    os.makedirs(path_write_nnunetinp,exist_ok=True)
    
    arr = nib.Nifti1Image(arr, np.eye(4))
    resize_file_path = os.path.join(path_write_nnunetinp, fname.replace('.ims', '_0000.nii.gz'))
    nib.save(arr, resize_file_path)
    return resize_file_path
    

def data_read_resize_tiff(raw_data_path, resize_factor, path_write):
    """
    Expect tiff file. Resizes it and saves in nii.gz format
    """
    
    resize_factor_inv = resize_factor
    resize_factor = 1/resize_factor
    fname = os.path.basename(raw_data_path)
    arr = imread(raw_data_path)

    if len(arr.shape)>3:
        raise Exception('Invalid shape of the input tiff data {}'.format(arr.shape))
    
    print('The shape of the input data is {} and dtype {}'.format(arr.shape, arr.dtype))
    if arr.dtype!=np.uint8:
        arr = arr.astype(np.uint8)
    path_write_inp =os.path.join(path_write, 'input_datafiles')
    os.makedirs(path_write,exist_ok=True)
    if fname.endswith('.tif'):
        cv2.imwrite(os.path.join(path_write, fname.replace('.tif', '_maxproj_inputdata.jpg')), np.max(arr, axis=0))    
    else:
        cv2.imwrite(os.path.join(path_write, fname.replace('.tiff', '_maxproj_inputdata.jpg')), np.max(arr, axis=0))
    arr = arr.transpose(1,2,0)
    if resize_factor_inv==8:
        arr = cv2.resize(arr, dsize=(512, 512))
    else:
        arr = cv2.resize(arr, fx = resize_factor, fy=resize_factor, dsize = None)
    path_write_nnunetinp =os.path.join(path_write_inp, 'nnunet_input')
    if os.path.exists(path_write_nnunetinp):
        shutil.rmtree(path_write_nnunetinp)
    os.makedirs(path_write_nnunetinp,exist_ok=True)
    
    arr = nib.Nifti1Image(arr, np.eye(4))  # Save axis for data (just identity)
    if fname.endswith('.tif'):
        resize_file_path = os.path.join(path_write_nnunetinp, fname.replace('.tif', '_0000.nii.gz'))
    else:
        resize_file_path = os.path.join(path_write_nnunetinp, fname.replace('.tiff', '_0000.nii.gz'))
    nib.save(arr, resize_file_path)
    return resize_file_path



def getK_cc3dscipy(img_3d,k, connectivity=26, NeuronSizeVoxel=0.35e6):
    
    """
    Finds connected components in the 3D binary mask and returns the top k connected components that are atleast half the size of the largest component. 
    Returns each component and combined mask of top k components as a list of numpy arrays. The combined mask is the last element in the list.
    img_3d: 3D binary mask (numpy array)
    k: number of top connected components to return
    connectivity: 6 or 26 (default 26)
    NeuronSizeVoxel: minimum size of neuron in voxels, if the component size is greater than this, it will be included in the final mask even if its relative size is less than 0.5 the maximum component size.
    """
    
    if connectivity==26:
        structure = ndimage.generate_binary_structure(3, 2)
    elif connectivity==6:
        structure = ndimage.generate_binary_structure(3, 1)
    else:
        raise Exception('Invalid connectivity')
    labeled_array, num_features = ndimage.label(img_3d, structure=structure)
    print(f"Found {num_features} connected components in image {labeled_array.shape}")
    component_sizes = np.bincount(labeled_array.ravel())
    label_size_pairs = [(label, size) for label, size in enumerate(component_sizes) if label != 0]
    label_size_pairs.sort(key=lambda x: x[1], reverse=True)
    
    top_labels = [label for label, size in label_size_pairs[:k]]
    top_size = [(np.divide(size, label_size_pairs[0][1]), size) for label, size in label_size_pairs[:k]]
    print(f"Top {k} components: Labels {top_labels}, RelativeSize: {top_size}")
    
    container = np.zeros(img_3d.shape, dtype=np.uint8)
    all_imgs = []
    for seg_id,(size,size_pix) in dict(zip(top_labels, top_size)).items():
        extracted_image = labeled_array * (labeled_array == seg_id)
        extracted_image = extracted_image.astype(np.uint8)
        all_imgs.append((extracted_image>0).astype(bool).astype(np.uint8)*255)
        if size>=0.5 or size_pix>NeuronSizeVoxel:
            container = cv2.bitwise_or(container, extracted_image)
    container = (container>0).astype(bool).astype(np.uint8)*255
    all_imgs.append(container)
    return all_imgs


def project_modelResultsOnInput(model_mask, filename, file_3d, path_output, obj_id, fillbackgrd=True):
    
    inp_dtype = file_3d.dtype
    if file_3d.dtype==np.uint8:
        inp_dtype = np.uint8
    elif file_3d.dtype==np.uint16:
        inp_dtype = np.uint16
    print(filename)
    model_mask = model_mask.transpose(1,2,0)
    model_mask = cv2.resize(model_mask, (file_3d.shape[2], file_3d.shape[1]))
    model_mask = model_mask.transpose(2,0,1)
    
    if model_mask.shape[0]!=file_3d.shape[0]:
        diff = file_3d.shape[0]-model_mask.shape[0]
        if diff<0:
            raise Exception('The raw file has less number of Z stack than model prediction')
        top_offset = diff//2
        bot_offset = diff-top_offset
        file_3d = file_3d[top_offset:]
        file_3d = file_3d[:-bot_offset]
    
    result = np.zeros(model_mask.shape, dtype=inp_dtype)
    for i in range(model_mask.shape[0]):
        result[i]= cv2.bitwise_and(file_3d[i], file_3d[i], mask=model_mask[i])
    
    if fillbackgrd.lower()=='true':
        for zidx in range(result.shape[0]):
            img = result[zidx]
            hist = cv2.calcHist([img], [0], model_mask[zidx], [65536], [0, 65536])
            max_val = int(np.argmax(hist))
            if max_val<10:
                bkgrd = None
            elif max_val>90 and max_val<170:
                bkgrd=max_val
            
            if not bkgrd:
                hist = cv2.calcHist([file_3d[zidx]], [0], None, [65536], [0, 65536])
                bkgrd = int(np.argmax(hist))
            img = np.where(img<15, bkgrd, img)
            result[zidx] = img
    
    cv2.imwrite(os.path.join(path_output, filename.split('.nii.gz')[0]+'input_vs_overlayedResult_{}.jpg'.format(obj_id)), np.hstack((np.max(file_3d, axis=0), np.max(result, axis=0))))
    # imwrite(os.path.join(path_output, filename.split('.nii.gz')[0] + '_projected_{}.tiff'.format(obj_id)), result)
    # imwrite(os.path.join(path_output, filename.split('.nii.gz')[0]+'refined_mask_{}.tiff'.format(obj_id)), model_mask)
    cv2.imwrite(os.path.join(path_output, filename.split('.nii.gz')[0]+'refined_mask_{}.jpg'.format(obj_id)),np.max(model_mask,axis=0))
    
    result = result.transpose(1,2,0)
    result = nib.Nifti1Image(result, np.eye(4))  # Save axis for data (just identity)
    # result.header.get_xyzt_units()
    result.set_data_dtype(inp_dtype)
    comp_path_img_dump = os.path.join(path_output, filename.split('.nii.gz')[0] + '_projected_{}.nii.gz'.format(obj_id))
    nib.save(result, comp_path_img_dump)
    #result.to_filename(comp_path_img_dump)
    return os.path.join(path_output, filename.split('.nii.gz')[0] + '_projected_{}.nii.gz'.format(obj_id)), result

def readFile_h5ims(raw_read_path):
    if raw_read_path.endswith('.h5'):
        f1 = h5py.File(raw_read_path, 'r')
        file_3d = f1['raw'][0]
    elif raw_read_path.endswith('.ims'):
        f1 = imsreader(raw_read_path)
        file_3d = f1[0,0]
        # file_3d = file_3d.astype(np.uint8)
    elif raw_read_path.endswith('.tiff') or raw_read_path.endswith('.tif'):
        file_3d= imread(raw_read_path)
    else:
        raise Exception('invalid raw file format {}'.format(raw_read_path))
    return file_3d


def nnunet_postprocessing(results_read_path, raw_read_path, backgroundFilling, watershedpostprocess, dilation_factor=1, erosion_factor=None):
    
    """
    reads nnunet model prediction, reads the raw file. Clean the nnunet prediction. Overlays it on the raw file. Fills the background with most frequent value in the corresponding z.
    writes the output after each operation and the final overlayed file.
    Returns the path to the final overlayed file.
    """
    dilation_factor = int(dilation_factor)
    
    results_write_path = os.path.join(results_read_path, 'nnunetpostprocessed')
    os.makedirs(results_write_path, exist_ok=True)
    file_input = os.path.basename(raw_read_path)
    
    if file_input.endswith('.tiff') or file_input.endswith('.tif'):
        file_input = file_input.split('.tif')[0]
    elif file_input.endswith('.ims'):
        file_input = file_input.split('.ims')[0]
    else:
        raise Exception('Invalid input file type {}'.format(file_input))
    
    all_projected_fnames = []
    for fname in os.listdir(results_read_path):
        if not fname.endswith('.nii.gz'):continue
        if not fname.split('.nii.gz')[0]==file_input:continue
        print(fname)
        pred_mask = nib.load(os.path.join(results_read_path, fname))
        pred_mask = pred_mask.get_fdata()
        pred_mask = pred_mask.astype(np.uint8)
        pred_mask = pred_mask.transpose(2,0,1)
        
        cv2.imwrite(os.path.join(os.path.dirname(results_write_path),fname.split('.nii.gz')[0]+'_rawmodelpred.jpg'), np.max(pred_mask, axis=0)*255)

        pred_mask = (ndimage.binary_dilation(pred_mask, structure=ball(radius=dilation_factor)))*255
        all_components2_skimg = getK_cc3dscipy(pred_mask, k=5)
    
        arr = all_components2_skimg[-1]
        if erosion_factor:
            erosion_factor = int(erosion_factor)
        
        arr = arr.astype(np.uint8)
        arr = arr.transpose(1,2,0)
        
        arr = arr.transpose(2,0,1)
        arr = arr.astype(np.uint8)
        if watershedpostprocess.lower()=='true':
            arr = watershed_segmentation(arr)
            cv2.imwrite(os.path.join(results_write_path,fname.split('.nii.gz')[0]+'_cc3d_vs_watershed_mask.jpg'), np.hstack((np.max(all_components2_skimg[-1], axis=0), np.max((arr>0).astype(np.uint8)*255, axis=0))))
            
        obj_ids = np.unique(arr)
        obj_ids = obj_ids[obj_ids!=0] #remove background
        print('Total objects found are', len(obj_ids), obj_ids)
        if len(obj_ids)==1:
            multi_objects = False
        else:
            multi_objects = True
        counter = 0
        final_mask = np.zeros(arr.shape, dtype=np.uint8)
        
        file_3d = readFile_h5ims(raw_read_path)
            
        for objid in obj_ids:
            arr_id = (arr==objid).astype(np.uint8)*255
            arr_id = arr_id.astype(np.uint8)
            if erosion_factor:
                arr_id = ((ndimage.binary_erosion(arr_id, structure=ball(radius=erosion_factor)))*255).astype(np.uint8)
            if multi_objects:
                projected_filename, _ = project_modelResultsOnInput(arr_id, fname, file_3d, results_write_path, counter+1, backgroundFilling)
                all_projected_fnames.append(projected_filename)
            if counter ==0:
                final_mask = arr_id
            else:
                for z in range(final_mask.shape[0]):
                    final_mask[z] = cv2.bitwise_or(final_mask[z], arr_id[z])
            counter+=1
        
        final_projected_filename, _ = project_modelResultsOnInput(final_mask, fname, file_3d, results_write_path, 'final', backgroundFilling)
        final_mask = final_mask.transpose(1,2,0)
        final_mask = nib.Nifti1Image(final_mask, np.eye(4))  # Save axis for data (just identity)
        final_mask.header.get_xyzt_units()
        comp_path_img_dump = os.path.join(results_write_path, fname.replace('.nii.gz', '_postprocessedMaskFinal.nii.gz'))
        final_mask.to_filename(comp_path_img_dump)
        break
    return final_projected_filename, comp_path_img_dump

def watershed_segmentation(arr, dendritic_neuron_area = 0.35e6):

    """
    Input: binary mask path (nii.gz file)
    Returns: watershed segmentation results
    """
    
    print(arr.dtype)
    if np.max(arr)==1:
        arr = (arr*255).astype(np.uint8)
    

    distance = distance_transform_edt(arr)
    
    minima = h_maxima(distance, np.max(distance)/2) #9
    markers = label(minima)

    labels = watershed(-distance, markers, mask=arr)
    labels = labels.astype(np.uint8)
    del distance
    del markers
    del arr
    gc.collect()
    props = regionprops(labels)
    all_areas = [prop.area for prop in props]
    try:
        max_area = max(all_areas)
    except:
        print('Unexpected behaviour in watershed segmentation, no object area found. Input sample has no neuron or the threshold parameters are incorrect. Aborting watershed.')
        return (labels>0).astype(np.uint8)*255
        
    all_areas = [np.round(np.divide(prop.area, max_area),2) for prop in props]
    if len(all_areas)>2:
        ids_exclude = [prop.label for prop in props if prop.area<dendritic_neuron_area and np.round(np.divide(prop.area, max_area),2)<0.5]
    else:
        ids_exclude = []

    if len(ids_exclude)>0:
        for objid in ids_exclude:
            labels = np.where(labels==objid, 0, labels)
    return labels

def analyze_spines(binary, source="unknown", visualize=True, percentile_range=(5, 95), voxel_size=1.0, write_path=None):
    """
    Analyze connected components (spines) in a 3D binary mask and report physical units.

    Parameters:
    - binary: 3D numpy array (binary mask where spines = 1)
    - source: str, name or identifier of the source file
    - visualize: bool, if True show histogram plot
    - percentile_range: tuple (low, high), percentiles for voxel size filtering
    - voxel_size: scalar (isotropic) or tuple of 3 floats (z, y, x) in microns
    - write_path: str, path to save the histogram plot

    Returns:
    - df: pandas DataFrame with geometry metrics in µm, µm², µm³ + source column
    """
    filename = os.path.basename(source)
    filename = filename.split('.nii.gz')[0]
    # Label connected objects
    labeled, num = label(binary, return_num=True)
    print(f"Initial spine count: {num} for source: {filename}")

    # Measure voxel sizes
    sizes = np.bincount(labeled.ravel())[1:]  # exclude background

    # Thresholds
    low_p, high_p = percentile_range
    low_thresh = np.percentile(sizes, low_p)
    high_thresh = np.percentile(sizes, high_p)

    if visualize:
        plt.figure(figsize=(8, 5))
        plt.hist(sizes, bins=30, color='gray', edgecolor='black')
        plt.title(f"Histogram of Spine Sizes (Source: {filename})")
        plt.xlabel("Voxel count")
        plt.ylabel("Frequency")
        plt.axvline(low_thresh, color='red', linestyle='--', label=f'{low_p}th percentile')
        plt.axvline(high_thresh, color='blue', linestyle='--', label=f'{high_p}th percentile')
        plt.legend()
        plt.tight_layout()
        if write_path is not None:
            plt.savefig(os.path.join(write_path, f"{filename}_spine_size_histogram.png"))
        plt.show()

    print(f"Keeping spines between {low_thresh:.0f} and {high_thresh:.0f} voxels for {filename}")

    # 4️⃣ Process spines
    results = []

    for spine_id in range(1, num + 1):
        spine_mask = (labeled == spine_id)
        spine_size = sizes[spine_id - 1]

        if spine_size < low_thresh or spine_size > high_thresh:
            continue

        try:
            if isinstance(voxel_size, (tuple, list)) and len(voxel_size) == 3:
                spacing = voxel_size
            else:
                spacing = (voxel_size, voxel_size, voxel_size)

            verts, faces, _, _ = marching_cubes(spine_mask, level=0.5, spacing=spacing)
        except ValueError:
            continue

        mesh = Trimesh(vertices=verts, faces=faces, process=False)
        if mesh.is_empty:
            continue

        area_um2 = mesh.area  # Already in µm² due to spacing
        volume_um3 = abs(mesh.volume)  # Ensure positive volume
        bbox = mesh.bounding_box_oriented.extents
        length_um = np.max(bbox)  # Largest dimension in µm due to spacing

        results.append({
            "source": filename,
            "spine_id": spine_id,
            "volume_um3": volume_um3,
            "surface_area_um2": area_um2,
            "length_um": length_um,
            "voxel_count": spine_size
        })

    # 5️⃣ Output DataFrame
    df = pd.DataFrame(results)
    return df


def spineMorphologyAnalysis(folder_path, voxel_size, visualize_histograms=True):
    """
    Implementation of spine morphology analysis
    Parameters:
    folder_path: path to folder containing nii.gz chunks of spine segmentation results
    voxel_size: tuple of 3 floats indicating voxel size in microns (z, y, x)
    """
    spineresultsPath = os.path.join(os.path.dirname(folder_path), 'spine_morphology_results')
    os.makedirs(spineresultsPath, exist_ok=True)
    df_new = pd.DataFrame()

    # Scan all files and classify them
    # all_files = [f for f in os.listdir(folder_path) if f.endswith('.tiff') and not f.startswith('.')]
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz') and not f.startswith('.')]
    total_files = len(all_files)

    empty_files = []
    non_empty_files = []

    for file in all_files:
        file_path = os.path.join(folder_path, file)
        # image = imread(tiff_path)
        image = nib.load(file_path)
        image = image.get_fdata()
        image = image.astype(np.uint8)
        if np.any(image):
            non_empty_files.append(file)
        else:
            empty_files.append(file)

    print(f"Total files: {total_files}")
    print(f"Non-empty files: {len(non_empty_files)}")
    print(f"Empty files: {len(empty_files)}")
    if visualize_histograms:
        spineresultsHistogramPath = os.path.join(spineresultsPath, 'histograms')
        os.makedirs(spineresultsHistogramPath, exist_ok=True)
    # Process non-empty files with progress bar
    for file in tqdm(non_empty_files, desc="Processing non-empty files"):
        file_path = os.path.join(folder_path, file)
        # image = imread(tiff_path)
        image = nib.load(file_path)
        image = image.get_fdata()
        image = image.astype(np.uint8)
        # basename = os.path.splitext(file)[0]
        basename = file.split('.nii.gz')[0]

        df = analyze_spines(image, source=file_path, visualize=visualize_histograms, percentile_range=(5, 95), voxel_size=voxel_size, write_path = spineresultsHistogramPath if visualize_histograms else None)
        print(f'Processed {basename}, {len(df)} records')
        df_new = pd.concat([df_new, df], axis=0, ignore_index=True)
        print(f'Concatenated {basename} to df_new')

    print('Finished processing all non-empty files')

    # Save summary dataframe
    output_csv = os.path.join(spineresultsPath, 'spine_metrics.csv')
    df_new.to_csv(output_csv, index=False)
    print(f'Saved summary to {output_csv}')
    
        
        

        
        
