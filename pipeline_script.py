import os
import argparse
from model_inference import nnUNET, nnUNETSpineDendriteSoma
from nest_utils import data_read_resize_ims, data_read_resize_tiff, nnunet_postprocessing, spineMorphologyAnalysis




if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('-i1', '--input_file', help = 'path to the tiff/ims file with z,y,x order', required = True)
    args.add_argument('-i2', '--resize_factor', help = 'resize factor, accepts 2,4,8,16', default= '8')
    args.add_argument('-i3', '--backgroundFilling', help = 'Fill background with when projecting model mask', default='true')
    args.add_argument('-i4', '--typeofPostprocessing', help = 'To use watershed segmentation for postprocessing', default='true')
    args.add_argument('-i5', '--dilation_factor', help='dilation factor in postprocessing', default = 1)
    args.add_argument('-i6', '--numConnectedComponents', help='number of largest connected components to be included', default = 5)
    args.add_argument('-i7', '--erosion_factor', help='erosion factor in postprocessing', default = None)
    args.add_argument('-i8', '--chunkSizeStage2', help='chunk size for stage 2 model', default = 512)
    args.add_argument('-i9', '--Zresize_factorStage2', help='resize factor in the z axis', default = 6)
    args.add_argument('-i10', '--voxelsize', help='voxel size for spine morphology analysis', default = (0.0833,0.054, 0.054))
    args.add_argument('-i11', '--deepd3_model', help='DeepD3_model_file', default = None)
    
    args.add_argument('-o1', '--output_path', help='folder path to write all the results', default = './NEST_output')
    args=args.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    try:
        factor = int(args.resize_factor)
        if not factor in [2,4,8,16]:
            raise Exception('Resize factor can one of these values i.e. 2 4 8 16. Passed {}'.format(factor))
    except:
        raise Exception('Resize factor must be an integer {}'.format(args.resize_factor))
    
    
    try:        Zresize_factorStage2 = int(args.Zresize_factorStage2)
    except:     raise Exception('Z resize factor for Stage 2 must be an integer {}'.format(args.Zresize_factorStage2))
    
    try:        chunkSizeStage2 = int(args.chunkSizeStage2)
    except:     raise Exception('Chunk size for Stage 2 must be an integer {}'.format(args.chunkSizeStage2))
    
    filename = os.path.basename(args.input_file)
    if args.input_file.endswith('.tiff') or args.input_file.endswith('.tif'):
        foldername = filename.split('.tif')[0]
        resized_file_path = data_read_resize_tiff(args.input_file, factor, args.output_path)
    elif args.input_file.endswith('.ims'):
        foldername = filename.split('.ims')[0]
        resized_file_path = data_read_resize_ims(args.input_file, factor, args.output_path)
    else:
        raise Exception('Invalid input file path {}'.format(args.input_file))
    
    
    write_path_nnunet = nnUNET(args.output_path, resized_file_path, foldername)
    nnunetpostprocessedfilename, final_mask_path = nnunet_postprocessing(write_path_nnunet, args.input_file, args.backgroundFilling, args.typeofPostprocessing, args.dilation_factor, args.erosion_factor)
    
    
    write_path_nnunetStage2 = nnUNETSpineDendriteSoma(nnunetpostprocessedfilename, args.output_path, foldername, model_identifier='3d_fullres_SpineDend6xZnorm', factor =Zresize_factorStage2, chunk_size=chunkSizeStage2)
    spineMorphologyAnalysis(write_path_nnunetStage2, args.voxelsize)
