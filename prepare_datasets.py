#==========================================================
#
#  This prepare the hdf5 datasets of the NEW database
#
#============================================================

import os, errno
import h5py
import numpy as np
from PIL import Image
import configparser
import sys
sys.path.insert(0, './lib/')
from extract_patches import extract_ordered_overlap
from pre_processing import my_PreProc


#config file to read from
config = configparser.RawConfigParser()
config.read('./global_config.txt')

height = 565
width = 565

patch_h = int(config.get('data attributes', 'patch_height'))
patch_w = int(config.get('data attributes', 'patch_width'))
patch_size = (patch_h, patch_w)

average_mode = config.get('global', 'average_mode')
stride_h = int(config.get('global', 'stride_height'))
stride_w = int(config.get('global', 'stride_width'))
stride_size = (stride_h, stride_w)

N_patches_per_img = ((height + stride_h - (height - patch_h) % stride_h - patch_h)//stride_h + 1)*((width + stride_w - (width - patch_w) % stride_w - patch_w)//stride_w + 1)

def write_hdf5(arr,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)

def mkdirs(newdir):
    try: os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory 
        if err.errno != errno.EEXIST or not os.path.isdir(newdir): 
            raise

def get_filename(dataset_path, suffix, train_test, fileCounter):
    return dataset_path + train_test + "/" + suffix + "/" + str(fileCounter) + ".png"

def save_samples(img_patches, gt_patches):


def get_datasets(
    dataset_path,
    Nimgs,
    imgs_dir,
    groundTruth_dir,
    borderMasks_dir,
    N_subimgs,
    train_test="null"
):
    fileCounter = 0

    mkdirs(dataset_path + train_test + "/imgs")
    mkdirs(dataset_path + train_test + "/groundtruths")

    for _, _, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            img = np.asarray(Image.open(imgs_dir + files[i]).convert('L'))
            g_truth = np.asarray(Image.open(groundTruth_dir + files[i]).convert('L'))
            img = np.reshape(img, (1, 1, img.shape[0], img.shape[1]))
            g_truth = np.reshape(g_truth, (1, 1, g_truth.shape[0], g_truth.shape[1]))

            if i % 100 == 99:
                print('processing img ' + str(i + 1))
            
            # test imgs
            assert(img.shape == (1, 1, height, width))
            # test g_truths
            assert(np.max(g_truth)==255)
            assert(np.min(g_truth)==0)
            assert(g_truth.shape == (1, 1, height, width))

            # extract patches
            img = my_PreProc(img)
            img_data = extract_ordered_overlap(img, patch_size, stride_size)
            # preprocess img
            gt_data  = extract_ordered_overlap(g_truth, patch_size, stride_size)
            

            for i in range(img_data.shape[0]):
            encoded_image_string = cv2.imencode(‘.jpg’, image)[1].tostring()
                filename_imgs = get_filename(dataset_path, 'imgs', train_test, fileCounter)
                # print("writing " + filename_imgs)
                Image.fromarray(img_data[i, 0].astype(np.uint8), 'L').save(filename_imgs)
                filename_gts = get_filename(dataset_path, 'groundtruths', train_test, fileCounter)
                # print("writing " + filename_gts)
                Image.fromarray(gt_data[i, 0].astype(np.uint8), 'L').save(filename_gts)
                fileCounter += 1

def prepare_dataset(configuration):

    dataset_path = configuration['path']

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    #getting the testing datasets
    get_datasets(
        dataset_path,
        int(configuration['N_imgs_test']),        
        configuration['original_imgs_test'],
        configuration['groundTruth_imgs_test'],
        configuration['borderMasks_imgs_test'],
        int(configuration['N_subimgs']),
        "test"
    )
    print ("test data done!")

    #getting the training datasets
    get_datasets(
        dataset_path,
        int(configuration['N_imgs_train']),
        configuration['original_imgs_train'],
        configuration['groundTruth_imgs_train'],
        configuration['borderMasks_imgs_train'],
        int(configuration['N_subimgs']),
        "train"
    )
    print("train data done!")

def get_data(imgs, gts, subimgs, test_train):
    if test_train == 'train':
        img_patches, gt_patches = get_data_training(
            imgs,
            gts,
            patch_h,
            patch_w,
            subimgs,
            inside_FOV = config.getboolean('data attributes', 'inside_FOV') #select the patches only inside the FOV  (default == True)
        )
        print(img_patches.shape)
        permutation = np.random.permutation(img_patches.shape[0])
        img_patches, gt_patches = img_patches[permutation], gt_patches[permutation]
    elif average_mode:
        img_patches, _, _, gt_patches = get_data_testing_overlap(
            imgs,
            gts,
            imgs.shape[0],
            patch_h,
            patch_w,
            stride_h,
            stride_w
        )
    else:
        img_patches, _, _, gt_patches = get_data_testing(
            imgs,
            gts,
            imgs.shape[0],
            patch_h,
            patch_w
        )

    return img_patches, gt_patches

print("")
print("processing DRIVE dataset")
prepare_dataset(config['DRIVE'])
print("")
print("processing Synth dataset")
prepare_dataset(config['Synth'])
