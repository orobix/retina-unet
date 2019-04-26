#==========================================================
#
#  This prepare the hdf5 datasets of the NEW database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image
import configparser
import sys
sys.path.insert(0, './lib/')
from extract_patches import get_data_training


#config file to read from
config = configparser.RawConfigParser()
config.read('./global_config.txt')

batch_size = int(config['global']['hdf5_batch_size'])
patch_h = int(config['data attributes']['patch_height'])
patch_w = int(config['data attributes']['patch_width'])

def write_hdf5(arr,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)

def write_virtual_layout(layout, output):
    with h5py.File(output, 'w', libver='latest') as f:
        f.create_virtual_dataset('image', layout, fillvalue=-5)

def get_filename(dataset_path, suffix, train_test, fileCounter):
    return dataset_path + "dataset_" + suffix + "_" + train_test + str(fileCounter) + ".hdf5"

def get_datasets(
    dataset_path,
    Nimgs,
    imgs_dir,
    groundTruth_dir,
    borderMasks_dir,
    N_subimgs,
    train_test="null"
):
    channels = 3
    height = 565
    width = 565
    total_imgs = Nimgs

    if Nimgs > batch_size:
        Nimgs = batch_size

    shape_imgs = (Nimgs,height,width, channels)
    imgs = np.empty(shape_imgs, dtype=np.uint8)
    layout_imgs = h5py.VirtualLayout(shape=(total_imgs * N_subimgs, 1, patch_h, patch_w), dtype=np.uint8)

    shape_gts = (Nimgs, height, width)
    g_truths = np.empty(shape_gts, dtype=np.uint8)
    layout_gts = h5py.VirtualLayout(shape=(total_imgs * N_subimgs, 1, patch_h, patch_w), dtype=np.uint8)

    fileCounter = 0
    readFile = False

    for _, _, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            img = Image.open(imgs_dir+files[i])
            g_truth = Image.open(groundTruth_dir + files[i]).convert('L')
            imgs[i % batch_size] = np.asarray(img)
            g_truths[i % batch_size] = np.asarray(g_truth)
            readFile = True

            if i % 100 == 0:
                print('reading img ' + str(i))
            
            if i % batch_size == (batch_size - 1):
                # test imgs
                imgs = np.transpose(imgs,(0,3,1,2))
                assert(imgs.shape == (Nimgs,channels,height,width))

                # test g_truths
                assert(np.max(g_truths)==255)
                assert(np.min(g_truths)==0)
                g_truths = np.reshape(g_truths,(Nimgs,1,height,width))
                assert(g_truths.shape == (Nimgs,1,height,width))

                # extract patches
                img_data, gt_data = get_data(imgs, g_truths, N_subimgs, train_test)
                
                filename_imgs = get_filename(dataset_path, 'imgs', train_test, fileCounter)
                print("writing " + filename_imgs)
                write_hdf5(img_data, filename_imgs)
                vsource_imgs = h5py.VirtualSource(filename_imgs, 'image', shape=img_data.shape)
                layout_imgs[fileCounter * Nimgs * N_subimgs: (fileCounter + 1) * Nimgs * N_subimgs] = vsource_imgs
                
                filename_gts = get_filename(dataset_path, 'groundTruths', train_test, fileCounter)
                print("writing " + filename_gts)
                write_hdf5(gt_data, filename_gts)
                vsource_gts = h5py.VirtualSource(filename_gts, 'image', shape=gt_data.shape)
                layout_gts[fileCounter * Nimgs * N_subimgs: (fileCounter + 1) * Nimgs * N_subimgs] = vsource_gts

                fileCounter += 1
                readFile = False
                imgs = np.empty(shape_imgs, dtype=np.uint8)
                g_truths = np.empty(shape_gts, dtype=np.uint8)

    if readFile:
        # test imgs
        imgs = np.transpose(imgs[:total_imgs % batch_size], (0,3,1,2))
        assert(imgs.shape == (total_imgs % batch_size,channels,height,width))

        # test g_truths
        assert(np.max(g_truths)==255)
        assert(np.min(g_truths)==0)
        g_truths = np.reshape(g_truths[:total_imgs % batch_size], (total_imgs % batch_size,1,height,width))
        assert(g_truths.shape == (total_imgs % batch_size,1,height,width))

        # extract patches
        img_data, gt_data = get_data(imgs, g_truths, N_subimgs, train_test)
        print(img_data.shape)
        filename_imgs = get_filename(dataset_path, 'imgs', train_test, fileCounter)
        print("writing " + filename_imgs)
        write_hdf5(img_data, filename_imgs)
        vsource_imgs = h5py.VirtualSource(filename_imgs, 'image', shape=img_data.shape)
        layout_imgs[fileCounter * Nimgs * N_subimgs:] = vsource_imgs
        
        filename_gts = get_filename(dataset_path, 'groundTruths', train_test, fileCounter)
        print("writing " + filename_gts)
        write_hdf5(gt_data, filename_gts)
        vsource_gts = h5py.VirtualSource(filename_gts, 'image', shape=gt_data.shape)
        layout_gts[fileCounter * Nimgs * N_subimgs:] = vsource_gts

    # write layout
    write_virtual_layout(layout_imgs, dataset_path + "dataset_imgs_" + train_test + ".hdf5")
    write_virtual_layout(layout_gts, dataset_path + "dataset_groundTruths_" + train_test + ".hdf5")

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
    train, test = get_data_training(
        imgs,
        gts,
        patch_height = int(config.get('data attributes', 'patch_height')),
        patch_width = int(config.get('data attributes', 'patch_width')),
        N_subimgs = subimgs,
        inside_FOV = config.getboolean('data attributes', 'inside_FOV') #select the patches only inside the FOV  (default == True)
    )

    permutation = np.random.permutation(train.shape[0])
    return train[permutation], test[permutation]

print("")
print('batch_size: ' + str(batch_size))
prepare_dataset(config['DRIVE'])
print("")
print('batch_size: ' + str(batch_size))
prepare_dataset(config['Synth'])