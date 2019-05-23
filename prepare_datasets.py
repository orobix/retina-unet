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
import tensorflow as tf
import cv2
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

def mkdirs(newdir):
    try: os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory 
        if err.errno != errno.EEXIST or not os.path.isdir(newdir): 
            raise

def get_filename_img(dataset_path, suffix, train_test, fileCounter):
    return dataset_path + train_test + "/" + suffix + "/" + str(fileCounter) + ".png"

def get_filename_tfrecord(dataset_path, suffix, train_test, fileCounter):
    return dataset_path + "dataset_" + suffix + "_" + train_test + str(fileCounter) + ".tfrecord"

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_datasets(
    dataset_path,
    Nimgs,
    imgs_dir,
    groundTruth_dir,
    borderMasks_dir,
    train_test="null"
):
    fileCounter = 0
    writer = tf.python_io.TFRecordWriter(get_filename_tfrecord(dataset_path, '', train_test, fileCounter))

    new_image_size = (0., 0.)
    n_subimgs = 0
    n_imgs

    mean_img = 0.
    var_img = 0.
    mean_gt = 0.
    var_gt = 0.

    for _, _, files in os.walk(imgs_dir): #list all files, directories in the path
        n_imgs = len(files)
        for i in range(n_imgs):
            #original
            img = np.asarray(Image.open(imgs_dir + files[i]).convert('L'))
            g_truth = np.asarray(Image.open(groundTruth_dir + files[i]).convert('L'))
            img = np.reshape(img, (1, 1, img.shape[0], img.shape[1]))
            g_truth = np.reshape(g_truth, (1, 1, g_truth.shape[0], g_truth.shape[1]))

            print('processing img ' + str(i + 1))
            
            # test imgs
            assert(np.max(img)<=255)
            assert(np.min(img)>=0)
            assert(img.shape == (1, 1, height, width))
            # test g_truths
            assert(np.max(g_truth)<=255)
            assert(np.min(g_truth)>=0)
            assert(g_truth.shape == (1, 1, height, width))

            # extract patches
            img = my_PreProc(img)
            img_data, new_image_size, n_subimgs = extract_ordered_overlap(img, patch_size, stride_size)
            img_data = np.transpose(img_data, (0, 2, 3, 1)).astype(np.uint8)
            mean_img += np.mean(img_data)
            var_img += np.var(img_data)
            # preprocess img
            gt_data, _, _ = extract_ordered_overlap(g_truth, patch_size, stride_size)
            gt_data = np.transpose(gt_data, (0, 2, 3, 1)).astype(np.uint8)
            mean_gt += np.mean(gt_data)
            var_gt += np.var(gt_data)

            for j in range(img_data.shape[0]):
                encoded_img_string = cv2.imencode('.png', img_data[j])[1].tostring()
                encoded_gt_string = cv2.imencode('.png', gt_data[j])[1].tostring()
                feature = {
                    'image': _bytes_feature(tf.compat.as_bytes(encoded_img_string)),
                    'label': _bytes_feature(tf.compat.as_bytes(encoded_gt_string)),
                }
                tf_example = tf.train.Example(
                    features = tf.train.Features(feature=feature)
                )
                writer.write(tf_example.SerializeToString())
            
            # create each 20 images a new tfrecord (ca 500mb)
            if i % 20 == 19:
                writer.close()
                fileCounter += 1
                print('create new writer: ' + get_filename_tfrecord(dataset_path, '', train_test, fileCounter))
                writer = tf.python_io.TFRecordWriter(get_filename_tfrecord(dataset_path, '', train_test, fileCounter))
    
    print('mean_img: ' + str(mean_img) + ' / ' + str(n_imgs) + ' = ' + str(mean_img / n_imgs))
    mean_img = mean_img / n_imgs
    print('std_img: sqrt(' + str(var_img) + ' / ' + str(n_imgs) + ') = ' + str(np.sqrt(var_img / n_imgs)))
    std_img = np.sqrt(var_img / n_imgs)
    print('mean_gt: ' + str(mean_gt) + ' / ' + str(n_imgs) + ' = ' + str(mean_gt / n_imgs))
    mean_gt = mean_gt / n_imgs
    print('std_gt: sqrt(' + str(var_gt) + ' / ' + str(n_imgs) + ') = ' + str(np.sqrt(var_gt / n_imgs)))
    std_gt = np.sqrt(var_gt / n_imgs)
    
    with open(dataset_path + 'stats_' + train_test + '.csv', 'w') as f:
        f.write('[statistics]')
        f.write('new_image_height = ' + str(new_image_size[0]))
        f.write('new_image_width = ' + str(new_image_size[1]))
        f.write('subimages_per_image = ' + str(n_subimgs))
        f.write('mean_images = ' + str(mean_img))
        f.write('std_images = ' + str(std_img))
        f.write('mean_groundtruths = ' + str(mean_gt))
        f.write('std_groundtruths = ' + str(std_gt))
    writer.close()

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
        "test"
    )
    print("test data done!")

    #getting the training datasets
    get_datasets(
        dataset_path,
        int(configuration['N_imgs_train']),
        configuration['original_imgs_train'],
        configuration['groundTruth_imgs_train'],
        configuration['borderMasks_imgs_train'],
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
