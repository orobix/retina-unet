import numpy as np
import random
import configparser

from help_functions import load_hdf5
from help_functions import visualize
from help_functions import group_images

from skimage.util.shape import view_as_blocks
from skimage.util.shape import view_as_windows
from skimage.util import pad

#Divide all the full_imgs in pacthes
def extract_ordered(full_imgs, patch_size):
    return _extract_patches(full_imgs, patch_size, (0, 0), False)

def extract_ordered_overlap(full_imgs, patch_size, stride_size):
    return _extract_patches(full_imgs, patch_size, stride_size, True)

# patch_size and stride_size => h x w
def _extract_patches(full_imgs, patch_size, stride_size, overlap):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3

    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image

    # prepare padding
    img_shape = np.array(full_imgs[0,0].shape)
    patch_size = np.array(patch_size)
    if overlap:
        residual = (img_shape - patch_size) / stride_size
        N_patches = np.ceil((img_shape - patch_size) // stride_size + 1).astype(np.int)
        pad_by = np.array(stride_size) - residual
    else:
        residual = img_shape % patch_size
        N_patches = np.ceil(img_shape / patch_size).astype(np.int)
        pad_by = patch_size - residual

    needs_padding = pad_by[0] > 0 or pad_by[1] > 0
    # can be padded only on one side because border is 0 anyway
    if needs_padding:
        pad_by = ((0, 0), (0, pad_by[0]), (0, pad_by[1])) # only pad img dont add another channel
    patches_per_img = N_patches[0] * N_patches[1]
    # print("number of patches per image: " + str(patches_per_img))
    N_patches_tot = patches_per_img * full_imgs.shape[0]
    patches = np.empty((N_patches_tot, full_imgs.shape[1], patch_size[0], patch_size[1]))

    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        img = full_imgs[i]
        # pad if needed
        if needs_padding:
            img = pad(img, pad_by, 'constant', constant_values=0)
        
        if not overlap:
            patches_of_img = view_as_blocks(img, patch_size).reshape(patches_per_img, 1, patch_size[0], patch_size[1])
        else:
            patches_of_img = view_as_windows(img[0], patch_size, stride_size)
        patches[i * patches_per_img : (i + 1) * patches_per_img] = patches_of_img.reshape(N_patches_tot, 1, patch_size[0], patch_size[1])
        iter_tot += patches_per_img
    assert (iter_tot == N_patches_tot)
    return patches  #array with all the full_imgs divided in patches

def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape)==4)  #4D arrays
    assert (preds.shape[1]==1 or preds.shape[1]==3)  #check the channel is 1 or 3
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    print("N_patches_h: " +str(N_patches_h))
    print("N_patches_w: " +str(N_patches_w))
    print("N_patches_img: " +str(N_patches_img))

    assert (preds.shape[0] % N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    print("According to the dimension inserted, there are " + str(N_full_imgs) + " full images (of " + str(img_h) + "x" + str(img_w) + " each)")
    full_prob = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))  #itialize to zero mega array with sum of Probabilities
    full_sum  = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,:, h * stride_h: h * stride_h + patch_h, w * stride_w: w * stride_w + patch_w] += preds[k]
                full_sum[i, :, h * stride_h: h * stride_h + patch_h, w * stride_w: w * stride_w + patch_w] += 1
                k += 1
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)  #at least one
    final_avg = full_prob/full_sum
    print(final_avg.shape)
    assert(np.max(final_avg)<=1.0) #max value for a pixel is 1.0
    assert(np.min(final_avg)>=0.0) #min value for a pixel is 0.0
    return final_avg


#Recompone the full images with the patches
def recompone(data, N_h, N_w):
    assert (data.shape[1]==1 or data.shape[1]==3)  #check the channel is 1 or 3
    assert(len(data.shape)==4)
    N_pacth_per_img = N_w * N_h
    assert(data.shape[0] % N_pacth_per_img == 0)
    N_full_imgs = data.shape[0]/N_pacth_per_img
    patch_h = data.shape[2]
    patch_w = data.shape[3]
    N_pacth_per_img = N_w * N_h
    #define and start full recompone
    full_recomp = np.empty((N_full_imgs,data.shape[1], N_h * patch_h, N_w * patch_w))
    k = 0  #iter full img
    s = 0  #iter single patch
    while (s<data.shape[0]):
        #recompone one:
        single_recon = np.empty((data.shape[1], N_h * patch_h, N_w * patch_w))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[:, h * patch_h: h * (patch_h + 1), w * patch_w: w * (patch_w + 1)] = data[s]
                s += 1
        full_recomp[k] = single_recon
        k += 1
    assert (k == N_full_imgs)
    return full_recomp


#return only the pixels contained in the FOV, for both images and masks
def pred_only_FOV(data_imgs,data_masks,original_imgs_border_masks):
    assert (len(data_imgs.shape)==4 and len(data_masks.shape)==4)  #4D arrays
    assert (data_imgs.shape[0]==data_masks.shape[0])
    assert (data_imgs.shape[2]==data_masks.shape[2])
    assert (data_imgs.shape[3]==data_masks.shape[3])
    assert (data_imgs.shape[1]==1 and data_masks.shape[1]==1)  #check the channel is 1
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE(i,x,y,original_imgs_border_masks)==True:
                    new_pred_imgs.append(data_imgs[i,:,y,x])
                    new_pred_masks.append(data_masks[i,:,y,x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs, new_pred_masks

#function to set to black everything outside the FOV, in a full image
def kill_border(data):
    assert (len(data.shape)==4)  #4D arrays
    assert (data.shape[1]==1 or data.shape[1]==3)  #check the channel is 1 or 3
    height = data.shape[2]
    width = data.shape[3]
    for i in range(data.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if is_inside_FOV(x, y, width, height)==False:
                    data[i,:,y,x]=0.0


def is_inside_FOV(x,y,img_w,img_h):
    x_ = x - int(img_w/2) # origin (0,0) shifted to image center
    y_ = y - int(img_h/2)  # origin (0,0) shifted to image center
    R_inside = 270
    radius = np.sqrt((x_*x_)+(y_*y_))
    if radius < R_inside:
        return True
    else:
        return False
