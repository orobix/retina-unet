###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

#Python
import numpy as np
import configparser
from matplotlib import pyplot as plt

#Keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow.keras.backend as K

import sys
sys.path.insert(0, './lib/')
from help_functions import *
from loader import load_testset
from extract_patches import recompone
from extract_patches import recompone_overlap

session = K.get_session()
def weighted_cross_entropy(y_true, y_pred):
    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 1.)

#========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('configuration.txt')

path_data = config.get('data paths', 'path_local')
test_data_path = config.get('data paths', 'test_data_path')

full_img_height = config.get('data attributes', 'img_height')
full_img_width = config.get('data attributes', 'img_width')

# dimension of the patches
patch_size = (int(config.get('data attributes', 'patch_height')), int(config.get('data attributes', 'patch_width')))

#the stride in case output with average
stride_size = (int(config.get('testing settings', 'stride_height')), int(config.get('testing settings', 'stride_width')))
assert (stride_size[0] < patch_size[0] and stride_size[1] < patch_size[1])

#model name
name_experiment = config.get('experiment', 'name')
arch = config.get('experiment', 'arch')
experiment_path = path_data + '/' + name_experiment + '_' + arch

#N full images to be predicted
imgs_to_visualize = int(config.get('testing settings', 'imgs_to_visualize'))
N_subimgs = int(config.get('testing settings', 'N_subimgs'))

#Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))
batch_size = int(config.get('training settings', 'batch_size'))


#================= Load the data =====================================================
dataset = load_testset(test_data_path, batch_size)

#================ Run the prediction of the patches ==================================
best_last = config.get('testing settings', 'best_last')

#Load the saved model
model = model_from_json(
    open(experiment_path + '/' + name_experiment +'_architecture.json').read()
)
model.compile(
    optimizer = 'adam',
    loss = weighted_cross_entropy,
    metrics = ['accuracy']
)
model.load_weights(experiment_path + '/' + name_experiment + '_'+best_last+'_weights.h5')

#Calculate the predictions
predictions = model.predict(
    dataset,
    batch_size = batch_size,
    steps = int(N_subimgs / batch_size)
)

print("predicted images size :")
print(predictions.shape)
print(dataset.shape)

#========================== Evaluate the results ===================================
print("\n\n========  Evaluate the results =======================")

#Area under the ROC curve
auc, update_op_auc = tf.metrics.auc(dataset, predictions)
print(session.run([auc, update_op_auc]))
# print("\nArea under the ROC curve: " +str(auc))
# roc_curve=plt.figure()
# plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % auc )
# plt.title('ROC curve')
# plt.xlabel("FPR (False Positive Rate)")
# plt.ylabel("TPR (True Positive Rate)")
# plt.legend(loc="lower right")
# plt.savefig(path_experiment+"ROC.png")

#Precision-recall curve
recall, update_op_recall = tf.metrics.recall(dataset, predictions)
precision, update_op_precision = tf.metrics.precision(dataset, predictions)
print(session.run([recall, update_op_recall, precision, update_op_precision]))
# print("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
# prec_rec_curve = plt.figure()
# plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
# plt.title('Precision - Recall curve')
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.legend(loc="lower right")
# plt.savefig(path_experiment+"Precision_recall.png")

#Confusion matrix
threshold_confusion = 0.5
print("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
tp, u_op_fn = tf.metrics.true_positives_at_thresholds(dataset, predicitons, [threshold_confusion, threshold_confusion])
tn, u_op_fn = tf.metrics.true_negatives_at_thresholds(dataset, predicitons, [threshold_confusion, threshold_confusion])
fp, u_op_fn = tf.metrics.false_positives_at_thresholds(dataset, predicitons, [threshold_confusion, threshold_confusion])
fn, u_op_fn = tf.metrics.false_negatives_at_thresholds(dataset, predicitons, [threshold_confusion, threshold_confusion])
print(session.run([tp, u_op_fn, ]))
# confusion = np.array([[tp, fp], [tn, fn]])
# print(confusion)
# accuracy = 0
# if float(np.sum(confusion))!=0:
#     accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
# print("Global Accuracy: " +str(accuracy))
# specificity = 0
# if float(confusion[0,0]+confusion[0,1])!=0:
#     specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
# print("Specificity: " +str(specificity))
# sensitivity = 0
# if float(confusion[1,1]+confusion[1,0])!=0:
#     sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
# print("Sensitivity: " +str(sensitivity))
# precision = 0
# if float(confusion[1,1]+confusion[0,1])!=0:
#     precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
# print("Precision: " +str(precision))

# #Save the results
# file_perf = open(path_experiment+'performances.txt', 'w')
# file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
#                 + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
#                 + "\nJaccard similarity score: " +str(jaccard_index)
#                 + "\nF1 score (F-measure): " +str(F1_score)
#                 +"\n\nConfusion matrix:"
#                 +str(confusion)
#                 +"\nACCURACY: " +str(accuracy)
#                 +"\nSENSITIVITY: " +str(sensitivity)
#                 +"\nSPECIFICITY: " +str(specificity)
#                 +"\nPRECISION: " +str(precision)
#                 )
# file_perf.close()

# #===== Convert the prediction arrays in corresponding images
# # TODO!!!!
# pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "threshold")


# #========== Elaborate and visualize the predicted images ====================
# pred_imgs = None
# orig_imgs = None
# gtruth_masks = None

# ### TODO: batchwise recomponment
# if average_mode == True:
#     pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
#     orig_imgs = my_PreProc(test_imgs_orig[:20,:,:,:])    #originals
#     gtruth_masks = masks_test  #ground truth masks
# else:
#     pred_imgs = recompone(pred_patches,13,12)       # predictions
#     orig_imgs = recompone(patches_imgs_test,13,12)  # originals
#     gtruth_masks = recompone(patches_masks_test,13,12)  #masks
# # apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
# # kill_border(pred_imgs, test_border_masks)  #DRIVE MASK  #only for visualization
# ## back to original dimensions
# orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
# pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
# gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]
# print("Orig imgs shape: " +str(orig_imgs.shape))
# print("pred imgs shape: " +str(pred_imgs.shape))
# print("Gtruth imgs shape: " +str(gtruth_masks.shape))
# visualize(group_images(orig_imgs,N_visual),path_experiment+"all_originals")#.show()
# visualize(group_images(pred_imgs,N_visual),path_experiment+"all_predictions")#.show()
# visualize(group_images(gtruth_masks,N_visual),path_experiment+"all_groundTruths")#.show()
# #visualize results comparing mask and prediction:
# assert (orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
# N_predicted = orig_imgs.shape[0]
# group = N_visual
# assert (N_predicted%group==0)
# for i in range(int(N_predicted/group)):
#     orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)
#     masks_stripe = group_images(gtruth_masks[i*group:(i*group)+group,:,:,:],group)
#     pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)
#     total_img = np.concatenate((orig_stripe,masks_stripe,pred_stripe),axis=0)
#     visualize(total_img,path_experiment+name_experiment +"_Original_GroundTruth_Prediction"+str(i))#.show()
