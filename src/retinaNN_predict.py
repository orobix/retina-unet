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
test_data_stats = config.get('data paths', 'test_data_stats')

stats_config = configparser.RawConfigParser()
stats_config.read(test_data_stats)
full_img_height = int(config.get('statistics', 'new_image_height'))
full_img_width = int(config.get('statistics', 'new_image_width'))

# dimension of the patches
patch_size = (int(config.get('data attributes', 'patch_height')), int(config.get('data attributes', 'patch_width')))

#the stride in case output with average
stride_size = (int(config.get('testing settings', 'stride_height')), int(config.get('testing settings', 'stride_width')))
assert (stride_size[0] < patch_size[0] and stride_size[1] < patch_size[1])

#model name
name_experiment = config.get('experiment', 'name')
arch = config.get('experiment', 'arch')
testset = config.get('experiment', 'testset')
experiment_path = path_data + '/' + name_experiment + '_' + arch
save_path = experiment_path + '/' + testset

#N full images to be predicted
imgs_to_visualize = int(config.get('testing settings', 'imgs_to_visualize'))
N_subimgs = int(config.get('testing settings', 'N_subimgs'))
patches_per_img = int(stats_config.get('statistics', 'subimages_per_image'))

#Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))
batch_size = int(config.get('training settings', 'batch_size'))


#================= Load the data =====================================================
dataset = load_testset(test_data_path, batch_size)
patches_imgs_samples, patches_gts_samples = load_images_labels(
        test_data_path,
        batch_size,
        N_subimgs,
        test = True
    )

patches_embedding = patches_imgs_samples[:patches_per_img * imgs_to_visualize]
patches_embedding_gt = tf.reshape(patches_gts_samples[:patches_per_img * imgs_to_visualize, 1], (patches_per_img * imgs_to_visualize, 1, patch_size[0], patch_size[1]))
patches_embedding, patches_embedding_gt = session.run([patches_embedding, patches_embedding_gt])

#================ Run the prediction of the patches ==================================
best_last = config.get('testing settings', 'best_last')

#Load the saved model
model = model_from_json(
    open(experiment_path + '/' + name_experiment +'_architecture.json').read()
)
model.compile(
    optimizer = 'sgd',
    loss = weighted_cross_entropy,
    metrics = [
        tf.keras.metrics.SensitivityAtSpecificity(), # auc roc
        tf.keras.metrics.SpecificityAtSensetivity(), # auc roc
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.FalsePositives(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.FalseNegatives() # confusion
    ]
)
model.load_weights(experiment_path + '/' + name_experiment + '_' + best_last + '_weights.h5')

print("start prediction")
#Calculate the predictions
predictions = model.predict(
    dataset,
    batch_size = batch_size,
    steps = int(N_subimgs / batch_size)
)

print("predicted images size :")
print(predictions.shape)

#===== Convert the prediction arrays in corresponding images
# get patches for visualization
vis_patches_predictions = predictions[:patches_per_img * imgs_to_visualize]

pred_patches = pred_to_imgs(vis_patches_predictions, patch_size[0], patch_size[1], "threshold")
print(np.max(pred_patches))
print(np.min(pred_patches))

# #========== Elaborate and visualize the predicted images ====================
pred_imgs = recompone_overlap(
    pred_patches,
    full_img_height,
    full_img_width,
    stride_size[0],
    stride_size[1]
) * 255
orig_imgs = recompone_overlap(
    patches_embedding,
    full_img_height,
    full_img_width,
    stride_size[0],
    stride_size[1]
) * 255
gtruth_masks = recompone_overlap(
    patches_embedding_gt,
    full_img_height,
    full_img_width,
    stride_size[0],
    stride_size[1]
) * 255

print("Orig imgs shape: " +str(orig_imgs.shape))
print("pred imgs shape: " +str(pred_imgs.shape))
print("Gtruth imgs shape: " +str(gtruth_masks.shape))
visualize(group_images(orig_imgs, N_visual), save_path + "_all_originals")#.show()
visualize(group_images(pred_imgs, N_visual), save_path + "_all_predictions")#.show()
visualize(group_images(gtruth_masks,N_visual), save_path + "_all_groundTruths")#.show()
#visualize results comparing mask and prediction:
assert (orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
N_predicted = orig_imgs.shape[0]
group = N_visual
assert (N_predicted%group==0)
for i in range(int(N_predicted/group)):
    fr = i * group
    to = i * group + group
    orig_stripe =  group_images(orig_imgs[fr: to], group)
    masks_stripe = group_images(gtruth_masks[fr: to], group)
    pred_stripe =  group_images(pred_imgs[fr: to], group)
    total_img = np.concatenate((orig_stripe, masks_stripe, pred_stripe), axis=0)
    visualize(total_img, save_path + "_Original_GroundTruth_Prediction" + str(i))#.show()

#========================== Evaluate the results ===================================
print("\n\n========  Evaluate the results =======================")

sensitivities, \ 
specificities, \
true_positives, \
false_positives, \
true_negatives, \
false_negatives = model.evaluate(
    dataset,
    batch_size = batch_size,
    steps = int(N_subimgs / batch_size)
)

# Area under the ROC curve
roc_curve=plt.figure()
print(sensitivities.shape)
print(specificities.shape)
# plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % auc )
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc = "lower right")
plt.savefig(save_path + "_ROC.png")

# Precision-recall curve
# print("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
prec_rec_curve = plt.figure()
# plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc = "lower right")
plt.savefig(path_experiment + "_Precision_recall.png")

# Confusion matrix
confusion = np.array([[true_positives, false_positives], [true_negatives, false_negatives]])
print(confusion)
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print("Global Accuracy: " +str(accuracy))
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print("Specificity: " +str(specificity))
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print("Sensitivity: " +str(sensitivity))
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print("Precision: " +str(precision))

#Save the results
with open(path_experiment+'performances.txt', 'w') as file:
    file.write(
        "Confusion matrix:"
        + str(confusion)
        + "\nACCURACY: " + str(accuracy)
        + "\nSENSITIVITY: " + str(sensitivity)
        + "\nSPECIFICITY: " + str(specificity)
        + "\nPRECISION: " + str(precision)
    )
    file.close()
