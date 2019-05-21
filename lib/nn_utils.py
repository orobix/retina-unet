import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops

import sys
sys.path.insert(0, './lib/')
from loader import load_images_labels
from help_functions import visualize, group_images

def weighted_cross_entropy(weight):
    def loss(y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, weight)
    return loss

# output of the net is between -1 and 1
# labels are between 0 and 1
def accuracy(y_true, y_pred):
    y = tf.cast(y_true > 0, y_true.dtype)
    y_ = tf.cast(y_pred > 0.5, y_pred.dtype)
    return K.mean(math_ops.equal(y, y_))

def visualize_samples(
    session,
    experiment_path,
    patches_imgs_samples,
    patches_gts_samples,
    patch_size
):

    patches_imgs_samples = (patches_imgs_samples[0:20] + 3) * 255. / 6.
    patches_gts_samples = tf.cast(patches_gts_samples[0:20] * 255., tf.float32)
    patches_gts_samples = tf.reshape(
        patches_gts_samples,
        (20, 1, patch_size[0], patch_size[1])
    )


    imgs_samples = session.run(
        tf.concat([patches_imgs_samples, patches_gts_samples], 0)
    )
    visualize(group_images(imgs_samples, 5), experiment_path + '/' + "sample_input")

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    if channel == 1:
        tensor = tensor.reshape((height, width))
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(
        height=height,
        width=width,
        colorspace=channel,
        encoded_image_string=image_string
    )

def sigmoid(x):
  return 1/(1+np.exp(-x))

class TensorBoardOutputCallback(tf.keras.callbacks.Callback):
    def __init__(self, tag, logdir, testdata, batch_size, patch_size, freq = 1):
        super().__init__() 
        self.tag = tag
        self.logdir = logdir
        self.testdata = np.array(testdata[0])
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.freq = freq

    def on_train_begin(self, logs=None):
        data = self.testdata.transpose((0, 2, 3, 1))
        data = (data + 3.) * 255 / 6.
        data = data.astype('uint8')
        values = []
        writer = tf.summary.FileWriter(self.logdir)
        for i in range(data.shape[0]):
            img = make_image(data[i])
            values.append(tf.Summary.Value(tag=self.tag + ' input', image=img))
        summary = tf.Summary(value=values)
        writer.add_summary(summary, 0)
        writer.close()
        return

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.freq == 0:
            images = self.model.predict(
                self.testdata,
                batch_size = 32)
            images = sigmoid(images) * 255.
            assert(np.max(images) <= 255.)
            assert(np.min(images) >= 0.)
            images = images.astype('uint8').reshape([
                self.batch_size,
                self.patch_size[0],
                self.patch_size[1],
                1
            ])

            values = []
            writer = tf.summary.FileWriter(self.logdir)
            for i in range(images.shape[0]):
                img = make_image(images[i])
                values.append(tf.Summary.Value(tag=self.tag + ' output', image=img))
            summary = tf.Summary(value=values)
            writer.add_summary(summary, epoch)
            writer.close()

        return
