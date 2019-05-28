import configparser
import csv
import tensorflow as tf
import glob
from tensorflow.io import decode_png
import tensorflow.keras.backend as K

config = configparser.RawConfigParser()
config.read('configuration.txt')

PATCH_SIZE = (
    int(config.get('data attributes', 'patch_height')),
    int(config.get('data attributes', 'patch_width'))
)

def load_testset(filepath, batch_size):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(glob.glob(filepath))

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    return dataset.map(_parse_function, num_parallel_calls=8) \
        .map(normalize, num_parallel_calls=8) \
        .batch(batch_size, drop_remainder=True) \
        .repeat()




def load_trainset(filepath, batch_size, N_imgs):
    dataset = tf.data.TFRecordDataset(glob.glob(filepath))
    
    # Set the number of datapoints you want to load and shuffle
    # do not reshuffle each iteration
    print('total samples:')
    print(N_imgs)
    dataset = dataset.shuffle(250000)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8) \
        .map(normalize, num_parallel_calls=8)

    #split in testing and training
    test_data = dataset.take(int(N_imgs / 10)) \
        .batch(batch_size, drop_remainder=True) \
        .repeat()

    train_data = dataset.skip(int(N_imgs / 10)) \
        .batch(batch_size, drop_remainder=True) \
        .repeat()

    return test_data, train_data




def load_images_labels(filepath, batch_size, N_imgs):
    _, train_data = load_trainset(filepath, batch_size, N_imgs)
    
    # Create an iterator
    iterator = train_data.make_one_shot_iterator()
    
    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    return image, label


# ================================ HELPER =======================================
def normalize(image, label):
    # image = (image - MEAN) / STD
    image = tf.cast(image, tf.float32) / 255. * 6. - 3.    # normalize to [-3, 3]
    label = tf.cast(label, tf.float32) / 255.         # label from 0 - 1

    foreground = tf.cast(label > 0.4, tf.float32)
    bkgrnd = 1 - foreground
    label = tf.concat([bkgrnd, foreground], 0)

    assert_max_label = tf.Assert(tf.less_equal(tf.reduce_max(label), 1.), [label])
    assert_max_image = tf.Assert(tf.less_equal(tf.reduce_max(image), 3.), [image])
    assert_min_label = tf.Assert(tf.greater_equal(tf.reduce_min(label), 0.), [label])
    assert_min_image = tf.Assert(tf.greater_equal(tf.reduce_min(image), -3.), [image])
    with tf.control_dependencies([
        assert_max_label,
        assert_max_image,
        assert_min_label,
        assert_min_image
    ]):
        return image, label

def _parse_function(proto):
    global PATCH_SIZE
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                        'label': tf.FixedLenFeature([], tf.string)}
    
    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)
    
    # Turn your saved image string into an array
    image = decode_png(parsed_features['image'])
    label = decode_png(parsed_features['label'])


    # Bring your picture back in shape
    image = tf.reshape(image, [1, PATCH_SIZE[0], PATCH_SIZE[1]])
    label = tf.reshape(label, [1, PATCH_SIZE[0] * PATCH_SIZE[1]])
    
    return image, label
