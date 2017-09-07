"""Vgg16 in Keras 2.0 and Python +3.5.

For more information see the Keras 2.0 release notes:
https://github.com/fchollet/keras/wiki/Keras-2.0-release-notes
"""
from keras import backend as keras_backedn
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Lambda
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image

import json
import numpy as np
import os


# In case we are going to use the TensorFlow backend we need to explicitly set
# the Theano image ordering
keras_backedn.set_image_dim_ordering('th')

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


vgg_mean = np.array(
    [123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))


def vgg_preprocess(x):
    """
        Subtracts the mean RGB value, and transposes RGB to BGR.
        The mean RGB was computed on the image set used to train the VGG model.

        Args:
            x: Image array (height x width x channels)
        Returns:
            Image array (height x width x transposed_channels)
    """
    x = x - vgg_mean
    return x[:, ::-1]  # reverse axis rgb->bgr


class Vgg16:
    """The VGG 16 Imagenet model."""

    def __init__(
        self,
        file_path='http://files.fast.ai/models/',
        vgg_weights='vgg16.h5',
        imagenet_class_index='imagenet_class_index.json'
    ):
        self.FILE_PATH = file_path
        self.vgg_weights = vgg_weights
        self.imagenet_class_index = imagenet_class_index
        self.create()
        self.get_classes()

    def get_classes(self):
        """Downloads the Imagenet classes index file and loads it.

        It loads Imagenet classes index file to self.classes.
        The file is downloaded only if it not already in the cache.
        """

        # fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(os.path.join(self.FILE_PATH, self.imagenet_class_index)) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def predict(self, imgs, details=False):
        """Predict the labels of a set of images using the VGG16 model.

            Args:
                imgs (ndarray)    : An array of N images (size:
                    N x width x height x channels).
                details : ??

            Returns:
                preds (np.array) : Highest confidence value of the predictions
                                   for each image.
                idxs (np.ndarray): Class index of the predictions with the max
                                   confidence.
                classes (list)   : Class labels of the predictions with the max
                                   confidence.
        """
        # predict probability of each class for each image
        all_preds = self.model.predict(imgs)
        # for each image get the index of the class with max probability
        idxs = np.argmax(all_preds, axis=1)
        # get the values of the highest probability for each image
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
        # get the label of the class with the highest probability for each image
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds), idxs, classes

    def ConvBlock(self, layers, filters):
        """Add ZeroPadding2D, Conv2D and MaxPooling2D layers.

        Adds a specified number of ZeroPadding and Covolution layers to the
        model, and a MaxPooling layer at the very end.

        Args:
            layers (int):   The number of zero padded convolution layers
                            to be added to the model.
            filters (int):  The number of convolution filters to be
                            created for each layer.
        """
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Conv2D(filters, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def FCBlock(self):
        """Add a fully connected layer of 4096 neurons to the model.

        Adds a fully connected layer of 4096 neurons to the model with a
        Dropout of 0.5

        Args:   None
        Returns:   None
        """
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

    def create(self):
        """Create the VGG16 network achitecture and loads the pretrained weights.  # noqa

        Args:   None
        Returns:   None
        """
        model = self.model = Sequential()
        model.add(Lambda(
            vgg_preprocess,
            input_shape=(3, 224, 224),
            output_shape=(3, 224, 224),
        ))

        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        model.add(Dense(1000, activation='softmax'))

        # model.load_weights(
        #     get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))
        model.load_weights(os.path.join(self.FILE_PATH, self.vgg_weights))

    def get_batches(
        self,
        path: str,
        gen=image.ImageDataGenerator(),
        shuffle: bool=True,
        batch_size: int=8,
        class_mode: str='categorical',
    ):
        """Generate batches of normalized data.

        Takes the path to a directory, and generates batches of
        augmented/normalized data. Yields batches indefinitely, in an infinite
        loop.

        See Keras documentation: https://keras.io/preprocessing/image/
        """
        return gen.flow_from_directory(
            path,
            target_size=(224, 224),
            class_mode=class_mode,
            shuffle=shuffle,
            batch_size=batch_size,
        )

    def replace_with_dense(self, num: int):
        """Replace the last layer with `Dense` layer.

        Replace the last layer of the model with a Dense (fully connected)
        layer of num neurons.

        Will also lock the weights of all layers except the new layer so that
        we only learn weights for the last layer in subsequent training.

        Args:
            num (int) : Number of neurons in the Dense layer
        Returns:
                None
        """
        model = self.model
        model.pop()

        for layer in model.layers:
            layer.trainable = False

        model.add(Dense(num, activation='softmax'))
        self.compile()

    def finetune(self, batches: int):
        """

        Modifies the original VGG16 network architecture and updates
        self.classes for new training data.

        Args:
            batches : A keras.preprocessing.image.ImageDataGenerator object.
                      See definition for get_batches().
        """
        self.replace_with_dense(batches.num_class)
        # get a list of all the class labels
        classes = list(iter(batches.class_indices))

        # batches.class_indices is a dict with the class name as key and an
        # index as value eg. {'cats': 0, 'dogs': 1}

        # sort the class labels by index according to batches.class_indices
        # and update model.classes

        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
        self.classes = classes

    def compile(self, lr: float=0.001):
        """Configures the model for training.

        See Keras documentation: https://keras.io/models/model/
        """
        self.model.compile(
            optimizer=Adam(lr=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    def fit(self, batches, val_batches, epoch=1):
        """Trains the model for a fixed number of epochs on a dataset.

        Fits the model on data yielded batch-by-batch by a Python generator.

        See Keras documentation: https://keras.io/models/model/
        """
        # fits the model on batches with real-time data augmentation:
        self.model.fit_generator(
            batches,
            steps_per_epoch=batches.samples,
            epochs=epoch,
            validation_data=val_batches,
            validation_steps=val_batches.samples,
        )

    def test(self, path: str, batch_size: int=8):
        """Test the model.

        Predicts the classes using the trained model on data yielded
        batch-by-batch.

        Args:
            path (string):  Path to the target directory. It should contain
                            one subdirectory per class.
            batch_size (int): The number of images to be considered in each
                              batch.

        Returns:
            test_batches, numpy array(s) of predictions for the test_batches.
        """
        test_batches = self.get_batches(
            path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(
            test_batches, test_batches.samples)
