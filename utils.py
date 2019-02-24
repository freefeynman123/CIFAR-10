import pickle
from collections import defaultdict
import numpy as np
import os
import time
from typing import Union
from functools import wraps
from skimage.transform import resize
from sklearn.utils import shuffle
from bs4 import BeautifulSoup
from urllib.request import urlopen, urlretrieve
import cv2
import re
import tarfile
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def download():
    """Function to download CIFAR-10 data"""
    try:
        html_body = urlopen("https://www.cs.toronto.edu/~kriz/cifar.html")
    except ValueError:
        print("The webpage address has changed")
        raise ValueError
    soup = BeautifulSoup(html_body, 'html.parser')
    file_to_download = soup.find(href=re.compile('^.*python.tar.gz$'))
    file = file_to_download.get('href')
    # Omit cifar.html from original address and replace it by href found in file_to_download
    url_to_download = '/'.join(html_body.geturl().split('/')[:-1] + [file])
    urlretrieve(url_to_download, "cifar_data.tar.gz")
    tar = tarfile.open('cifar_data.tar.gz', 'r:gz')
    tar.extractall()
    tar.close()
    #Since we have extracted data from tar.gz we can get rid of it
    os.remove('cifar_data.tar.gz')


def unpickle(file: str):
    """
    Function taken from Alex Krizhevsky's homepage to load CIFAR-10 data
    Args:
        file: Path to the file with batch of the data

    Returns:
        dictionary: Unpickled data
    """
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def load_cifar_data():
    """
    Function to load cifar data downloaded with download function
    """
    train = defaultdict(list)
    for index in range(1, 5):
        for k, v in unpickle('cifar-10-batches-py/data_batch_{i}'.format(i=index)).items():
            train[k].extend(v)

    X_train = np.asarray([np.moveaxis(array.reshape(3, 32, 32), 0, -1) for array in train[b'data']])
    Y_train = np.asarray(train[b'labels'])

    validation = unpickle('cifar-10-batches-py/data_batch_5')
    X_val = np.asarray([np.moveaxis(array.reshape(3, 32, 32), 0, -1) for array in validation[b'data']])
    Y_val = np.asarray(validation[b'labels'])

    test = unpickle('cifar-10-batches-py/test_batch')
    X_test = np.asarray([np.moveaxis(array.reshape(3, 32, 32), 0, -1) for array in test[b'data']])
    Y_test = np.asarray(test[b'labels'])

    labels = unpickle('cifar-10-batches-py/batches.meta')
    label_names = [label.decode('utf-8') for label in labels[b'label_names']]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, label_names


def hog_features(image: np.ndarray):
    """
    Args:
        image: Image to be processed to histogram of oriented gradients.

    Returns:
        Numpy array with processed features
    """
    g_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    g_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(g_x, g_y)
    bin_n = 16  #Number of bins
    bins = np.int32(bin_n*angle/(2*np.pi))

    size_x = size_y = 8
    cell_y_num = int(image.shape[0]/size_y)
    cell_x_num = int(image.shape[1]/size_x)
    bin_cells = [bins[cell_y*size_y:(cell_y+1)*size_y, cell_x*size_x : (cell_x+1)*size_x] 
                 for cell_y in range(cell_y_num) for cell_x in range(cell_x_num)]
    mag_cells = [magnitude[cell_y*size_y:(cell_y + 1) * size_y, cell_x*size_x:(cell_x+1)*size_x]
                 for cell_y in range(cell_y_num) for cell_x in range(cell_x_num)]

    hists = [np.bincount(bin_cell.ravel(), mag_cell.ravel(), bin_n) for bin_cell, mag_cell in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= np.linalg.norm(hist) + eps
    return hist

def resize_images(data: Union[np.ndarray, list], target_shape: tuple, preserve_range: bool = True):
    """
    Function to resize images, in case of the cifar-10 classification mostly used to fit the default
    input shape of pretrained CNN.
    Args:
        data:
        target_shape:
        preserve_range:

    Returns:
        Resized images

    """
    resized_images = np.array([resize(data[iterator], target_shape, preserve_range=preserve_range)
                               for iterator in range(data.shape[0])]
                             ).astype('float32')
    return resized_images


def data_generator(X: Union[np.ndarray, list], Y: Union[np.ndarray, list], batch_size: int,
                   target_shape: tuple = (224, 224, 3), if_shuffle: bool = False, augment: bool = False,
                   rotation_range: int = 30, horizontal_flip: bool = True):
    """
    Function to generate mini batches of data with possibility of performing data augmentation
    Args:
        X : Images to take the batches from for training data
        Y: Images to take the batches from for test data
        batch_size: Size of the batch used
        target_shape (tuple): Shape of an image to be transformed to
        if_shuffle (bool): Whether to randomly shuffle dataset
        augment (bool): Whether to augment the dataset
        rotation_range (int): Rotation range for an image if it is augmented
        horizontal_flip (bool): Whether to flip image horizontally

    Yields:
        Mini-batch of data
    """
    start = 0
    end = start + batch_size
    num_of_batches = X.shape[0]
    if if_shuffle:
        X, Y = shuffle(X, Y)
    if augment:
        data_augmenter = ImageDataGenerator(rotation_range=rotation_range, horizontal_flip=horizontal_flip)
    while True:
        X_batch = X[start:end]
        Y_batch = Y[start:end]
        X_batch_resized = resize_images(X_batch, target_shape)
        if augment:
            X_batch_resized = np.array([data_augmenter.random_transform(image, seed=42) for
                                       image in X_batch_resized])
        X_preprocessed = preprocess_input(X_batch_resized)
        start += batch_size
        end += batch_size
        if start >= num_of_batches:
            start = 0
            end = batch_size
            if if_shuffle:
                X, Y = shuffle(X, Y)
        yield (X_preprocessed, Y_batch)


class Timer:
    """
    Class providing time execution of our scripts with simple "with" statement
    """
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        print(f'Time passed: {time.time() - self.start:.2f} seconds. \n')


def timeit(f):
    """Decorator to report on training time when defining a function"""
    @wraps(f)
    def wrapper(*args, **kwds):
        start_time = time.time()
        result = f(*args, **kwds)
        elapsed_time = time.time() - start_time
        print('Elapsed computation time: {:.3f} secs'
              .format(elapsed_time))
        return result

    return wrapper
