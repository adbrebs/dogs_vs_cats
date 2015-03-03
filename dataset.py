__author__ = 'adeb'

import os
import numpy
import scipy
import matplotlib.pyplot as plt
import tables
import time

from scipy import misc
from pylearn2.utils.string_utils import preprocess
from pylearn2.datasets import cache
import theano


class Dataset():
    _default_seed = 2015 + 1 + 17

    def __init__(self, start, stop, data_augmentator):
        self.start = start
        self.stop = stop
        self.n_data = stop - start
        self.data_augmentator = data_augmentator

        path=os.path.join(
            '${PYLEARN2_DATA_PATH}', 'dogs_vs_cats', 'train.h5')
        data_node='Data'

        # Locally cache the files before reading them
        path = preprocess(path)
        dataset_cache = cache.datasetCache
        path = dataset_cache.cache_file(path)

        self.h5file = tables.openFile(path, mode="r")
        node = self.h5file.getNode('/', data_node)

        self.x = getattr(node, 'X')
        self.s = getattr(node, 's')
        self.y = getattr(node, 'y')

        # If you want to display some images
        # a = scipy.ndimage.interpolation.zoom(self.x[0].reshape(self.s[0]), (3, 3, 1), order=0)
        # plt.imshow(a)
        # plt.show()

    def get_iterator(self, batch_size, n_batches=None):
        return DataIterator(self, batch_size, self.data_augmentator, n_batches=n_batches)


class DataIterator(object):
    def __init__(self, dataset, batch_size, data_augmentator, n_batches=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_augmentator = data_augmentator
        if n_batches is None:
            self.n_batches = dataset.n_data / batch_size
        else:
            self.n_batches = n_batches
        self.batch_id = 0
        self.permutation = dataset.start + numpy.random.permutation(dataset.n_data)
        self.shape_batch = [(batch_size,) + sh for sh in
                            data_augmentator.get_final_shape()]
        self.n_chunks = data_augmentator.get_number_chunks()

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        if self.batch_id < self.n_batches:
            pos = self.batch_id * self.batch_size
            self.batch_id += 1
            idx = self.permutation[pos: pos + self.batch_size]
            xs = []
            for chunk in range(self.n_chunks):
                xs.append(numpy.zeros(self.shape_batch[chunk],
                                      dtype=theano.config.floatX))
            y = numpy.zeros((self.batch_size,), dtype='int32')

            for i in xrange(self.batch_size):
                real_idx = idx[i]
                img = numpy.reshape(self.dataset.x[real_idx], self.dataset.s[real_idx])
                new_imgs = self.data_augmentator.process_image(img)
                if not isinstance(new_imgs, list):
                    new_imgs = [new_imgs]
                # plt.imshow(img)
                # plt.show()
                for a, img in enumerate(new_imgs):
                    xs[a][i] = img
                y[i] = self.dataset.y[real_idx]

            for a in range(len(xs)):
                xs[a] = numpy.swapaxes(xs[a], 3, 2)
                xs[a] = numpy.swapaxes(xs[a], 2, 1)
                xs[a] = xs[a]/255.0 - 0.5

            return xs, y
        else:
            raise StopIteration()


class DataAugmentator():
    def __init__(self, child=None):
        self.child = child

    def process_image(self, img):
        # plt.imshow(img)
        # plt.show()
        if self.child is not None:
            img = self.child.process_image(img)
        img = self.process_image_virtual(img)
        # plt.imshow(img)
        # plt.show()
        return img

    def get_final_shape(self):
        if self.child is not None:
            return self.child.get_final_shape()

    def get_number_chunks(self):
        return 1

    def process_image_virtual(self, img):
        raise NotImplementedError


class Randomize(DataAugmentator):
    def __init__(self, d_a, probability=0.5):
        DataAugmentator.__init__(self, child=d_a.child)
        self.d_a = d_a
        self.probability = probability

    def process_image_virtual(self, img):
        if numpy.random.uniform() < self.probability:
            img = self.d_a.process_image_virtual(img)
        return img


class ScaleMinSide(DataAugmentator):
    def __init__(self, min_side, child=None):
        DataAugmentator.__init__(self, child)
        self.min_side = min_side

    def process_image_virtual(self, img):
        min_axis = min(img.shape[0:2])
        ratio = self.min_side / float(min_axis)
        # a = scipy.ndimage.interpolation.zoom(img, (ratio, ratio, 1), order=0)
        a = misc.imresize(img, ratio)
        return a


class ScaleMinSideOnlyIfSmaller(DataAugmentator):
    """
    If the smallest side of the image is greater than min_side, do nothing
    """
    def __init__(self, min_side, child=None):
        DataAugmentator.__init__(self, child)
        self.min_side = min_side

    def process_image_virtual(self, img):
        min_axis = min(img.shape[0:2])
        if min_axis >= self.min_side:
            return img
        ratio = self.min_side / float(min_axis)
        a = misc.imresize(img, ratio)
        return a


class Scale(DataAugmentator):
    def __init__(self, new_scale, child=None):
        DataAugmentator.__init__(self, child)
        self.new_scale = new_scale

    def process_image_virtual(self, img):
        ratio0 = self.new_scale[0] / img.shape[0]
        ratio1 = self.new_scale[1] / img.shape[1]
        a = scipy.ndimage.interpolation.zoom(img, (ratio0, ratio1, 1), order=0)
        return a


class ScaleOnlyIfSmaller(DataAugmentator):
    def __init__(self, new_scale, child=None):
        DataAugmentator.__init__(self, child)
        self.new_scale = new_scale

    def process_image_virtual(self, img):
        ratio0 = max(self.new_scale[0] / img.shape[0], img.shape[0])
        ratio1 = max(self.new_scale[1] / img.shape[1], img.shape[1])
        a = scipy.ndimage.interpolation.zoom(img, (ratio0, ratio1, 1), order=0)
        return a


class Zoom(DataAugmentator):
    def __init__(self, zoom, child=None):
        DataAugmentator.__init__(self, child)
        self.zoom = zoom

    def process_image_virtual(self, img):
        a = misc.imresize(img, self.zoom)
        return a


class RandomCropping(DataAugmentator):
    def __init__(self, crop_size, child=None):
        DataAugmentator.__init__(self, child)
        self.crop_size = crop_size

    def get_final_shape(self):
        return self.crop_size, self.crop_size, 3

    def process_image_virtual(self, img):
        lx, ly, _ = img.shape
        pos_x = numpy.random.randint(0, 1 + lx - self.crop_size)
        pos_y = numpy.random.randint(0, 1 + ly - self.crop_size)
        return img[pos_x:pos_x+self.crop_size, pos_y:pos_y+self.crop_size, :]


class HorizontalFlip(DataAugmentator):
    def __init__(self, child=None):
        DataAugmentator.__init__(self, child)

    def process_image_virtual(self, img):
        new_image = numpy.empty_like(img)
        new_image[:,:,0] = numpy.fliplr(img[:,:,0])
        new_image[:,:,1] = numpy.fliplr(img[:,:,1])
        new_image[:,:,2] = numpy.fliplr(img[:,:,2])
        return new_image


class RandomRotate(DataAugmentator):
    def __init__(self, max_angle, child=None):
        DataAugmentator.__init__(self, child)
        self.max_angle = max_angle

    def process_image_virtual(self, img):
        angle = numpy.random.randint(-self.max_angle, self.max_angle)
        return scipy.ndimage.interpolation.rotate(img, angle, reshape=False, mode="nearest", order=0)


class RotatePreciseAngle(DataAugmentator):
    def __init__(self, angle, child=None):
        DataAugmentator.__init__(self, child)
        self.angle = angle

    def process_image_virtual(self, img):
        return scipy.ndimage.interpolation.rotate(img, self.angle, reshape=False, mode="nearest", order=0)


class ParallelDataAugmentator(DataAugmentator):
    """
    Generate several augmented images from a single one.
    process_image returns a list!
    """
    def __init__(self, data_agumentators, child=None):
        DataAugmentator.__init__(self, child)
        self.data_augmentators = data_agumentators

    def process_image_virtual(self, img):
        new_images = []
        for d_a in self.data_augmentators:
            new_images.append(d_a.process_image(img))
        return new_images

    def get_number_chunks(self):
        return len(self.data_augmentators)

    def get_final_shape(self):
        return [d_a.get_final_shape() for d_a in self.data_augmentators]

