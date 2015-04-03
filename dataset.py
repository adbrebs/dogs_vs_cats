__author__ = 'adeb'

import os
import numpy
import scipy
import matplotlib.pyplot as plt
import tables
import time
import shutil
import math
import multiprocessing
from sys import getsizeof

from scipy import misc
from pylearn2.utils.string_utils import preprocess
from pylearn2.datasets import cache
import theano


class Dataset():
    def __init__(self, start, stop, path=None):
        self.start = start
        self.stop = stop
        self.n_data = self.stop - self.start

        if path is None:
            path=os.path.join(
                '${PYLEARN2_DATA_PATH}', 'dogs_vs_cats', 'train.h5')
        # Locally cache the files before reading them
        path = preprocess(path)
        dataset_cache = cache.datasetCache
        self.path = dataset_cache.cache_file(path)

    def get_iterator(self, data_augmentator, batch_size, n_procs, n_batches=None):
        return DataIterator(self, batch_size, data_augmentator, n_procs, n_batches=n_batches)


class DataIterator(object):
    """
    Manages the batch creation with a data_augmentator scheme
    """
    def __init__(self, dataset, batch_size, data_augmentator, n_procs, n_batches=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_augmentator = data_augmentator
        self.n_procs = n_procs
        if n_batches is None:
            self.n_batches = dataset.n_data / batch_size
        else:
            self.n_batches = n_batches

        self.permutation = numpy.random.permutation(dataset.n_data)

        self.shape_batch = [(batch_size,) + sh for sh in
                            data_augmentator.get_final_shape([None])]

    def get_batch(self, idx_batch):
        """
        Returns a single batch corresponding to idx_batch
        """
        pos = idx_batch * self.batch_size
        indices = self.dataset.start + self.permutation[pos: pos + self.batch_size]
        return indices

    def get_batches(self, idx_batches):
        """
        Returns several batches correponding to idx_batches
        """
        indices = [0]*len(idx_batches)
        for i, idx in enumerate(idx_batches):
            indices[i] = self.get_batch(idx)
        return indices

    def generate_single_batch(self, batch_of_indices, h5_x, h5_s, h5_y):

        # Initialize containers
        x = []
        for shape in self.shape_batch:
            x.append(numpy.zeros(shape, dtype=theano.config.floatX))
        y = numpy.zeros((self.batch_size,), dtype='int32')

        for i, idx in enumerate(batch_of_indices):
            img, s, target = h5_x[idx], h5_s[idx], h5_y[idx]

            img = numpy.reshape(img, s)
            # plt.imshow(img)
            # plt.show()
            new_imgs = self.data_augmentator.process_image([img])
            # plt.imshow(new_imgs[0])
            # plt.show()

            y[i] = target
            for a, img in enumerate(new_imgs):
                x[a][i] = img

        for a in range(len(x)):
            x[a] = self.swap_axis_and_norm(x[a])

        return x + [y], batch_of_indices + self.dataset.start

    def create_batch_indices(self):
        n_batches = self.dataset.n_data / self.batch_size
        batches = range(n_batches)
        n_batches_per_proc = int(math.ceil(float(n_batches) / self.n_procs))
        proc_idx_batches = [batches[i:i + n_batches_per_proc]
                            for i in range(0, (self.n_procs-1)*n_batches_per_proc, n_batches_per_proc)]
        proc_idx_batches.append(batches[(self.n_procs-1)*n_batches_per_proc:])
        return proc_idx_batches, n_batches

    def generate_batches_in_parallel(self, n_times=1, num_cached=30):
        """
        Generates batches in multiple processes and load the batches in a queue.
        All the processes are responsible for the same number of batches.
        n_times is useful if you want to use several branches in you multi-view
        architecture (same data augmentation for each view)
        """
        try:
            queue = multiprocessing.Queue(maxsize=num_cached)

            proc_idx_batches, n_batches = self.create_batch_indices()

            # define producer (putting items into queue)
            def producer(batches_of_indices):

                try:
                    h5file = tables.openFile(self.dataset.path, mode="r")
                    node = h5file.getNode('/', 'Data')
                    h5_x = getattr(node, 'X')
                    h5_s = getattr(node, 's')
                    h5_y = getattr(node, 'y')


                    for batch_of_indices in batches_of_indices:
                        queue.put([self.generate_single_batch(batch_of_indices, h5_x, h5_s, h5_y)
                                   for _ in range(n_times)])
                except:
                    "print close hdf5 files"
                    h5file.close()

            processes = []
            for i in xrange(self.n_procs):
                thread = multiprocessing.Process(
                    target=producer,
                    args=(self.get_batches(proc_idx_batches[i]),))
                time.sleep(0.01)
                thread.start()
                processes.append(thread)

            # run as consumer (read items from queue, in current thread)
            for i in range(n_batches):
                yield queue.get(), queue.qsize()
        except:
            print "SALUT TOI"
            for th in processes:
                th.terminate()
            queue.close()
            raise

    def generate_batches_sequentially(self, n_times=1):
        """
        Generates batches sequentitially (in one process)
        """
        if self.n_procs != 1:
            print "generate batches sequentially only works with a single " \
                  "process. Setting the variable to 1."
            self.n_procs = 1

        h5file = tables.openFile(self.dataset.path, mode="r")
        node = h5file.getNode('/', 'Data')
        h5_x = getattr(node, 'X')
        h5_s = getattr(node, 's')
        h5_y = getattr(node, 'y')

        proc_idx_batches, n_batches = self.create_batch_indices()
        idx_batches = proc_idx_batches[0]

        for idx_batch in idx_batches:
            batch_of_indices = self.get_batch(idx_batch)
            batch = [self.generate_single_batch(
                batch_of_indices, h5_x, h5_s, h5_y) for _ in range(n_times)]
            yield batch, 0

    def swap_axis_and_norm(self, a):
        a = numpy.swapaxes(a, 3, 2)
        a = numpy.swapaxes(a, 2, 1)
        return a/255.0 - 0.5


class DataAugmentator():
    def __init__(self, child=None):
        self.child = child

    def process_image(self, ls_imgs):
        """
        Takes a list of images as argument.
        Returns a list of images (can be a list with a single element.
        """
        # plt.imshow(img)
        # plt.show()
        if self.child is not None:
            ls_imgs = self.child.process_image(ls_imgs)

        ls_new_imgs = []
        for img in ls_imgs:
            ls_new_imgs += self.process_image_virtual(img)
        # plt.imshow(img)
        # plt.show()
        return ls_new_imgs

    def process_image_virtual(self, img):
        """
        Takes a single image (not in a list) as argument.
        Returns a list of images (can be a list with a single element.
        """
        raise NotImplementedError

    def get_final_shape(self, ls_shapes):
        """
        If you know that a child or a parent will return a fixed shape, ls_shapes
        can be set to None.
        """
        if self.child is not None:
            ls_shapes = self.child.get_final_shape(ls_shapes)

        ls_new_shapes = []
        for shape in ls_shapes:
            ls_new_shapes += self.get_final_shape_virtual(shape)

        return ls_new_shapes

    def get_final_shape_virtual(self, shape):
        return [shape]


class RandomizeSingle(DataAugmentator):
    def __init__(self, d_a, probability=0.5, child=None):
        DataAugmentator.__init__(self, child=child)
        self.d_a = d_a
        self.probability = probability

    def process_image_virtual(self, img):
        if numpy.random.uniform() < self.probability:
            ls_imgs = self.d_a.process_image([img])
            return ls_imgs
        else:
            return [img]


class RandomizeMultiple(DataAugmentator):
    """
    assumes all the da have the final shape and same number of chunks
    """
    def __init__(self, ls_d_a, ls_probabilities=None, child=None):
        DataAugmentator.__init__(self, child=child)
        self.ls_d_a = ls_d_a
        self.n_da = len(ls_d_a)
        if ls_probabilities is not None:
            self.ls_probabilities = ls_probabilities
        else:
            self.ls_probabilities = [1.0/self.n_da]*self.n_da

    def process_image_virtual(self, img):
        id_da = numpy.random.choice(self.n_da, p=self.ls_probabilities)
        ls_imgs = self.ls_d_a[id_da].process_image([img])
        return ls_imgs

    def get_final_shape_virtual(self, shape):
        return self.ls_d_a[0].get_final_shape([shape])


class ScaleMinSide(DataAugmentator):
    def __init__(self, min_side, child=None):
        DataAugmentator.__init__(self, child)
        self.min_side = min_side

    def process_image_virtual(self, img):
        min_axis = numpy.min(img.shape[0:2])
        ratio = self.min_side / float(min_axis)
        new_shape = int(round(img.shape[0]*ratio)), int(round(img.shape[1]*ratio))
        # a = scipy.ndimage.interpolation.zoom(img, (ratio, ratio, 1), order=0)
        a = misc.imresize(img, new_shape)
        return [a]

    def get_final_shape_virtual(self, shape):
        if shape is None:
            return [None]
        min_axis = min(shape[0:2])
        ratio = self.min_side / float(min_axis)
        return [(shape[0] * ratio, shape[1] * ratio, shape[2])]


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
        return [a]

    def get_final_shape_virtual(self, shape):
        min_axis = min(shape[0:2])
        if min_axis >= self.min_side:
            return [shape]
        ratio = self.min_side / float(min_axis)
        return [(shape[0] * ratio, shape[1] * ratio, shape[2])]


class Scale(DataAugmentator):
    def __init__(self, new_scale, child=None):
        DataAugmentator.__init__(self, child)
        self.new_scale = new_scale

    def process_image_virtual(self, img):
        ratio0 = self.new_scale[0] / float(img.shape[0])
        ratio1 = self.new_scale[1] / float(img.shape[1])
        a = scipy.ndimage.interpolation.zoom(img, (ratio0, ratio1, 1), order=0)
        return [a]

    def get_final_shape_virtual(self, shape):
        return [self.new_scale]


class ScaleOnlyIfSmaller(DataAugmentator):
    def __init__(self, new_scale, child=None):
        DataAugmentator.__init__(self, child)
        self.new_scale = new_scale

    def process_image_virtual(self, img):
        ratio0 = max(self.new_scale[0] / img.shape[0], img.shape[0])
        ratio1 = max(self.new_scale[1] / img.shape[1], img.shape[1])
        a = scipy.ndimage.interpolation.zoom(img, (ratio0, ratio1, 1), order=0)
        return [a]

    def get_final_shape_virtual(self, shape):
        ratio0 = max(self.new_scale[0] / shape[0], shape[0])
        ratio1 = max(self.new_scale[1] / shape[1], shape[1])
        return [(shape[0] * ratio0, shape[1] * ratio1, shape[2])]


class Zoom(DataAugmentator):
    def __init__(self, zoom, child=None):
        DataAugmentator.__init__(self, child)
        self.zoom = zoom

    def process_image_virtual(self, img):
        a = misc.imresize(img, self.zoom)
        return [a]

    def get_final_shape_virtual(self, shape):
        return [(shape[0] * self.zoom, shape[1] * self.zoom, shape[2])]


class RandomZoom(DataAugmentator):
    def __init__(self, max_zoom, child=None):
        DataAugmentator.__init__(self, child)
        self.max_zoom = max_zoom

    def process_image_virtual(self, img):
        zoom = numpy.random.uniform(1.0, self.max_zoom)
        return [misc.imresize(img, zoom)]

    def get_final_shape_virtual(self, shape):
        return [None]


class RandomCropping(DataAugmentator):
    def __init__(self, crop_size, child=None):
        DataAugmentator.__init__(self, child)
        self.crop_size = crop_size

    def process_image_virtual(self, img):
        lx, ly, _ = img.shape
        pos_x = numpy.random.randint(0, 1 + lx - self.crop_size)
        pos_y = numpy.random.randint(0, 1 + ly - self.crop_size)
        return [img[pos_x:pos_x+self.crop_size, pos_y:pos_y+self.crop_size, :]]

    def get_final_shape_virtual(self, shape):
        return [(self.crop_size, self.crop_size, 3)]


class Cropping5Views(DataAugmentator):
    def __init__(self, crop_size, random_shift, child=None):
        DataAugmentator.__init__(self, child)
        self.crop_size = crop_size
        self.random_shift = random_shift

    def generate_shift(self):
        shift_x = numpy.random.randint(0, 1 + self.random_shift)
        shift_y = numpy.random.randint(0, 1 + self.random_shift)
        return shift_x, shift_y

    def process_image_virtual(self, img):
        lx, ly, _ = img.shape

        # top left corner
        x, y = self.generate_shift()
        tl_img = img[x:x+self.crop_size, -y-self.crop_size-1:-y-1, :]

        # top right corner
        x, y = self.generate_shift()
        tr_img = img[-x-self.crop_size-1:-x-1, -y-self.crop_size-1:-y-1, :]

        # bottom right corner
        x, y = self.generate_shift()
        br_img = img[-x-self.crop_size-1:-x-1, y:y+self.crop_size, :]

        # bottom left corner
        x, y = self.generate_shift()
        bl_img = img[x:x+self.crop_size, y:y+self.crop_size, :]

        # central image
        mid_x, mid_y = lx/2, ly/2
        b = (self.crop_size + self.random_shift)/2
        pos_x, pos_y = mid_x - b, mid_y - b
        x, y = self.generate_shift()
        pos_x += x
        pos_y += y
        c_img = img[pos_x:pos_x+self.crop_size, pos_y:pos_y+self.crop_size, :]

        return [tl_img, tr_img, br_img, bl_img, c_img]

    def get_final_shape_virtual(self, shape):
        return [(self.crop_size, self.crop_size, 3)]*5


class HorizontalFlip(DataAugmentator):
    def __init__(self, child=None):
        DataAugmentator.__init__(self, child)

    def process_image_virtual(self, img):
        new_image = numpy.empty_like(img)
        new_image[:,:,0] = numpy.fliplr(img[:,:,0])
        new_image[:,:,1] = numpy.fliplr(img[:,:,1])
        new_image[:,:,2] = numpy.fliplr(img[:,:,2])
        return [new_image]


class VerticalFlip(DataAugmentator):
    def __init__(self, child=None):
        DataAugmentator.__init__(self, child)

    def process_image_virtual(self, img):
        new_image = numpy.empty_like(img)
        new_image[:,:,0] = numpy.flipud(img[:,:,0])
        new_image[:,:,1] = numpy.flipud(img[:,:,1])
        new_image[:,:,2] = numpy.flipud(img[:,:,2])
        return [new_image]


class RandomRotate(DataAugmentator):
    def __init__(self, max_angle, child=None):
        DataAugmentator.__init__(self, child)
        self.max_angle = max_angle

    def process_image_virtual(self, img):
        angle = numpy.random.randint(-self.max_angle, self.max_angle+1)
        return [scipy.ndimage.interpolation.rotate(img, angle, reshape=False, mode="nearest", order=0)]


class RotatePreciseAngle(DataAugmentator):
    def __init__(self, angle, child=None):
        DataAugmentator.__init__(self, child)
        self.angle = angle

    def process_image_virtual(self, img):
        return [scipy.ndimage.interpolation.rotate(img, self.angle, reshape=False, mode="nearest", order=0)]


class ParallelDataAugmentator(DataAugmentator):
    """
    Generate several augmented images from a single one.
    """
    def __init__(self, data_agumentators, child=None):
        DataAugmentator.__init__(self, child)
        self.data_augmentators = data_agumentators

    def process_image_virtual(self, img):
        ls_new_imgs = []
        for d_a in self.data_augmentators:
            ls_new_imgs += d_a.process_image([img])
        return ls_new_imgs

    def get_final_shape_virtual(self, shape):
        ls_new_shapes = []
        for d_a in self.data_augmentators:
            ls_new_shapes += d_a.get_final_shape([shape])
        return ls_new_shapes

