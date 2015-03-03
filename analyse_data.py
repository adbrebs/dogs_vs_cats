import os
import numpy
import scipy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import tables
import time
from scipy import misc
from pylearn2.utils.data_specs import is_flat_specs
from pylearn2.utils.iteration import SubsetIterator, resolve_iterator_class
from pylearn2.space import VectorSpace, IndexSpace, Conv2DSpace, CompositeSpace
from pylearn2.utils import safe_izip, wraps
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils.rng import make_np_rng
from pylearn2.datasets import cache
import theano

path=os.path.join(
    '${PYLEARN2_DATA_PATH}', 'dogs_vs_cats', 'train.h5')
data_node='Data'

# Locally cache the files before reading them
path = preprocess(path)
datasetCache = cache.datasetCache
path = datasetCache.cache_file(path)

h5file = tables.openFile(path, mode="r")
node = h5file.getNode('/', data_node)

x = getattr(node, 'X')
s = getattr(node, 's')
y = getattr(node, 'y')

s_np = s[:]

s_0 = s_np[:,1]
mu = s_0.mean()
sigma = s_0.std()

weights = numpy.ones_like(s_0)/float(len(s_0))
n, bins, patches = plt.hist(s_0, 20, weights=weights, facecolor='green', alpha=0.75)

plt.xlabel('Widths')
plt.ylabel('Probability')
plt.title("Distribution of the widths, mu={}, sigma={}".format(mu, sigma))
plt.xlim([0, 600])
plt.grid(True)

plt.savefig("distribution_widths.jpeg")
plt.show()