
from __future__ import print_function

__authors__ = "Ian Goodfellow, Harm Aarts"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import gc
import numpy as np
import sys
import cPickle

from theano.compat.six.moves import input, xrange
from pylearn2.utils import serial
from theano.printing import _TagGenerator
from pylearn2.utils.string_utils import number_aware_alphabetical_key
from pylearn2.utils import contains_nan, contains_inf
import argparse

channels = {}

def unique_substring(s, other, min_size=1):
    """
    .. todo::

        WRITEME
    """
    size = min(len(s), min_size)
    while size <= len(s):
        for pos in xrange(0,len(s)-size+1):
            rval = s[pos:pos+size]
            fail = False
            for o in other:
                if o.find(rval) != -1:
                    fail = True
                    break
            if not fail:
                return rval
        size += 1
    # no unique substring
    return s

def unique_substrings(l, min_size=1):
    """
    .. todo::

        WRITEME
    """
    return [unique_substring(s, [x for x in l if x is not s], min_size)
            for s in l]

def main():
    """
    .. todo::

        WRITEME
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("model_path", nargs='+')
    parser.add_argument("--yrange", help='The y-range to be used for plotting, e.g.  0:1')
    
    options = parser.parse_args()
    model_path = options.model_path[0]

    model = serial.load(model_path)

    this_model_channels = model.monitor.channels

    cPickle.dump(this_model_channels, open("./" + options.out, "wb"))


if __name__ == "__main__":
    main()