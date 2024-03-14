""" from https://github.com/jaywalnut310/glow-tts """

from __future__ import absolute_import
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'monotonic_align',
    ext_modules = cythonize("core.pyx"),
    include_dirs=[numpy.get_include()]
)
