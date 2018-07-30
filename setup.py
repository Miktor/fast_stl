from __future__ import print_function

import sys
import textwrap
import pkg_resources

from setuptools import setup, Extension


def is_installed(requirement):
    try:
        pkg_resources.require(requirement)
    except pkg_resources.ResolutionError:
        return False
    else:
        return True


if not is_installed('numpy>=1.13.0'):
    print(textwrap.dedent("""
            Error: numpy needs to be installed first. You can install it via:

            $ pip install numpy
            """), file=sys.stderr)
    exit(1)


def ext_modules():
    import numpy as np

    fast_stl_impl = Extension('fast_stl_impl',
                              sources=['fast_stl/src/fast_stl_impl.cpp'],
                              language='C++',
                              include_dirs=[np.get_include()],
                              library_dirs=[],
                              extra_compile_args=[],
                              )

    return [fast_stl_impl]


setup(
    name="fast_stl",
    platforms=["win-amd64", 'win32'],
    author='Boris Shishov',
    author_email='borisshishov@gmail.com',
    license="BSD",
    url='https://github.com/Miktor/fast_stl',
    packages=['fast_stl'],
    package_data={
        'fast_stl': ['LICENSE'],
    },
    version='0.1',
    description='Time series decomposition written in C++',
    ext_package='fast_stl',
    ext_modules=ext_modules(),
    requires=['numpy'],
)
