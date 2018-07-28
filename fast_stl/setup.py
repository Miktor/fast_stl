from distutils.core import setup, Extension, DEBUG

sfc_module = Extension('fast_loess', sources = ['fast_loess.c'])

setup(name = 'fast_loess', version = '1.0',
    description = 'Python Package with cpp_loess C++ extension',
    ext_modules = [sfc_module]
    )