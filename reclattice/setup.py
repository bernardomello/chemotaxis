#Install command: pip3 install chetax/. --upgrade
#python3 setup.py build_ext --inplace
from distutils.core import setup, Extension

setup(name = 'reclattice',
      version = '1.0',
      description = 'Monte Carlo simulation of chemotaxis receptors',
      py_modules = ['reclattice'],
      ext_modules = [Extension('montecarlo',
            sources = ['montecarlo_module.c'], extra_compile_args = ["-Ofast"])],
      author='Bernardo Mello',
      author_email='bernardomello@unb.br')
