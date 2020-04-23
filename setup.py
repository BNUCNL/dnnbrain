import os
import glob

from os.path import join as pjoin
from setuptools import setup, find_packages

# the 'install_requires' parameter can't deal with non-PYPI packages!
setup(
    name='DNNBrain',
    version='1.0a',
    url='https://github.com/BNUCNL/dnnbrain',
    description='Exploring representations in both DNN and brain.',
    # install_requires=open('requirements.txt').read().splitlines(),
    packages=find_packages(),
    scripts=[i for i in glob.iglob(pjoin('bin', '*')) if os.path.isfile(i)]
)
