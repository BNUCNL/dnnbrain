import os
import glob

from os.path import join as pjoin
from setuptools import setup, find_packages

with open('README.md', 'r') as rf:
    long_description = rf.read()

# the 'install_requires' parameter can't deal with non-PYPI packages!
setup(
    name='dnnbrain',
    version='1.0a',
    author='Xiayu Chen, Ming Zhou, Zhengxin Gong, Wei Xu, Xingyu Liu, Taicheng Huang, Zonglei Zhen, Jia Liu',
    author_email='sunshine_drizzle@foxmail.com',
    description='Exploring representations in both DNN and brain.',
    long_description=long_description,
    url='https://github.com/BNUCNL/dnnbrain',
    # install_requires=open('requirements.txt').read().splitlines(),
    packages=find_packages(),
    scripts=[i for i in glob.iglob(pjoin('bin', '*')) if os.path.isfile(i)],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.7'
)
