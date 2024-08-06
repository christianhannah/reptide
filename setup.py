#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:17:09 2024

@author: christian
"""

from setuptools import setup, find_packages

setup(
    name='reptide',                  # Replace with your project's name
    version='0.1.0',                    # Replace with your project's version
    packages=['reptide'],           # Automatically find packages in the project
    install_requires=[                  # List your project's dependencies here
        'numpy',
        'scipy',
        'astropy',
        'tqdm',
        'mgefit'
    ],
    url='https://github.com/christianhannah/reptide.git',  # Replace with your GitHub URL
    author='Christian H. Hannah',                 # Replace with your name
    author_email='hannah.christian@utah.edu',  # Replace with your email
    description='This software is designed to compute the expected TDE rate for a galaxy based on 2-body relaxation under the assumption of spherical symmetry.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11.7',  # Specify your Python version requirements
    )
