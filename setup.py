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
    packages=find_packages(),           # Automatically find packages in the project
    install_requires=[                  # List your project's dependencies here
        'numpy>=1.26.4',
        'scipy>=1.13.0<1.15.0',
        'astropy>=5.3.4',
        'tqdm>=4.66.2',
        'mgefit>=5.0.15'
    ],
    url='https://github.com/christianhhannah/reptide',  # Replace with your GitHub URL
    author='Christian H. Hannah',                 # Replace with your name
    author_email='hannah.christian@utah.edu',  # Replace with your email
    description='This software is designed to compute the expected TDE rate for a galaxy based on 2-body relaxation under the assumption of spherical symmetry.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Replace with your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11.7',  # Specify your Python version requirements
    )