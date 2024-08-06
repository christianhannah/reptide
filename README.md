# reptide

## Overview

This software is designed to compute the expected TDE rate for a galaxy based on 2-body relaxation under the assumption of spherical symmetry. The primary inputs to this code are either a 1-D surface brightness profile or a 1-D stellar density profile and the central black hole mass. The code returns the TDE rate and other relevant quantities, such as the distribution function, angular momentum diffusion coefficients, etc.

## Features

- Convert surface brightness data to 3-D stellar density via MGE fitting.
- Works with discrete surface brightness and density profiles. 

## Installation

To install Reptide, follow these steps:

### Prerequisites

Make sure you have the following packages installed:
- numpy, scipy, astropy, tqdm, mgefit

This can be accomplished with the requirements.txt file, which gives specific version information.

### Using pip

pip install git+https://github.com/christianhannah/reptide.git

## Examples

Detailed example usage scripts can be found in the Examples folder.
