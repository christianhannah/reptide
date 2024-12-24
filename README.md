![alt text](https://github.com/christianhannah/reptide/blob/main/reptide/Images/REPTIDE_LOGO_FINAL_white.png?raw=true)

## Overview

This software is designed to compute the expected TDE rate for a galaxy based on 2-body relaxation under the assumption of spherical symmetry. The primary inputs to this code are either a 1-D surface brightness profile or a 1-D stellar density profile and the central black hole mass. The code returns the TDE rate and other relevant quantities, such as the distribution function, angular momentum diffusion coefficients, etc. See the REPTiDE_Manual.pdf file for additional details on usage as well as input/output parameters.

## Features

- Convert surface brightness data to 3-D stellar density via MGE fitting.
- Works with discrete surface brightness and density profiles. 

## Installation

pip install git+https://github.com/christianhannah/reptide.git

## Examples

Detailed example usage scripts can be found in the Examples folder.
