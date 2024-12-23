#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 23:51:00 2024

@author: christian
"""

import _reptide_ as rep
import numpy as np


# To use a 1D SB profile as input, you must first perform an MGE fit to 
# convert to a 3D stellar density profile.

radii = np.linspace(0.05,2,100) # in arcsec
SBs = np.linspace(100,1,100) # in solar luminosties/pc^2
M_L = 1 # mass-to-light ratio for the system
distance = 1e6 # distance in pc
ax_ratio = 1 # observed axial ratio
inc = 90 # inclination angle

# This returns the radii, densities, individual gaussians used to fit the SB, and
# the summed gaussian fit to the SB.
rads, dens, gausses, summed_gauss = rep.SB_to_Density(radii,SBs,M_L,distance,ax_ratio=1,inc=90)

