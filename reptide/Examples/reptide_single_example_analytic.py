#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:42:55 2024

Example script highlighting the use of reptide.

@author: christian
"""

import _reptide_ as rep
import numpy as np


# =============================================================================
# =================== DEFINE INPUT PARAMETER ARRAYS ===========================
# =============================================================================

# NOTE: All values used for the input arrays must be in a list

name = np.array(['Example Galaxy'])
slope = np.array([1.77])
rho_5pc =  np.array([3000])*rep.M_SOL/rep.PC_TO_M**3 # in kg/m^3
    
# if the slope is too steep, set the bw_cusp parameter to true 
if np.abs(slope) >= 2.25:
    bw_cusp = np.array([True])
else:
    bw_cusp = np.array([False])

bw_rad = np.array([5.53285169288588e+16]) # 1.79 pc in m (median of just BW sample)

# define the black hole mass
MBH_SI = np.array([8e7])*rep.M_SOL # in M_sol

# define the start and width of the exponential decay
decay_start = np.array([50])*rep.PC_TO_M
decay_width = np.array([100])*rep.PC_TO_M

# =============================================================================
# =============================================================================
# =============================================================================


# =============================================================================
# ============== CREATE THE INPUT .FITS FILE FOR REPTIDE ======================
# =============================================================================

# Create the input fits files using the built in functions
in_table = rep.create_analytic_input_table(name, slope, rho_5pc, MBH_SI, 
                                           decay_start, decay_width, bw_cusp, 
                                           bw_rad, no_print=False)

# =============================================================================
# =============================================================================
# =============================================================================


# =============================================================================
# ========================= CALL REPTIDE ======================================
# =============================================================================

output_table = rep.run_reptide(in_table, analytic=True)

# =============================================================================
# =============================================================================
# =============================================================================
