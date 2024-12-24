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

# NOTE: All values used for the input arrays must be in SI units when applicable

name = 'Example Galaxy'
slope = 1.77 # slope must always be positive
rho_5pc =  3000*rep.M_SOL/rep.PC_TO_M**3 # in kg/m^3

# define the black hole mass
MBH_SI = 8e7*rep.M_SOL # in kg    

# define the start and width of the exponential decay
decay_start = 50*rep.PC_TO_M
decay_width = 100*rep.PC_TO_M

# if the slope is too steep, set the bw_cusp parameter to true 
# Note that slopes >= 2.25 extrapolated all the way to the MBH cause a divergent TDE rate 
if slope >= 2.25:
    bw_cusp = True
else:
    bw_cusp = False

bw_rad = 5e+16 # 1.79 pc in m 


# =============================================================================
# =============================================================================
# =============================================================================


# =============================================================================
# ============== CREATE THE INPUT .FITS FILE FOR REPTIDE ======================
# =============================================================================

# Create the input fits files using the built in functions
in_table = rep.create_analytic_input_table(name, slope, rho_5pc, MBH_SI, 
                                           decay_start, decay_width, bw_cusp, 
                                           bw_rad, quiet=False)

# =============================================================================
# =============================================================================
# =============================================================================


# =============================================================================
# ========================= CALL REPTIDE ======================================
# =============================================================================

output_table = rep.run_reptide(in_table, analytic=True, n_energies=100, EHS=False)

# =============================================================================
# =============================================================================
# =============================================================================
