#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 12:54:17 2024

@author: christian
"""

import _reptide_ as REP
import numpy as np
from tqdm import tqdm
from astropy.table import Table, vstack
import multiprocessing as mp
import time
import istarmap


###############################################################################
###############################################################################
########################## USER DEFINED SECTION ###############################
###############################################################################
###############################################################################

# NOTE: All values used for the input arrays must be in SI units when applicable


# output filename (specify entire path)
out_fn = 'reptide_batch_output.fits'


# specify the input type and other setup parameters
analytic = False
cpu_count = mp.cpu_count() # Default to the max number of cores available.
n_energies = 1000
EHS = False


# read in example data file used to define input arrays
example_data_filename = '../Data/example_data.fits'
gal_data = Table.read(example_data_filename, unit_parse_strict='silent') 



# =================== DISCRETE INPUT PARAMETER ARRAYS ===========================

# define the necessary input arrays
name = gal_data['name']
M_BH = gal_data['mbh']
rads = 10**gal_data['lograd']
dens = 10**gal_data['logdens']

# convert everything to SI units for modeling
rads_SI = rads*REP.PC_TO_M # m
dens_SI = dens*(REP.M_SOL/REP.PC_TO_M**3) # kg/m^3
MBH_SI = M_BH*REP.M_SOL # kg


# define the fixed inner slope flags and values
sflag = np.ones(len(name)).astype(bool)
s = np.abs(gal_data['slope']) # slope must be positive (negatives are applied within reptide) 


# Optional Parameters

# define if you would like to apply a Bachall-Wolf cusp to each galaxy
# Below applys one if the slope is steeper than a standard Bachall-Wolf cusp
# Note that slopes >= 2.25 extrapolated all the way to the MBH cause a divergent TDE rate 
bw_cusp = np.zeros(len(gal_data)).astype(bool)
for i in range(len(s)):
    if np.abs(s[i]) >= 1.75:
        bw_cusp[i] = True
    else:
        bw_cusp[i] = False

# define the radial extent of the Bachall-Wolf cusps
bw_rad = rads_SI[:,0] # [m]


# =============================================================================


# ================== ANALYTIC INPUT PARAMETER ARRAYS ==========================

# define the necessary input arrays
name = gal_data['name']
M_BH = gal_data['mbh']*REP.M_SOL # kg
slope = np.abs(gal_data['slope']) # slope must be positive (negatives are applied within reptide) 
rho_5pc = gal_data['dens_at_5pc']*REP.M_SOL/REP.PC_TO_M**3 # kg/m^3
decay_starts = np.ones(len(name))*50*REP.PC_TO_M
decay_widths = np.ones(len(name))*100*REP.PC_TO_M


# Optional Parameters

# define if you would like to apply a Bachall-Wolf cusp to each galaxy
# Below applys one if the slope is steeper than a standard Bachall-Wolf cusp
# Note that slopes >= 2.25 extrapolated all the way to the MBH cause a divergent TDE rate 
bw_cusp = np.zeros(len(gal_data)).astype(bool)
for i in range(len(s)):
    if np.abs(s[i]) >= 1.75:
        bw_cusp[i] = True
    else:
        bw_cusp[i] = False

# define the radial extent of the Bachall-Wolf cusps
bw_rad = rads_SI[:,0] # [m]



# =============================================================================


###############################################################################
###############################################################################
######################## END USER DEFINED SECTION #############################
###############################################################################
###############################################################################






def batch_TDE_rate(input_data,analytic,cpu_count,n_energies,EHS=False):

    input_data = input_data
    
    print()
    print('Beginning TDE rate computation...')
    print()
    start = time.time()
    
    if analytic:
        if len(input_data) == 1:
            output_table = REP.get_TDE_rate_analytic(input_data[0][0],input_data[0][1],input_data[0][2],
                                                 input_data[0][3],input_data[0][4],input_data[0][5],
                                                 input_data[0][6],input_data[0][7],input_data[0][8],
                                                 input_data[0][9],input_data[0][10],input_data[0][11],
                                                 n_energies, EHS)
        else:   
            inputs = []
            for i in range(len(input_data)):
                inputs.append((input_data[i][0],input_data[i][1],input_data[i][2],
                               input_data[i][3],input_data[i][4],input_data[i][5],
                               input_data[i][6],input_data[i][7],input_data[i][8],
                               input_data[i][9],input_data[i][10],input_data[0][11],
                               n_energies, EHS))

            print('Batch Info: ')
            print('\t # of cores: {}'.format(mp.cpu_count()))
            print('\t # of runs: {}'.format(len(input_data)))
            print()
            with mp.Pool(cpu_count) as p:
                results = list(tqdm(p.istarmap(REP.get_TDE_rate_analytic, inputs), 
                                    total=len(inputs)))
                p.close()
                p.join()
            
            
            output_table = Table()
            for r in results:
                output_table = vstack([output_table,r])

        
    else:
        if len(input_data) == 1:
            #pdb.set_trace()
            
            output_table = REP.get_TDE_rate_discrete(input_data[0][0],input_data[0][1],input_data[0][2],
                                                 input_data[0][3],input_data[0][4],input_data[0][5],
                                                 input_data[0][6],input_data[0][7],input_data[0][8],
                                                 input_data[0][9],input_data[0][9],n_energies,EHS)
        else:   
            inputs = []
            for i in range(len(input_data)):
                inputs.append((input_data[i][0],input_data[i][1],input_data[i][2],
                               input_data[i][3],input_data[i][4],input_data[i][5],
                               input_data[i][6],input_data[i][7],input_data[i][8],
                               input_data[i][9],input_data[i][10],n_energies,EHS))

            print('Batch Info: ')
            print('\t # of cores: {}'.format(cpu_count))
            print('\t # of runs: {}'.format(len(input_data)))
            print()
            with mp.Pool(cpu_count) as p:
                results = list(tqdm(p.istarmap(REP.get_TDE_rate_discrete, inputs), 
                                    total=len(inputs)))
                p.close()
                p.join()
            
            
            output_table = Table()
            for r in results:
                output_table = vstack([output_table,r])

    
    end = time.time()
    print()    
    print('Done.')
    print()
    print('Runtime: {:.2f} minutes / {:.2f} hours'.format(round(end-start,3)/60,
                                                          round(end-start,3)/3600))
    print()

    return output_table



def main(in_table, out_fn, analytic, cpu_count, n_energies, EHS=False):
    
    out_table = batch_TDE_rate(in_table, analytic, cpu_count, n_energies, EHS)
    
    out_table.write(out_fn, format='fits', overwrite=True)
    
    print('Output saved as: '+out_fn)
    
    return out_table



if __name__ == '__main__':
    
    
    # ============== CREATE THE INPUT .FITS FILE FOR REPTIDE ======================

    if not analytic:
        in_table = REP.create_discrete_input_table(name, rads_SI, dens_SI, MBH_SI, sflag, s, 
                                               bw_cusp, bw_rad)
    else:
        in_table = REP.create_analytic_input_table(name, slope, rho_5pc, MBH_SI, decay_starts, decay_widths,
                                               bw_cusp, bw_rad)

    # =============================================================================
    
    main(in_table,out_fn,analytic,cpu_count,n_energies,EHS)
    
    