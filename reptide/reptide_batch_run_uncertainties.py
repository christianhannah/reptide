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


cpu_count = mp.cpu_count() # Default to the max number of cores available.
n_energies = 1000


# specify the input type
analytic = False


# =================== DISCRETE INPUT PARAMETER ARRAYS ===========================

model_gal_filename = './master_sample.fits'
gal_data = Table.read(model_gal_filename, unit_parse_strict='silent') 

select = np.ones(len(gal_data)).astype(bool)
name = gal_data['name'][select]
slope = np.abs(gal_data['slope'][select])
rho5 = gal_data['dens_at_5pc'][select] * (REP.M_SOL/REP.PC_TO_M**3)
M_BH = gal_data['mbh'][select]
rads = 10**gal_data['lograd'][select]
dens = 10**gal_data['logdens'][select]

bw_cusp = np.zeros(len(gal_data)).astype(bool)
for i in range(len(slope)):
    if np.abs(slope[i]) >= 2.25:
        bw_cusp[i] = True
    else:
        bw_cusp[i] = False

# convert everything to SI units for modeling
rads_SI = rads*REP.PC_TO_M # m
dens_SI = dens*(REP.M_SOL/REP.PC_TO_M**3) # kg/m^3
MBH_SI = M_BH*REP.M_SOL # kg

# define the fixed inner slope flags and values
sflag = np.ones(len(name)).astype(bool)
s = -slope


def get_t_relax(r, bh_mass, rho5pc, gamma):
    r5pc = 5*REP.PC_TO_M
    kappa = 0.34
    masses = np.linspace(0.08,1,50)*REP.M_SOL
    avg_M_sq = REP.get_avg_M_sq(masses)
    avg_M = REP.get_avg_M(masses)
    sigma_cubed = np.abs((REP.G*(bh_mass+(4*np.pi*rho5pc*r5pc**gamma)/(3-gamma)*r**(3-gamma)))/(r*(1+gamma)))**(3/2)
    t_relax = kappa*sigma_cubed*avg_M/(REP.G**2*rho5pc*(r/r5pc)**-gamma*avg_M_sq*np.log(0.4*bh_mass/avg_M)) 
    #pdb.set_trace()
    return t_relax/REP.SEC_PER_YR

def r_relax_func(r, bh_mass, rho5pc, gamma, t_age):
    r5pc = 5*REP.PC_TO_M
    kappa = 0.34
    masses = np.linspace(0.08,1,50)*REP.M_SOL
    avg_M_sq = REP.get_avg_M_sq(masses)
    avg_M = REP.get_avg_M(masses)
    sigma_cubed = np.abs((REP.G*(bh_mass+(4*np.pi*rho5pc*r5pc**gamma)/(3-gamma)*r**(3-gamma)))/(r*(1+gamma)))**(3/2)
    t_relax = kappa*sigma_cubed*avg_M/(REP.G**2*rho5pc*(r/r5pc)**-gamma*avg_M_sq*np.log(0.4*bh_mass/avg_M)) 
    return t_relax - t_age

from scipy.optimize import fsolve, root
def get_r_relax(t, r0, bh_mass, rho5pc, gamma):
    #rooty = fsolve(r_relax_func, r0, args=(bh_mass, rho5pc, gamma, t))[0]
    rooty = root(r_relax_func, r0, method='lm', args=(bh_mass, rho5pc, gamma, t)).x[0]
# =============================================================================
#     if rooty == r0:
#         pdb.set_trace()
# =============================================================================
    return rooty


bw_rad = np.zeros(len(name))
t_r_reslim = np.zeros(len(name))
rl_inds = []
ht_inds = []
bw_rl = np.zeros(len(name))
bw_ht = np.zeros(len(name))
res_lims = np.zeros(len(name))
st_inds = []
for i in range(len(name)):
    t_r_reslim[i] = get_t_relax(rads_SI[i,0], MBH_SI[i], rho5[i], slope[i])
    bw_rad[i] = rads_SI[i,0]
    res_lims[i] = rads_SI[i,0]
    if slope[i] >= 1.75 :
        st_inds.append(i)
        bw_rl[i] = rads_SI[i,0]
        bw_ht[i] = get_r_relax(13e9*REP.SEC_PER_YR, rads_SI[i,0]-0*REP.PC_TO_M, MBH_SI[i], rho5[i], slope[i])
        if t_r_reslim[i] <= 13e9:
            rl_inds.append(i)
        else:
            bw_rad[i] = get_r_relax(13e9*REP.SEC_PER_YR, rads_SI[i,0], MBH_SI[i], rho5[i], slope[i])
            #print(rads_SI[i,0]/REP.PC_TO_M - bw_rad[i]/REP.PC_TO_M)
            ht_inds.append(i)


import matplotlib.pyplot as plt

plt.figure(dpi=600)
plt.scatter(bw_rl[rl_inds]/REP.PC_TO_M, bw_ht[rl_inds]/REP.PC_TO_M)
plt.scatter(bw_rl[ht_inds]/REP.PC_TO_M, bw_ht[ht_inds]/REP.PC_TO_M)
plt.plot([0,5],[0,5],'k--')
plt.xlabel('Res Limit')
plt.ylabel('BW Radius over a Hubble Time')
plt.show()


plt.figure(dpi=600)
plt.scatter(bw_rad/REP.PC_TO_M,res_lims/REP.PC_TO_M)
plt.scatter(bw_rad[ht_inds]/REP.PC_TO_M,res_lims[ht_inds]/REP.PC_TO_M,c='r',label='t$_{relax}$(res limit) > Hubble Time')
plt.plot([0,5],[0,5],'k--')
#plt.colorbar()
plt.legend()
plt.xlabel('Res Limit')
plt.ylabel('BW Radius over a Hubble Time')
plt.show()




# =============================================================================

mbh_table = Table.read('MBH_and_SIGMA_DATA_w_errors.fits', unit_parse_strict='silent')
MBH_low = mbh_table['mbh_err_l']
MBH_high = mbh_table['mbh_err_h']

# =============================================================================
# HERE WE NEED TO DEFINE THE RUNS TO GAUGE UNCERTAINTIES

# let's deine 3 sets of new arrays for the input parameters with 3 entries 
# for each galaxy in each.

names_3 = np.zeros((len(name)*3)).astype(str)
slope_3 = np.zeros((len(name)*3)).astype(float)
M_BH_3 = np.zeros((len(name)*3)).astype(float)
rads_3 = np.zeros((len(name)*3,len(rads[0,:]))).astype(float)
dens_3 = np.zeros((len(name)*3,len(rads[0,:]))).astype(float)
bw_cusp_3 = np.zeros((len(name)*3)).astype(bool)
bw_rad_3 = np.zeros((len(name)*3)).astype(bool)
sflag_3 = np.ones((len(name)*3)).astype(bool)
s_3 = np.zeros((len(name)*3)).astype(float)

gal_count = 0
for i in range(0, len(name)*3, 3):
    names_3[i] = name[gal_count]
    names_3[i+1] = name[gal_count]
    names_3[i+2] = name[gal_count]

    slope_3[i] = slope[gal_count]
    slope_3[i+1] = slope[gal_count]
    slope_3[i+2] = slope[gal_count]
    
    rads_3[i,:] = rads_SI[gal_count]
    rads_3[i+1,:] = rads_SI[gal_count]
    rads_3[i+2,:] = rads_SI[gal_count]
    
    dens_3[i,:] = dens_SI[gal_count]
    dens_3[i+1,:] = dens_SI[gal_count]
    dens_3[i+2,:] = dens_SI[gal_count]
    
    bw_cusp_3[i] = bw_cusp[gal_count]
    bw_cusp_3[i+1] = bw_cusp[gal_count]
    bw_cusp_3[i+2] = bw_cusp[gal_count]
    
    bw_rad_3[i] = bw_rad[gal_count]
    bw_rad_3[i+1] = bw_rad[gal_count]
    bw_rad_3[i+2] = bw_rad[gal_count]
    
    sflag_3[i] = sflag[gal_count]
    sflag_3[i+1] = sflag[gal_count]
    sflag_3[i+2] = sflag[gal_count]
    
    s_3[i] = -slope[gal_count]
    s_3[i+1] = -slope[gal_count]
    s_3[i+2] = -slope[gal_count]
    
    M_BH_3[i] = MBH_SI[gal_count]
    M_BH_3[i+1] = MBH_low[gal_count]*REP.M_SOL
    M_BH_3[i+2] = MBH_high[gal_count]*REP.M_SOL

    gal_count += 1
    
# ============== CREATE THE INPUT .FITS FILE FOR REPTIDE ======================

in_table_mbh_uncert = REP.create_discrete_input_table(names_3, rads_3, dens_3, M_BH_3, sflag_3, s_3, 
                                       bw_cusp_3, bw_rad_3)  

# =============================================================================

#%%



names_3 = np.zeros((len(name)*3)).astype(str)
slope_3 = np.zeros((len(name)*3)).astype(float)
M_BH_3 = np.zeros((len(name)*3)).astype(float)
rads_3 = np.zeros((len(name)*3,len(rads[0,:]))).astype(float)
dens_3 = np.zeros((len(name)*3,len(rads[0,:]))).astype(float)
bw_cusp_3 = np.zeros((len(name)*3)).astype(bool)
bw_rad_3 = np.zeros((len(name)*3)).astype(bool)
sflag_3 = np.ones((len(name)*3)).astype(bool)
s_3 = np.zeros((len(name)*3)).astype(float)

gal_count = 0
for i in range(0, len(name)*3, 3):
    names_3[i] = name[gal_count]
    names_3[i+1] = name[gal_count]
    names_3[i+2] = name[gal_count]

    slope_3[i] = slope[gal_count]
    slope_3[i+1] = slope[gal_count]-0.2
    slope_3[i+2] = slope[gal_count]+0.2
    
    rads_3[i,:] = rads_SI[gal_count]
    rads_3[i+1,:] = rads_SI[gal_count]
    rads_3[i+2,:] = rads_SI[gal_count]
    
    dens_3[i,:] = dens_SI[gal_count]
    dens_3[i+1,:] = dens_SI[gal_count]
    dens_3[i+2,:] = dens_SI[gal_count]
    
    bw_cusp_3[i] = bw_cusp[gal_count]
    if slope_3[i+1] >= 1.75:
        bw_cusp_3[i+1] = True
    else:
        bw_cusp_3[i+1] = False
    if slope_3[i+2] >= 1.75:
        bw_cusp_3[i+2] = True
    else:
        bw_cusp_3[i+2] = False
        
    bw_rad_3[i] = bw_rad[gal_count]
    bw_rad_3[i+1] = bw_rad[gal_count]
    bw_rad_3[i+2] = bw_rad[gal_count]
    
    sflag_3[i] = sflag[gal_count]
    sflag_3[i+1] = sflag[gal_count]
    sflag_3[i+2] = sflag[gal_count]
    
    s_3[i] = -slope_3[i]
    s_3[i+1] = -slope_3[i+1]
    s_3[i+2] = -slope_3[i+2]
    
    M_BH_3[i] = MBH_SI[gal_count]
    M_BH_3[i+1] = MBH_SI[gal_count]
    M_BH_3[i+2] = MBH_SI[gal_count]

    gal_count += 1

# ============== CREATE THE INPUT .FITS FILE FOR REPTIDE ======================

in_table_slope_uncert = REP.create_discrete_input_table(names_3, rads_3, dens_3, M_BH_3, sflag_3, s_3, 
                                       bw_cusp_3, bw_rad_3)  

# =============================================================================




names_3 = np.zeros((len(name)*3)).astype(str)
slope_3 = np.zeros((len(name)*3)).astype(float)
M_BH_3 = np.zeros((len(name)*3)).astype(float)
rads_3 = np.zeros((len(name)*3,len(rads[0,:]))).astype(float)
dens_3 = np.zeros((len(name)*3,len(rads[0,:]))).astype(float)
bw_cusp_3 = np.zeros((len(name)*3)).astype(bool)
bw_rad_3 = np.zeros((len(name)*3)).astype(bool)
sflag_3 = np.ones((len(name)*3)).astype(bool)
s_3 = np.zeros((len(name)*3)).astype(float)

gal_count = 0
for i in range(0, len(name)*3, 3):
    names_3[i] = name[gal_count]
    names_3[i+1] = name[gal_count]
    names_3[i+2] = name[gal_count]

    slope_3[i] = slope[gal_count]
    slope_3[i+1] = slope[gal_count]
    slope_3[i+2] = slope[gal_count]
    
    rads_3[i,:] = rads_SI[gal_count]
    rads_3[i+1,:] = rads_SI[gal_count]
    rads_3[i+2,:] = rads_SI[gal_count]
    
    dens_3[i,:] = dens_SI[gal_count]
    dens_3[i+1,:] = dens_SI[gal_count]-10**0.11*(REP.M_SOL/REP.PC_TO_M**3)
    dens_3[i+1,dens_3[i+1,:]<0] = 0
    dens_3[i+2,:] = dens_SI[gal_count]+10**0.11*(REP.M_SOL/REP.PC_TO_M**3)
    
    bw_cusp_3[i] = bw_cusp[gal_count]
    bw_cusp_3[i+1] = bw_cusp[gal_count]
    bw_cusp_3[i+2] = bw_cusp[gal_count]
    
    bw_rad_3[i] = bw_rad[gal_count]
    bw_rad_3[i+1] = bw_rad[gal_count]
    bw_rad_3[i+2] = bw_rad[gal_count]
    
    sflag_3[i] = sflag[gal_count]
    sflag_3[i+1] = sflag[gal_count]
    sflag_3[i+2] = sflag[gal_count]
    
    s_3[i] = -slope[gal_count]
    s_3[i+1] = -slope[gal_count]
    s_3[i+2] = -slope[gal_count]
    
    M_BH_3[i] = MBH_SI[gal_count]
    M_BH_3[i+1] = MBH_SI[gal_count]
    M_BH_3[i+2] = MBH_SI[gal_count]

    gal_count += 1

# ============== CREATE THE INPUT .FITS FILE FOR REPTIDE ======================

in_table_dens_uncert = REP.create_discrete_input_table(names_3, rads_3, dens_3, M_BH_3, sflag_3, s_3, 
                                       bw_cusp_3, bw_rad_3)  

# =============================================================================




###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################



def batch_TDE_rate(input_data,analytic,cpu_count,n_energies):

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
                                                 input_data[0][9],input_data[0][10],input_data[0][11],n_energies)
        else:   
            inputs = []
            for i in range(len(input_data)):
                inputs.append((input_data[i][0],input_data[i][1],input_data[i][2],
                               input_data[i][3],input_data[i][4],input_data[i][5],
                               input_data[i][6],input_data[i][7],input_data[i][8],
                               input_data[i][9],input_data[i][10],input_data[0][11],n_energies))

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
                                                 input_data[0][9],n_energies)
        else:   
            inputs = []
            for i in range(len(input_data)):
                inputs.append((input_data[i][0],input_data[i][1],input_data[i][2],
                               input_data[i][3],input_data[i][4],input_data[i][5],
                               input_data[i][6],input_data[i][7],input_data[i][8],
                               input_data[i][9],n_energies))

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



def main(in_table, out_fn, analytic, cpu_count, n_energies):
    
    out_table = batch_TDE_rate(in_table, analytic, cpu_count, n_energies)
    
    out_table.write(out_fn, format='fits', overwrite=True)
    
    print('Output saved as: '+out_fn)
    
    return out_table



if __name__ == '__main__':
    
    main(in_table_mbh_uncert,'reptide_output_mbh_uncert_1000.fits',analytic,cpu_count,n_energies)
    
    main(in_table_slope_uncert,'reptide_output_slope_uncert_1000.fits',analytic,cpu_count,n_energies)
    
    main(in_table_dens_uncert,'reptide_output_dens_uncert_1000.fits',analytic,cpu_count,n_energies)
    
    
    