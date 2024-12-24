#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:24:49 2024

THIS IS REPTiDE...

@author: christian
"""

import numpy as np
from scipy import integrate
from scipy.optimize import curve_fit, fsolve
from astropy.table import Table
from astropy.io import fits
from tqdm import tqdm
import time
from mgefit.mge_fit_1d import mge_fit_1d
import matplotlib.pyplot as plt

np.seterr(invalid='ignore')
np.seterr(divide='ignore')

# CONSTANTS
PC_TO_M = 3.08567758128e16 
M_SOL =  1.989e30 # kg
R_SOL = 696.34e6 # m
G = 6.6743e-11 # m^3 s^-2 kg^-1
ARCSEC_TO_RADIAN = 4.8481e-6
SEC_PER_YR = 3.154e+7

# ============================ FUNCTIONS ======================================

# =============================================================================
# Function to fit an input 1D SB profile with an MGE and return the 3D density 

def gauss(height, sig, r):
    return height*np.exp((-0.5)*r**2/sig**2)

def SB_to_Density(rads,SBs,M_L,distance,ax_ratio=1,inc=90):
    # SB needs to be in L_sol/pc^2, rads in arcsec
    
    num_rad = 100
    rad = np.geomspace(np.min(rads), np.max(rads), num_rad)
    rad_pc = (rad*ARCSEC_TO_RADIAN)*distance # in pc
    SBs = 10**np.interp(np.log10(rad), np.log10(rads), np.log10(SBs))

    mge = mge_fit_1d(rad, SBs, ngauss=20, inner_slope=4, outer_slope=1,plot=False)

    densities = np.zeros_like(rad)

    heights = []
    sigs = []
    tots = []
    # compute 3-d density assuming spherical symmetry (so, as f(r))
    for j in range(len(rad)):
        for i in range(len(mge.sol[0])):
            
            # store total luminosity and width of each gaussian component
            L_tot_i = mge.sol[0][i] # in solar luminosities
            height = L_tot_i/(np.sqrt(2*np.pi)*mge.sol[1][i]) #height of 1-d Gaussian
            sig_i = mge.sol[1][i] # in arcsec
            
            # convert sigma from arcsec to pc using distance
            sig = (sig_i*ARCSEC_TO_RADIAN)*distance # in pc
            L_tot = (height)*2*np.pi*(sig)**2*ax_ratio # total area under 2-d gaussian
            
            if j == 0:
                tots.append(L_tot_i)
                heights.append(height)
                sigs.append(sig_i)
           
            # convert total luminosity to total mass using M/L ratio
            M_tot = L_tot*M_L
            
            q_prime = ax_ratio
            q = np.sqrt((q_prime**2 - (np.cos(inc)**2))/(np.sin(inc)**2))
            if np.isnan(q):
                print('Inclination inconsistent with observed q. Setting intrinsic axial ratio to 1.')
                q = 1
            
            # compute desity contribution from one gaussian component
            # (Spherically averaged by setting the polar angle to pi/6)
            dens_comp = (M_tot/((2*np.pi)**(3/2)*sig**3*q))*np.exp(-(rad_pc[j]**2*(np.sin(np.pi/6)**2+np.cos(np.pi/6)**2/q**2)/(2*sig**2))) # M_SOL/pc^3
            
            # add density from each gaussian to total density
            densities[j] += dens_comp

    gausses = [] 
    for i in range(len(mge.sol[0])):
        gausses.append(gauss(heights[i],sigs[i],rad))

    summed_gauss = np.zeros_like(rad)
    for i in range(len(rad)):
        for j in range(len(mge.sol[0])):
            summed_gauss[i] += gausses[j][i]

    return rad_pc, densities, gausses, summed_gauss

# =============================================================================



# =============================================================================
# function to ignore the divide by zero error in safe_interp
def safe_interp(x, xp, fp):
    np.seterr(divide='ignore')
    result = np.interp(x, xp, fp)
    return result
# =============================================================================


# =============================================================================
# a little function for finding maximum outliers
def get_n_max(arr, n):
    temp_arr = np.copy(arr)
    maxes = np.zeros((n))
    maxes_idx = np.zeros((n)).astype(int)
    for i in range(n):
        maxes[i] = np.max(temp_arr)
        maxes_idx[i] = np.argmax(temp_arr)
        temp_arr[maxes_idx[i]] = -999999
    return maxes, maxes_idx

# =============================================================================

# =============================================================================
# a little function for finding minimum outliers
def get_n_min(arr, n):
    temp_arr = np.copy(arr)
    mins = np.zeros((n))
    mins_idx = np.zeros((n)).astype(int)
    for i in range(n):
        mins[i] = np.min(temp_arr)
        mins_idx[i] = np.argmin(temp_arr)
        temp_arr[mins_idx[i]] = 999999
    return mins, mins_idx

# =============================================================================   



# =============================================================================
# function to return power-law inner density profile
def get_rho_r(r,slope,rho_5pc,decay_start,decay_width,bw_cusp,bw_rad,smooth):
    pc5_in_m = 1.54283879064e+17
    if bw_cusp:
        # 5.53285169288588e+16 # 1.79 pc in m (median of just BW sample)
        r_b = bw_rad
        rho_b = rho_5pc*(r_b/pc5_in_m)**(-slope)
        return np.piecewise(r, [r>=decay_start, r<decay_start], 
                        [lambda r,slope,rho_b,r_b,smooth,decay_start,decay_width : rho_b*(r/r_b)**(-7/4)*(0.5*(1+(r/r_b)**(1/smooth)))**((7/4-slope)*smooth)*np.exp(-(r-decay_start)/decay_width), 
                         lambda r,slope,rho_b,r_b,smooth,decay_start,decay_width : rho_b*(r/r_b)**(-7/4)*(0.5*(1+(r/r_b)**(1/smooth)))**((7/4-slope)*smooth)], slope,rho_b,r_b,smooth,decay_start,decay_width)
    else:
        return np.piecewise(r, [r>=decay_start, r<decay_start], 
                        [lambda r,slope,rho_5pc,smooth,decay_start,decay_width : rho_5pc*(r/pc5_in_m)**(-slope)*np.exp(-(r-decay_start)/decay_width), 
                         lambda r,slope,rho_5pc,smooth,decay_start,decay_width : rho_5pc*(r/pc5_in_m)**(-slope)], slope,rho_5pc,smooth,decay_start,decay_width)
# =============================================================================

# =============================================================================
# function to return power-law inner density profile modified for broken powerlaw test
def line(x,a,b):
    return a*x+b
def get_rho_r_discrete(r,dens_rad,dens,sflag,s,bw_cusp,bw_rad):
    
    # Interpolate the y-values for the new x-data within the original range
    y_interp = 10**safe_interp(np.log10(r),np.log10(dens_rad),np.log10(dens))
    
    # Add the exponential decay beyond the original radii
    x_max = np.max(dens_rad)
    x_extrapolate = r[r > x_max]
    y_extrapolate = y_interp[r > x_max]
    y_extrapolate = y_extrapolate*np.exp(-(x_extrapolate - x_max)/(1000*PC_TO_M))
    
    # uncomment below to extrapolate the outer power-law density slope rather than exp decay
# =============================================================================
#     pars1, cov1 = curve_fit(f=line, xdata=np.log10(dens_rad[-5:]), ydata=np.log10(dens[-5:]))
#     slope1 = pars1[0] 
#     y_extrapolate = 10**(slope1 * (np.log10(x_extrapolate) - np.log10(dens_rad[-1])) + np.log10(dens[-1]))
# =============================================================================
    
    y_interp[r > x_max] = y_extrapolate
    
    # set anything beyond 1e9 pc as zero
    y_interp[r > 10**9*PC_TO_M] = 0

    # Extrapolate using the measured power law slope or fit a slope   
    if sflag:
        slope = s
    else:
        max_ind = 20
        x1 = np.log10(dens_rad[0:max_ind+1])
        y1 = np.log10(dens[0:max_ind+1])
        pars, cov = curve_fit(f=line, xdata=x1, ydata=y1)
        slope = pars[0] 
    x_extrap = r[r < dens_rad[0]]
    y_extrap = y_interp[r < dens_rad[0]]
    y_extrap = 10**(slope * (np.log10(x_extrap) - np.log10(dens_rad[0])) + np.log10(dens[0]))
    y_interp[r < dens_rad[0]] = y_extrap
    
    r_ref = np.geomspace(bw_rad, dens_rad[0], 10**3)
    y_ref = 10**(slope * (np.log10(r_ref) - np.log10(dens_rad[0])) + np.log10(dens[0]))
    den_at_bw = y_ref[0]
    
    # apply the BW cusp
    if bw_cusp:
        x_ext = r[r < bw_rad]
        y_ext = y_interp[r < bw_rad]
        y_ext = 10**((-7/4) * (np.log10(x_ext) - np.log10(bw_rad)) + np.log10(den_at_bw))
        y_interp[r < bw_rad] = y_ext
        
    return y_interp

# =============================================================================


# =============================================================================
def get_encm_pot_integrand(r,slope,rho_5pc,decay_start,decay_width,bw_cusp,bw_rad,smooth):
    return r**2*get_rho_r(r,slope,rho_5pc,decay_start,decay_width,bw_cusp,bw_rad,smooth)    
# function to compute the mass enclosed from density profile 
def get_enc_mass(r,slope,rho_5pc,max_ind,decay_start,decay_width,bw_cusp,bw_rad,smooth):
    if max_ind == 0:
        return 0
    else:
        return 4*np.pi*integrate.trapezoid(get_encm_pot_integrand(r[0:max_ind+1],slope,rho_5pc,decay_start,decay_width,bw_cusp,bw_rad,smooth), r[0:max_ind+1])
# =============================================================================

# =============================================================================
def get_encm_pot_integrand_discrete(r,dens_rad,dens,sflag,s,bw_cusp,bw_rad):
    return r**2*get_rho_r_discrete(r,dens_rad,dens,sflag,s,bw_cusp,bw_rad)    
# function to compute the mass enclosed from density profile 
def get_enc_mass_discrete(r,dens_rad,dens,max_ind,sflag,s,bw_cusp,bw_rad):
    if max_ind == 0:
        return 0
    else:
        return 4*np.pi*integrate.trapezoid(get_encm_pot_integrand_discrete(r[0:max_ind+1],dens_rad,dens,sflag,s,bw_cusp,bw_rad), r[0:max_ind+1])
# =============================================================================




# =============================================================================
def get_ext_pot_integrand(r,slope,rho_5pc,min_ind,decay_start,decay_width,bw_cusp,bw_rad,smooth):
    return r*get_rho_r(r,slope,rho_5pc,decay_start,decay_width,bw_cusp,bw_rad,smooth)   
# function to compute the contribution to the potential of the galaxy at 
# larger radii
def get_ext_potential(r,slope,rho_5pc,min_ind,decay_start,decay_width,bw_cusp,bw_rad,smooth):
    return 4*np.pi*G*integrate.trapezoid(get_ext_pot_integrand(r[min_ind:],slope,rho_5pc,min_ind,decay_start,decay_width,bw_cusp,bw_rad,smooth),r[min_ind:])
# =============================================================================

# =============================================================================
def get_ext_pot_integrand_discrete(r,dens_rad,dens,min_ind,sflag,s,bw_cusp,bw_rad):
    return r*get_rho_r_discrete(r,dens_rad,dens,sflag,s,bw_cusp,bw_rad)   
# function to compute the contribution to the potential of the galaxy at 
# larger radii
def get_ext_potential_discrete(r,dens_rad,dens,min_ind,sflag,s,bw_cusp,bw_rad):
    return 4*np.pi*G*integrate.trapezoid(get_ext_pot_integrand_discrete(r[min_ind:],dens_rad,dens,min_ind,sflag,s,bw_cusp,bw_rad),r[min_ind:])
# =============================================================================




# =============================================================================
# derive the total gravitational potential (psi(r)) as a function of r
def get_psi_r(r,M_BH,slope,rho_5pc,decay_start,decay_width,bw_cusp,bw_rad,smooth):
    
    psi_1 = G*M_BH/r
    
    M_enc = np.zeros_like(r)
    for i in range(len(M_enc)):
        M_enc[i] = get_enc_mass(r,slope,rho_5pc,i,decay_start,decay_width,bw_cusp,bw_rad,smooth)
    psi_2 = G*M_enc/r

    psi_3 = np.zeros_like(r)
    for i in range(len(psi_3)):
        psi_3[i] = get_ext_potential(r,slope,rho_5pc,i,decay_start,decay_width,bw_cusp,bw_rad,smooth)
    
    # remove zero values to avoid divide by zero when taking log10
    psi_1[psi_1 == 0] = 1e-300
    psi_2[psi_2 == 0] = 1e-300
    psi_3[psi_3 == 0] = 1e-300
        
        
    return psi_1+psi_2+psi_3,psi_1,psi_2,psi_3,M_enc
# =============================================================================

# =============================================================================
# derive the total gravitational potential (psi(r)) as a function of r
def get_psi_r_discrete(r,M_BH,dens_rad,dens,sflag,s,bw_cusp,bw_rad):
    
    psi_1 = G*M_BH/r
    
    M_enc = np.zeros_like(r)
    for i in range(len(M_enc)):
        M_enc[i] = get_enc_mass_discrete(r,dens_rad,dens,i,sflag,s,bw_cusp,bw_rad)
    psi_2 = G*M_enc/r

    psi_3 = np.zeros_like(r)
    for i in range(len(psi_3)):
        psi_3[i] = get_ext_potential_discrete(r,dens_rad,dens,i,sflag,s,bw_cusp,bw_rad)
    
    return psi_1+psi_2+psi_3,psi_1,psi_2,psi_3,M_enc
# =============================================================================






# =============================================================================
# define function to find value in an array nearest to a supplied value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.abs(array - value).argmin()
           
    return idx
# =============================================================================   



# =============================================================================  
def get_avg_M_sq(masses):
    PDMF = get_PDMF(masses)
    return integrate.trapezoid((PDMF/M_SOL)*(masses)**2, masses)
# =============================================================================  

# =============================================================================
def get_avg_M(masses):
    PDMF = get_PDMF(masses)
    return integrate.trapezoid(masses*(PDMF/M_SOL),masses)
# =============================================================================

# =============================================================================
def get_M_0(masses):
    PDMF = get_PDMF(masses)
    return integrate.trapezoid((PDMF/M_SOL),masses)
# =============================================================================


# =============================================================================  
def get_PDMF(masses):
    # Kroupa present-day stellar mass function from S&M+16
    solar_masses = masses/M_SOL
    PDMF = np.zeros_like(masses)
    for i in range(len(masses)):
        if solar_masses[i] < 0.5:
            PDMF[i] = (1/1.61)*(solar_masses[i]/0.5)**(-1.3)
            #PDMF[i] = 0.98*solar_masses[i]**(-1.3)
        else:
            PDMF[i] = (1/1.61)*(solar_masses[i]/0.5)**(-2.3)
            #PDMF[i] = 2.4*solar_masses[i]**(-2.3)
            
    # normalize the PDMF to have an area of 1
    area = integrate.trapezoid(PDMF,solar_masses)
    norm_fac = (1/area)
    #norm_fac = 1/np.min(PDMF)
    
    #pdb.set_trace()
    return norm_fac*PDMF
# =============================================================================  

# =============================================================================
# Function to compute the Hills mass
def get_Hills_mass(masses):
    rad_fac = (masses/M_SOL)**0.8
    return ((2.998e8)**2 / (2*G) * (R_SOL*rad_fac)/masses**(1/3))**(3/2)
# =============================================================================


# =============================================================================
def get_psi_t(e,t):
    return (e/2)*(np.tanh(np.pi/2*np.sinh(t))+1)
# =============================================================================


# =============================================================================

def get_psi_r_prime(r_prime,r_apo,M_BH,dens_rad,dens,sflag,s,bw_cusp,bw_rad):
    
    r = r_apo/2*(r_prime+1)
    
    psi_1 = G*M_BH/r
    
    psi_1[np.isnan(psi_1)] = 0
    
    M_enc = np.zeros_like(r)
    for i in range(len(M_enc)):
        M_enc[i] = get_enc_mass_discrete(r,dens_rad,dens,i,sflag,s,bw_cusp,bw_rad)
    psi_2 = G*M_enc/r
    psi_2[np.isnan(psi_2)] = 0
    

    psi_3 = np.zeros_like(r)
    for i in range(len(psi_3)):
        psi_3[i] = get_ext_potential_discrete(r,dens_rad,dens,i,sflag,s,bw_cusp,bw_rad)
        
    psi_3[np.isnan(psi_3)] = 0
    
#    return psi_1+r_apo/2*(psi_2+psi_3)
    return r_apo/2*(psi_1+psi_2+psi_3)

# =============================================================================



# =============================================================================
def integrand_p(rs,r,psi_r,e):
    if len(rs) == 1:
        new_r = np.arange(0.01,2,0.01)*rs
        psi_r_p = 10**safe_interp(np.log10(new_r),np.log10(r),np.log10(psi_r))[99] 
        out = (2*(psi_r_p-e)**(-1/2))
        if np.isnan(out): out = 0
    else:
        new_r = rs
        psi_r_p = 10**safe_interp(np.log10(new_r),np.log10(r),np.log10(psi_r))
        out = (2*(psi_r_p-e)**(-1/2))
        out[np.isnan(out)] = 0
    return out

def integrand_p_quad(rs,r,psi_r,e):
    new_r = rs
    psi_r_p = 10**safe_interp(np.log10(new_r),np.log10(r),np.log10(psi_r))
    out = (2*(psi_r_p-e)**(-1/2))

    return out

# =============================================================================



# =============================================================================
def integrand_I_12(es,psi_i,e_DF,DF):
    DF_interp = 10**safe_interp(np.log10(es),np.log10(e_DF),np.log10(DF))
    return (2*(psi_i-es))**(1/2)*DF_interp
# =============================================================================



# =============================================================================
def integrand_I_32(es,psi_i,e_DF,DF):
    DF_interp = 10**safe_interp(np.log10(es),np.log10(e_DF),np.log10(DF))
    return (2*(psi_i-es))**(3/2)*DF_interp
# =============================================================================


import pdb
# =============================================================================
def integrand_mu(rs_mu,r,psi_r,e_DF_i,e_DF,DF,I_0,M_BH,avg_M_sq,J_c_e):
    if isinstance(rs_mu, float):
        rs_mu_len = 1
        new_r = np.arange(0,2+0.1,0.1)*rs_mu
        psi_rs_mu = np.array(10**safe_interp(np.log10(new_r),np.log10(r),np.log10(psi_r))[10])
    else:
        rs_mu_len = len(rs_mu)
        new_r = rs_mu
        psi_rs_mu = 10**safe_interp(np.log10(new_r),np.log10(r),np.log10(psi_r))

    I_12_r = np.zeros(rs_mu_len)
    I_32_r = np.zeros(rs_mu_len)
    
    for j in range(rs_mu_len):
        if rs_mu_len == 1:
            psi_i = psi_rs_mu[0]
        else:
            psi_i = psi_rs_mu[j]
        es_i = np.linspace(e_DF_i,psi_i,10**3)
        DF_interp_i = 10**safe_interp(np.log10(es_i),np.log10(e_DF),np.log10(DF))
        
        
        I_12_r[j] = (2*(psi_i-e_DF_i))**(-1/2)*integrate.trapezoid((2*(psi_i-es_i))**(1/2)*DF_interp_i,es_i)
        I_32_r[j] = (2*(psi_i-e_DF_i))**(-3/2)*integrate.trapezoid((2*(psi_i-es_i))**(3/2)*DF_interp_i,es_i)
        if np.isnan(I_12_r[j]): I_12_r[j] = 0
        if np.isnan(I_32_r[j]): I_32_r[j] = 0
    
    lim_thing_r = (32*np.pi**2*rs_mu**2*G**2*avg_M_sq*np.log(0.4*M_BH/M_SOL))/(3*J_c_e**2)* \
                    (3*I_12_r - I_32_r + 2*I_0)


    out = lim_thing_r/np.sqrt(2*(psi_rs_mu-e_DF_i))
    if len(out) == 1:
        if np.isnan(out): out == 0
    else:
        out[np.isnan(out)] = 0
    
    return out
# =============================================================================


# =============================================================================
def integrand_mu_temp(r,psi_r,e_DF_i,e_DF,DF,I_0,M_BH,avg_M_sq,J_c_e):
    
    rs_mu_len = len(r)
    psi_rs_mu = psi_r

    I_12_r = np.zeros(rs_mu_len)
    I_32_r = np.zeros(rs_mu_len)
    
    for j in range(rs_mu_len):
        psi_i = psi_rs_mu[j]
        es_i = np.linspace(e_DF_i,psi_i,10**3)
        DF_interp_i = 10**safe_interp(np.log10(es_i),np.log10(e_DF),np.log10(DF))
        
        
        I_12_r[j] = (2*(psi_i-e_DF_i))**(-1/2)*integrate.trapezoid((2*(psi_i-es_i))**(1/2)*DF_interp_i,es_i)
        I_32_r[j] = (2*(psi_i-e_DF_i))**(-3/2)*integrate.trapezoid((2*(psi_i-es_i))**(3/2)*DF_interp_i,es_i)
        if np.isnan(I_12_r[j]): I_12_r[j] = 0
        if np.isnan(I_32_r[j]): I_32_r[j] = 0
    
    lim_thing_r = (32*np.pi**2*r**2*G**2*avg_M_sq*np.log(0.4*M_BH/M_SOL))/(3*J_c_e**2)* \
                    (3*I_12_r - I_32_r + 2*I_0)
    
    return lim_thing_r 
# =============================================================================


# =============================================================================
# 
def write_dat_file(filename, names, black_hole_masses, radii_arrays, densities_arrays):
    """
    Write data to a .dat file with specified rows.

    Parameters:
    filename (str): The name of the .dat file to write.
    names (list of str): List of text names.
    black_hole_masses (list of float): List of black hole masses.
    radii_arrays (list of list of float): List of arrays of radii.
    densities_arrays (list of list of float): List of arrays of densities.
    """
    if not (len(names) == len(black_hole_masses) == len(radii_arrays) == len(densities_arrays)):
        raise ValueError("All input lists must have the same length.")
    
    with open(filename, 'w') as file:
        for name, mass, radii, densities in zip(names, black_hole_masses, radii_arrays, densities_arrays):
            if len(radii) != len(densities):
                raise ValueError("Radii and densities arrays must be of equal length.")
            file.write(f"{name}\n")
            file.write(f"{mass}\n")
            file.write(' '.join(map(str, radii)) + '\n')
            file.write(' '.join(map(str, densities)) + '\n')
            file.write('\n')  # Optional: add an extra newline for separation between entries

# =============================================================================

# =============================================================================
# 
def read_dat_file(filename):
    """
    Read data from a .dat file and store each row in its own variable.

    Parameters:
    filename (str): The name of the .dat file to read.

    Returns:
    tuple: A tuple containing four lists: names, black_hole_masses, radii_arrays, and densities_arrays.
    """
    names = []
    black_hole_masses = []
    radii_arrays = []
    densities_arrays = []

    with open(filename, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)
        
        i = 0
        while i < num_lines:
            name = lines[i].strip()
            mass = float(lines[i + 1].strip())
            radii = list(map(float, lines[i + 2].strip().split()))
            densities = list(map(float, lines[i + 3].strip().split()))

            names.append(name)
            black_hole_masses.append(mass)
            radii_arrays.append(radii)
            densities_arrays.append(densities)
            
            i += 5  # Move to the next set of rows (4 rows of data + 1 empty line)
    
    return names[0], float(black_hole_masses[0]), np.array(radii_arrays[0]).astype(float), np.array(densities_arrays[0]).astype(float)

# =============================================================================

#%%

# =============================================================================
# Function to create the input fits file for an analytic run

def create_analytic_input_table(names, slopes, rho_5pc, M_BHs, decay_start, decay_width, bw_cusp=False, bw_rad=1e-300,
                      quiet=True, M_min=0.08, M_max=1, smooth=0.1):
    """
    Create a FITS table with specified columns and values.

    Parameters:
    names (array-like): Array of string identifiers.
    slopes (array-like): Array of float, power law slope of density profile.
    rho_5pc (array-like): Array of float, 3D density at 5pc (kg/m^3).
    M_BH (array-like): Array of float, BH mass in kg.
    decay_params (array-like): Array of float tuples (radius, width) of exponential decay (m).
    bw_cusp (array-like): Array of booleans, indicate BW cusp needed.
    bw_rad (array-like): Array of float, radius at which BW cusp will begin (-7/4 power-law).
    quiet (boolean): Only want this false for a single run.
    M_min (float, optional): Minimum mass for the PDMF. Default is 0.08.
    M_max (float, optional): Maximum mass for the PDMF. Default is 1.
    smooth (float, optional): Smoothness of exponential decay transition. Default is 0.1.
    filename (str): Name of the output FITS file.

    Returns:
    None
    """
    
    if not isinstance(names, np.ndarray): names = np.array([names])
    if not isinstance(slopes, np.ndarray): slopes = np.array([slopes])
    if not isinstance(rho_5pc, np.ndarray): rho_5pc = np.array([rho_5pc])
    if not isinstance(M_BHs, np.ndarray): M_BHs = np.array([M_BHs])
    if not isinstance(decay_start, np.ndarray): decay_start = np.array([decay_start])
    if not isinstance(decay_width, np.ndarray): decay_width = np.array([decay_width])
    if not isinstance(quiet, np.ndarray): quiet = np.array([quiet])
    if not isinstance(M_min, np.ndarray): M_min = np.array([M_min])
    if not isinstance(M_max, np.ndarray): M_max = np.array([M_max])
    if not isinstance(smooth, np.ndarray): smooth = np.array([smooth])
    
    
    # Define the columns for the FITS table
    col1 = fits.Column(name='name', format='20A', array=np.array(names))
    col2 = fits.Column(name='slope', format='D', array=np.array(slopes))
    col3 = fits.Column(name='rho_5pc', format='D', array=np.array(rho_5pc))
    col4 = fits.Column(name='M_BH', format='D', array=np.array(M_BHs))        
    col5 = fits.Column(name='decay_start', format='D', array=np.array(decay_start))
    col6 = fits.Column(name='decay_width', format='D', array=np.array(decay_width)) 
    if isinstance(bw_cusp, bool):
        col7 = fits.Column(name='bw_cusp', format='L', array=np.ones_like(names).astype(bool)*bw_cusp)
        col8 = fits.Column(name='bw_rad', format='D', array=np.ones_like(names).astype(float)*bw_rad)
    else:
        col7 = fits.Column(name='bw_cusp', format='L', array=np.array(bw_cusp))
        col8 = fits.Column(name='bw_rad', format='D', array=np.array(bw_rad))
    col9 = fits.Column(name='quiet', format='L', array=np.ones_like(names).astype(bool) * quiet)
    col10 = fits.Column(name='M_min', format='D', array=np.ones_like(names).astype(float) * M_min)
    col11 = fits.Column(name='M_max', format='D', array=np.ones_like(names).astype(float) * M_max)
    col12 = fits.Column(name='smooth', format='D', array=np.ones_like(names).astype(float) * smooth)
    # Create the FITS table with the defined columns
    cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12])
    hdu = fits.BinTableHDU.from_columns(cols)

    # Write the FITS table to a file
    #hdu.writeto(filename, overwrite=True)

    return Table.read(hdu)

# =============================================================================

#%%

# =============================================================================
# Function to create the input fits file for an discrete run

def create_discrete_input_table(names, rads, dens, M_BH, sflag, s, bw_cusp=False, bw_rad=1e-300, 
                                quiet=True, M_min=0.08, M_max=1):
    """)
    Create a FITS table with specified columns and values.

    Parameters:
    names (array-like): Array of string identifiers.
    rads (array-like): Array of numpy arrays, each containing radii (m).
    dens (array-like): Array of numpy arrays, each containing 3D densities (kg/m^3).
    M_BH (array-like): Array of float, BH mass in kg.
    sflag (array-like): Array of booleans, indicate fixed inner slope.
    s (array-like): Array of float, fixed slope values.
    bw_cusp (array-like): Array of booleans, indicate inner slope to BW cusp (-7/4).
    bw_rad (array-like): Array of float, radius at which BW cusp will begin (-7/4 power-law).
    quiet (boolean): Only want this false for a single run.
    M_min (float, optional): Minimum mass for the PDMF. Default is 0.08.
    M_max (float, optional): Maximum mass for the PDMF. Default is 1.
    filename (str): Name of the output FITS file.

    Returns:
    None
    """

    if not isinstance(names, np.ndarray): names = np.array([names])
    if len(rads.shape) == 1: rads = np.array([rads])
    if len(dens.shape) == 1: dens = np.array([dens])
    if not isinstance(M_BH, np.ndarray): M_BH = np.array([M_BH])
    if not isinstance(sflag, np.ndarray): sflag = np.array([sflag])
    if not isinstance(s, np.ndarray): s = np.array([s])
    if not isinstance(bw_cusp, np.ndarray): bw_cusp = np.array([bw_cusp])
    if not isinstance(bw_rad, np.ndarray): bw_rad = np.array([bw_rad])
    if not isinstance(quiet, np.ndarray): quiet = np.array([quiet])
    if not isinstance(M_min, np.ndarray): M_min = np.array([M_min])
    if not isinstance(M_max, np.ndarray): M_max = np.array([M_max])
    
    # Define the columns for the FITS table
    col1 = fits.Column(name='name', format='20A', array=np.array(names))
    col2 = fits.Column(name='M_BH', format='D', array=np.array(M_BH))
    col3 = fits.Column(name='sflag', format='L', array=np.array(sflag))
    col4 = fits.Column(name='s', format='D', array=np.array(s))
    if isinstance(bw_cusp, bool):
        col5 = fits.Column(name='bw_cusp', format='L', array=np.ones_like(names).astype(bool)*bw_cusp)
        col6 = fits.Column(name='bw_rad', format='D', array=np.ones_like(names).astype(float)*bw_rad)
    else:
        col5 = fits.Column(name='bw_cusp', format='L', array=np.array(bw_cusp))
        col6 = fits.Column(name='bw_rad', format='D', array=np.array(bw_rad))
    col7 = fits.Column(name='quiet', format='L', array=np.ones_like(names).astype(bool) * quiet)
    col8 = fits.Column(name='M_min', format='D', array=np.ones_like(names).astype(float) * M_min)
    col9 = fits.Column(name='M_max', format='D', array=np.ones_like(names).astype(float) * M_max)
    
    # Convert rads and dens to variable-length arrays
    rads_vla = fits.Column(name='rads', format='PE()', array=np.array(rads,dtype=float))
    dens_vla = fits.Column(name='dens', format='PE()', array=np.array(dens,dtype=float))
    
    # Create the FITS table with the defined columns
    cols = fits.ColDefs([col1, rads_vla, dens_vla, col2, col3, col4, col5, col6, col7, col8, col9])
    hdu = fits.BinTableHDU.from_columns(cols)

    return Table.read(hdu)

# =============================================================================


# =============================================================================
# Functions to compute the decay width necessary to ensure correct galaxy mass in
# the analytic version

def integral_difference(decay_width, r, slope, rho_5pc, decay_start, bw_cusp, bw_rad, smooth, galm):
    tot_mass = get_enc_mass(r,slope,rho_5pc,-2,decay_start,decay_width,bw_cusp,bw_rad,smooth)
    return tot_mass - galm

def find_decay_width(r, slope, rho_5pc, decay_start, bw_cusp, bw_rad, smooth, galm, initial_guess=3e16):
    sol = fsolve(integral_difference, initial_guess, args=(r, slope, rho_5pc, decay_start, bw_cusp, bw_rad, smooth, galm))
    return sol[0]

# =============================================================================
# r = np.geomspace(1e-10,1e37,10**3)
# galm = 4.25240659601994e+39 # NGC 4486B in kg
# slope = 1.0032002171007708
# rho_5pc = 1.9526285662497317e-16
# decay_start = 1.54283879064e+18
# bw_rad = 0
# bw_cusp = False
# smooth = 0.1
# dw = find_decay_width(r, slope, rho_5pc , decay_start, bw_cusp, bw_rad, smooth, galm)
# 
# =============================================================================

# =============================================================================


# =========================== END OF FUNCTIONS ================================



###############################################################################
###############################################################################
###############################################################################
# =============================================================================
# ============================= MAIN FUNCTION =================================
# =============================================================================
###############################################################################
###############################################################################
###############################################################################

def get_TDE_rate_analytic(name,slope,rho_5pc,M_BH,decay_start,decay_width,bw_cusp,bw_rad,
                 quiet=True,M_min=0.08,M_max=1,smooth=0.1,n_energies=1000,EHS=False):

    dis = True
# =========================== DF Computation ==================================
# =============================================================================
# Compute the stellar distribution function (DF) as a function of specific 
# orbital energy (f(epsilon))
# =============================================================================
# =============================================================================
    # use the potential at the tidal radius and 1,000,000 pc to set the orbital energy bounds
    r_t = R_SOL*(M_BH/M_SOL)**(1/3)
    radys = np.geomspace(r_t/PC_TO_M,10**6,10**4)*PC_TO_M
    psis = get_psi_r(radys,M_BH,slope,rho_5pc,decay_start,decay_width,bw_cusp,bw_rad,smooth)[0]
    r_for_e_max = 100*r_t
    ind_for_e_max = find_nearest(radys, r_for_e_max)
    e_max = psis[ind_for_e_max]
    e_min = psis[-1]
    
    # specify the range of specific orbital energies to consider
    e = np.geomspace(e_min,e_max,n_energies+2)
    
    # STEP 1: Define wide range of t
    t_bound = 1.5
    num_t = 10**3
    t = np.linspace(-t_bound, t_bound, num_t)
    
    integrals = np.zeros_like(e)
    
    r_min = 1000 # meters
    r_max = 10**12*PC_TO_M # meters
    num_rad = 10**4
    r_ls = np.geomspace(r_min,r_max,num_rad)
    
    psi_r_init,psi_bh_init,psi_enc_init,psi_ext_init,enc_masses = get_psi_r(r_ls,M_BH,slope,rho_5pc,decay_start,decay_width,bw_cusp,bw_rad,smooth)

    if not quiet:
        dis = False
        print('Computing DF...')
        print()
    time.sleep(1)
    def compute_DF(r_ls,psi_r_init):
        for j in tqdm(range(len(e)), position=0, leave=True, disable=dis):
        #for j in range(len(e)):
            # STEP 2: Compute array of psi(t)
            psi_t = get_psi_t(e[j],t)
        
            # since I am not using the true psi_r to step the radii, double check 
            # that the final radii do indeed give psi_r covering psi_t
            if np.min(psi_r_init) >= np.min(psi_t) or np.max(psi_r_init) <= np.max(psi_t):
                raise ValueError('*** ERROR: psi(r) failed to cover psi(t); Increase r_min and/or r_max at lines 927-928 ***')
    
            # STEP 4: Evaluate drho/dpsi at all values of psi_t
            d_rho_d_psi_t = np.zeros_like(psi_t)
            num_r = 10**4
            drhodpsi_ratios = np.zeros(len(psi_t))
            def get_drhodpsi(d_rho_d_psi_t,drhodpsi_ratios):
                for i in range(len(psi_t)):
                    if psi_t[i] >= 1e-20:
                        r_ind = find_nearest(psi_r_init,psi_t[i])
                        r_closest = r_ls[r_ind]
                        
                        spacing = ((r_closest+0.2*r_closest)-(r_closest-0.2*r_closest))/(num_r-1)
                        r = np.arange(0,num_r,1)*spacing+(r_closest-0.2*r_closest)
                        psi_r = 10**safe_interp(np.log10(r),np.log10(r_ls),np.log10(psi_r_init))
                        
                        rho_r = get_rho_r(r,slope,rho_5pc,decay_start,decay_width,bw_cusp,bw_rad,smooth)
                    
                        rho_r[np.where(rho_r == 0.0)] = 1e-323
                    
                        psi_t_ind = find_nearest(psi_r,psi_t[i])
    
                        if psi_r[psi_t_ind]-psi_t[i] >= psi_t[i]*0.001:
                            raise ValueError('*** ERROR: Finer grid needed for initial psi_r/psi_t match; Increase num_rad at line 929 ***')
    
                        d_rho_d_psi_t[i] = (rho_r[psi_t_ind-1]-rho_r[psi_t_ind+1])/(psi_r[psi_t_ind-1]-psi_r[psi_t_ind+1])
                       
                return d_rho_d_psi_t
        
            d_rho_d_psi_t = get_drhodpsi(d_rho_d_psi_t,drhodpsi_ratios)
    
    
            # STEP 5: Use t and drho/dpsi to tabulate drho/dt
            d_rho_d_t = (e[j]/2)*(np.pi/2*np.cosh(t))/(np.cosh(np.pi/2*np.sinh(t)))**2*d_rho_d_psi_t
    
    
            # STEP 6: Tabulate the other factor from the double exponential transformed 
            # version of the DF integral
            frac_fac_t = 1/np.sqrt(e[j] - (e[j]/2*(np.tanh(np.pi/2*np.sinh(t)) + 1)))    
    
    
            integrands = d_rho_d_t * frac_fac_t
    
            # STEP 7: Evaluate the integral for all values of epsilon (e)
            integrals[j] = integrate.trapezoid(integrands,t)
            
        return integrals
      
    try:
        integrals = compute_DF(r_ls,psi_r_init)
    except ValueError as err:
        print(f"An error occurred: {err}")
    
    # STEP 8: Compute the derivative of the integral values vs. epsilon for all 
    # all values of epsilon
    d_int_d_e = np.zeros(len(e)-2)
    for i in range(len(e)-2):
        d_int_d_e[i] = (integrals[(i+1)-1]-integrals[(i+1)+1])/(e[(i+1)-1]-e[(i+1)+1])
    
    
    DF = np.abs(1/(np.sqrt(8)*np.pi**2*M_SOL)*d_int_d_e)
    orb_ens = e[1:-1]
    
    if not quiet:
        print('DF computation complete.')
        print()

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================



#%%
# ==================== q, mu, and P Computation ===============================
# =============================================================================
# Compute the orbit averaged angular momentum diffusion coefficient for 
# highly eccentric orbits, mu(epsilon), periods, and q (unitless change in 
# squared angular momentum per orbit)
# =============================================================================
# =============================================================================

    mu_e = np.zeros(len(orb_ens))
    periods_e = np.zeros(len(orb_ens))
    J_c_e = np.zeros(len(orb_ens))
    R_LC_e = np.zeros(len(orb_ens))
    
    int_fac_rs = []
    
    avg_M_sq = M_SOL**2
    
    if not quiet:
        print('Computing q...')
        print()
    time.sleep(1)
    
    def compute_q(r_ref,psi_r_ref,psi_bh_ref,psi_enc_ref,psi_ext_ref,enc_masses):
        #for i in range(len(orb_ens)):
        for i in tqdm(range(len(orb_ens)), position=0, leave=True, disable=dis):   
            eps_apo = psi_r_ref
            eps = (psi_bh_ref+psi_enc_ref)/2 + psi_ext_ref
            J_c_r = np.sqrt(G*(enc_masses+M_BH)*r_ref)
            R_LC_r = (2*M_BH*r_t)/((enc_masses+M_BH)*r_ref)
            
            if orb_ens[i] > eps[0]:
                periods_e[i] = 0
                R_LC_e[i] = 1
                mu_e[i] = 0
            else:   
                r_ind = find_nearest(eps_apo,orb_ens[i])
                if np.abs(eps_apo[r_ind] - orb_ens[i])/orb_ens[i] >= 0.001:
                    new_r_ref = np.geomspace(r_ref[r_ind]-0.1*r_ref[r_ind],r_ref[r_ind]+0.1*r_ref[r_ind],10**3)
                    new_eps_apo = 10**safe_interp(np.log10(new_r_ref),np.log10(r_ref),np.log10(eps_apo))
                    new_r_ind = find_nearest(new_eps_apo,orb_ens[i])
                    if np.abs(new_eps_apo[new_r_ind] - orb_ens[i])/orb_ens[i] >= 0.001:
                        raise ValueError('*** Error in R_apo calculation: R_apo energy value is greater than energy array value by > 0.1%. Finer sampling needed; Adjust r_ref at line 1111. ***')
                    r_apo = new_r_ref[new_r_ind]
                else:
                    r_apo = r_ref[r_ind]
                
                r_ind_LC = find_nearest(eps,orb_ens[i])
                if np.abs(eps[r_ind_LC] - orb_ens[i])/orb_ens[i] >= 0.001:
                    new_r_ref_LC = np.geomspace(r_ref[r_ind_LC]-0.2*r_ref[r_ind_LC],r_ref[r_ind_LC]+0.2*r_ref[r_ind_LC],10**3)
                    new_eps = 10**safe_interp(np.log10(new_r_ref_LC),np.log10(r_ref),np.log10(eps))
                    new_J_c_r = 10**safe_interp(np.log10(new_r_ref_LC),np.log10(r_ref),np.log10(J_c_r))
                    new_r_ind_LC = find_nearest(new_eps,orb_ens[i])
                    if np.abs(new_eps[new_r_ind_LC]- orb_ens[i])/orb_ens[i] >= 0.001:
                        raise ValueError('Error in R_LC calculations: R_LC energy value is greater than energy array value by > 0.1%. Finer sampling needed; Adjust r_ref at line 1111. ***')
                    new_R_LC_r = 10**safe_interp(np.log10(new_r_ref_LC),np.log10(r_ref),np.log10(R_LC_r))
                    R_LC_e[i] = new_R_LC_r[new_r_ind_LC]
                    J_c_e[i] = new_J_c_r[new_r_ind_LC]
                else:
                    R_LC_e[i] = R_LC_r[r_ind_LC]
                    J_c_e[i] = J_c_r[r_ind_LC]
            
            r2 = np.linspace(1e-10,r_apo,10**5)
            psi2 = 10**np.interp(np.log10(r2),np.log10(r_ref),np.log10(eps_apo))
            
            r2_prime = 2*r2/r_apo - 1
            psi2_prime = psi2 
            
            t_of_r2_prime_temp = np.arcsinh((2*np.arctanh(r2_prime))/(np.pi))
            psi2_t_temp = psi2_prime
            
            sort_t2 = np.argsort(t_of_r2_prime_temp)
            t_of_r2_prime = t_of_r2_prime_temp[sort_t2]
            psi2_t = psi2_t_temp[sort_t2]
            
            t_of_r2_prime[t_of_r2_prime == -1*np.inf] = -3
            t_of_r2_prime[t_of_r2_prime == np.inf] = 3
            
            prefacs2 = np.sqrt(4*np.arctanh(np.tanh(np.pi/2*np.sinh(t_of_r2_prime)))**2 + np.pi**2)/(2*np.cosh(np.pi/2*np.sinh(t_of_r2_prime))**2)
            integrands2 = prefacs2/(np.sqrt(2*(psi2_t-orb_ens[i])))
            integrands2[np.isnan(integrands2)] = 0
            integrands2[np.isinf(integrands2)] = 0
            
            final_t2 = np.linspace(t_of_r2_prime[0],t_of_r2_prime[-1],10**3)
            
            final_integrand2 = np.interp(final_t2,t_of_r2_prime,integrands2)
            
            periods_e[i] = r_apo*integrate.trapezoid(final_integrand2,final_t2)
    
            es = np.geomspace(orb_ens[0],orb_ens[i],10**3)
            DF_interp = 10**safe_interp(np.log10(es),np.log10(orb_ens),np.log10(DF))
            I_0 = integrate.trapezoid(DF_interp,es)
    
            int_fac_r = integrate.quadrature(integrand_mu,r_t,r_apo,args=(r_ref,psi_r_ref,orb_ens[i],orb_ens,DF,I_0,M_BH,avg_M_sq,J_c_e[i]),
                                       rtol=1,maxiter=400)[0]
            
            if r_t > r_apo:
                int_fac_r = 0
    
            int_fac_rs.append(int_fac_r)
    
            mu_e[i] = 2*int_fac_r/periods_e[i]
    
        q_discrete = mu_e*periods_e/R_LC_e
    
        if not quiet:    
            print('q computation complete.')
            print()
            
        return np.pi/2*q_discrete, np.pi/2*mu_e, periods_e
    
    try:
        r_ref = np.geomspace(1e-10,1e37,10**3)
        psi_r_ref,psi_bh_ref,psi_enc_ref,psi_ext_ref,enc_masses = get_psi_r(r_ref,M_BH,slope,rho_5pc,decay_start,decay_width,bw_cusp,bw_rad,smooth)
        q, mu_e, periods_e = compute_q(r_ref,psi_r_ref,psi_bh_ref,psi_enc_ref,psi_ext_ref,enc_masses)
        
        #q, mu_e, periods_e = compute_q()
    except ValueError as err:
        print(f"An error occurred: {err}")
        
    
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================



#%%
# ==================== LC FLUX Computation ===============================
# =============================================================================
# Compute the flux of stars that scatter into the loss cone per unit time 
# and energy
# =============================================================================
# =============================================================================

    def get_LC_flux(orb_ens,G,M_BH,DF,mu_e,q,R_LC_e,periods_e,J_c_e):
        ln_R_0 = np.zeros_like(orb_ens)
    
        for i in range(len(orb_ens)):
            if q[i] > 1:
                ln_R_0[i] = (q[i] - np.log(R_LC_e[i]))
            else:
                ln_R_0[i] = ((0.186*q[i]+0.824*np.sqrt(q[i])) - np.log(R_LC_e[i]))
            
        return (4*np.pi**2)*periods_e*J_c_e**2*mu_e*(DF/ln_R_0)


# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================



#%%
# ======================== TDE RATE Computation ===============================
# =============================================================================
# Compute the TDE rate by integrating the total flux into the loss cone for 
# stars of a given mass, and then by integrating over the stellar mass function.
# =============================================================================
# =============================================================================
    
    # compute the TDE rates for a pure population of solar mass stars 
    LC_flux_solar = get_LC_flux(orb_ens,G,M_BH,DF,mu_e,q,R_LC_e,periods_e,J_c_e)
    
    #LC_flux_solar[LC_flux_solar < 0] = 0.0
    
    TDE_rate_solar = integrate.trapezoid(LC_flux_solar, orb_ens)*SEC_PER_YR
    
    
    #%%
    masses = np.linspace(M_min,M_max,50)*M_SOL
    R_stars = (masses/M_SOL)**0.8*R_SOL # m
    
    # get the total number of stars contributed to the LC for each mass
    LC_contributions = np.zeros(len(masses))
    LC_flux_per_mass = []
    
    if not quiet:
        print('Computing LC Flux per mass in PDMF:')
        print()
    time.sleep(1)
    
    for j in range(len(masses)):
        r_t_adj = R_stars[j]*(M_BH/masses[j])**(1/3)
        R_LC_e_adj = np.zeros(len(orb_ens))
        
        for i in range(len(orb_ens)):
            eps = (psi_bh_ref+psi_enc_ref)/2 + psi_ext_ref
            R_LC_r = (2*M_BH*r_t_adj)/((enc_masses+M_BH)*r_ref)
        
            if orb_ens[i] > eps[0]:
                R_LC_e_adj[i] = 1
            else:   
                r_ind_LC = find_nearest(eps,orb_ens[i])
                if np.abs(eps[r_ind_LC] - orb_ens[i])/orb_ens[i] >= 0.001:
                    new_r_ref_LC = np.geomspace(r_ref[r_ind_LC]-0.2*r_ref[r_ind_LC],r_ref[r_ind_LC]+0.2*r_ref[r_ind_LC],10**3)
                    new_eps = 10**safe_interp(np.log10(new_r_ref_LC),np.log10(r_ref),np.log10(eps))
                    new_r_ind_LC = find_nearest(new_eps,orb_ens[i])
                    if np.abs(new_eps[new_r_ind_LC] - orb_ens[i])/orb_ens[i] >= 0.001:
                        print('Possible Error: R_LC calculations.')
                        print(np.abs(new_eps[new_r_ind_LC] - orb_ens[i])/orb_ens[i])
                        pdb.set_trace()
                    new_R_LC_r = 10**safe_interp(np.log10(new_r_ref_LC),np.log10(r_ref),np.log10(R_LC_r))
                    R_LC_e_adj[i] = new_R_LC_r[new_r_ind_LC]
                else:
                    R_LC_e_adj[i] = R_LC_r[r_ind_LC]
                
       
        DF_adj = DF * (M_SOL/get_avg_M(masses))
        mu_e_adj = mu_e * (get_avg_M_sq(masses)/M_SOL**2) * (M_SOL/get_avg_M(masses))
        q_adj = mu_e_adj*periods_e/R_LC_e_adj
        
        LC_flux_e = get_LC_flux(orb_ens,G,M_BH,DF_adj,mu_e_adj,q_adj,R_LC_e_adj,periods_e,J_c_e)
        #LC_flux_e[LC_flux_e < 0] = 0.0
        LC_flux_per_mass.append(LC_flux_e)
        LC_contributions[j] = integrate.trapezoid(LC_flux_e, orb_ens)
        
        
    if not quiet:
        print('LC Flux per Mass computation complete.')
        print()

    LC_flux_per_mass = np.array(LC_flux_per_mass)
    
    if EHS:
        EHS_flag = np.zeros(len(masses)).astype(bool)
        Hill_masses = get_Hills_mass(masses)
        EHS_flag[M_BH<=Hill_masses] = True
    else:
        EHS_flag = np.ones(len(masses)).astype(bool)
    
    PDMF = get_PDMF(masses)
    TDE_rate_full = integrate.trapezoid(LC_contributions[EHS_flag]*PDMF[EHS_flag], masses[EHS_flag]/M_SOL)*SEC_PER_YR

    all_output = {'name': [name],
                  'slope': [slope],
                  'rho_5pc': [rho_5pc],
                  'smooth': [smooth],
                  'M_BH': [M_BH],
                  'decay_start': [decay_start],
                  'decay_width': [decay_width],
                  'TDE_rate_solar': [TDE_rate_solar],
                  'TDE_rate_full': [TDE_rate_full],
                  'orb_ens': [orb_ens],
                  'DF': [DF],
                  'q': [q],
                  'LC_flux_solar': [LC_flux_solar],
                  'masses': [masses],
                  'LC_contributions_per_mass': [LC_contributions],
                  'LC_flux_per_mass': [LC_flux_per_mass],
                  'psi_rads': [r_ls],
                  'psi_tot': [psi_r_init],
                  'psi_bh': [psi_bh_init],
                  'psi_enc': [psi_enc_init],
                  'psi_ext': [psi_ext_init],
                  'DF_integrals': [integrals],
                  'mu_integrals': [int_fac_rs],
                  'periods': [periods_e],
                  'mu': [mu_e],
                  'R_LC': [R_LC_e],
                  'bw_cusp': [bw_cusp],
                  'bw_rad': [bw_rad],
                  'EHS': [EHS]}

    output_table = Table(all_output)

    return output_table


###############################################################################
###############################################################################
###############################################################################
# =============================================================================
# ======================== END OF MAIN FUNCTION ===============================
# =============================================================================
###############################################################################
###############################################################################
###############################################################################





def get_TDE_rate_discrete(name,dens_rad,dens,M_BH,sflag,s,bw_cusp,bw_rad,
                 quiet=True,M_min=0.08,M_max=1,n_energies=1000,EHS=False):

    dis = True
# =========================== DF Computation ==================================
# =============================================================================
# Compute the stellar distribution function (DF) as a function of specific 
# orbital energy (f(epsilon))
# =============================================================================
# =============================================================================
    # use the potential at 0.1 of the tidal radius and 1,000,000 pc to set the orbital energy bounds
    r_t = R_SOL*(M_BH/M_SOL)**(1/3)
    radys = np.geomspace(10**-10,10**6,10**4)*PC_TO_M
    psis = get_psi_r_discrete(radys,M_BH,dens_rad,dens,sflag,s,bw_cusp,bw_rad)[0]
    r_for_e_max = 100*r_t
    ind_for_e_max = find_nearest(radys, r_for_e_max)
    e_max = psis[ind_for_e_max]
    e_min = psis[-1]
    
    #psi_out = get_psi_r_discrete(radys,M_BH,dens_rad,dens,sflag,s,bw_cusp,bw_rad)
    #pdb.set_trace()
    
    
    # specify the range of specific orbital energies to consider
    e = np.geomspace(e_min,e_max,n_energies+2)

    # STEP 1: Define wide range of t
    t_bound = 1.5
    num_t = 10**3
    t = np.linspace(-t_bound, t_bound, num_t)

    integrals = np.zeros_like(e)

    r_min = 1000 # meters
    r_max = 10**12*PC_TO_M # meters
    num_rad = 10**4
    r_ls = np.geomspace(r_min,r_max,num_rad)

    psi_r_init,psi_bh_init,psi_enc_init,psi_ext_init,enc_masses = get_psi_r_discrete(r_ls,M_BH,dens_rad,dens,sflag,s,bw_cusp,bw_rad)
    rho_r_init = get_rho_r_discrete(r_ls,dens_rad,dens,sflag,s,bw_cusp,bw_rad)

    if not quiet:
        dis = False
        print('Computing DF...')
        print()
    time.sleep(1)
    def compute_DF(r_ls,psi_r_init):
        for j in tqdm(range(len(e)), position=0, leave=True, disable=dis):
        #for j in range(len(e)):
            # STEP 2: Compute array of psi(t)
            psi_t = get_psi_t(e[j],t)
        
            # since I am not using the true psi_r to step the radii, double check 
            # that the final radii do indeed give psi_r covering psi_t
            if np.min(psi_r_init) >= np.min(psi_t) or np.max(psi_r_init) <= np.max(psi_t):
                raise ValueError('*** ERROR: psi(r) failed to cover psi(t); Increase r_min and/or r_max at lines 1319-1320 ***')

            # STEP 4: Evaluate drho/dpsi at all values of psi_t
            d_rho_d_psi_t = np.zeros_like(psi_t)
            num_r = 10**4
            drhodpsi_ratios = np.zeros(len(psi_t))
            def get_drhodpsi(d_rho_d_psi_t,drhodpsi_ratios):
                for i in range(len(psi_t)):
                    if psi_t[i] >= 1e-20:
                        r_ind = find_nearest(psi_r_init,psi_t[i])
                        r_closest = r_ls[r_ind]
                        
                        spacing = ((r_closest+0.2*r_closest)-(r_closest-0.2*r_closest))/(num_r-1)
                        r = np.arange(0,num_r,1)*spacing+(r_closest-0.2*r_closest)
                        #psi_r = get_psi_r(r,G,M_BH,slope,rho_b,r_b,decay_start,decay_width)
                        psi_r = 10**safe_interp(np.log10(r),np.log10(r_ls),np.log10(psi_r_init))
                        rho_r = get_rho_r_discrete(r,dens_rad,dens,sflag,s,bw_cusp,bw_rad)
                    
                        psi_t_ind = find_nearest(psi_r,psi_t[i])
    
                        if psi_r[psi_t_ind]-psi_t[i] >= psi_t[i]*0.001:
                            raise ValueError('ERROR: Finer grid needed for initial psi_r/psi_t match; Increase num_rad at line 1321 ***')
    
                        d_rho_d_psi_t[i] = (rho_r[psi_t_ind-1]-rho_r[psi_t_ind+1])/(psi_r[psi_t_ind-1]-psi_r[psi_t_ind+1])
                    
                return d_rho_d_psi_t
        
            d_rho_d_psi_t = get_drhodpsi(d_rho_d_psi_t,drhodpsi_ratios)

    
            # STEP 5: Use t and drho/dpsi to tabulate drho/dt
            d_rho_d_t = (e[j]/2)*(np.pi/2*np.cosh(t))/(np.cosh(np.pi/2*np.sinh(t)))**2*d_rho_d_psi_t


            # STEP 6: Tabulate the other factor from the double exponential transformed 
            # version of the DF integral
            frac_fac_t = 1/np.sqrt(e[j] - (e[j]/2*(np.tanh(np.pi/2*np.sinh(t)) + 1)))    


            integrands = d_rho_d_t * frac_fac_t

            # STEP 7: Evaluate the integral for all values of epsilon (e)
            integrals[j] = integrate.trapezoid(integrands,t)
    
  
        return integrals
  
    try:
        integrals = compute_DF(r_ls,psi_r_init)
    except ValueError as err:
        print(f"An error occurred: {err}")

    # STEP 8: Compute the derivative of the integral values vs. epsilon for all 
    # all values of epsilon
    d_int_d_e = np.zeros(len(e)-2)
    for i in range(len(e)-2):
        d_int_d_e[i] = (integrals[(i+1)-1]-integrals[(i+1)+1])/(e[(i+1)-1]-e[(i+1)+1])


    DF = np.abs(1/(np.sqrt(8)*np.pi**2*M_SOL)*d_int_d_e)
    orb_ens = e[1:-1]

    e_ind = find_nearest(orb_ens,5.58e13)

    #pdb.set_trace()

    if not quiet:
        print('DF computation complete.')
        print()

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================



#%%
# ==================== q, mu, and P Computation ===============================
# =============================================================================
# Compute the orbit averaged angular momentum diffusion coefficient for 
# highly eccentric orbits, mu(epsilon), periods, and q (unitless change in 
# squared angular momentum per orbit)
# =============================================================================
# =============================================================================

    mu_e = np.zeros(len(orb_ens))
    periods_e = np.zeros(len(orb_ens))
    R_LC_e = np.zeros(len(orb_ens))
    J_c_e = np.zeros(len(orb_ens))

    int_fac_rs = []

    avg_M_sq = M_SOL**2

    if not quiet:
        print('Computing q...')
        print()
    time.sleep(1)
    
    def compute_q(r_ref,psi_r_ref,psi_bh_ref,psi_enc_ref,psi_ext_ref,enc_masses):
        for i in tqdm(range(len(orb_ens)), position=0, leave=True, disable=dis):

            eps_apo = psi_r_ref
            eps = (psi_bh_ref+psi_enc_ref)/2 + psi_ext_ref
            J_c_r = np.sqrt(G*(enc_masses+M_BH)*r_ref)
            R_LC_r = (2*M_BH*r_t)/((enc_masses+M_BH)*r_ref)
            
            if orb_ens[i] > eps[0]:
                periods_e[i] = 0
                R_LC_e[i] = 1
                mu_e[i] = 0
            else:   
            
                r_ind = find_nearest(eps_apo,orb_ens[i])
                if np.abs(eps_apo[r_ind] - orb_ens[i])/orb_ens[i] >= 0.001:
                    new_r_ref = np.geomspace(r_ref[r_ind]-0.2*r_ref[r_ind],r_ref[r_ind]+0.2*r_ref[r_ind],10**3)
                    new_eps_apo = 10**safe_interp(np.log10(new_r_ref),np.log10(r_ref),np.log10(eps_apo))
                    new_r_ind = find_nearest(new_eps_apo,orb_ens[i])
                    if np.abs(new_eps_apo[new_r_ind] - orb_ens[i])/orb_ens[i] >= 0.001:
                        raise ValueError('*** Error in R_apo calculation: R_apo energy value is greater then energy array value by > 0.1%. Finer sampling needed. Adjust r_ref at line 1506 ***')
                    r_apo = new_r_ref[new_r_ind]
                else:
                    r_apo = r_ref[r_ind]
                
                r_ind_LC = find_nearest(eps,orb_ens[i])
                if np.abs(eps[r_ind_LC] - orb_ens[i])/orb_ens[i] >= 0.001:
                    new_r_ref_LC = np.geomspace(r_ref[r_ind_LC]-0.2*r_ref[r_ind_LC],r_ref[r_ind_LC]+0.2*r_ref[r_ind_LC],10**3)
                    new_eps = 10**safe_interp(np.log10(new_r_ref_LC),np.log10(r_ref),np.log10(eps))
                    new_J_c_r = 10**safe_interp(np.log10(new_r_ref_LC),np.log10(r_ref),np.log10(J_c_r))
                    new_r_ind_LC = find_nearest(new_eps,orb_ens[i])
                    if np.abs(new_eps[new_r_ind_LC] - orb_ens[i])/orb_ens[i] >= 0.001:
                        raise ValueError('*** Error in R_LCcalculation: R_LC energy value is greater then energy array value by > 0.1%. Finer sampling needed. Adjust r_ref at line 1506 ***')
                    new_R_LC_r = 10**safe_interp(np.log10(new_r_ref_LC),np.log10(r_ref),np.log10(R_LC_r))
                    R_LC_e[i] = new_R_LC_r[new_r_ind_LC]
                    J_c_e[i] = new_J_c_r[new_r_ind_LC]
                else:
                    R_LC_e[i] = R_LC_r[r_ind_LC]
                    J_c_e[i] = J_c_r[r_ind_LC]
                
            
            r2 = np.linspace(1e-10,r_apo,10**5)
            psi2 = 10**np.interp(np.log10(r2),np.log10(r_ref),np.log10(eps_apo))
            
            r2_prime = 2*r2/r_apo - 1
            psi2_prime = psi2 
            
            t_of_r2_prime_temp = np.arcsinh((2*np.arctanh(r2_prime))/(np.pi))
            psi2_t_temp = psi2_prime
            
            sort_t2 = np.argsort(t_of_r2_prime_temp)
            t_of_r2_prime = t_of_r2_prime_temp[sort_t2]
            psi2_t = psi2_t_temp[sort_t2]
            
            t_of_r2_prime[t_of_r2_prime == -1*np.inf] = -3
            t_of_r2_prime[t_of_r2_prime == np.inf] = 3
            
            prefacs2 = np.sqrt(4*np.arctanh(np.tanh(np.pi/2*np.sinh(t_of_r2_prime)))**2 + np.pi**2)/(2*np.cosh(np.pi/2*np.sinh(t_of_r2_prime))**2)
            integrands2 = prefacs2/(np.sqrt(2*(psi2_t-orb_ens[i])))
            integrands2[np.isnan(integrands2)] = 0
            integrands2[np.isinf(integrands2)] = 0
            
            final_t2 = np.linspace(t_of_r2_prime[0],t_of_r2_prime[-1],10**3)
            
            final_integrand2 = np.interp(final_t2,t_of_r2_prime,integrands2)
            
            periods_e[i] = r_apo*integrate.trapezoid(final_integrand2,final_t2)
            
            es = np.geomspace(orb_ens[0],orb_ens[i],10**3)
            DF_interp = 10**safe_interp(np.log10(es),np.log10(orb_ens),np.log10(DF))
            I_0 = integrate.trapezoid(DF_interp,es)
            
    
            int_fac_r = integrate.quadrature(integrand_mu,1e-10,r_apo,args=(r_ref,psi_r_ref,orb_ens[i],orb_ens,DF,I_0,M_BH,avg_M_sq,J_c_e[i]),
                                             rtol=1,maxiter=400)[0]
            
            # let's pull out the limit factor in mu to compare with Nick
            #if i == e_ind:
            #    pdb.set_trace()
            #    datay = np.column_stack((r_ref,integrand_mu_temp(r_ref,psi_r_ref,orb_ens[i],orb_ens,DF,I_0,M_BH,avg_M_sq,J_c_e[i])))
            #    np.savetxt("cusp_mu_lim_fac_orbe_{:.3e}.dat".format(orb_ens[i]), datay)
            
            if r_t > r_apo:
                int_fac_r = 0
                
            int_fac_rs.append(int_fac_r)

            mu_e[i] = 2*int_fac_r/periods_e[i]

        q_discrete = mu_e*periods_e/R_LC_e



        if not quiet:
            print('q computation complete.')
            print()
        return np.pi/2*q_discrete, np.pi/2*mu_e, periods_e, 

    try:
        r_ref = np.geomspace(1e-10,1e37,10**3)
        psi_r_ref,psi_bh_ref,psi_enc_ref,psi_ext_ref,enc_masses = get_psi_r_discrete(r_ref,M_BH,dens_rad,dens,sflag,s,bw_cusp,bw_rad)
        q, mu_e, periods_e = compute_q(r_ref,psi_r_ref,psi_bh_ref,psi_enc_ref,psi_ext_ref,enc_masses)
    except ValueError as err:
        print(f"An error occurred: {err}")


# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================



#%%
# ==================== LC FLUX Computation ===============================
# =============================================================================
# Compute the flux of stars that scatter into the loss cone per unit time 
# and energy
# =============================================================================
# =============================================================================

    def get_LC_flux(orb_ens,G,M_BH,DF,mu_e,q,R_LC_e,periods_e,J_c_e):
        #J_c = G*M_BH/np.sqrt(2*orb_ens)
        ln_R_0 = np.zeros_like(orb_ens)
    
        for i in range(len(orb_ens)):
            if q[i] > 1:
                ln_R_0[i] = q[i] - np.log(R_LC_e[i])
            else:
                ln_R_0[i] = ((0.186*q[i]+0.824*np.sqrt(q[i])) - np.log(R_LC_e[i]))

        return (4*np.pi**2)*periods_e*J_c_e**2*mu_e*(DF/ln_R_0)


# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================



#%%
# ======================== TDE RATE Computation ===============================
# =============================================================================
# Compute the TDE rate by integrating the total flux into the loss cone for 
# stars of a given mass, and then by integrating over the stellar mass function.
# =============================================================================
# =============================================================================
    
    # compute the TDE rates for a pure population of solar mass stars 
    LC_flux_solar = get_LC_flux(orb_ens,G,M_BH,DF,mu_e,q,R_LC_e,periods_e,J_c_e)
    
    #LC_flux_solar[LC_flux_solar < 0] = 0.0
    
    TDE_rate_solar = integrate.trapezoid(LC_flux_solar, orb_ens)*SEC_PER_YR

    masses = np.linspace(M_min,M_max,50)*M_SOL
    R_stars = (masses/M_SOL)**0.8*R_SOL # m

    # get the total number of stars contributed to the LC for each mass
    LC_contributions = np.zeros(len(masses))
    LC_flux_per_mass = []

    if not quiet:
        print('Computing LC Flux per mass in PDMF:')
        print()
    time.sleep(1)


    #for j in tqdm(range(len(masses)), position=0, leave=True):
    for j in range(len(masses)):
        r_t_adj = R_stars[j]*(M_BH/masses[j])**(1/3)
        R_LC_e_adj = np.zeros(len(R_LC_e))

        for i in range(len(orb_ens)):
            eps = (psi_bh_ref+psi_enc_ref)/2 + psi_ext_ref
            R_LC_r = (2*M_BH*r_t_adj)/((enc_masses+M_BH)*r_ref)
        
            if orb_ens[i] > eps[0]:
                R_LC_e_adj[i] = 1
            else:          
                r_ind_LC = find_nearest(eps,orb_ens[i])
                if np.abs(eps[r_ind_LC] - orb_ens[i])/orb_ens[i] >= 0.001:
                    new_r_ref_LC = np.geomspace(r_ref[r_ind_LC]-0.2*r_ref[r_ind_LC],r_ref[r_ind_LC]+0.2*r_ref[r_ind_LC],10**3)
                    new_eps = 10**safe_interp(np.log10(new_r_ref_LC),np.log10(r_ref),np.log10(eps))
                    new_r_ind_LC = find_nearest(new_eps,orb_ens[i])
                    if np.abs(new_eps[new_r_ind_LC] - orb_ens[i])/orb_ens[i] >= 0.001:
                        print('Possible Error: R_LC calculations.')
                        print(np.abs(new_eps[new_r_ind_LC] - orb_ens[i])/orb_ens[i])
                        pdb.set_trace()
                    new_R_LC_r = 10**safe_interp(np.log10(new_r_ref_LC),np.log10(r_ref),np.log10(R_LC_r))
                    R_LC_e_adj[i] = new_R_LC_r[new_r_ind_LC]
                else:
                    R_LC_e_adj[i] = R_LC_r[r_ind_LC]

        DF_adj = DF * (M_SOL/get_avg_M(masses))
        mu_e_adj = mu_e * (get_avg_M_sq(masses)/M_SOL**2) * (M_SOL/get_avg_M(masses))
        q_adj = mu_e_adj*periods_e/R_LC_e_adj
        
        LC_flux_e = get_LC_flux(orb_ens,G,M_BH,DF_adj,mu_e_adj,q_adj,R_LC_e_adj,periods_e,J_c_e)
        
        #LC_flux_e[LC_flux_e < 0] = 0.0
        
        LC_flux_per_mass.append(LC_flux_e)
        LC_contributions[j] = integrate.trapezoid(LC_flux_e, orb_ens)

    
    if not quiet:
        print('LC Flux per Mass computation complete.')
        print()


    LC_flux_per_mass = np.array(LC_flux_per_mass)
    
    if EHS:
        EHS_flag = np.zeros(len(masses)).astype(bool)
        Hill_masses = get_Hills_mass(masses)
        EHS_flag[M_BH<=Hill_masses] = True
    else:
        EHS_flag = np.ones(len(masses)).astype(bool)
    
    PDMF = get_PDMF(masses)
    TDE_rate_full = integrate.trapezoid(LC_contributions[EHS_flag]*PDMF[EHS_flag], masses[EHS_flag]/M_SOL)*SEC_PER_YR

    all_output = {'name': [name],
                  'radii': [r_ls],
                  'dens': [rho_r_init],
                  'M_BH': [M_BH],
                  'TDE_rate_solar': [TDE_rate_solar],
                  'TDE_rate_full': [TDE_rate_full],
                  'orb_ens': [orb_ens],
                  'DF': [DF],
                  'q': [q],
                  'LC_flux_solar': [LC_flux_solar],
                  'masses': [masses],
                  'LC_contributions_per_mass': [LC_contributions],
                  'LC_flux_per_mass': [LC_flux_per_mass],
                  'psi_rads': [r_ls],
                  'psi_tot': [psi_r_init],
                  'psi_bh': [psi_bh_init],
                  'psi_enc': [psi_enc_init],
                  'psi_ext': [psi_ext_init],
                  'DF_integrals': [integrals],
                  'mu_integrals': [int_fac_rs],
                  'periods': [periods_e],
                  'mu': [mu_e],
                  'R_LC': [R_LC_e],
                  'bw_cusp': [bw_cusp],
                  'bw_rad': [bw_rad],
                  'sflag': [sflag],
                  's': [s],
                  'EHS':[EHS]}

    output_table = Table(all_output)

    return output_table

###############################################################################
###############################################################################
###############################################################################
# =============================================================================
# ======================== END OF MAIN FUNCTION ===============================
# =============================================================================
###############################################################################
###############################################################################
###############################################################################



def run_reptide(in_table, analytic, n_energies=1000, EHS=False):
    
    print()
    print('Beginning TDE rate computation...')
    print()
    start = time.time()
    
    if analytic:
        output_table = get_TDE_rate_analytic(in_table[0][0],in_table[0][1],in_table[0][2],
                                                 in_table[0][3],in_table[0][4],in_table[0][5],
                                                 in_table[0][6],in_table[0][7],in_table[0][8],
                                                 in_table[0][9],in_table[0][10],in_table[0][11],
                                                 n_energies, EHS)
            
                
    else:
        output_table = get_TDE_rate_discrete(in_table[0][0],in_table[0][1],in_table[0][2],
                                                 in_table[0][3],in_table[0][4],in_table[0][5],
                                                 in_table[0][6],in_table[0][7],in_table[0][8],
                                                 in_table[0][9],in_table[0][10],
                                                 n_energies,EHS)

    
    end = time.time()
    print()    
    print('Done.')
    print()
    print('Runtime: {:.2f} minutes / {:.2f} hours'.format(round(end-start,3)/60,
                                                          round(end-start,3)/3600))
    print()

    return output_table



