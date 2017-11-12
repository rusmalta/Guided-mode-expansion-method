# -*- coding: utf-8 -*-
"""
Module for 1D photonic crystal slab: GME - eigenvalue problem
"""
import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import scipy as sc

import mod_eff_w as wg
import mod_eps as fourier

####Part 2:  Parameters of basis set of guided modes: constants, normalization

#function to find electromagnetic field coefficients (TE polarization) for given g and omega
def Coeff_TE(hi1, hi3, q, g, epsilon):

    A2 = (q - 1j * hi1) / (2. * q) * np.exp(1j * q / 2.)
    B2 = (q + 1j * hi1) / (2. * q) * np.exp(-1j * q / 2.)
    A3 = ((q * (hi3 - hi1) * np.cos(q) 
          + (q ** 2. + hi1 * hi3) * np.sin(q)) / (2. * q * hi3))
    
    #normalization condition
    B_term1 = (hi1 ** 2. + g ** 2.) / (2. * hi1)
    B_term2 = (hi3 ** 2. + g ** 2.) / (2. * hi3) * abs(A3) ** 2.
    B_term3 = (g ** 2. + q ** 2.) * (abs(A2) ** 2. + abs(B2) ** 2.)
    B_term4 = (g ** 2. - q ** 2.) * (np.conj(A2) * B2 + np.conj(B2) * A2) * np.sin(q)/q
    B1 = 1 / np.sqrt( B_term1 + B_term2 + B_term3 + B_term4)
    
    A2 = A2 * B1
    B2 = B2 * B1
    A3 = A3 * B1
    res = [A2, A3, B1, B2]
    
    return res

#function to find electromagnetic field coefficients (TM polarization) for given g and omega
def Coeff_TM(hi1, hi3, q, epsilon):
    q = q / epsilon[1]
    hi1 = hi1 / epsilon[0]
    hi3 = hi3 / epsilon[2]
    C2 = (q - 1j * hi1) / (2. * q) * np.exp(1j * q * epsilon[1] / 2.)
    D2 = (q + 1j * hi1) / (2. * q) * np.exp(-1j * q * epsilon[1] / 2.)
    C3 = ((q * (hi3 - hi1) * np.cos(q * epsilon[1])+
        (q ** 2. + hi1 * hi3) * np.sin(q * epsilon[1])) / (2. * q * hi3))
    
    #normalization condition 
    D_term1 = 1 / (2. * hi1 * epsilon[0])
    D_term2 = 1 / (2. * hi3 * epsilon[2]) * abs(C3) ** 2.
    D_term3 = abs(C2) ** 2. + abs(D2) ** 2.
    D_term4 = (np.conj(C2) * D2 + np.conj(D2) * C2) * np.sin(q * epsilon[1])/(q * epsilon[1])
    D1 = 1 / np.sqrt(D_term1 + D_term2 + D_term3 + D_term4)
    
    C2 = C2 * D1
    D2 = D2 * D1
    C3 = C3 * D1
    res = [C2, C3, D1, D2]

    return res


####Part 3:  Matrix elements for eigenvalue problem

##Part 3.1: reindexing for further convenience

# mu_TE and mu_TM : functions that
# (a) change indexing from (g,alpha) to mu=(g,m1,m2,alpha)
#    Here m1 numerates reciprocal wave vectors in k-space in x-direction, 
#    m2 numerates reciprocal wave vectors in k-space in y-direction.
#    Array of (m1,m2) sets a grid of vectors all of which are equivalent 
#    to chosen point K in the first Brillouin zone.
# (b) truncate frequencies that correspond to leaky modes. 
#    mu=(g,m1,m2,alpha) contains only guided nonleaky modes

def mu_TE(K, n_PW, N_alpha, G_grid, epsilon):
    
    n_size = (n_PW + 1) 
    omega = np.zeros((2, (N_alpha) * n_size))
    k = 0  #counter
    for i1 in range(0, n_PW + 1):
            g = abs(G_grid[i1] + K) 
            x = wg.Dispersion_TE(g, N_alpha, epsilon)
            x_min = g / np.sqrt(epsilon[1])
            x_max = g / max(np.sqrt(epsilon[0]), np.sqrt(epsilon[2]))
            for i in range(0, N_alpha):
                if ((x[i] > x_min) & (x[i] < x_max)):
                    omega[0, k] = x[i]
                    omega[1, k] = i1
                    k = k + 1
    
    mu=np.zeros((2, k))
    for i in range(0, 2):
        mu[i, :] = omega[i,0 : k]

    return mu


def mu_TM(K, n_PW, N_alpha, G_grid, epsilon):
    
    n_size = (n_PW + 1) 
    omega = np.zeros((2, (N_alpha) * n_size))
    k = 0  #counter
    for i1 in range(0,n_PW+1):
            g = abs(G_grid[i1] + K) 
            x = wg.Dispersion_TM(g, N_alpha, epsilon)
            x_min = g / np.sqrt(epsilon[1])
            x_max = g / max(np.sqrt(epsilon[0]), np.sqrt(epsilon[2]))
            for i in range(0, N_alpha):
                if ((x[i] > x_min) & (x[i] < x_max)):
                    omega[0, k] = x[i]
                    omega[1, k] = i1
                    k = k + 1
    
    mu=np.zeros((2, k))
    for i in range(0, 2):
        mu[i, :] = omega[i,0 : k]

    return mu

####Part 3.2: calculation of average permittivities of effective waveguide and Fourier of dielectric function

def Grid(n_PW, G_basis):
    
    G = G_basis
    
    G_grid = np.zeros(n_PW + 1)
    
    i = 0
    for k in range(-int(n_PW / 2), int(n_PW / 2) + 1):
            G_grid[i] = k * G
            i = i + 1
        
    return G_grid
 

def H_TE_TE(omega_TE, g_TE, eta, epsilon, C_TE, hi1_TE, hi3_TE, q_TE):
  
    e_unit = g_TE / abs(g_TE)
    e_1, e_2 = np.meshgrid(e_unit, e_unit, indexing = 'ij')
    e_inner = e_1 * e_2
 
    hi1_1, hi1_2 = np.meshgrid(hi1_TE, hi1_TE, indexing = 'ij')
    hi3_1, hi3_2 = np.meshgrid(hi3_TE, hi3_TE, indexing = 'ij')
    q_1, q_2 = np.meshgrid(q_TE, q_TE, indexing = 'ij')
    
    I1 = 1 / (hi1_1 + hi1_2)
    I3 = 1 / (hi3_1 + hi3_2)
    I2p = np.sin(1 / 2. * (q_1 + q_2 + 1e-12)) / ((q_1 + q_2 + 1e-12) / 2.)
    I2m = np.sin(1 / 2. * (q_1 - q_2 + 1e-12)) / ((q_1 - q_2 + 1e-12) / 2.)

    A2_1, A2_2 = np.meshgrid(C_TE[0], C_TE[0], indexing = 'ij'); 
    A3_1, A3_2 = np.meshgrid(C_TE[1], C_TE[1], indexing = 'ij');  
    B1_1, B1_2 = np.meshgrid(C_TE[2], C_TE[2], indexing = 'ij');  
    B2_1, B2_2 = np.meshgrid(C_TE[3], C_TE[3], indexing = 'ij');
    
    omega_1, omega_2 = np.meshgrid(omega_TE, omega_TE, indexing = 'ij')

    fact = omega_1 ** 2 * omega_2 ** 2 * e_inner
    H_term1 = epsilon[0] ** 2 * eta[0] * np.conj(B1_1) * B1_2 * I1
    H_term2 = epsilon[2] ** 2 * eta[2] * np.conj(A3_1) * A3_2 * I3
    H_term3 = (epsilon[1] ** 2 * eta[1] * ((np.conj(A2_1) * A2_2 + np.conj(B2_1) * B2_2) * I2m + 
               (np.conj(A2_1) * B2_2 + np.conj(B2_1) * A2_2) * I2p))
    H = fact * (H_term1 + H_term2 + H_term3)
           
    return H

def H_TM_TM(omega_TM, g_TM, eta, epsilon, C_TM, hi1_TM, hi3_TM, q_TM):
    
    g_1, g_2 = np.meshgrid(abs(g_TM), abs(g_TM), indexing = 'ij')
    
    g_1_proj, g_2_proj = np.meshgrid(g_TM / abs(g_TM), g_TM / abs(g_TM), indexing = 'ij')
    g_inner = g_1_proj * g_2_proj
 
    hi1_1, hi1_2 = np.meshgrid(hi1_TM, hi1_TM, indexing = 'ij')
    hi3_1, hi3_2 = np.meshgrid(hi3_TM, hi3_TM, indexing = 'ij')
    q_1, q_2 = np.meshgrid(q_TM, q_TM, indexing = 'ij')
    
    I1 = 1 / (hi1_1 + hi1_2)
    I3 = 1 / (hi3_1 + hi3_2)
    I2p = np.sin(1 / 2. * (q_1 + q_2 + 1e-12)) / ((q_1 + q_2 + 1e-12) / 2.)
    I2m = np.sin(1 / 2. * (q_1 - q_2 + 1e-12)) / ((q_1 - q_2 + 1e-12) / 2.)
    
    C2_1, C2_2 = np.meshgrid(C_TM[0], C_TM[0], indexing = 'ij'); 
    C3_1, C3_2 = np.meshgrid(C_TM[1], C_TM[1], indexing = 'ij');  
    D1_1, D1_2 = np.meshgrid(C_TM[2], C_TM[2], indexing = 'ij');  
    D2_1, D2_2 = np.meshgrid(C_TM[3], C_TM[3], indexing = 'ij');
    
    omega_1, omega_2 = np.meshgrid(omega_TM, omega_TM, indexing = 'ij')
    
    H_term1 = eta[0] * np.conj(D1_1) * D1_2 * (hi1_1 * hi1_2 * g_inner + g_1 * g_2) * I1
    H_term2 = eta[2] * np.conj(C3_1) * C3_2 * (hi3_1 * hi3_2 * g_inner + g_1 * g_2) * I3
    H_term3 = eta[1] * ((np.conj(C2_1) * C2_2 + np.conj(D2_1) * D2_2) *
                        (q_1 * q_2 * g_inner + g_1 * g_2) * I2m)
    H_term4 = eta[1] * ((np.conj(C2_1) * D2_2 + np.conj(D2_1) * C2_2) *
                        (-q_1 * q_2 * g_inner + g_1 * g_2) * I2p)
    H = H_term1 + H_term2 + H_term3 + H_term4
            
    return H  
  
    
#density of leaky modes of effective waveguide   
def DOS(omega, g, epsilon):
    x = omega ** 2. - g ** 2. / epsilon
    if ( abs(x) >= 0) :
        rho = np.sqrt(epsilon) / (4. * np.pi) / np.sqrt(x)
    else:
        rho = 0. + 0 * 1j
    return rho
    
def Grid_loss(G_basis, K, omega, epsilon):
    
    G = G_basis
    
    k1_max = int(( omega * np.sqrt(epsilon) - K) / G ) 
    k1_min = int((-omega * np.sqrt(epsilon) - K) / G) 
    
    G_grid = np.zeros((k1_max - k1_min + 1))
    
    i = 0
    
    for k in range(k1_min, k1_max + 1):
            G_grid[i] = k * G
            i = i + 1
    
    return G_grid  
    
def r_index(G_grid_l):  
    
    size = len(G_grid_l)
    r = np.zeros(size)
    k = 0  #counter
    for i1 in range(0, size):
            r[k] = i1
            k = k + 1
            
    return r     
   
def Coeff_TE_low(q1_nu, q2_nu, q3_nu, epsilon):
    q1 = q1_nu
    q2 = q2_nu
    q3 = q3_nu
    ex = np.exp(1j * q2 / 2.)
    
    T1_00 = (q2 + q1) * ex / (2. * q2) 
    T1_01 = (q2 - q1) * ex / (2. * q2) 
    T1_10 = (q2 - q1) * np.conj(ex) / (2. * q2) 
    T1_11 = (q2 + q1) * np.conj(ex) / (2. * q2)
    
    T2_00 = (q3 + q2) * ex / (2. * q3)
    T2_01 = (q3 - q2) * np.conj(ex) / (2. * q3) 
    T2_10 = (q3 - q2) * ex / (2. * q3)
    T2_11 = (q3 + q2) * np.conj(ex) / (2. * q3)
    
    T_00 = T2_00 * T1_00 + T2_01 * T1_10 
    T_01 = T2_00 * T1_01 + T2_01 * T1_11 
    
    N1_1 = 1. / np.sqrt(epsilon[0])
    N1_0 = - T_01 / T_00 * N1_1    
    N2_0 = T1_00 * N1_0 + T1_01 * N1_1
    N2_1 = T1_10 * N1_0 + T1_11 * N1_1
    N3_0 = 0.  
    N3_1 = T2_10 * N2_0 + T2_11 * N2_1
    
#    print (N1_0, N2_0, N3_0, N1_1, N2_1, N3_1)
    return [[N1_0, N2_0, N3_0], [N1_1, N2_1, N3_1]]

def Coeff_TE_up(q1_nu, q2_nu, q3_nu, epsilon): 
      
    q1 = q1_nu
    q2 = q2_nu
    q3 = q3_nu
    ex = np.exp(1j * q2 / 2.)
  
    T1_00 = (q2 + q1) * np.conj(ex) / (2. * q1) 
    T1_01 = (q1 - q2) * ex / (2. * q1) 
    T1_10 = (q1 - q2) * np.conj(ex) / (2. * q1) 
    T1_11 = (q2 + q1) * ex / (2. * q1)
    
    T2_00 = (q3 + q2) * np.conj(ex) / (2. * q2)
    T2_01 = (q2 - q3) * np.conj(ex) / (2. * q2) 
    T2_10 = (q2 - q3) * ex / (2. * q2)
    T2_11 = (q3 + q2) * ex / (2. * q2)
    
    T_10 = T1_10 * T2_00 + T1_11 * T2_10 
    T_11 = T1_10 * T2_01 + T1_11 * T2_11 
    
    N3_0 = 1. / np.sqrt(epsilon[2])
    N3_1 = - T_10 / T_11 * N3_0  

    N2_0 = T2_00 * N3_0 + T2_01 * N3_1
    N2_1 = T2_10 * N3_0 + T2_11 * N3_1
    N1_0 = T1_00 * N2_0 + T1_01 * N2_1
    N1_1 = 0.  
   
    return [[N1_0, N2_0, N3_0], [N1_1, N2_1, N3_1]]
    
def Coeff_TM_low(q1_nu,q2_nu,q3_nu,epsilon):

    q1 = q1_nu / epsilon[0]
    q2 = q2_nu / epsilon[1]
    q3 = q3_nu / epsilon[2]
    ex = np.exp(1j * epsilon[1] * q2 / 2.)
   
    T1_00 = (q2 + q1) * ex / (2. * q2) 
    T1_01 = (q2 - q1) * ex / (2. * q2) 
    T1_10 = (q2 - q1) * np.conj(ex) / (2. * q2) 
    T1_11 = (q2 + q1) * np.conj(ex) / (2. * q2)
    
    T2_00 = (q3 + q2) * ex / (2. * q3)
    T2_01 = (q3 - q2) * np.conj(ex) / (2. * q3) 
    T2_10 = (q3 - q2) * ex / (2. * q3)
    T2_11 = (q3 + q2) * np.conj(ex) / (2. * q3)
    
    T_00 = T2_00 * T1_00 + T2_01 * T1_10 
    T_01 = T2_00 * T1_01 + T2_01 * T1_11  
    
    N1_1 = 1. 
    N1_0 = - T_01 / T_00 * N1_1    
    N2_0 = T1_00 * N1_0 + T1_01 * N1_1
    N2_1 = T1_10 * N1_0 + T1_11 * N1_1
    N3_0 = 0.  
    N3_1 = T2_10 * N2_0 + T2_11 * N2_1
   
    return [[N1_0, N2_0, N3_0], [N1_1, N2_1, N3_1]]

def Coeff_TM_up(q1_nu, q2_nu, q3_nu, epsilon):
    
    q1 = q1_nu / epsilon[0]
    q2 = q2_nu / epsilon[1]
    q3 = q3_nu / epsilon[2]
    ex = np.exp(1j * epsilon[1] * q2 / 2.)
  
    T1_00 = (q2 + q1) * np.conj(ex) / (2. * q1) 
    T1_01 = (q1 - q2) * ex / (2. * q1) 
    T1_10 = (q1 - q2) * np.conj(ex) / (2. * q1) 
    T1_11 = (q2 + q1) * ex / (2. * q1)
    
    T2_00 = (q3 + q2) * np.conj(ex) / (2. * q2)
    T2_01 = (q2 - q3) * np.conj(ex) / (2. * q2) 
    T2_10 = (q2 - q3) * ex / (2. * q2)
    T2_11 = (q3 + q2) * ex / (2. * q2)
     
    T_10 = T1_10 * T2_00 + T1_11 * T2_10 
    T_11 = T1_10 * T2_01 + T1_11 * T2_11 
    
    N3_0 = 1. 
    N3_1 = - T_10 / T_11 * N3_0  

    N2_0 = T2_00 * N3_0 + T2_01 * N3_1
    N2_1 = T2_10 * N3_0 + T2_11 * N3_1
    N1_0 = T1_00 * N2_0 + T1_01 * N2_1
    N1_1 = 0.  
   
    return [[N1_0, N2_0, N3_0], [N1_1, N2_1, N3_1]]


 
def H_TE_TE_los(omega_mu, omega_nu, epsilon, eta, g_mu, g_nu, hi1_mu, hi3_mu, q_mu, q1_nu, q2_nu, q3_nu, C_mu, C_nu):

    e_1, e_2 = np.meshgrid(g_mu / abs(g_mu), g_nu / abs(g_nu), indexing = 'ij')
    e_inner = e_1 * e_2
 
    hi1_1, q1_2 = np.meshgrid(hi1_mu, q1_nu, indexing = 'ij')
    hi3_1, q3_2 = np.meshgrid(hi3_mu, q3_nu, indexing = 'ij')
    q_1, q2_2 = np.meshgrid(q_mu, q2_nu, indexing = 'ij') 
        
    I1p = 1 / (hi1_1 + 1j * q1_2)
    I1m = 1 / (hi1_1 - 1j * q1_2)
    I3p = 1 / (hi3_1 + 1j * q3_2)
    I3m = 1 / (hi3_1 - 1j * q3_2)
    I2p = np.sin(1 / 2. * (q_1 + q2_2 + 1e-25)) / ((q_1 + q2_2 + 1e-25) / 2.)
    I2m = np.sin(1 / 2. * (q_1 - q2_2 + 1e-25)) / ((q_1 - q2_2 + 1e-25) / 2.)
    
    
    A2_1, W2_2 = np.meshgrid(C_mu[0], C_nu[0][1], indexing = 'ij'); 
    A3_1, W3_2 = np.meshgrid(C_mu[1], C_nu[0][2], indexing = 'ij');
    A3_1, X3_2 = np.meshgrid(C_mu[1], C_nu[1][2], indexing = 'ij');  
    B1_1, W1_2 = np.meshgrid(C_mu[2], C_nu[0][0], indexing = 'ij');  
    B1_1, X1_2 = np.meshgrid(C_mu[2], C_nu[1][0], indexing = 'ij');  
    B2_1, X2_2 = np.meshgrid(C_mu[3], C_nu[1][1], indexing = 'ij');
    
    omega_1, omega_2 = np.meshgrid(omega_mu, g_nu, indexing = 'ij')

    fact = (omega_1) ** 2. * omega_nu * e_inner
    H_term1 = epsilon[0] ** 2. * eta[0] * np.conj(B1_1) * (W1_2 * I1p + X1_2 * I1m)
    H_term2 = epsilon[2] ** 2. * eta[2] * np.conj(A3_1) * (W3_2 * I3m + X3_2 * I3p)
    H_term3 = epsilon[1] ** 2. * eta[1] * (I2m * (np.conj(A2_1) * W2_2 + np.conj(B2_1) * X2_2)
                                           + I2p * (np.conj(A2_1) * X2_2 + np.conj(B2_1) * W2_2))
    H = fact * (H_term1 + H_term2 + H_term3)
           
    return H

def H_TM_TM_los(omega_mu, omega_nu, epsilon, eta, g_mu, g_nu, hi1_mu, hi3_mu, q_mu, q1_nu, q2_nu, q3_nu, C_mu, C_nu):
    
    g_1, g_2 = np.meshgrid(abs(g_mu), abs(g_nu), indexing = 'ij')
    
    g_1_proj, g_2_proj = np.meshgrid(g_mu / abs(g_mu), g_nu / abs(g_nu), indexing = 'ij')
    g_inner = g_1_proj * g_2_proj
    
    hi1_1, q1_2 = np.meshgrid(hi1_mu, q1_nu, indexing = 'ij')
    hi3_1, q3_2 = np.meshgrid(hi3_mu, q3_nu, indexing = 'ij')
    q_1, q2_2 = np.meshgrid(q_mu, q2_nu, indexing = 'ij')
    
    I1p = 1 / (hi1_1 + 1j * q1_2)
    I1m = 1 / (hi1_1 - 1j * q1_2)
    I3p = 1 / (hi3_1 + 1j * q3_2)
    I3m = 1 / (hi3_1 - 1j * q3_2)
    I2p = np.sin(1 / 2. * (q_1 + q2_2 + 1e-12)) / ((q_1 + q2_2 + 1e-12) / 2.)
    I2m = np.sin(1 / 2. * (q_1 - q2_2 + 1e-12)) / ((q_1 - q2_2 + 1e-12) / 2.)
    
    C2_1, Y2_2 = np.meshgrid(C_mu[0], C_nu[0][1], indexing = 'ij'); 
    C3_1, Y3_2 = np.meshgrid(C_mu[1], C_nu[0][2], indexing = 'ij');
    C3_1, Z3_2 = np.meshgrid(C_mu[1], C_nu[1][2], indexing = 'ij');  
    D1_1, Y1_2 = np.meshgrid(C_mu[2], C_nu[0][0], indexing = 'ij');  
    D1_1, Z1_2 = np.meshgrid(C_mu[2], C_nu[1][0], indexing = 'ij');  
    D2_1, Z2_2 = np.meshgrid(C_mu[3], C_nu[1][1], indexing = 'ij');

    H_term1 = eta[0] * np.conj(D1_1) * (Y1_2 * I1p * (g_1 * g_2 + 1j * hi1_1 * q1_2 * g_inner) + 
                                        Z1_2 * I1m * (g_1 * g_2 - 1j * hi1_1 * q1_2 * g_inner))
    
    H_term2 = eta[2] * np.conj(C3_1) * (Y3_2 * I3m * (g_1 * g_2 - 1j * hi3_1 * q3_2 * g_inner) +
                                        Z3_2 * I3p * (g_1 * g_2 + 1j * hi3_1 * q3_2 * g_inner))
    
    H_term3 = eta[1] * (I2m * (np.conj(C2_1) * Y2_2 + np.conj(D2_1) * Z2_2) * (g_1 * g_2 + q_1 * q2_2 * g_inner)+
                        I2p * (np.conj(C2_1) * Z2_2 + np.conj(D2_1) * Y2_2) * (g_1 * g_2 - q_1 * q2_2 * g_inner))                                
    H = H_term1 + H_term2 + H_term3
           
    return H


#Eigenproblem
    
    
def Eig_Solve(K, n_PW, N_alpha, G_basis, G_grid, a, L, eps_exact, epsilon, eta, N_eig):
    
    # TE polarization
    # TE polarization
    # TE polarization
    TE = mu_TE(K, n_PW, N_alpha, G_grid, epsilon) 
    size_TE = TE[0, :].size
    omega_TE = TE[0, :]
    g_TE = np.zeros(size_TE)
    eta_TE = np.zeros((3, size_TE, size_TE), dtype = complex)
    
    for n1 in range(0, size_TE):
        g_TE[n1] = G_grid[int(TE[1,n1])] + K
        for n2 in range(0, size_TE): 
            eta_TE[:, n1, n2] = eta[:, int(TE[1, n1]), int(TE[1,n2])]
    
    hi1_TE = np.sqrt(g_TE ** 2. - epsilon[0] * omega_TE ** 2.)
    hi3_TE = np.sqrt(g_TE ** 2. - epsilon[2] * omega_TE ** 2.)
    q_TE = np.sqrt(epsilon[1] * omega_TE ** 2. - g_TE ** 2.)    
    
    C_TE = Coeff_TE(hi1_TE, hi3_TE, q_TE, g_TE, epsilon)    

    H= H_TE_TE(omega_TE, g_TE, eta_TE, epsilon, C_TE, hi1_TE, hi3_TE, q_TE)  
    
    ##Eigensolver
    eigres_RE = sc.linalg.eigh(H, eigvals = (0, N_eig))
    
    #clear memory
    H = 0
    
    ####################################################
    omega = np.sqrt(eigres_RE[0])
    eigval_mu = eigres_RE[1]
    
    omega_re = np.zeros(N_eig)
    omega_im = np.zeros(N_eig)

    for N in range(0, N_eig):
        omega_nu = omega[N]
        
        eigval = np.conj(eigval_mu[:, N])
        
        #Additional grid to calculate the sum over leaky states
        G_grid_l = Grid_loss(G_basis, K, omega_nu, max(epsilon[0], epsilon[2]))
        
        size_1 = len(G_grid)
        size_l_1 = len(G_grid_l)
        
        nu = r_index(G_grid_l)
        size_nu = nu.size
        
        eta_l = np.zeros((3, size_1, size_l_1), dtype = complex)
        eta_l[0] = fourier.dielectric_matrix(G_grid, G_grid_l, 1 / eps_exact[0], L , a)
        eta_l[1] = fourier.dielectric_matrix(G_grid, G_grid_l, 1 / eps_exact[1], L , a)
        eta_l[2] = fourier.dielectric_matrix(G_grid, G_grid_l, 1 / eps_exact[2], L , a) 
        
        rho_low = np.zeros((size_nu), dtype = complex)
        rho_up = np.zeros((size_nu), dtype = complex)
        
                 
        g_nu = np.zeros(size_nu, dtype = complex)
        for n2 in range(0, size_nu):
            g_nu[n2] = complex(G_grid_l[int(nu[n2])] + K)
        
        q1_nu = np.sqrt(- g_nu ** 2. + epsilon[0] * omega_nu ** 2.)
        q2_nu = np.sqrt(- g_nu ** 2. + epsilon[1] * omega_nu ** 2.)
        q3_nu = np.sqrt(- g_nu ** 2. + epsilon[2] * omega_nu ** 2.) 
        
        eta_l_TE = np.zeros((3, size_TE, size_nu), dtype = complex)
        
        for n2 in range(0, size_nu):
            rho_low[n2] = DOS(omega_nu, g_nu[n2], epsilon[0])
            rho_up[n2] = DOS(omega_nu, g_nu[n2], epsilon[2])
            for n1 in range(0, size_TE):
                eta_l_TE[:, n1, n2]  = eta_l[:, int(TE[1, n1]) , int(nu[n2])]
                                  
        C_nu_1_TE = Coeff_TE_low(q1_nu, q2_nu, q3_nu, epsilon)
        C_nu_3_TE = Coeff_TE_up(q1_nu, q2_nu, q3_nu, epsilon)
        
        
        eigval = eigval.reshape(len(eigval), 1)        
        
        H = eigval * H_TE_TE_los(omega_TE, omega_nu, epsilon, eta_l_TE, g_TE, g_nu, hi1_TE, hi3_TE, q_TE, q1_nu, q2_nu, q3_nu, C_TE, C_nu_1_TE)
        
        H_term1 = sum(abs(sum(H)) ** 2. * rho_low)

        H = eigval * H_TE_TE_los(omega_TE, omega_nu, epsilon, eta_l_TE, g_TE, g_nu, hi1_TE, hi3_TE, q_TE, q1_nu, q2_nu, q3_nu, C_TE, C_nu_3_TE)
        
        H_term2 = sum(abs(sum(H)) ** 2. * rho_up)
        
        fact = -1. / (2. * omega_nu) * np.pi 
        
        omega_im[N] = abs(np.real(fact * (H_term1 + H_term2)))
        omega_re[N] = omega_nu


#
#    ## TM polarization
#    ## TM polarization
#    ## TM polarization
#    
#    TM = mu_TM(K, n_PW, N_alpha, G_grid, epsilon) 
#    size_TM = TM[0, :].size
#    omega_TM = TM[0, :]
#    eta_TM = np.zeros((3, size_TM, size_TM), dtype = complex)
#    g_TM = np.zeros(size_TM)
#    
#    for n1 in range(0, size_TM):
#        g_TM[n1] = G_grid[int(TM[1,n1])] + K
#        for n2 in range(0, size_TM): 
#            eta_TM[:, n1, n2] = eta[:, int(TM[1, n1]), int(TM[1,n2])]
#
#    hi1_TM = np.sqrt(g_TM ** 2. - epsilon[0] * omega_TM ** 2.)
#    hi3_TM = np.sqrt(g_TM ** 2. - epsilon[2] * omega_TM ** 2.)
#    q_TM = np.sqrt(epsilon[1] * omega_TM ** 2. - g_TM ** 2.)  
#    
#    C_TM = Coeff_TM(hi1_TM, hi3_TM, q_TM, epsilon)    
#            
#    H = H_TM_TM(omega_TM, g_TM, eta_TM, epsilon, C_TM, hi1_TM, hi3_TM, q_TM)        
#
#    ##Eigensolver
#    eigres_RE = sc.linalg.eigh(H, eigvals = (0, N_eig))
#
#    #clear memory
#    H = 0
#    
#      
#    ####################################################
#    omega = np.sqrt(eigres_RE[0])
#    eigval_mu = eigres_RE[1]
#    
#    omega_re = np.zeros(N_eig)
#    omega_im = np.zeros(N_eig)
#
#    for N in range(0, N_eig):
#        omega_nu = omega[N]
#        eigval = np.conj(eigval_mu[:, N])
#
#        #Additional grid to calculate the sum over leaky states
#        G_grid_l = Grid_loss(G_basis, K, omega_nu, max(epsilon[0], epsilon[2]))
#        
#        size_1 = len(G_grid)
#        size_l_1 = len(G_grid_l)
#        
#        nu = r_index(G_grid_l)
#        size_nu = nu.size
#        
#        eta_l = np.zeros((3, size_1, size_l_1), dtype = complex)
#        eta_l[0] = fourier.dielectric_matrix(G_grid, G_grid_l, 1 / eps_exact[0], L , a)
#        eta_l[1] = fourier.dielectric_matrix(G_grid, G_grid_l, 1 / eps_exact[1], L , a)
#        eta_l[2] = fourier.dielectric_matrix(G_grid, G_grid_l, 1 / eps_exact[2], L , a) 
#        
#        rho_low = np.zeros((size_nu), dtype = complex)
#        rho_up = np.zeros((size_nu), dtype = complex)
#        
#                 
#        g_nu = np.zeros(size_nu, dtype = complex)
#        for n2 in range(0, size_nu):
#            g_nu[n2] = complex(G_grid_l[int(nu[n2])] + K)
#        
#        q1_nu = np.sqrt(- g_nu ** 2. + epsilon[0] * omega_nu ** 2.)
#        q2_nu = np.sqrt(- g_nu ** 2. + epsilon[1] * omega_nu ** 2.)
#        q3_nu = np.sqrt(- g_nu ** 2. + epsilon[2] * omega_nu ** 2.) 
#        
#        eta_l_TM = np.zeros((3, size_TM, size_nu), dtype = complex)
#        
#        for n2 in range(0, size_nu):
#            rho_low[n2] = DOS(omega_nu, g_nu[n2], epsilon[0])
#            rho_up[n2] = DOS(omega_nu, g_nu[n2], epsilon[2])   
#            for n1 in range(0, size_TM):
#                eta_l_TM[:, n1, n2]  = eta_l[:, int(TM[1, n1]) , int(nu[n2])] 
#                                  
#
#        C_nu_1_TM = Coeff_TM_low(q1_nu, q2_nu, q3_nu, epsilon)
#        C_nu_3_TM = Coeff_TM_up(q1_nu, q2_nu, q3_nu, epsilon)
#        
#        eigval = eigval.reshape(len(eigval), 1)        
#           
#        H = eigval * H_TM_TM_los(omega_TM, omega_nu, epsilon, eta_l_TM, g_TM, g_nu, hi1_TM, hi3_TM, q_TM, q1_nu, q2_nu, q3_nu, C_TM, C_nu_1_TM)
#        
#        H_term1 = sum(abs(sum(H)) ** 2. * rho_low)
#      
#        H = eigval * H_TM_TM_los(omega_TM, omega_nu, epsilon, eta_l_TM, g_TM, g_nu, hi1_TM, hi3_TM, q_TM, q1_nu, q2_nu, q3_nu, C_TM, C_nu_3_TM)
#        
#        H_term2 = sum(abs(sum(H)) ** 2. * rho_up)
#        
#        fact = -1. / (2. * omega_nu) * np.pi 
#        
#        omega_im[N] = abs(np.real(fact * (H_term1 + H_term2)))
#        omega_re[N] = omega_nu
    
    
    
    return omega_re, omega_im   
    


   