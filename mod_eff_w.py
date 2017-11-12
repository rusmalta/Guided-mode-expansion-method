@@ -0,0 +1,91 @@
# -*- coding: utf-8 -*-
"""
Module for 2D photonic crystal slab: dispersion of effective waveguide - For ALL MODES
"""
import os, sys
sys.path.append(os.path.dirname(__file__))


import numpy as np
import cmath
from scipy.optimize import brentq

####Part 1: calculation of effective waveguide spectrum

#function to find roots of dispersion equation for TE-polarization: k_transverse for given k_lateral
def Solve_disp_TE(x, g, epsilon):
    term1_l = (epsilon[1] - epsilon[0]) / epsilon[1] * g ** 2.
    term2_l = - epsilon[0] / epsilon[1] * x ** 2.
    term1_u = (epsilon[1] - epsilon[2]) / epsilon[1] * g ** 2.
    term2_u = - epsilon[2] / epsilon[1] * x ** 2.
    hi1 = cmath.sqrt(term1_l + term2_l) #k_transverse inside lower cladding
    hi3 = cmath.sqrt(term1_u + term2_u) #k_transverse inside upper cladding
    eq1 = x * (hi1 + hi3) * np.cos(x)
    eq2 = (hi1 * hi3 - x ** 2.) * np.sin(x)
    f = (eq1 + eq2) #dispersion equation
    return (f.real)

    
#function to find roots of dispersion equation for TM-polarization: k_transverse for given k_lateral  
def Solve_disp_TM(x, g, epsilon):
    term1_l = (epsilon[1] - epsilon[0]) / epsilon[1] * g ** 2.
    term2_l = - epsilon[0] / epsilon[1] * x ** 2.
    term1_u = (epsilon[1] - epsilon[2]) / epsilon[1] * g ** 2.
    term2_u = - epsilon[2] / epsilon[1] * x ** 2.
    hi1 = cmath.sqrt(term1_l + term2_l) #k_transverse inside lower cladding
    hi3 = cmath.sqrt(term1_u + term2_u) #k_transverse inside upper cladding
    eq1 = x / epsilon[1] * (hi1 / epsilon[0] + hi3 / epsilon[2]) * np.cos(x)
    eq2 = (hi1 / epsilon[0] * hi3 / epsilon[2] - x ** 2. / epsilon[1] ** 2.) * np.sin(x)
    f = (eq1 + eq2) #dispersion equation
    return (f.real)
    

#function to find array of N_alpha frequencies for given k-lateral wavevector (for TE-polarization)
def Dispersion_TE(g, N_alpha, epsilon):
  
    x_max = np.pi * (N_alpha + 2)
    x_step = 0.3 * np.pi
    x0 = np.arange(1e-6, x_max, x_step) #initial guess of transverse (quantized) wavevector
    
    temp = np.zeros(2 * (N_alpha + 1))
    q = np.zeros(N_alpha) #transverse (quantized) wavevector
    omega = np.zeros(N_alpha) #frequency
    
    k = 0 #counter
    for i in range(0, len(x0) - 1):
        #condition that boundaries of current k_lateral-interval must have different sings
        s1 = Solve_disp_TE(x0[i], g, epsilon)
        s2 = Solve_disp_TE(x0[i + 1], g, epsilon)
        if (s1 * s2 < 0):
            temp[k] = brentq(Solve_disp_TE, x0[i], x0[i + 1], args = (g, epsilon))
            k = k + 1

    q[:] = temp[0 : N_alpha]  #we need only N_alpha first roots, without lowest frequency which is aa[N_alpha] 
    
    omega = np.sqrt((q ** 2. + g ** 2.) / epsilon[1])     
    return omega
 
#function to find array of N_alpha frequencis for given k-lateral wavevector (for TM-polarization)   
def Dispersion_TM(g, N_alpha, epsilon):
  
    x_max = np.pi * (N_alpha + 2)
    x_step = 0.3 * np.pi
    x0 = np.arange(1e-6, x_max, x_step) #initial guess of transverse (quantized) wavevector
    
    temp = np.zeros(2 * (N_alpha + 1))
    q = np.zeros(N_alpha) #transverse (quantized) wavevector
    omega = np.zeros(N_alpha) #frequency
    
    k = 0 #counter
    for i in range(0, len(x0) - 1):
        #condition that boundaries of current k_lateral-interval must have different sings
        s1 = Solve_disp_TM(x0[i], g, epsilon)
        s2 = Solve_disp_TM(x0[i + 1], g, epsilon)
        if (s1 * s2 < 0):
            temp[k] = brentq(Solve_disp_TM, x0[i], x0[i+1], args = (g, epsilon))
            k = k + 1

    q[:] = temp[0 : N_alpha]  #we need only N_alpha first roots, without lowest frequency which is aa[N_alpha] 
    
    omega = np.sqrt((q ** 2. + g ** 2.) / epsilon[1])     
    return omega
