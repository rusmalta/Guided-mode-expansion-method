# -*- coding: utf-8 -*-
"""
Module for 2D photonic crystal slab: Permittivity Fourier - SQUARE LATTICE + CIRCULAR HOLES
"""
import numpy as np
import cmath
# Function calculates Fourier of dielectric function for 1D lattice
# Particularly, it finds average permittivies using actual ones 
     
def dielectric(eps_exact, G, L, a):
    
    f = (L / a)   #filling factor     
    
    g = L * G / 2
   
    if ((G == 0)):
        epsilon = f * eps_exact[0] + ( 1 - f) * eps_exact[1]
    else:
        epsilon = f *(eps_exact[0] - eps_exact[1]) * cmath.exp(-1j * g) * cmath.sin(g) / g
        
    return epsilon

##function calculates matrix of permittivities
def dielectric_matrix(G_grid, G_grid_l, eps_exact, L, a):
    
    i1_max = len(G_grid)
    j1_max = len(G_grid_l)
    
    epsilon = np.zeros((i1_max, j1_max), dtype = complex)

    for i1 in range(0, i1_max):
            for j1 in range(0, j1_max):
                    dG = G_grid[i1] - G_grid_l[j1]
                    epsilon[i1, j1] = dielectric(eps_exact, dG, L, a)
    
    return epsilon
   