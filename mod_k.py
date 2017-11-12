"""
Module for 2D photonic crystal slab: SQUARE BRAVAIS LATTICE (k-space)
"""


import numpy as np

def Lattice(a):
    
    ### Analysis of k-space
    #high-symmetry points:
    #1) G=[0,0]
    #2) X=[pi,0]

    #basis reciprocal vector
    G_basis = np.pi * 2 / a   

    return G_basis

