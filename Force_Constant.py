# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:51:27 2021

@author: YZ
"""
import numpy as np
import sys
import phonopy
from sklearn.neighbors import Ne
import time
kev2ev = 0.00008617328149741
aum = 1.6605390666e-27
ev2J = 1.60218e-19
scale_f = ev2J/(1e-10**2)/aum

def cross_corr_funs(input_dat, dn =0):
    """
    input_dat: velocity, force, position from lammps dump file
             its shale should be [timesteps, number of atom, and directions(xyz)]
	dn: the step for correlation, dn = 0 for selfcorrelation

    """
    n_step, n_atom,_ = input_dat.shape
    result = np.zeros([n_atom*3, n_atom*3])

    for i_time in range(n_step):
        d_arr1 = input_dat[i_time]
        d_arr_tot1 = d_arr1.sum(0)
        d_arr1 -= d_arr_tot1/n_atom
        d_arr1 = d_arr1.ravel()
        j_time = i_time +dn
        if j_time >= n_step:
            j_time -= n_step
        d_arr2 = input_dat[j_time]
        d_arr_tot2 = d_arr2.sum(0)
        d_arr2 -= d_arr_tot2/n_atom
        d_arr2 = d_arr2.ravel()
        result += np.outer(d_arr1, d_arr2)
    result /= n_step
    return result

def scale(para):
    mass = 28.0855
    if para == 'velocity':
        result = mass*0.0001036472073616271
    elif para == 'force':
        result = 1
    elif para == 'displacement':
        result == 1
    return result


def main_run(para, T, dn):
    in_dat =  np.load('%s_T%s.npy'%(para, T))
    matRenorm = cross_corr_funs(in_dat, dn)
    matRenorm = matRenorm/(kev2ev*float(T))*scale(para)
    np.save('%s_CorrRenormalized_%s_%s'%(para,T,dn), matRenorm)
    return 0
	


def main():
    main_run(sys.argv[1], sys.argv[2], sys.argv[3])



if __name__ == '__main__':
    main()
