# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:41:02 2024

@author: fer
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def fit_linear(x,E,eE):
    def pol(x,a,b):
        return a+b*x   
    popt, pcov = curve_fit(pol, x,E,sigma=eE,absolute_sigma=True)
    ne=np.sqrt(pcov[0,0])
    nez=10**(-int(np.log10(ne))+2)*ne
    me=np.sqrt(pcov[1,1])
    mez=10**(-int(np.log10(me))+2)*me
    print('Linear fit:',popt[0],'+-',ne,popt[1],'+-',me,'*x' )
    plt.errorbar(x,E,yerr=eE,fmt='o')
    DP=np.linspace(x[0], x[-1], num=100)
    plt.plot(DP,pol(DP,popt[0],popt[1]),label='y=%.5g$ (%.2g) -  %.7g(%.2g)\cdot L $'%(popt[0],nez,-popt[1],mez))
    plt.legend( prop={'size': 7})
    plt.xlabel('L')
    plt.ylabel('Energy/A')
    #plt.savefig('Fit_beta_50_dir.pdf',bbox_inches='tight')
    plt.show()
    return popt,pcov
def plot_linear(x,E,eE,x2,E2,eE2):
    def pol(x,a,b):
        return a+b*x
    xtot=np.concatenate((x2, x),axis=None)
    Etot=np.concatenate((E2, E),axis=None)
    eEtot=np.concatenate((eE2, eE),axis=None)
    popt, pcov = curve_fit(pol, x,E*x,sigma=eE*x,absolute_sigma=True)
    ne=np.sqrt(pcov[0,0])
    nez=10**(-int(np.log10(ne))+2)*ne
    me=np.sqrt(pcov[1,1])
    mez=10**(-int(np.log10(me))+2)*me
    print('Linear fit:',popt[0],'+-',ne,popt[1],'+-',me,'*x' )
    plt.errorbar(xtot,Etot*xtot,yerr=eEtot*xtot,fmt='o')
    DP=np.linspace(xtot[0], xtot[-1], num=1000)
    plt.plot(DP,pol(DP,popt[0],popt[1]),label='y=%.5g$ (%.2g) -  %.7g(%.2g)\cdot L $'%(popt[0],nez,-popt[1],mez))
    plt.legend( prop={'size': 7})
    plt.xlabel('L')
    plt.ylabel('Energy/A')
    #plt.xlim([1.6, 4.4])
    #plt.ylim([-0.06, -0.015])
    plt.savefig('Fit_beta_50_dir_all.pdf',bbox_inches='tight')
    plt.show()
    return popt,pcov