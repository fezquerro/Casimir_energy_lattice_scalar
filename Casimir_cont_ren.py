# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:24:14 2022

@author: fer

Casimir energy from the second-difference renormalization E(N_L)+E(N_L+2L0)-2E(N_L+L0),
periodic and Dirichlet in one figure: Casimir_scalar_renorm.pdf
"""
import matplotlib.pyplot as plt
from modules.energy_formulas import casimir_second_diff
from modules.plotting import flat, plot_continuum, symlog, save, LABELS, COLORS, MASSES

L_periodic=[96,96]#t and x dimensions of the lattice
L0=48
Lv=[range(12,25),range(8,25),range(8,25),range(8,25)]#N_L values for each mass

plt.figure()
for bound in ['per','dir']:
 C=[[casimir_second_diff(L_periodic,NL,L0,a,bound)/a**2 for NL in NLs] for a,NLs in zip(MASSES,Lv)]
 x=[[a*NL for NL in NLs] for a,NLs in zip(MASSES,Lv)]
 xdata=flat(x)
 plt.plot(xdata,flat(C),'o',color=COLORS[bound],label=LABELS[bound]+' lattice')
 dx=(max(xdata)-min(xdata))/100
 plot_continuum(min(xdata)-dx,max(xdata)+dx,2,bound,label=LABELS[bound]+' continuum',num=102,color=COLORS[bound])
plt.xlabel('$mL$',fontsize=14)
plt.ylabel('$E_U/(Am^2)$',fontsize=14)
symlog(0.0000000000032,[-0.1,-0.001,-0.00001,-0.0000001,-0.000000001,-0.00000000001,0])
plt.legend(prop={'size': 8},ncol=2,loc=2)
plt.title('Second-difference renormalization, $D=2$, $N_A=%d$, $L_0=%d$'%(L_periodic[0],L0),fontsize=11)
save('Casimir_scalar_renorm.pdf')
