# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:24:14 2022

@author: fer

Finite-size checks of the exact lattice Casimir energy (casimir_exact, N_L0 -> infinity) against the continuum.
Only the sizes that enter the formula are varied:
  'A'  size N_A of the periodic directions (time included), masses 0.1-0.4, N_L=8-24
  'L'  size N_L of the confined direction, masses M=(1+j/2)/N_L so that mL=1..9 for every N_L
       (lattice spacing a ~ 1/N_L at fixed mL). To keep the finite-N_A effects out, M*N_A >= 10:
       D=1,2 use N_A=max(96,16 N_L); D=3,4 keep N_A=96 (48 for D=4) to keep the run time reasonable and drop the points with M*N_A < 10.
One figure per scan and dimension D=1..4, a panel per boundary condition: Casimir_size_<scan>_D<D>.pdf
The panels show E_U/(A m^D) against mL with the continuum curve. Dirichlet and Neumann share a panel:
their lattice Casimir energies are exactly the same (they differ only in the boundary energy C1).
"""
import matplotlib.pyplot as plt
from modules.energy_formulas import casimir_scan
from modules.plotting import flat, plot_continuum, save, LETTER, LABELS_DN, PLOT_BOUNDS, MASSES, LV, NA

MIN_MNA=10#smallest M*N_A kept in the 'L' scan
VALUES={'A':{1:[16,32,48,96,160],2:[16,32,48,96,160],3:[16,32,48,96,160],4:[16,24,32,48]},
        'L':{D:[2,6,10,16,24,32] for D in range(1,5)}}
LABEL={'A':'$N_A=%g$','L':'$N_L=%g$'}
TITLE={'A':'Size $N_A$ of the periodic directions, $D=%d$, $M=0.1$-$0.4$, $N_L=8$-$24$',
       'L':'Size $N_L$ of the confined direction, $D=%d$, $M=(1+j/2)/N_L$, %s'}
NA_L={1:'$N_A=\\max(96,16N_L)$',2:'$N_A=\\max(96,16N_L)$',
      3:'$N_A=%d$, $MN_A\\geq%d$'%(NA[3],MIN_MNA),4:'$N_A=%d$, $MN_A\\geq%d$'%(NA[4],MIN_MNA)}


def scan(kind,D,bound,v):
    """E_U/(A m^D) and mL for one value v of the size that is varied."""
    if kind=='A':
        return casimir_scan([v]*D,LV,MASSES,bound)
    m=[(1+0.5*j)/v for j in range(17)]
    if D<=2:
        N=max(NA[D],16*v)#M*N_A >= 16
    else:
        N=NA[D]
        m=[a for a in m if a*N>=MIN_MNA]
    return casimir_scan([N]*D,[[v]]*len(m),m,bound)


for kind in ['A','L']:
    for D in range(1,5):
        fig,axes=plt.subplots(2,2,figsize=(9,7.5))
        for ax,bound in zip(axes.flat,PLOT_BOUNDS):
            plt.sca(ax)
            xall=[]
            for v in VALUES[kind][D]:
                C,x=scan(kind,D,bound,v)
                plt.plot(flat(x),flat(C),'o',markersize=3,label=LABEL[kind]%v)
                xall+=list(flat(x))
            plot_continuum(min(xall),max(xall),D,bound,label='Continuum')
            if bound in ['anti','zar']:#positive Casimir energy
                plt.yscale('log')
            else:#negative Casimir energy
                plt.yscale('symlog',linthresh=1e-13)
            plt.title(LABELS_DN[bound],fontsize=11)
            plt.xlabel('$mL$',fontsize=12)
            plt.ylabel('$E_%s/(Am^%d)$'%(LETTER[bound],D),fontsize=12)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=len(labels),fontsize=10)
        fig.suptitle(TITLE[kind]%((D,) if kind=='A' else (D,NA_L[D])),fontsize=13)
        fig.tight_layout(rect=(0,0.05,1,1))#leave room for the legend
        save('Casimir_size_%s_D%d.pdf'%(kind,D))
        print('Casimir_size_%s_D%d.pdf'%(kind,D))
