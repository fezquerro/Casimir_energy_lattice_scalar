# -*- coding: utf-8 -*-
"""
Casimir energy of the finite lattice (casimir_exact, periodic directions of size N_A, time included) compared with
the Casimir energy of infinite walls at zero temperature (casimir_inf, N_A -> infinity).
The difference comes from the modes that wind around the time direction (finite temperature) or a parallel direction
(finite area); it is controlled by m N_A and by l_U/N_A, with l_U = N_L for P, A and 2N_L for D, N, Z, so it is
largest for the lightest mass and the longest N_L.
One figure per dimension D=1..4, a panel per boundary condition, Casimir_infinite_D<D>.pdf: E_U/A against N_L at
M=0.1 for several N_A (points) and E_U^inf (line). Dirichlet and Neumann share a panel.
"""
import matplotlib.pyplot as plt
from modules.energy_formulas import casimir_exact, casimir_inf
from modules.plotting import save, LETTER, LABELS_DN, PLOT_BOUNDS, NA

M=0.1#lightest mass of the thesis figures: largest finite-N_A effect
NLS=list(range(4,25))
SIZES={1:[16,24,32,48,96],2:[16,24,32,48,96],3:[16,24,32,48,96],4:[16,24,32,48,100]}#N_A values

print('E_U/E_U^inf - 1 on the thesis lattices')
print('D  M    N_L  '+''.join('%-11s'%LETTER[b] for b in PLOT_BOUNDS))
for D in range(1,5):
    for m,NL in [(0.1,8),(0.1,24),(0.4,24)]:
        print('%d  %.1f  %-4d '%(D,m,NL)+''.join('%-11.1e'%(casimir_exact([NA[D]]*D,NL,m,b)/casimir_inf(D,NL,m,b)-1)
                                                for b in PLOT_BOUNDS))

for D in range(1,5):
    fig,axes=plt.subplots(2,2,figsize=(9,7.5))
    for ax,bound in zip(axes.flat,PLOT_BOUNDS):
        plt.sca(ax)
        for N in SIZES[D]:
            plt.plot(NLS,casimir_exact([N]*D,NLS,M,bound),'o',markersize=3,label='$N_A=%d$'%N)
        plt.plot(NLS,casimir_inf(D,NLS,M,bound),'-',color='k',lw=1,label='$N_A\\to\\infty$')#black: the colors are the N_A values
        if bound in ['anti','zar']:#positive Casimir energy
            plt.yscale('log')
        else:#negative Casimir energy
            plt.yscale('symlog',linthresh=1e-13)
        plt.title(LABELS_DN[bound],fontsize=11)
        plt.xlabel('$N_L$',fontsize=12)
        plt.ylabel('$E_%s/A$'%LETTER[bound],fontsize=12)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=len(labels),fontsize=10)
    fig.suptitle('Finite lattice vs infinite walls at zero temperature, $D=%d$, $M=%g$'%(D,M),fontsize=13)
    fig.tight_layout(rect=(0,0.05,1,1))#leave room for the legend
    save('Casimir_infinite_D%d.pdf'%D)
    print('Casimir_infinite_D%d.pdf'%D)
