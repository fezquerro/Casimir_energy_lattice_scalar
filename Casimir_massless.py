# -*- coding: utf-8 -*-
"""
Massless lattice Casimir energy (casimir_exact at M=0) compared with the massless continuum
E_U/A = c/L^D (casimir_continuum_massless). The panels show E_U L^D/A against N_L: the lattice points approach the
continuum value c as N_L grows, with O(1/N_L^2) corrections.
At M=0 the finite size N_A of the periodic directions only decays as a power of N_L/N_A, so N_A = r*N_L with a large
ratio r (larger for small D). Dirichlet and Neumann have exactly the same lattice Casimir energy and share a panel.
One figure per dimension D=1..4, a panel per boundary condition: Casimir_massless_D<D>.pdf
"""
import matplotlib.pyplot as plt
from modules.energy_formulas import casimir_exact, phys_length
from modules.continuum import casimir_continuum_massless
from modules.plotting import save, LETTER, LABELS_DN, PLOT_BOUNDS, COLORS

# r is the smallest ratio for which the finite-N_A effect stays below the lattice-spacing one (Dirichlet is the most
# sensitive); the largest N_L keeps the number of distinct transverse modes below ~3e7 (memory)
#     D: (N_L values,                                    r=N_A/N_L)
SIZES={1:([2,3,4,6,8,12,16,24,32,48,64,96,128],          65536),
       2:([2,3,4,6,8,12,16,24,32,48,64],                 128),
       3:([2,3,4,6,8,10,12,16,20,24],                    32),
       4:([2,3,4,5,6,8,10,12,14],                        16)}

for D in range(1,5):
    NLs,r=SIZES[D]
    fig,axes=plt.subplots(2,2,figsize=(9,7.5))
    print('D=%d, N_A=%d N_L: E_U L^D/A / continuum - 1'%(D,r))
    for ax,bound in zip(axes.flat,PLOT_BOUNDS):
        plt.sca(ax)
        c=casimir_continuum_massless(D,bound)
        y=[casimir_exact([r*NL]*D,NL,0.0,bound)*phys_length(NL,bound)**D for NL in NLs]#E_U L^D/A
        print('   %-4s '%bound+'  '.join('N_L=%d: %+.1e'%(NL,v/c-1) for NL,v in zip(NLs,y)))
        plt.plot(NLs,y,'o',markersize=4,color=COLORS[bound],label='Lattice, $M=0$')
        plt.plot([NLs[0],NLs[-1]],[c,c],'--',color=COLORS[bound],lw=1,label='Continuum')#plot, not axhline: the y axis must include it
        plt.title(LABELS_DN[bound],fontsize=11)
        plt.xlabel('$N_L$',fontsize=12)
        plt.ylabel('$E_%s L^%d/A$'%(LETTER[bound],D),fontsize=12)
        plt.legend(prop={'size': 8})
    fig.suptitle('Massless Casimir energy, $D=%d$, $N_A=%dN_L$'%(D,r),fontsize=13)
    fig.tight_layout()
    save('Casimir_massless_D%d.pdf'%D)
