# -*- coding: utf-8 -*-
"""
Exact lattice Casimir energy (casimir_exact, N_L0 -> infinity) compared with the renormalization prescription
of the thesis (casimir_finite_ref, eqs. 3.79-3.87: subtraction of a reference lattice of size N_L0).
The prescription differs from the exact result by the Casimir energy of the reference lattice, ~exp(-m N_L0),
plus the rounding error of subtracting O(1) energies, which dominates when E_U is very small.
One figure per dimension D=1..4 with all the boundary conditions, Casimir_prescription_D<D>.pdf:
dots exact, crosses prescription with N_L0=N_A, lines continuum (Dirichlet = Neumann).
The largest relative difference for N_L0 = N_A/2, N_A, 2N_A is printed.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from modules.energy_formulas import casimir_scan
from modules.plotting import flat, plot_continuum, symlog, save, LABELS_DN, PLOT_BOUNDS, COLORS, MASSES, LV, NA

for D in range(1,5):
    N=NA[D]
    NL0S=[N//2,N,2*N]
    print('D=%d, N_A=%d: max |prescription/exact - 1|'%(D,N))
    print('      '+''.join('N_L0=%-9d'%n for n in NL0S))
    plt.figure(figsize=(6.4,5.2))
    for bound in PLOT_BOUNDS:
        Ce,x=casimir_scan([N]*D,LV,MASSES,bound)#exact
        Cr={n:casimir_scan([N]*D,LV,MASSES,bound,NL0=n)[0] for n in NL0S}#prescription
        xd=flat(x)
        print('%-6s'%bound+''.join('%-14.1e'%max(abs(flat(Cr[n])/flat(Ce)-1)) for n in NL0S))
        plt.plot(xd,flat(Ce),'o',markersize=4,color=COLORS[bound])
        plt.plot(xd,flat(Cr[N]),'x',markersize=5,color=COLORS[bound])
        plot_continuum(min(xd),max(xd),D,bound,label=None,color=COLORS[bound])
    handles=[Line2D([],[],color=COLORS[b],marker='s',ls='',label=LABELS_DN[b]) for b in PLOT_BOUNDS]
    handles+=[Line2D([],[],color='k',marker='o',ls='',markersize=4,label='Exact ($N_{L_0}\\to\\infty$)'),
              Line2D([],[],color='k',marker='x',ls='',label='Prescription, $N_{L_0}=%d$'%N),
              Line2D([],[],color='k',label='Continuum')]
    plt.legend(handles=handles,prop={'size': 8},loc='upper left',bbox_to_anchor=(1.02,1))#outside the axes: no overlap with the points
    symlog(1e-13,[-1e-2,-1e-5,-1e-8,-1e-11,0,1e-11,1e-8,1e-5,1e-2])
    plt.xlabel('$mL$',fontsize=14)
    plt.ylabel('$E_U/(Am^%d)$'%D,fontsize=14)
    plt.title('Exact vs prescription, $D=%d$, $N_A=%d$'%(D,N),fontsize=11)
    save('Casimir_prescription_D%d.pdf'%D)
