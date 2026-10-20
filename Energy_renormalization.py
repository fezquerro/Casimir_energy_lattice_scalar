# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:05:20 2024

@author: usuario

Divergent bulk and boundary contributions of the lattice energy, D=1..4 (N_A=100, 48 for D=4).
The total energy per unit area splits exactly as N_L*e_U(N_L) = C0*L_U + C1^U + E_U/A,
with L_U = N_L (N_L-1/2 for Zaremba). This script compares:
1. the renormalization method, one figure per boundary condition with a panel per D (dots),
   with the exact C0 and C1 (dashed lines):
     periodic, anti-periodic     e_U(N_L)                           -> C0
     Dirichlet, Neumann, Zaremba N_L*e_U(N_L) - L_U*e_P(N_ref)      -> C1^U   (L_U bulk units subtracted)
2. the bulk energy of the lattice with one dimension less (the periodic directions only),
   e0 at mass m and e1 at mass sqrt(m^2+4), with the combinations of boundary terms that give them:
     e0 = C1^N - C1^D,   e1 = -(C1^D + C1^N) = -2 C1^Z               (Boundary_vs_bulk_lower_dim.pdf, a panel per D)
"""
import numpy as np
import matplotlib.pyplot as plt
from modules.energy_formulas import ener_fast, ener_periodic, bulk_C0, boundary_C1, phys_length
from modules.plotting import LABELS, save
plt.rcParams.update({
    'text.usetex': True
})
plt.rc('text.latex', preamble=r'\usepackage{braket}')

NA={1:100,2:100,3:100,4:48}#size of the periodic directions (time included); 48 for D=4 to keep the run time reasonable
m=[0.2,0.22,0.25,0.3,0.4]
Lv=list(range(10,101,10))
NLref=Lv[-1]#periodic bulk subtracted from dir/neu/zar


# 1. Renormalization method vs exact C0 and C1
BULK='$\\braket{\\mathcal E_%s}/(ALm^{%d})$'
SUB='$(\\braket{\\mathcal E_%s}-\\frac{%s}{%d}\\braket{\\mathcal E_P})/(Am^{%d})$'
#            bound   letter  subtracted  file name                              exact line label
FIGURES=[('per', 'P',    None,       'Periodic_divergence_scalar.pdf',     'Exact $C_0$'),
         ('dir', 'D',    'N_L',      'Dirichlet_divergence_scalar.pdf',    'Exact $C_1^D$'),
         ('anti','A',    None,       'AntiPeriodic_divergence_scalar.pdf', 'Exact $C_0$'),
         ('neu', 'N',    'N_L',      'Neumann_divergence_scalar.pdf',      'Exact $C_1^N$'),
         ('zar', 'Z',    'N_L-1/2',  'Zaremba_divergence_scalar.pdf',      'Exact $C_1^Z$')]

print('Renormalization method at N_L=%d vs exact: max |diff| over the masses'%NLref)
for bound,letter,sub,fname,exact_label in FIGURES:
    fig,axes=plt.subplots(2,2,figsize=(9,7.5))
    for ax,D in zip(axes.flat,range(1,5)):
        plt.sca(ax)
        L_periodic=[NA[D]]*D
        diff=0
        for k,a in enumerate(m):
            E=[ener_fast(L_periodic,NL,a,bound) for NL in Lv]
            if bound in ['per','anti']:
                y=[e/a**(D+1) for e in E]
                exact=bulk_C0(L_periodic,a)/a**(D+1)
            else:
                Eper=ener_fast(L_periodic,NLref,a,'per')
                y=[(NL*e-phys_length(NL,bound)*Eper)/a**D for NL,e in zip(Lv,E)]#subtract L_U bulk units
                exact=boundary_C1(L_periodic,a,bound)/a**D
            plt.scatter(Lv,y,s=12,label='$M$=%g'%a,color='C%d'%k)
            plt.axhline(exact,color='C%d'%k,ls='--',lw=0.8,label=exact_label if k==0 else None)
            diff=max(diff,abs(y[-1]-exact))
        print('  %-4s D=%d  %.1e'%(bound,D,diff))
        plt.xlabel('$N_L$',fontsize=12)
        if sub is None:
            plt.ylabel(BULK%(letter,D+1),fontsize=12)
        else:
            plt.ylabel(SUB%(letter,sub,NLref,D),fontsize=11)
        plt.title('$D=%d$, $N_A=%d$'%(D,NA[D]),fontsize=11)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=len(labels),fontsize=10)
    fig.suptitle('%s: renormalization method (dots) and exact (dashed)'%LABELS[bound],fontsize=13)
    fig.tight_layout(rect=(0,0.05,1,1))#leave room for the legend
    save(fname)


# 2. Bulk energy one dimension less from the boundary terms
M=np.linspace(0.05,2,40)
print('Bulk energy one dimension less vs combinations of boundary terms (max |diff| over M=%g-%g)'%(M[0],M[-1]))
fig,axes=plt.subplots(2,2,figsize=(9,7.5))
for ax,D in zip(axes.flat,range(1,5)):
    plt.sca(ax)
    L_periodic=[NA[D]]*D
    e0=np.array([ener_periodic(L_periodic,a) for a in M])#the lattice of the periodic directions only
    e1=np.array([ener_periodic(L_periodic,np.sqrt(a*a+4)) for a in M])
    C1={b:np.array([boundary_C1(L_periodic,a,b) for a in M]) for b in ['dir','neu','zar']}
    #            exact  combination of C1        label               marker  color
    RELATIONS=[(e0,     C1['neu']-C1['dir'],     '$C_1^N-C_1^D$',    'o',    'C0'),
               (e1,     -(C1['dir']+C1['neu']),  '$-(C_1^D+C_1^N)$', 'o',    'C1'),
               (e1,     -2*C1['zar'],            '$-2C_1^Z$',        'x',    'C3')]
    plt.plot(M,e0,color='C0',label='$e_0$: bulk, one dimension less, mass $m$')
    plt.plot(M,e1,color='C1',label='$e_1$: bulk, one dimension less, mass $\\sqrt{m^2+4}$')
    for exact,comb,comb_name,marker,color in RELATIONS:
        plt.plot(M,comb,marker,markersize=4,color=color,label=comb_name)
        print('  D=%d %-16s %.1e'%(D,comb_name.strip('$'),np.max(np.abs(comb-exact))))
    plt.xlabel('$M$',fontsize=12)
    plt.ylabel('Energy per site',fontsize=12)
    plt.title('$D=%d$, $N_A=%d$'%(D,NA[D]),fontsize=11)
handles,labels=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=10)
fig.suptitle('Bulk energy in one dimension less from the boundary terms',fontsize=13)
fig.tight_layout(rect=(0,0.08,1,1))#leave room for the legend
save('Boundary_vs_bulk_lower_dim.pdf')
