# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:24:14 2022

@author: fer

Lattice Casimir energy from the exact formula (casimir_exact: closed-form sum over the confined direction,
limit N_L0 -> infinity), compared with the continuum.
Dirichlet and Neumann have exactly the same lattice Casimir energy and are shown together.
Figures:
  Casimir_lattice_2ds.pdf      D=2, all the boundary conditions together
  Casimir_lattice_3d.pdf       D=3, all the boundary conditions together
  Casimir_lattice_D<bc>s.pdf   D=1..4 together, one figure per boundary condition (DDirichlets is Dirichlet = Neumann)
The comparison with the finite reference lattice prescription of the thesis is in Casimir_prescription_compare.py.
"""
import matplotlib.pyplot as plt
from modules.energy_formulas import casimir_scan
from modules.plotting import (flat, plot_continuum, plot_Casimir, style_D, symlog, save,
                              NAMES, LABELS_DN, PLOT_BOUNDS, COLORS, MASSES, LV, NA)


def plot_all_bounds(D,fname):
 """All the boundary conditions in one figure for dimension D."""
 plt.figure()
 for bound in PLOT_BOUNDS:
  C,x=casimir_scan([NA[D]]*D,LV,MASSES,bound)
  plt.plot(flat(x),flat(C),'o',markersize=4,color=COLORS[bound],label=LABELS_DN[bound]+' lattice')
  plot_continuum(min(flat(x)),max(flat(x)),D,bound,label=LABELS_DN[bound]+' continuum',color=COLORS[bound])
 plt.xlabel('$mL$',fontsize=14)
 plt.ylabel('$E_U/(Am^%d)$'%D,fontsize=14)
 symlog(1e-13,[-1e-2,-1e-5,-1e-8,-1e-11,0,1e-11,1e-8,1e-5,1e-2])
 plt.legend(prop={'size': 7},loc='center',ncol=2)#positive values above, negative below: the middle is empty
 plt.title('All boundary conditions, $D=%d$, $N_A=%d$, $N_{L_0}\\to\\infty$'%(D,NA[D]),fontsize=11)
 save(fname)


plot_all_bounds(2,'Casimir_lattice_2ds.pdf')
plot_all_bounds(3,'Casimir_lattice_3d.pdf')


# D=1..4, one figure per boundary condition
sizes=', '.join('%d'%NA[d] for d in range(1,5))
for bound in PLOT_BOUNDS:
 plt.figure()
 for d in range(1,5):
  C,x=casimir_scan([NA[d]]*d,LV,MASSES,bound)
  print(NAMES[bound],'d=%d'%d,'mL=%.2f-%.2f'%(min(flat(x)),max(flat(x))))
  plot_Casimir(C,x,bound,d)
 style_D(bound)
 plt.title('%s, $N_A=%s$ for $D=1,2,3,4$, $N_{L_0}\\to\\infty$'%(LABELS_DN[bound],sizes),fontsize=11)
 save('Casimir_lattice_D'+NAMES[bound]+'s.pdf')
