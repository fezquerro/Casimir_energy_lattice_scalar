# -*- coding: utf-8 -*-
"""
Lattice Casimir energy from the closed-form sum over the confined direction
(closed_form_note/closed_form_casimir.tex). Same parameters as Casimir_d_dimensions.py,
but no data files and no reference lattice: casimir_exact gives the N_L0 -> infinity limit directly.
"""
import numpy as np
import matplotlib.pyplot as plt
from modules.energy_formulas import casimir_exact
from modules.fit_casimir_sc import plot_Casimir

m=[0.1,0.2,0.3,0.4]#masses
Lv=[list(range(8,25)) for i in range(len(m))]#N_L values for each mass
bconds=[('per','periodic'),('anti','antiperiodic'),('dir','Dirichlet'),('neu','Neumann'),('zar','Zaremba')]

for bound,bname in bconds:
 plt.figure()#one figure per boundary condition
 for d in range(1,5):
  Lt=96 #t dimension of the lattice
  L=96 #transverse dimensions of the lattice
  if d==4:
     Lt=48
     L=48
  L_periodic=[Lt]+[L]*(d-1)#time + (d-1) transverse directions
  Casimir=[]
  xdata=[]
  for i in range(len(m)):
     Casimir.append([casimir_exact(L_periodic,NL,m[i],bound)/m[i]**d for NL in Lv[i]])#E_U/(A m^d)
     if bound=='zar':
         xdata.append([m[i]*(NL-0.5) for NL in Lv[i]])#physical length N_L-1/2
     else:
         xdata.append([m[i]*NL for NL in Lv[i]])
  print(bname,'d=%d'%d,'mL=%.2f-%.2f'%(min(np.concatenate(xdata)),max(np.concatenate(xdata))))
  plot_Casimir(Casimir,xdata,m,bname,'lin',d,False)#draw lattice points and continuum curve only
 #the exact values go far below the symlog thresholds tuned in plot_Casimir, so the axis is set here
 if bound in ['anti','zar']:#positive Casimir energy
     plt.yscale('log')
 else:#negative Casimir energy
     plt.yscale('symlog',linthresh=1e-13)
 plt.legend(prop={'size': 8},ncol=2)
 plt.savefig('Casimir_exact_D'+bname+'.pdf',bbox_inches='tight')
 plt.close()
