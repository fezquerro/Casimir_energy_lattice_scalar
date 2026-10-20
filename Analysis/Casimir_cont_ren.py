# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:24:14 2022

@author: fer
"""
from modules.create_data import create_2nd_norm
from modules.read_sc import read_data_2nd,create_Lz
from modules.compute_casimir_sc import compute_casimir_renor
from modules.fit_casimir_sc import cas_m_compare





Lt=96 #t dimension of the lattice
L=96 #x dimension of the lattice
L0=48
Lz_ref=[[96],[96],[96],[96],[96],[96],[96],[96],[96],[96]] #y dimension of the lattice of the reference value
m=[0.1,0.2,0.3,0.4]
Lz=[[12,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24]]#min and max y dimension for each beta
Lv=create_Lz(Lz, 1,False)#compute an array with all the y values, jump is the distance between them
boundary_condition='per'#select boundary condition; per is periodic, dir is Dirichlet

#E=create_2nd_norm(Lt,L,Lv,L0,m,boundary_condition)
E=read_data_2nd(Lt,L,Lv,L0,m,boundary_condition)
print(E)


print_values=True
Casimir,xdata=compute_casimir_renor(E,Lv,m,print_values,boundary_condition)#compute casimir, error and physical distance of each point

boundary_condition='periodic'#select boundary condition; per is periodic, dir is Dirichlet
dif_m=False#plot different m in color with True
y_scale='log'#log for log scale
cas_m_compare(Casimir,xdata,m,boundary_condition,dif_m,y_scale)

Lt=96 #t dimension of the lattice
L=96 #x dimension of the lattice
L0=48
Lz_ref=[[96],[96],[96],[96],[96],[96],[96],[96],[96],[96]] #y dimension of the lattice of the reference value
m=[0.1,0.2,0.3,0.4]
Lz=[[12,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24]]#min and max y dimension for each beta
Lv=create_Lz(Lz, 1,False)#compute an array with all the y values, jump is the distance between them
boundary_condition='dir'#select boundary condition; per is periodic, dir is Dirichlet

#E=create_2nd_norm(Lt,L,Lv,L0,m,boundary_condition)
E=read_data_2nd(Lt,L,Lv,L0,m,boundary_condition)
print(E)


print_values=True
Casimir,xdata=compute_casimir_renor(E,Lv,m,print_values,boundary_condition)#compute casimir, error and physical distance of each point

boundary_condition='Dirichlet'#select boundary condition; per is periodic, dir is Dirichlet
dif_m=False#plot different m in color with True
y_scale='log'#log for log scale
cas_m_compare(Casimir,xdata,m,boundary_condition,dif_m,y_scale)




