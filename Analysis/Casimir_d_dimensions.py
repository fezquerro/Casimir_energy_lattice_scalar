# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:24:14 2022

@author: fer
"""
from modules.create_data import create_points,create_ref
from modules.read_sc import read_data_an,read_data_an_ref,create_Lz
from modules.compute_casimir_sc import compute_casimir_analytic
from modules.fit_casimir_sc import plot_Casimir





for d in range(1,5):

 Lt=96 #t dimension of the lattice
 L=96 #x dimension of the lattice
 Lz_ref=[[96],[96],[96],[96],[96],[96],[96],[96],[96],[96]] #y dimension of the lattice of the reference value
 if d==4:
    Lt=48 #t dimension of the lattice
    L=48 #x dimension of the lattice
    Lz_ref=[[48],[48],[48],[48],[96],[96],[96],[96],[96],[96]] #y dimension of the lattice of the reference value
     
 m=[0.1,0.2,0.3,0.4]
 Lz=[[12,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24]]#min and max y dimension for each beta
 Lv=create_Lz(Lz, 1,False)#compute an array with all the y values, jump is the distance between them
 Lv=[[12,14,16,18,20,22,24],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],[8,9,10,11,12,13,14,15,16,17,18,19,20]]

 boundary_condition='per'#select boundary condition; per is periodic, dir is Dirichlet   
 #E=create_points(Lt,L,Lv,m,boundary_condition,d)
 #ERef=create_ref(Lt,L,Lz_ref,m,boundary_condition,d)

 E=read_data_an(Lt,L,Lv,m,boundary_condition,d)
 ERef=read_data_an_ref(Lt,L,Lz_ref,m,boundary_condition,d)
 #print(E)
 #print(ERef)

 print_values=False
 Casimir,xdata=compute_casimir_analytic(E,Lv,m,ERef,Lz_ref,print_values,boundary_condition,d)#compute casimir, error and physical distance of each point
 print(Casimir)
 boundary_condition='periodic'#select boundary condition; per is periodic, dir is Dirichlet
 y_scale='log'#log for log scale
 if d==4:
     pshow=True
 else:
     pshow=False
 plot_Casimir(Casimir,xdata,m,boundary_condition,y_scale,d,pshow)
 

for d in range(1,5):

 Lt=96 #t dimension of the lattice
 L=96 #x dimension of the lattice
 Lz_ref=[[96],[96],[96],[96],[96],[96],[96],[96],[96],[96]] #y dimension of the lattice of the reference value
 if d==4:
    Lt=48 #t dimension of the lattice
    L=48 #x dimension of the lattice
    Lz_ref=[[48],[48],[48],[48],[96],[96],[96],[96],[96],[96]] #y dimension of the lattice of the reference value
     
 m=[0.1,0.2,0.3,0.4]
 Lz=[[12,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24]]#min and max y dimension for each beta
 Lv=create_Lz(Lz, 1,False)#compute an array with all the y values, jump is the distance between them
 Lv=[[12,14,16,18,20],[8,9,10,11,12,13,14,15,16,17,18,19],[8,9,10],[8,9,10]]
 boundary_condition='dir'#select boundary condition; per is periodic, dir is Dirichlet   
 #E=create_points(Lt,L,Lv,m,boundary_condition,d)
 #ERef=create_ref(Lt,L,Lz_ref,m,'dir',d)
 #ERef=create_ref(Lt,L,Lz_ref,m,boundary_condition,d)

 E=read_data_an(Lt,L,Lv,m,boundary_condition,d)
 ERef=read_data_an_ref(Lt,L,Lz_ref,m,'per',d)
 ERefD=read_data_an_ref(Lt,L,Lz_ref,m,'dir',d)
 #print(E)
 #print(ERef)

 print_values=True
 Casimir,xdata=compute_casimir_analytic(E,Lv,m,[ERef,ERefD],Lz_ref,print_values,boundary_condition,d)#compute casimir, error and physical distance of each point
 print(Casimir)
 boundary_condition='Dirichlet'#select boundary condition; per is periodic, dir is Dirichlet
 y_scale='log'#log for log scale
 if d==4:
     pshow=True
 else:
     pshow=False
 plot_Casimir(Casimir,xdata,m,boundary_condition,y_scale,d,pshow)


for d in range(1,5):

 Lt=96 #t dimension of the lattice
 L=96 #x dimension of the lattice
 Lz_ref=[[96],[96],[96],[96],[96],[96],[96],[96],[96],[96]] #y dimension of the lattice of the reference value
 if d==4:
    Lt=48 #t dimension of the lattice
    L=48 #x dimension of the lattice
    Lz_ref=[[48],[48],[48],[48],[96],[96],[96],[96],[96],[96]] #y dimension of the lattice of the reference value
     
 m=[0.1,0.2,0.3,0.4]
 Lz=[[12,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24]]#min and max y dimension for each beta
 Lv=create_Lz(Lz, 1,False)#compute an array with all the y values, jump is the distance between them
 Lv=[[12,14,16,18,20,22,24],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],[8,9,10,11,12,13,14,15,16,17,18,19,20]]

 boundary_condition='anti'#select boundary condition; per is periodic, dir is Dirichlet   
 #E=create_points(Lt,L,Lv,m,boundary_condition,d)
 #ERef=create_ref(Lt,L,Lz_ref,m,boundary_condition,d)

 E=read_data_an(Lt,L,Lv,m,boundary_condition,d)
 ERef=read_data_an_ref(Lt,L,Lz_ref,m,'per',d)
 #print(E)
 #print(ERef)

 print_values=False
 Casimir,xdata=compute_casimir_analytic(E,Lv,m,ERef,Lz_ref,print_values,boundary_condition,d)#compute casimir, error and physical distance of each point
 print(Casimir)
 boundary_condition='antiperiodic'#select boundary condition; per is periodic, dir is Dirichlet
 y_scale='log'#log for log scale
 if d==4:
     pshow=True
 else:
     pshow=False
 plot_Casimir(Casimir,xdata,m,boundary_condition,y_scale,d,pshow)


for d in range(1,5):

 Lt=96 #t dimension of the lattice
 L=96 #x dimension of the lattice
 Lz_ref=[[96],[96],[96],[96],[96],[96],[96],[96],[96],[96]] #y dimension of the lattice of the reference value
 if d==4:
    Lt=48 #t dimension of the lattice
    L=48 #x dimension of the lattice
    Lz_ref=[[48],[48],[48],[48],[96],[96],[96],[96],[96],[96]] #y dimension of the lattice of the reference value
     
 m=[0.1,0.2,0.3,0.4]
 Lz=[[12,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24]]#min and max y dimension for each beta
 Lv=create_Lz(Lz, 1,False)#compute an array with all the y values, jump is the distance between them
 Lv=[[12,14,16,18,20],[8,9,10,11,12,13,14,15,16,17,18,19],[8,9,10],[8,9,10]]
 boundary_condition='neu'#select boundary condition; per is periodic, dir is Dirichlet   
 #E=create_points(Lt,L,Lv,m,boundary_condition,d)
 #ERef=create_ref(Lt,L,Lz_ref,m,boundary_condition,d)

 E=read_data_an(Lt,L,Lv,m,boundary_condition,d)
 ERef=read_data_an_ref(Lt,L,Lz_ref,m,'per',d)
 ERefD=read_data_an_ref(Lt,L,Lz_ref,m,'neu',d)
 #print(E)
 #print(ERef)

 print_values=True
 Casimir,xdata=compute_casimir_analytic(E,Lv,m,[ERef,ERefD],Lz_ref,print_values,boundary_condition,d)#compute casimir, error and physical distance of each point
 print(Casimir)
 boundary_condition='Neumann'#select boundary condition; per is periodic, dir is Dirichlet
 y_scale='log'#log for log scale
 if d==4:
     pshow=True
 else:
     pshow=False
 plot_Casimir(Casimir,xdata,m,boundary_condition,y_scale,d,pshow)


for d in range(1,5):

 Lt=96 #t dimension of the lattice
 L=96 #x dimension of the lattice
 Lz_ref=[[96],[96],[96],[96],[96],[96],[96],[96],[96],[96]] #y dimension of the lattice of the reference value
 if d==4:
    Lt=48 #t dimension of the lattice
    L=48 #x dimension of the lattice
    Lz_ref=[[48],[48],[48],[48],[96],[96],[96],[96],[96],[96]] #y dimension of the lattice of the reference value
     
 m=[0.1,0.2,0.3,0.4]
 Lz=[[12,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24]]#min and max y dimension for each beta
 Lv=create_Lz(Lz, 1,False)#compute an array with all the y values, jump is the distance between them
 Lv=[[12,14,16,18,20],[8,9,10,11,12,13,14,15,16,17,18,19],[8,9,10],[8,9,10]]
 if d==4:
    Lv=[[12,14,16],[8,9,10,11,12,13,14,15,16],[8,9,10],[8,9,10]]
 #Lv=[[8,9,10,11,12,13,14,15,16,17,18,19,20]]
 boundary_condition='zar'#select boundary condition; per is periodic, dir is Dirichlet   
 #E=create_points(Lt,L,Lv,m,boundary_condition,d)
 #ERef=create_ref(Lt,L,Lz_ref,m,boundary_condition,d)

    
 E=read_data_an(Lt,L,Lv,m,boundary_condition,d)
 ERef=read_data_an_ref(Lt,L,Lz_ref,m,'per',d)
 ERefD=read_data_an_ref(Lt,L,Lz_ref,m,'zar',d)
 #print(E)
 #print(ERef)

 print_values=True
 Casimir,xdata=compute_casimir_analytic(E,Lv,m,[ERef,ERefD],Lz_ref,print_values,boundary_condition,d)#compute casimir, error and physical distance of each point
 print(Casimir)
 boundary_condition='Zaremba'#select boundary condition; per is periodic, dir is Dirichlet
 y_scale='log'#log for log scale
 if d==4:
     pshow=True
 else:
     pshow=False
 plot_Casimir(Casimir,xdata,m,boundary_condition,y_scale,d,pshow)
