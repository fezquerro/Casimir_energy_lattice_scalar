# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:24:14 2022

@author: fer
"""
#from modules.create_data import create_points,create_ref
from modules.read_sc import read_data_an,read_data_an_ref,create_Lz
from modules.compute_casimir_sc import compute_casimir_analytic
#from modules.fit_casimir_sc import plot_Casimir

import numpy as np
import matplotlib.pyplot as plt
import math
import mpmath
plt.rcParams.update({
    'text.usetex': True
})


def eigen(i,L):
    return (2*np.sin(np.pi*i/L))**2
def eigend(i,L):
    return (2*np.sin(np.pi*i/L/2))**2
def ener_3d(Lt,Lx,Ly,Lz,m,bound):
 aux=0;
 if bound=='dir':
  for t in range(Lt):
     for x in range(Lx):
      for y in range(Ly):
         for z in range(1,Lz):
               a=eigen(t,Lt)
               aux+=eigen(t,Lt)/(eigen(x,Lx)+eigen(y,Ly)+eigend(z,Lz)+m**2+a)
  return 0.5*(1-1/Lz)-aux/Lt/Lx/Ly/Lz
 if bound=='per':
   for t in range(Lt):
     for x in range(Lx):
         for y in range(Ly):
             for z in range(Lz):
                a=eigen(t,Lt)
                aux+=a/(eigen(x,Lx)+eigen(y,Ly)+eigen(z,Lz)+m**2+a)
   return 0.5-aux/Lt/Lx/Ly/Lz


def create_ref(Lt,L,Lz,m,bound,dim):
    E_ref=np.zeros(len(m))
    for i in range (len(m)):
      E_ref[i]=ener_3d(Lt, L,L, Lz[i][0], m[i], bound)
    return E_ref
def write_data(Lt,L,Lz,m,energy,bound,dim):
    Name='../Energy_analytical/Energy_an3d_'+str(Lt)+'_'
    for i in range(1,dim):
     Name=Name+str(L)+'_'
    Name=Name+'m_'+'%.4f'%m+'_'+bound+'.txt'
    f = open(Name, "w")
    f.write("M Lz Energy\n")
    for i in range(np.size(Lz)):
     f.write(str(m)+' '+str(Lz[i])+' '+str(energy[i])+'\n')
    f.close()
def create_points(Lt,L,Lz,m,bound,dim):
    E=[]
    for i in range (len(m)):
        E.append([])
        for j in range (len(Lz[i])):
             E[i].append(ener_3d(Lt, L,L, Lz[i][j], m[i], bound))
        write_data(Lt,L,Lz[i],m[i],E[i],bound,dim)
    return E
def read_data_an(Lt,L,Lz,m,boundary,dim):
 data=[]
 bdata=np.size(m)
 E=[]
 for j in range(bdata):
  Name='../Energy_analytical/Energy_an3d_'+str(Lt)+'_'
  for i in range(1,dim):
    Name=Name+str(L)+'_'
  Name=Name+'m_'+'%.4f'%m[j]+'_'+boundary+'.txt'
  data.append(np.loadtxt(Name,dtype=float,delimiter=' ',skiprows=1,ndmin=2))
 for j in range(bdata):
     lut={int(row[1]):row[2] for row in data[j]}#Lz -> energy
     E.append([lut[int(Lzj)] for Lzj in Lz[j]])#KeyError if a requested Lz is not in the file
 return np.array(E,dtype=tuple)
def Cas_Pergen(mL,D,ep):
    auxD=(D+1)/2
    s=mpmath.besselk(auxD,mL)
    aux2=s
    c=1
    while(abs(aux2/s)>ep):
        c=c+1
        aux2=mpmath.besselk(auxD,mL*c)/pow(c,auxD)
        #print(aux2/s)
        s=s+aux2
    return -s/(pow(2,auxD-1)*pow(math.pi,auxD)*pow(mL,auxD-1))

def Cas_Dirgen(mL,D,ep):
    auxD=(D+1)/2
    s=mpmath.besselk(auxD,2*mL)
    aux2=s
    c=1
    while(abs(aux2/s)>ep):
        c=c+1
        aux2=mpmath.besselk(auxD,2*mL*c)/pow(c,auxD)
        #print(aux2/s)
        s=s+aux2
    return -s/(pow(2,D)*pow(math.pi,auxD)*pow(mL,auxD-1))

def plot_Casimir(C,x,m,bcond,yscale,dimension,plot):
    xdata=np.array(np.concatenate(x).flat)#fuse all the x
    E=np.array(np.concatenate(C).flat)#fuse all the Cas energy
    DP=np.linspace(min(xdata), max(xdata), num=100)#for plot space x
    
    Econt=[]
    if bcond=='periodic':
            plt.plot(xdata,E,'o',label='Periodic lattice')
            for mL in DP:
                Econt.append(Cas_Pergen(mL,3,1e-9))
            plt.plot(DP,Econt,label='Periodic contiuum ')
    else:
            plt.plot(xdata,E,'o',label='Dirichlet lattice') 
            for mL in DP:
                Econt.append(Cas_Dirgen(mL,3,1e-9))
            plt.plot(DP,Econt,label='Dirichlet contiuum ')
            
    plt.xlabel('$mL$',fontsize=14)
    plt.ylabel('$E_U/(Am^3)$',fontsize=14)
    if yscale=='log':
      if bcond=='periodic':
        plt.yscale('symlog', linthresh=0.00000001)
        plt.yticks([-0.1,-0.0001,-0.000001,-0.00000001,0])
      else:
        plt.yscale('symlog', linthresh=0.00000000032)
        plt.yticks([-0.01,-0.0001,-0.000001,-0.00000001,0])
        #plt.ylim(-0.1,0.000000000001)
     #plt.yscale('log')
    plt.legend( prop={'size': 8},ncol=3,loc=2)
    #plt.title('V=%g %g %g' %(Lt,L,Lz[0][-1] ))
    
    #if bcond=='Dirichlet':
     #   plt.savefig('Casimir_scalar_3d_full.pdf',bbox_inches='tight')
    if plot:
     #plt.show()
     plt.savefig('Casimir_lattice_3d.pdf',bbox_inches='tight')
     plt.show()
d=3
Lt=96 #t dimension of the lattice
L=96 #x dimension of the lattice
Lz_ref=[[96],[96],[96],[96],[96],[96],[96],[96],[96],[96]] #y dimension of the lattice of the reference value
m=[0.1,0.2,0.3,0.4]
Lz=[[12,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24]]#min and max y dimension for each beta
Lv=create_Lz(Lz, 1,False)#compute an array with all the y values, jump is the distance between them
Lv=[[12,14,16,18,20,22,24],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[8,9,10,11,12,13,14,15,16,17]]

boundary_condition='per'#select boundary condition; per is periodic, dir is Dirichlet   
#E=create_points(Lt,L,Lv,m,boundary_condition,d)
#ERef=create_ref(Lt,L,Lz_ref,m,boundary_condition,d)

E=read_data_an(Lt,L,Lv,m,boundary_condition,d)
ERef=read_data_an_ref(Lt,L,Lz_ref,m,boundary_condition,d)
#print(E)
#print(ERef)

print_values=False
Casimir,xdata=compute_casimir_analytic(E,Lv,m,ERef,Lz_ref,print_values,boundary_condition,d)#compute casimir, error and physical distance of each point

boundary_condition='periodic'#select boundary condition; per is periodic, dir is Dirichlet
y_scale='log'#log for log scale

pshow=False
plot_Casimir(Casimir,xdata,m,boundary_condition,y_scale,d,pshow)
 
 
 
d=3

Lt=96 #t dimension of the lattice
L=96 #x dimension of the lattice
Lz_ref=[[96],[96],[96],[96],[96],[96],[96],[96],[96],[96]] #y dimension of the lattice of the reference value
 
     
m=[0.1,0.2,0.3,0.4]
Lz=[[12,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24]]#min and max y dimension for each beta
Lv=create_Lz(Lz, 1,False)#compute an array with all the y values, jump is the distance between them
Lv=[[12,14,16,18,20,22,24],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[8,9,10,11,12,13,14,15,16,17]]
boundary_condition='dir'#select boundary condition; per is periodic, dir is Dirichlet   
#E=create_points(Lt,L,Lv,m,boundary_condition,d)
#ERef=create_ref(Lt,L,Lz_ref,m,'per',d)
#ERefD=create_ref(Lt,L,Lz_ref,m,boundary_condition,d)

E=read_data_an(Lt,L,Lv,m,boundary_condition,d)
ERef=read_data_an_ref(Lt,L,Lz_ref,m,'per',d)
ERefD=read_data_an_ref(Lt,L,Lz_ref,m,'dir',d)
print(E)
print(ERef)

print_values=True
Casimir,xdata=compute_casimir_analytic(E,Lv,m,[ERef,ERefD],Lz_ref,print_values,boundary_condition,d)#compute casimir, error and physical distance of each point

boundary_condition='Dirichlet'#select boundary condition; per is periodic, dir is Dirichlet
y_scale='log'#log for log scale

pshow=True


plot_Casimir(Casimir,xdata,m,boundary_condition,y_scale,d,pshow)