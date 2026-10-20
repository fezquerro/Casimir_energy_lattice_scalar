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
def ener_2d(Lt,Lx,Ly,m,bound):
 aux=0;
 if bound=='dir':
  for t in range(Lt):
     for x in range(Lx):
         for y in range(1,Ly):
               a=eigen(t,Lt)
               aux+=a/(eigen(x,Lx)+eigend(y,Ly)+m**2+a)
  return 0.5*(1-1/Ly)-aux/Lt/Lx/Ly 
    
 if bound=='per':
   for t in range(Lt):
     for x in range(Lx):
         for y in range(Ly):
                a=eigen(t,Lt)
                aux+=a/(eigen(x,Lx)+eigen(y,Ly)+m**2+a)
   return 0.5-aux/Lt/Lx/Ly
def write_data(Lt,L,Lz,m,energy,bound,dim):
    Name='../Energy_analytical/Energy_anA2d_'+str(Lt)+'_'
    for i in range(1,dim):
     Name=Name+str(L)+'_'
    Name=Name+'m_'+'%.4f'%m+'_'+bound+'.txt'
    f = open(Name, "w")
    f.write("M Lz Energy\n")
    for i in range(np.size(Lz)):
     f.write(str(m)+' '+str(Lz[i])+' '+str(energy[i])+'\n')
    f.close()
def write_data_ref(Lt,L,Lz,m,energy,bound,dim):
    Name='../Energy_analytical/Energy_an_refA2d_'+str(Lt)+'_'
    for i in range(1,dim):
     Name=Name+str(L)+'_'
    Name=Name+str(Lz)+'m_'+'%.4f'%m+'_'+bound+'.txt'
    f = open(Name, "w")
    f.write("M Lz Energy\n")
    f.write(str(m)+' '+str(Lz)+' '+str(energy)+'\n')
    f.close()
def create_points(Lt,L,Lz,m,bound,dim):
    E=[]
    for i in range (len(m)):
        E.append([])
        for j in range (len(Lz[i])):
             E[i].append(ener_2d(Lt, L, Lz[i][j], m[i], bound))
        write_data(Lt,L,Lz[i],m[i],E[i],bound,dim)
    return E
def create_ref(Lt,L,Lz,m,bound,dim):
    E_ref=np.zeros(len(m))
    for i in range (len(m)):
             E_ref[i]=ener_2d(Lt, L, Lz[i][0], m[i], bound)
             write_data_ref(Lt,L,Lz[i][0],m[i],E_ref[i],bound,dim)
    return E_ref
def read_data_an(Lt,L,Lz,m,boundary,dim):
 data=[]
 bdata=np.size(m)
 E=[]
 for j in range(bdata):
  Name='../Energy_analytical/Energy_anA2d_'+str(Lt)+'_'
  for i in range(1,dim):
    Name=Name+str(L)+'_'
  Name=Name+'m_'+'%.4f'%m[j]+'_'+boundary+'.txt'
  data.append(np.loadtxt(Name,dtype=float,delimiter=' ',skiprows=1,ndmin=2))
 for j in range(bdata):
     lut={int(row[1]):row[2] for row in data[j]}#Lz -> energy
     E.append([lut[int(Lzj)] for Lzj in Lz[j]])#KeyError if a requested Lz is not in the file
 return np.array(E,dtype=tuple)
def read_data_an_ref(Lt,L,Lz,m,boundary,dim):
 data=[]
 bdata=np.size(m)
 E=[]
 for j in range(bdata):
  Name='../Energy_analytical/Energy_an_refA2d_'+str(Lt)+'_'
  for i in range(1,dim):
   Name=Name+str(L)+'_'
  Name=Name+str(Lz[j][0])+'m_'+'%.4f'%m[j]+'_'+boundary+'.txt'
  data.append(np.loadtxt(Name,dtype=float,delimiter=' ',skiprows=1,ndmin=2))
 for j in range(bdata):
         E.append(data[j][0][2])
 return np.array(E,dtype=tuple)


def Cas_Per(mL):
    aux2=math.exp(-mL)
    return -(mL*mpmath.polylog(2,aux2)+mpmath.polylog(3,aux2))/(2*math.pi*mL**2)
def Cas_Dir(mL):
    aux2=math.exp(-2*mL)
    return -(2*mL*mpmath.polylog(2,aux2)+mpmath.polylog(3,aux2))/(16*math.pi*mL**2)

def plot_Casimir(C,x,m,bcond,yscale,dimension,plot,L):
    xdata=np.array(np.concatenate(x).flat)#fuse all the x
    DP=np.linspace(min(xdata), max(xdata), num=100)#for plot space x
    Econt=[]
    if bcond=='periodic':
            
            for mL in DP:
                Econt.append(Cas_Per(mL))
            plt.plot(DP,Econt,label='Continuum')
            plt.ylabel('$E_P/(Am^2)$',fontsize=14)

    else:
            for mL in DP:
                Econt.append(Cas_Dir(mL))
            plt.plot(DP,Econt,label='Continuum')
            plt.ylabel('$E_D/(Am^2)$',fontsize=14)
    for i in range(len(L)):
                E=np.array(np.concatenate(C[i]).flat)#fuse all the Cas energy
                plt.plot(xdata,E,'o',markersize=4,label='$N_A=$%g' %L[i])
    plt.xlabel('$mL$',fontsize=14)

    if yscale=='log':
      if bcond=='periodic':
        plt.yscale('symlog', linthresh=0.000001)
        plt.yticks([-0.1,-0.001,-0.00001,0])
      else:
        plt.yscale('symlog', linthresh=0.000000000032)
        plt.yticks([-0.01,-0.0001,-0.000001,-0.00000001,-0.0000000001,0])
        #plt.ylim(-0.1,0.000000000001)
     #plt.yscale('log')
    plt.legend( prop={'size': 8},ncol=3,loc=4)
    #plt.title('V=%g %g %g' %(Lt,L,Lz[0][-1] ))
    
    #if bcond=='Dirichlet':
     #    plt.savefig('Casimir_scalar_2d_full.pdf',bbox_inches='tight')
    if plot:
     if bcond=='periodic':
      plt.savefig('Casimir_lattice_2d_sizeA_log_periodics.pdf',bbox_inches='tight')
     else:
      plt.savefig('Casimir_lattice_2d_sizeA_log_Dirs.pdf',bbox_inches='tight')
     plt.show()
def plot_quot(C,x,m,bcond,yscale,dimension,plot,L):
    xdata=np.array(np.concatenate(x).flat)#fuse all the x
    Econt=[]
    if bcond=='periodic':
        for mL in xdata:
                Econt.append(Cas_Per(mL))
        plt.ylabel('$E_P/(Am^2)$',fontsize=14)

    else:
            for mL in xdata:
                Econt.append(Cas_Dir(mL))
            plt.ylabel('$E_D/(Am^2)$',fontsize=14)
 
    for i in range(len(L)): 
        E=np.array(np.concatenate(C[i]).flat)#fuse all the Cas energy
        q=[]
        for j in range (len(xdata)):
         q.append(E[j]/Econt[j])
        plt.plot(xdata,q,'o',label='$N_A=$%g' %L[i])           
    plt.xlabel('$mL$',fontsize=14)

    if yscale=='log':
      if bcond=='periodic':
        plt.yscale('symlog', linthresh=0.000001)
        plt.yticks([-0.1,-0.001,-0.00001,0])
      else:
        plt.yscale('symlog', linthresh=0.0000000000032)
        plt.yticks([-0.01,-0.0001,-0.000001,-0.00000001,-0.0000000001,0])
        #plt.ylim(-0.1,0.000000000001)
     #plt.yscale('log')
    plt.legend( prop={'size': 8},ncol=3,loc=2)
    #plt.title('V=%g %g %g' %(Lt,L,Lz[0][-1] ))
    
    #if bcond=='Dirichlet':
     #    plt.savefig('Casimir_scalar_2d_full.pdf',bbox_inches='tight')
    if plot:
     if bcond=='periodic':
      plt.savefig('Casimir_lattice_2d_sizeA_periodic.pdf',bbox_inches='tight')
     else:
      plt.savefig('Casimir_lattice_2d_sizeA_log_Dir.pdf',bbox_inches='tight')
     plt.show()    
     

d=2
Lc=[32,48,64,96,160]
Casimir=[]
ERef=[]
for i in range(len(Lc)):
 Lt=Lc[i] #t dimension of the lattice
 L=Lc[i]  #x dimension of the lattice
 Lref=96
 Lz_ref=[[Lref],[Lref],[Lref],[Lref],[Lref],[Lref],[Lref],[Lref],[Lref],[Lref]] #y dimension of the lattice of the reference value
 m=[0.1,0.2,0.3,0.4]
 #m=[0.1]
 Lz=[[12,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24]]#min and max y dimension for each beta
 Lv=create_Lz(Lz, 1,False)#compute an array with all the y values, jump is the distance between them
 Lv=[[12,14,16,18,20,22,24],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]
 boundary_condition='per'#select boundary condition; per is periodic, dir is Dirichlet   
 #E=create_points(Lt,L,Lv,m,boundary_condition,d)
 #ERef=create_ref(Lt,L,Lz_ref,m,boundary_condition,d)

 E=read_data_an(Lt,L,Lv,m,boundary_condition,d)
 ERef=read_data_an_ref(Lt,L,Lz_ref,m,boundary_condition,d)
 #print(E)
 #print(ERef)
 Casimir.append([])
 print_values=False
 Casimir[i],xdata=compute_casimir_analytic(E,Lv,m,ERef,Lz_ref,print_values,boundary_condition,d)#compute casimir, error and physical distance of each point

boundary_condition='periodic'#select boundary condition; per is periodic, dir is Dirichlet
y_scale='log'#log for log scale

pshow=True
plot_Casimir(Casimir,xdata,m,boundary_condition,y_scale,d,pshow,Lc)

 

d=2
Lc=[32,48,64,96,160]
Casimir=[]
ERef=[]
for i in range(len(Lc)):
 Lt=Lc[i] #t dimension of the lattice
 L=Lc[i] #x dimension of the lattice
 Lz_ref=[[96],[96],[96],[96],[96],[96],[96],[96],[96],[96]] #y dimension of the lattice of the reference value
 
     
 m=[0.1,0.2,0.3,0.4]
 #m=[0.1]
 Lz=[[12,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24]]#min and max y dimension for each beta
 Lv=create_Lz(Lz, 1,False)#compute an array with all the y values, jump is the distance between them
 Lv=[[12,14,16,18,20,22,24],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]
 boundary_condition='dir'#select boundary condition; per is periodic, dir is Dirichlet   
  #E=create_points(Lt,L,Lv,m,boundary_condition,d)
 #ERef=create_ref(Lt,L,Lz_ref,m,'per',d)
 #ERefD=create_ref(Lt,L,Lz_ref,m,boundary_condition,d)

 E=read_data_an(Lt,L,Lv,m,boundary_condition,d)
 ERef=read_data_an_ref(Lt,L,Lz_ref,m,'per',d)
 ERefD=read_data_an_ref(Lt,L,Lz_ref,m,'dir',d)
 #print(E)
 #print(ERef)
 Casimir.append([])
 print_values=False
 Casimir[i],xdata=compute_casimir_analytic(E,Lv,m,[ERef,ERefD],Lz_ref,print_values,boundary_condition,d)#compute casimir, error and physical distance of each point

boundary_condition='Dirichlet'#select boundary condition; per is periodic, dir is Dirichlet
y_scale='log'#log for log scale

pshow=True


plot_Casimir(Casimir,xdata,m,boundary_condition,y_scale,d,pshow,Lc)
#plot_quot(Casimir,xdata,m,boundary_condition,y_scale,d,False,Lc)