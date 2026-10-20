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
    Name='../Energy_analytical/Energy_anL22d_'+str(Lt)+'_'
    for i in range(1,dim):
     Name=Name+str(L)+'_'
    Name=Name+'Lz_'+str(Lz)+'_'+bound+'.txt'
    f = open(Name, "w")
    f.write("Lz M Energy\n")
    for i in range(np.size(m)):
     f.write(str(Lz)+' '+str(m[i])+' '+str(energy[i])+'\n')
    f.close()
def write_data_ref(Lt,L,Lz,Lr,m,energy,bound,dim):
    Name='../Energy_analytical/Energy_an_refL22d_'+str(Lt)+'_'
    for i in range(1,dim):
     Name=Name+str(L)+'_'
    Name=Name+str(Lr)+'_Lc_'+str(Lz)+'_'+bound+'.txt'
    f = open(Name, "w")
    f.write("Lz M Energy\n")
    for i in range(np.size(m)):
     f.write(str(Lr)+' '+str(m[i])+' '+str(energy[i])+'\n')
    f.close()
def create_points(Lt,L,Lz,m,bound,dim):
    E=[]
    for i in range (len(Lz)):
        E.append([])
        for j in range (len(m[i])):
             E[i].append(ener_2d(Lt, L, Lz[i], m[i][j], bound))
        write_data(Lt,L,Lz[i],m[i],E[i],bound,dim)
    return E
def create_ref(Lt,L,Lz,Lr,m,bound,dim):
    E_ref=[]
    for i in range (len(Lz)):
        E_ref.append([])
        for j in range (len(m[i])):
             E_ref[i].append(ener_2d(Lt, L, Lr[i][0], m[i][j], bound))
        write_data_ref(Lt,L,Lz[i],Lr[i][0],m[i],E_ref[i],bound,dim)
    return E_ref
def read_data_an(Lt,L,Lz,m,boundary,dim):
 data=[]
 bdata=np.size(Lz)
 E=[]
 for j in range(bdata):
  Name='../Energy_analytical/Energy_anL22d_'+str(Lt)+'_'
  for i in range(1,dim):
    Name=Name+str(L)+'_'
  Name=Name+'Lz_'+str(Lz[j])+'_'+boundary+'.txt'
  data.append(np.loadtxt(Name,dtype=float,delimiter=' ',skiprows=1,ndmin=2))
 for j in range(bdata):
     E.append([])
     for i in range (len(data[j])):
         E[j].append(data[j][i][2])
 return np.array(E,dtype=tuple)
def read_data_an_ref(Lt,L,Lref,Lz,m,boundary,dim):
 data=[]
 bdata=np.size(Lz)
 E=[]
 for j in range(bdata):
   Name='../Energy_analytical/Energy_an_refL22d_'+str(Lt)+'_'
   for i in range(1,dim):
    Name=Name+str(L)+'_'
   Name=Name+str(Lref)+'_Lc_'+str(Lz[j])+'_'+boundary+'.txt'
   data.append(np.loadtxt(Name,dtype=float,delimiter=' ',skiprows=1,ndmin=2))
 c=0
 print(data)
 for j in range(bdata):
     E.append([])
     for i in range(np.size(m[j])):
         E[j].append(data[j][i][2])
         c=c+1
 return np.array(E,dtype=tuple)
def compute_casimir_analytic(E,Lz,m,E_ref,xref,cprint,boundary,dim):
    cas=[]
    e_cas=[]
    xdata=[]
    if boundary=='per':
     for i in range(np.size(Lz)):
        cas.append([])
        xdata.append([])
        for j in range(np.size(m[i])):
            a=m[i][j]
            cas[i].append((E[i][j]-E_ref[i][j])/a**dim*Lz[i])
            xdata[i].append(a*Lz[i])
            if cprint:
                print("Mass",m[i][j],"L",Lz[i],"x",np.round(xdata[i][j],3),"C",np.round(cas[i][j],9))
     return cas,xdata
    if boundary=='dir':
     for i in range(np.size(Lz)):
        cas.append([])
        xdata.append([])
        for j in range(np.size(m[i])):
            a=m[i][j]
            L=Lz[i]
            Ld=xref-L
            cas[i].append((E[i][j]*L-E_ref[1][i][j]*xref+Ld*E_ref[0][i][j])/a**dim)
            xdata[i].append(a*Lz[i])
            if cprint:
                print("\nMass",m[i][j],"L",Lz[i],"x",np.round(xdata[i][j],3),"C",np.round(cas[i][j],9))
    return cas,xdata

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
    for i in range(len(Lc)):
                E=[]
                xd=[]
                for j in range(np.size (m[i])):
                 E.append(C[i][j])#fuse all the Cas energy
                 xd.append(x[i][j])
                plt.plot(xd,E,'o',markersize=4,label='$N_{L}=$%g' %L[i])
    plt.xlabel('$mL$',fontsize=14)

    if yscale=='log':
      if bcond=='periodic':
        plt.yscale('symlog', linthresh=0.0000026)
        plt.yticks([-0.1,-0.001,-0.00001,0])
      else:
        plt.yscale('symlog', linthresh=0.000000000032)
        plt.yticks([-0.01,-0.0001,-0.000001,-0.00000001,-0.0000000001,0])
        #plt.ylim(-0.1,0.000000000001)
     #plt.yscale('log')
    plt.legend( prop={'size': 8},ncol=3,loc=2)
    #plt.title('V=%g %g %g' %(Lt,L,Lz[0][-1] ))
    
    #if bcond=='Dirichlet':
     #    plt.savefig('Casimir_scalar_2d_full.pdf',bbox_inches='tight')
    if plot:
     if bcond=='periodic':
      plt.savefig('Casimir_lattice_2d_sizeL_periodics.pdf',bbox_inches='tight')
     else:
      plt.savefig('Casimir_lattice_2d_sizeL_Dirs.pdf',bbox_inches='tight')
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
        plt.yscale('symlog', linthresh=0.0001)
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
      plt.savefig('Casimir_lattice_2d_sizeA_log_periodic.pdf',bbox_inches='tight')
     else:
      plt.savefig('Casimir_lattice_2d_sizeA_logDir.pdf',bbox_inches='tight')
     plt.show()    
     
     

d=2
#Lc=[2,4,6,8,10,12,16,20,24,28]
Lc=[2,6,10,16,24,32]
Casimir=[]
ERef=[]

Lt=96 #t dimension of the lattice
L=96  #x dimension of the lattice
Lref=96
Lz_ref=[[Lref],[Lref],[Lref],[Lref],[Lref],[Lref],[Lref],[Lref],[Lref],[Lref]] #y dimension of the lattice of the reference value
m=[1/14,1/12,0.1,0.125,1/6,0.2,0.25,1/3,0.5,1]
 #m=[0.1]
Lz=[[12,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24],[8,24]]#min and max y dimension for each beta
Lv=create_Lz(Lz, 1,False)#compute an array with all the y values, jump is the distance between them
Lv=[[28],[24],[20],[16],[12],[10],[8],[6],[4],[2]]
Lv=Lc
#m=[[0.5,1],[0.25,0.5],[0.5/3,1/3],[0.125,0.25],[0.1,0.2],[0.5/6,1/6],[0.0625,0.125],[0.05,0.1],[0.5/12,1/12],[0.5/14,1/14]]
m=[[0.5,0.625,0.75,0.875,1],[0.25,0.3125,0.375,0.4375,0.5],[0.125,0.15625,0.1875,0.21875,0.25],[0.0625,0.078125,0.09375,0.125],[0.5/12,5/96,0.0625,7/96,1/12],[0.03125,0.0390625,0.046875,0.0625,0.0546875]]
m=[]
for i in range(len(Lc)):
    m.append([])
    for j in range(17):
        m[i].append((1+0.5*j)/Lc[i])
print(m)
boundary_condition='per'#select boundary condition; per is periodic, dir is Dirichlet   
#E=create_points(Lt,L,Lv,m,boundary_condition,d)
#ERef=create_ref(Lt,L,Lv,Lz_ref,m,boundary_condition,d)
E=read_data_an(Lt,L,Lv,m,boundary_condition,d)
ERef=read_data_an_ref(Lt,L,Lref,Lc,m,boundary_condition,d)
Casimir.append([])
print_values=False
Casimir,xdata=compute_casimir_analytic(E,Lv,m,ERef,Lz_ref,print_values,boundary_condition,d)#compute casimir, error and physical distance of each point
boundary_condition='periodic'#select boundary condition; per is periodic, dir is Dirichlet
y_scale='log'#log for log scale

pshow=True
plot_Casimir(Casimir,xdata,m,boundary_condition,y_scale,d,pshow,Lc)

 

d=2
Lc=[2,4,6,8,12,16,20,24,28]
Lc=[2,6,10,16,24,32]
Casimir=[]
ERef=[]
Lt=96 #t dimension of the lattice
L=96 #x dimension of the lattice
Lref=96
Lz_ref=[[Lref],[Lref],[Lref],[Lref],[Lref],[Lref],[Lref],[Lref],[Lref],[Lref]] #y dimension of the lattice of the reference value

    
m=[]
#m=[0.1]
for i in range(len(Lc)):
   m.append([])
   for j in range(17):
       m[i].append((1+0.5*j)/Lc[i])
Lv=Lc
boundary_condition='dir'#select boundary condition; per is periodic, dir is Dirichlet   
#E=create_points(Lt,L,Lv,m,boundary_condition,d)
#ERef=create_ref(Lt,L,Lv,Lz_ref,m,'per',d)
#ERefD=create_ref(Lt,L,Lv,Lz_ref,m,boundary_condition,d)

E=read_data_an(Lt,L,Lv,m,boundary_condition,d)
ERef=read_data_an_ref(Lt,L,Lref,Lc,m,'per',d)
ERefD=read_data_an_ref(Lt,L,Lref,Lc,m,'dir',d)
#print(E)
#print(ERef)
Casimir.append([])
print_values=False
Casimir,xdata=compute_casimir_analytic(E,Lv,m,[ERef,ERefD],Lref,print_values,boundary_condition,d)#compute casimir, error and physical distance of each point
boundary_condition='Dirichlet'#select boundary condition; per is periodic, dir is Dirichlet
y_scale='log'#log for log scale

pshow=True


plot_Casimir(Casimir,xdata,m,boundary_condition,y_scale,d,pshow,Lc)
#plot_quot(Casimir,xdata,m,boundary_condition,y_scale,d,False,Lc)
