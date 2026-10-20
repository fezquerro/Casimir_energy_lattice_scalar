# -*- coding: utf-8 -*-
"""
Created on Fri May  9 17:49:57 2025

@author: fer
"""
import numpy as np
import scipy.linalg as linalg
def M_mat_mod_dir(Lt,Lx,Ly,Lz,m,a,b):#define the transference matrix
 V=Lt*Lx*Ly*Lz
 C=np.zeros((V,V))
 m2=m**2
 for i in range(0,V):
     C[i,i]=(6*a+2*b+m2*a)
     posz=i%Lz
     posy=int((i % (Lz*Ly)) / Lz)
     posx=int((i % (Lz*Ly*Lx)) / (Lz*Ly))
     post=int(i/(Lx*Ly*Lz))
    # print(posz,posy,posx)
     if(posz==0):
         C[i,i+Lz-1]=0
     else:
         C[i,i-1]+=-a
     if(posz==(Lz-1)):
         C[i,i-Lz+1]=0
     else:
         C[i,i+1]=-a
     if(posy==0):
          C[i,i+(Ly-1)*Lz]=-a
     else:
          C[i,i-Lz]=-a
     if(posy==(Ly-1)):
          C[i,i-Lz*(Ly-1)]=-a
     else:
          C[i,i+Lz]=-a
     if(posx==0):
         C[i,i+(Lx-1)*Ly*Lz]=-a
     else:
         C[i,i-Lz*Ly]+=-a
     if(posx==(Lx-1)):
         C[i,i-Ly*(Lx-1)*Lz]=-a
     else:
         C[i,i+Lz*Ly]=-a
     if(post==0):
             C[i,i+(Lt-1)*Lx*Ly*Lz]=-b
     else:
             C[i,i-Lz*Ly*Lx]+=-b
     if(post==(Lt-1)):
         C[i,i-Lx*Ly*(Lt-1)*Lz]=-b
     else:
         C[i,i+Lz*Ly*Lx]=-b

 return C

def M_mat_mod_neu(Lt,Lx,Ly,Lz,m,a,b):#define the transference matrix
 V=Lt*Lx*Ly*Lz
 C=np.zeros((V,V))
 m2=m**2
 for i in range(0,V):
     C[i,i]=(6*a+2*b+m2*a)
     posz=i%Lz
     posy=int((i % (Lz*Ly)) / Lz)
     posx=int((i % (Lz*Ly*Lx)) / (Lz*Ly))
     post=int(i/(Lx*Ly*Lz))
    # print(posz,posy,posx)
     if(posz==0):
         C[i,i+Lz-1]=-0
         C[i,i]=(5*a+2*b+m2*a)
     else:
         C[i,i-1]+=-a
     if(posz==(Lz-1)):
         C[i,i-Lz+1]=-0
         C[i,i]=(5*a+2*b+m2*a)
     else:
         C[i,i+1]=-a
     if(posy==0):
          C[i,i+(Ly-1)*Lz]=-a
     else:
          C[i,i-Lz]=-a
     if(posy==(Ly-1)):
          C[i,i-Lz*(Ly-1)]=-a
     else:
          C[i,i+Lz]=-a
     if(posx==0):
         C[i,i+(Lx-1)*Ly*Lz]=-a
     else:
         C[i,i-Lz*Ly]+=-a
     if(posx==(Lx-1)):
         C[i,i-Ly*(Lx-1)*Lz]=-a
     else:
         C[i,i+Lz*Ly]=-a
     if(post==0):
             C[i,i+(Lt-1)*Lx*Ly*Lz]=-b
     else:
             C[i,i-Lz*Ly*Lx]+=-b
     if(post==(Lt-1)):
         C[i,i-Lx*Ly*(Lt-1)*Lz]=-b
     else:
         C[i,i+Lz*Ly*Lx]=-b

 return C

def M_mat_mod_anti(Lt,Lx,Ly,Lz,m,a,b):#define the transference matrix
 V=Lt*Lx*Ly*Lz
 C=np.zeros((V,V))
 m2=m**2
 for i in range(0,V):
     C[i,i]=(6*a+2*b+m2*a)
     posz=i%Lz
     posy=int((i % (Lz*Ly)) / Lz)
     posx=int((i % (Lz*Ly*Lx)) / (Lz*Ly))
     post=int(i/(Lx*Ly*Lz))
    # print(posz,posy,posx)
     if(posz==0):
         C[i,i+Lz-1]=a
     else:
         C[i,i-1]+=-a
     if(posz==(Lz-1)):
         C[i,i-Lz+1]=a
     else:
         C[i,i+1]=-a
     if(posy==0):
          C[i,i+(Ly-1)*Lz]=-a
     else:
          C[i,i-Lz]=-a
     if(posy==(Ly-1)):
          C[i,i-Lz*(Ly-1)]=-a
     else:
          C[i,i+Lz]=-a
     if(posx==0):
         C[i,i+(Lx-1)*Ly*Lz]=-a
     else:
         C[i,i-Lz*Ly]+=-a
     if(posx==(Lx-1)):
         C[i,i-Ly*(Lx-1)*Lz]=-a
     else:
         C[i,i+Lz*Ly]=-a
     if(post==0):
             C[i,i+(Lt-1)*Lx*Ly*Lz]=-b
     else:
             C[i,i-Lz*Ly*Lx]+=-b
     if(post==(Lt-1)):
         C[i,i-Lx*Ly*(Lt-1)*Lz]=-b
     else:
         C[i,i+Lz*Ly*Lx]=-b

 return C

def M_mat_mod_Zar(Lt,Lx,Ly,Lz,m,a,b):#define the transference matrix
 V=Lt*Lx*Ly*Lz
 C=np.zeros((V,V))
 m2=m**2
 for i in range(0,V):
     C[i,i]=(6*a+2*b+m2*a)
     posz=i%Lz
     posy=int((i % (Lz*Ly)) / Lz)
     posx=int((i % (Lz*Ly*Lx)) / (Lz*Ly))
     post=int(i/(Lx*Ly*Lz))
    # print(posz,posy,posx)
     if(posz==0):
         C[i,i+Lz-1]=0
     else:
         C[i,i-1]+=-a
     if(posz==(Lz-1)):
         C[i,i-Lz+1]=0
         C[i,i]=(5*a+2*b+m2*a)
     else:
         C[i,i+1]=-a
     if(posy==0):
          C[i,i+(Ly-1)*Lz]=-a
     else:
          C[i,i-Lz]=-a
     if(posy==(Ly-1)):
          C[i,i-Lz*(Ly-1)]=-a
     else:
          C[i,i+Lz]=-a
     if(posx==0):
         C[i,i+(Lx-1)*Ly*Lz]=-a
     else:
         C[i,i-Lz*Ly]+=-a
     if(posx==(Lx-1)):
         C[i,i-Ly*(Lx-1)*Lz]=-a
     else:
         C[i,i+Lz*Ly]=-a
     if(post==0):
             C[i,i+(Lt-1)*Lx*Ly*Lz]=-b
     else:
             C[i,i-Lz*Ly*Lx]+=-b
     if(post==(Lt-1)):
         C[i,i-Lx*Ly*(Lt-1)*Lz]=-b
     else:
         C[i,i+Lz*Ly*Lx]=-b

 return C

def eigen(i,L):
    return (2*np.sin(np.pi*i/L))**2
def eigend(i,L):
    return (2*np.sin(np.pi*i/L/2))**2
def eigenn(i,L):
    return (2*np.sin(np.pi*i/(L)/2))**2
def eigenan(i,L):
    return (2*np.sin(np.pi*(i+0.5)/L))**2
def eigenZ(i,L):
    return (2*np.sin(np.pi*(i+0.5)/(2*L-1)))**2
def check_eigenvalues_mod_neu(Lt,Lx,Ly,Lz,m,a,b):
 aux=[]
 for i in range(Lx):
     for j in range(Ly):
         for k in range (0,Lz):
             for t in range (Lt):
                 aux.append(b*eigen(t,Lt)+a*eigen(i,Lx)+a*eigen(j,Ly)+a*eigenn(k,Lz)+a*m**2)

 return aux
def check_eigenvalues_mod_dir(Lt,Lx,Ly,Lz,m,a,b):
 aux=[]
 for i in range(Lx):
     for j in range(Ly):
         for k in range (1,Lz):
             for t in range (Lt):
                 aux.append(b*eigen(t,Lt)+a*eigen(i,Lx)+a*eigen(j,Ly)+a*eigend(k,Lz)+a*m**2)

 return aux
def check_eigenvalues_mod_anti(Lt,Lx,Ly,Lz,m,a,b):
 aux=[]
 for i in range(Lx):
     for j in range(Ly):
         for k in range (1,Lz+1):
             for t in range (Lt):
                 aux.append(b*eigen(t,Lt)+a*eigen(i,Lx)+a*eigen(j,Ly)+a*eigenan(k,Lz)+a*m**2)

 return aux

def check_eigenvalues_mod_Zar(Lt,Lx,Ly,Lz,m,a,b):
 aux=[]
 for i in range(Lx):
     for j in range(Ly):
         for k in range (0,Lz-1):
             for t in range (Lt):
                 aux.append(b*eigen(t,Lt)+a*eigen(i,Lx)+a*eigen(j,Ly)+a*eigenZ(k,Lz)+a*m**2)

 return aux
Lt=3
Lx=3
Ly=3
Lz=4
m=0
a=1
b=1

#print(np.sort(linalg.eigvals(M_mat_mod_neu(Lt,Lx,Ly,Lz,m,a,b))).real)
#print(np.sort(check_eigenvalues_mod_neu(Lt, Lx, Ly, Lz, m, a, b)))
#print(np.sum(np.sort(linalg.eigvals(M_mat_mod_neu(Lt,Lx,Ly,Lz,m,a,b)))-np.sort(check_eigenvalues_mod_neu(Lt, Lx, Ly, Lz, m, a, b))))
print(np.sort(linalg.eigvals(M_mat_mod_Zar(Lt,Lx,Ly,Lz-1,m,a,b))).real)
print(np.sort(check_eigenvalues_mod_Zar(Lt, Lx, Ly, Lz, m, a, b)))
print(np.sum(np.sort(linalg.eigvals(M_mat_mod_Zar(Lt,Lx,Ly,Lz-1,m,a,b)))-np.sort(check_eigenvalues_mod_Zar(Lt, Lx, Ly, Lz, m, a, b))))

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
 if bound=='neu':
   for t in range(Lt):
      for x in range(Lx):
          for y in range(Ly):
                a=eigen(t,Lt)
                aux+=a/(eigen(x,Lx)+eigenn(y,Ly)+m**2+a)
   return 0.5-aux/Lt/Lx/Ly 
 if bound=='anti':
   for t in range(Lt):
      for x in range(Lx):
          for y in range(Ly):
                a=eigen(t,Lt)
                aux+=a/(eigen(x,Lx)+eigenan(y,Ly)+m**2+a)
   return 0.5-aux/Lt/Lx/Ly
 if bound=='zar':
   for t in range(Lt):
      for x in range(Lx):
          for y in range(0,Ly-1):
                a=eigen(t,Lt)
                aux+=a/(eigen(x,Lx)+eigenZ(y,Ly)+m**2+a)
   return 0.5*(1-1/Ly)-aux/Lt/Lx/Ly  

def ener_2d2(Lt,Lx,Lz,m,bound):
 aux=0
 if bound=='dir':
  for t in range(Lt):
     for x in range(Lx):
             for z in range(1,Lz):
               aux+=(eigen(x,Lx)+eigend(z,Lz)+m**2-eigen(t,Lt))/(eigen(x,Lx)+eigend(z,Lz)+m**2+eigen(t,Lt))
  return aux/Lt/Lx/Lz/2
 if bound=='per':
  for t in range(Lt):
     for x in range(Lx):
             for z in range(Lz):
               aux+=(eigen(x,Lx)+eigen(z,Lz)+m**2-eigen(t,Lt))/(eigen(x,Lx)+eigen(z,Lz)+m**2+eigen(t,Lt))
  return aux/Lt/Lx/Lz/2
 if bound=='neu':
  for t in range(Lt):
     for x in range(Lx):
             for z in range(Lz):
               aux+=(eigen(x,Lx)+eigenn(z,Lz)+m**2-eigen(t,Lt))/(eigen(x,Lx)+eigenn(z,Lz)+m**2+eigen(t,Lt))
  return aux/Lt/Lx/Lz/2
 if bound=='anti':
  for t in range(Lt):
     for x in range(Lx):
             for z in range(Lz):
               aux+=(eigen(x,Lx)+eigenan(z,Lz)+m**2-eigen(t,Lt))/(eigen(x,Lx)+eigenan(z,Lz)+m**2+eigen(t,Lt))
  return aux/Lt/Lx/Lz/2
 if bound=='zar':
  for t in range(Lt):
     for x in range(Lx):
             for z in range(0,Lz-1):
               aux+=(eigen(x,Lx)+eigenZ(z,Lz)+m**2-eigen(t,Lt))/(eigen(x,Lx)+eigenZ(z,Lz)+m**2+eigen(t,Lt))
  return aux/Lt/Lx/Lz/2


L=96
Lz=18
m=0.3
print(ener_2d(32, 32, 4, 0.05, 'zar'))
#for i in range(8):
#   print((ener_2d2(96, 96, 10+10*i, 0.1, 'zar')-a)*(10+10*i))
#   print((ener_2d2(96, 96, 10+10*i, 0.1, 'dir')-a)*(10+10*i))
print((ener_2d(L, L, Lz, m, 'zar')*Lz-ener_2d(L, L, L, m, 'zar')*L-(Lz-L)*ener_2d(L, L, L, m, 'per'))/m**2)
print((ener_2d2(L, L, Lz, m, 'zar')*Lz-ener_2d2(L, L, L, m, 'zar')*L-(Lz-L)*ener_2d2(L,L, L, m, 'per'))/m**2)
print((ener_2d(L, L, Lz, m, 'dir')*Lz-ener_2d(L, L, L, m, 'dir')*L-(Lz-L)*ener_2d(L, L, L, m, 'per'))/m**2)
print((ener_2d2(L, L, Lz, m, 'dir')*Lz-ener_2d2(L, L, L, m, 'dir')*L-(Lz-L)*ener_2d2(L, L, L, m, 'per'))/m**2)
#print((ener_2d(L, L, Lz, m, 'anti')-ener_2d(L, L, L, m, 'anti'))*Lz/m**2)
#print((ener_2d(L, L, Lz, m, 'anti')-ener_2d(L, L, L, m, 'per'))*Lz/m**2)
import mpmath as mp


def Cas_Pergen(L,m,D,ep):
    aux=m*L
    auxD=(D+1)/2
    s=mp.besselk(auxD,aux)
    aux2=s
    c=1
    while(abs(aux2/s)>ep):
        c=c+1
        aux2=mp.besselk(auxD,aux*c)/pow(c,auxD)
        #print(aux2/s)
        s=s+aux2
    return -s/(pow(2,auxD-1)*pow(mp.pi,auxD)*pow(m*L,auxD-1))
def Cas_Dirgen(mL,D,ep):
    auxD=(D+1)/2
    s=mp.besselk(auxD,2*mL)
    aux2=s
    c=1
    while(abs(aux2/s)>ep):
        c=c+1
        aux2=mp.besselk(auxD,2*mL*c)/pow(c,auxD)
        s=s+aux2
    return -s/(pow(2,D)*pow(mp.pi,auxD)*pow(mL,auxD-1))

def Cas_AntiPergen(L,m,D,ep):
    aux=m*L
    auxD=(D+1)/2
    s=mp.besselk(auxD,aux)
    aux2=s
    c=1
    while(abs(aux2/s)>ep):
        c=c+1
        aux2=mp.besselk(auxD,aux*c)/pow(c,auxD)
        #print(aux2/s)
        s=s+aux2*(-1)**(c+1)
    return s/(pow(2,auxD-1)*pow(mp.pi,auxD)*pow(m*L,auxD-1))
def Cas_Zarembagen(L,m,D,ep):
    aux=m*L
    auxD=(D+1)/2
    s=mp.besselk(auxD,2*aux)
    aux2=s
    c=1
    while(abs(aux2/s)>ep):
        c=c+1
        aux2=mp.besselk(auxD,2*aux*c)/pow(c,auxD)
        #print(aux2/s)
        s=s+aux2*(-1)**(c+1)
    return s/(pow(2,D)*pow(mp.pi,auxD)*pow(m*L,auxD-1))
print(Cas_Zarembagen(Lz-1,m,2,1e-20))
print(Cas_Zarembagen(Lz-0.5,m,2,1e-20))
print(Cas_Zarembagen(Lz,m,2,1e-20))
print(Cas_Dirgen(Lz*m,2,1e-20))
#print(Cas_AntiPergen(Lz,m,2,1e-20))

