# -*- coding: utf-8 -*-
"""
Created on Fri May  9 13:49:19 2025

@author: usuario
"""
import math
import numpy as np
import mpmath as mp

def eigen(i,L):
    return (2*np.sin(np.pi*i/L))**2
def eigend(i,L):
    return (2*np.sin(np.pi*i/L/2))**2
def eigenm(i,L):
    return (2*np.sin(np.pi*i/(L+1)/2))**2

def ener_1d(Lt,Ly,m,bound):
 aux=0;
 if bound=='dir':
  for t in range(Lt):
         for y in range(1,Ly):
               a=eigen(t,Lt)
               aux+=a/(eigend(y,Ly)+m**2+a)
  return 0.5*(1-1/Ly)-aux/Lt/Ly
 if bound=='per':
   for t in range(Lt):
         for y in range(Ly):
               a=eigen(t,Lt)
               aux+=a/(eigen(y,Ly)+m**2+a)
   return 0.5-aux/Lt/Ly

def ener_2d(Lt,Lx,Ly,m,bound):
 #np.longdouble is just float64 on Windows, so the terms are summed with math.fsum (exactly rounded)
 terms=[]
 if bound=='dir':
  for t in range(Lt):
     for x in range(Lx):
         for y in range(1,Ly):
               a=eigen(t,Lt)
               terms.append(a/(eigen(x,Lx)+eigend(y,Ly)+m**2+a))
  return 0.5*(1-1/Ly)-math.fsum(terms)/Lt/Lx/Ly

 if bound=='per':
   for t in range(Lt):
     for x in range(Lx):
         for y in range(Ly):
                a=eigen(t,Lt)
                terms.append(a/(eigen(x,Lx)+eigen(y,Ly)+m**2+a))
   return 0.5-math.fsum(terms)/Lt/Lx/Ly

def ener_3d(Lt,Lx,Ly,Lz,m,bound):
 aux=0
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

def ener_3d2(Lt,Lx,Ly,Lz,m,bound):
 aux=0
 if bound=='dir':
  for t in range(Lt):
     for x in range(Lx):
      for y in range(Ly):
         for z in range(1,Lz):
               a=eigen(t,Lt)
               b=eigen(x,Lx)+eigen(y,Ly)+eigend(z,Lz)+m**2
               aux+=(b-a)/(a+b)
  return aux/Lt/Lx/Ly/Lz/2
 if bound=='per':
   for t in range(Lt):
     for x in range(Lx):
         for y in range(Ly):
             for z in range(Lz):
                a=eigen(t,Lt)
                b=eigen(x,Lx)+eigen(y,Ly)+eigen(z,Lz)+m**2
                aux+=(b-a)/(a+b)
   return aux/Lt/Lx/Ly/Lz/2

def ener_4d(Lt,Lx,Ly,Lk,Lz,m,bound):
 aux=0;
 if bound=='dir':
  for t in range(Lt):
     #print(t)
     for x in range(Lx):
      for y in range(Ly):
        for k in range(Lk):
         for z in range(1,Lz):
               a=eigen(t,Lt)
               aux+=eigen(t,Lt)/(eigen(x,Lx)+eigen(y,Ly)+eigen(k,Lk)+eigend(z,Lz)+m**2+a)
  return 0.5*(1-1/Lz)-aux/Lt/Lx/Ly/Lk/Lz
 if bound=='per':
   for t in range(Lt):
    # print(t)
     for x in range(Lx):
         for y in range(Ly):
           for k in range(Lk):
             for z in range(Lz):
                a=eigen(t,Lt)
                aux+=a/(eigen(x,Lx)+eigen(y,Ly)+eigen(k,Lk)+eigen(z,Lz)+m**2+a)
   return 0.5-aux/Lt/Lx/Ly/Lk/Lz

def ener_4d2(Lt,Lx,Ly,Lk,Lz,m,bound):
 aux=0;
 if bound=='dir':
  for t in range(Lt):
     #print(t)
     for x in range(Lx):
      for y in range(Ly):
        for k in range(Lk):
         for z in range(1,Lz):
               a=eigen(t,Lt)
               b=eigen(x,Lx)+eigen(y,Ly)+eigen(k,Lk)+eigend(z,Lz)+m**2
               aux+=(b-a)/(a+b)
  return aux/Lt/Lx/Ly/Lk/Lz/2
 if bound=='per':
   for t in range(Lt):
   #  print(t)
     for x in range(Lx):
         for y in range(Ly):
           for k in range(Lk):
             for z in range(Lz):
                a=eigen(t,Lt)
                b=eigen(x,Lx)+eigen(y,Ly)+eigen(k,Lk)+eigen(z,Lz)+m**2
                aux+=(b-a)/(a+b)
   return aux/Lt/Lx/Ly/Lk/Lz/2

def Cas_Pergen(mL,D,ep):
    auxD=(D+1)/2
    s=mp.besselk(auxD,mL)
    aux2=s
    c=1
    while(abs(aux2/s)>ep):
        c=c+1
        aux2=mp.besselk(auxD,mL*c)/pow(c,auxD)
        #print(aux2/s)
        s=s+aux2
    return -s/(pow(2,auxD-1)*pow(mp.pi,auxD)*pow(mL,auxD-1))

def Cas_Dirgen(mL,D,ep):
    auxD=(D+1)/2
    s=mp.besselk(auxD,2*mL)
    aux2=s
    c=1
    while(abs(aux2/s)>ep):
        c=c+1
        aux2=mp.besselk(auxD,2*mL*c)/pow(c,auxD)
        #print(c,aux2/s)
        s=s+aux2
    return -s/(pow(2,D)*pow(mp.pi,auxD)*pow(mL,auxD-1))

#mp.dps=16
L=48
Lr=48
Lz=24
m=0.4
"""
print(Lz)
a=ener_3d(L, L,L, Lz, m,'dir')
b=ener_3d(L, L,L, Lr, m,'dir')
c=ener_3d(L, L,L, Lr, m,'per')
Cas=(a*Lz-b*Lr-(Lz-Lr)*c)/m**3
print(a,b,c)
print(Cas)

a=ener_3d2(L, L,L, Lz, m,'dir')
b=ener_3d2(L, L,L, Lr, m,'dir')
c=ener_3d2(L, L,L, Lr, m,'per')
Cas=(a*Lz-b*Lr-(Lz-Lr)*c)/m**3
print(a,b,c)
print(Cas)


print(Cas_Dirgen(m*Lz,3,1e-20))
"""
print(Lz)
a=ener_4d(L, L,L,L, Lz, m,'per')
#b=ener_4d(L, L,L,L, Lr, m,'per')
b04=0.3036105659812618
b01=0.3002308834566648
Cas=(Lz*(a-b04))/m**4
print(a,b04)
print(Cas)

a=ener_4d2(L, L,L,L, Lz, m,'per')
#b=ener_4d2(L, L,L,L, Lr, m,'per')
b04=0.30361056598143404
Cas=(Lz*(a-b04))/m**4
print(a,b04)
print(Cas)


print(Cas_Pergen(m*Lz,4,1e-20))

