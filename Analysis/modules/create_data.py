# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:45:24 2024

@author: fer
"""
import numpy as np

def _ratio(a,den):#a/den, skipping the massless zero mode (a=den=0) of per/neu
    return a/den if den!=0 else 0.0
def _infr_pn(m,V):#infrared term for per/neu: the zero mode is removed when m=0
    return 0.5*(1-1/V) if m==0 else 0.5
def eigen(i,L):
    return (2*np.sin(np.pi*i/L))**2
def eigend(i,L):
    return (2*np.sin(np.pi*i/L/2))**2
def eigenan(i,L):
    return (2*np.sin(np.pi*(i+0.5)/L))**2
def eigenn(i,L):
    return (2*np.sin(np.pi*i/(L)/2))**2
def eigenz(i,L):
    return (2*np.sin(np.pi*(i+0.5)/(2*L-1)))**2
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
               aux+=_ratio(a,eigen(y,Ly)+m**2+a)
   return _infr_pn(m,Lt*Ly)-aux/Lt/Ly
 if bound=='anti':
   for t in range(Lt):
         for y in range(Ly):
               a=eigen(t,Lt)
               aux+=a/(eigenan(y,Ly)+m**2+a)
   return 0.5-aux/Lt/Ly
 if bound=='neu':
  for t in range(Lt):
         for y in range(Ly):
               a=eigen(t,Lt)
               aux+=_ratio(a,eigenn(y,Ly)+m**2+a)
  return _infr_pn(m,Lt*Ly)-aux/Lt/Ly
 if bound=='zar':
  for t in range(Lt):
         for y in range(0,Ly-1):
               a=eigen(t,Lt)
               aux+=a/(eigenz(y,Ly)+m**2+a)
  return 0.5*(1-1/Ly)-aux/Lt/Ly
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
                aux+=_ratio(a,eigen(x,Lx)+eigen(y,Ly)+m**2+a)
   return _infr_pn(m,Lt*Lx*Ly)-aux/Lt/Lx/Ly
 if bound=='anti':
   for t in range(Lt):
      for x in range(Lx):
          for y in range(Ly):
                a=eigen(t,Lt)
                aux+=a/(eigen(x,Lx)+eigenan(y,Ly)+m**2+a)
   return 0.5-aux/Lt/Lx/Ly
 if bound=='neu':
   for t in range(Lt):
      for x in range(Lx):
          for y in range(Ly):
                a=eigen(t,Lt)
                aux+=_ratio(a,eigen(x,Lx)+eigenn(y,Ly)+m**2+a)
   return _infr_pn(m,Lt*Lx*Ly)-aux/Lt/Lx/Ly
 if bound=='zar':
   for t in range(Lt):
      for x in range(Lx):
          for y in range(0,Ly-1):
                a=eigen(t,Lt)
                aux+=a/(eigen(x,Lx)+eigenz(y,Ly)+m**2+a)
   return 0.5*(1-1/Ly)-aux/Lt/Lx/Ly

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
                aux+=_ratio(a,eigen(x,Lx)+eigen(y,Ly)+eigen(z,Lz)+m**2+a)
   return _infr_pn(m,Lt*Lx*Ly*Lz)-aux/Lt/Lx/Ly/Lz
 if bound=='anti':
   for t in range(Lt):
     for x in range(Lx):
         for y in range(Ly):
             for z in range(Lz):
                a=eigen(t,Lt)
                aux+=a/(eigen(x,Lx)+eigen(y,Ly)+eigenan(z,Lz)+m**2+a)
   return 0.5-aux/Lt/Lx/Ly/Lz
 if bound=='neu':
   for t in range(Lt):
     for x in range(Lx):
         for y in range(Ly):
             for z in range(Lz):
                a=eigen(t,Lt)
                aux+=_ratio(a,eigen(x,Lx)+eigen(y,Ly)+eigenn(z,Lz)+m**2+a)
   return _infr_pn(m,Lt*Lx*Ly*Lz)-aux/Lt/Lx/Ly/Lz
 if bound=='zar':
  for t in range(Lt):
     for x in range(Lx):
      for y in range(Ly):
         for z in range(0,Lz-1):
               a=eigen(t,Lt)
               aux+=eigen(t,Lt)/(eigen(x,Lx)+eigen(y,Ly)+eigenz(z,Lz)+m**2+a)
  return 0.5*(1-1/Lz)-aux/Lt/Lx/Ly/Lz

def ener_4d(Lt,Lx,Ly,Lk,Lz,m,bound):
 aux=0;
 if bound=='dir':
  for t in range(Lt):
     for x in range(Lx):
      for y in range(Ly):
        for k in range(Lk):
         for z in range(1,Lz):
               a=eigen(t,Lt)
               aux+=eigen(t,Lt)/(eigen(x,Lx)+eigen(y,Ly)+eigen(k,Lk)+eigend(z,Lz)+m**2+a)
  return 0.5*(1-1/Lz)-aux/Lt/Lx/Ly/Lk/Lz
 if bound=='per':
   for t in range(Lt):
     for x in range(Lx):
         for y in range(Ly):
           for k in range(Lk):
             for z in range(Lz):
                a=eigen(t,Lt)
                aux+=_ratio(a,eigen(x,Lx)+eigen(y,Ly)+eigen(k,Lk)+eigen(z,Lz)+m**2+a)
   return _infr_pn(m,Lt*Lx*Ly*Lk*Lz)-aux/Lt/Lx/Ly/Lk/Lz
 if bound=='anti':
   for t in range(Lt):
     for x in range(Lx):
         for y in range(Ly):
           for k in range(Lk):
             for z in range(Lz):
                a=eigen(t,Lt)
                aux+=a/(eigen(x,Lx)+eigen(y,Ly)+eigen(k,Lk)+eigenan(z,Lz)+m**2+a)
   return 0.5-aux/Lt/Lx/Ly/Lk/Lz
 if bound=='neu':
   for t in range(Lt):
     for x in range(Lx):
         for y in range(Ly):
           for k in range(Lk):
             for z in range(Lz):
                a=eigen(t,Lt)
                aux+=_ratio(a,eigen(x,Lx)+eigen(y,Ly)+eigen(k,Lk)+eigenn(z,Lz)+m**2+a)
   return _infr_pn(m,Lt*Lx*Ly*Lk*Lz)-aux/Lt/Lx/Ly/Lk/Lz
 if bound=='zar':
  for t in range(Lt):
     for x in range(Lx):
      for y in range(Ly):
        for k in range(Lk):
         for z in range(0,Lz-1):
               a=eigen(t,Lt)
               aux+=eigen(t,Lt)/(eigen(x,Lx)+eigen(y,Ly)+eigen(k,Lk)+eigenz(z,Lz)+m**2+a)
  return 0.5*(1-1/Lz)-aux/Lt/Lx/Ly/Lk/Lz



def write_data(Lt,L,Lz,m,energy,bound,dim):
    Name='../Energy_analytical/Energy_an_'+str(Lt)+'_'
    for i in range(1,dim):
     Name=Name+str(L)+'_'
    Name=Name+'m_'+'%.4f'%m+'_'+bound+'.txt'
    f = open(Name, "w")
    f.write("M Lz Energy\n")
    for i in range(np.size(Lz)):
     f.write(str(m)+' '+str(Lz[i])+' '+str(energy[i])+'\n')
    f.close()
def write_data_ref(Lt,L,Lz,m,energy,bound,dim):
    Name='../Energy_analytical/Energy_an_ref_'+str(Lt)+'_'
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
            if dim==1:
             E[i].append(ener_1d(Lt, Lz[i][j], m[i], bound))
            if dim==2:
             E[i].append(ener_2d(Lt, L, Lz[i][j], m[i], bound))
            if dim==3:
             E[i].append(ener_3d(Lt, L,L, Lz[i][j], m[i], bound))
             print(Lz[i][j],m[i])
            if dim==4:
             E[i].append(ener_4d(Lt, L,L,L, Lz[i][j], m[i], bound))
             print(Lz[i][j],m[i])
        write_data(Lt,L,Lz[i],m[i],E[i],bound,dim)
    return E
def create_ref(Lt,L,Lz,m,bound,dim):
    E_ref=np.zeros(len(m))
    for i in range (len(m)):
            if dim==1:
             E_ref[i]=ener_1d(Lt, Lz[i][0], m[i], bound)
            if dim==2:
             E_ref[i]=ener_2d(Lt, L, Lz[i][0], m[i], bound)
            if dim==3:
             E_ref[i]=ener_3d(Lt, L,L, Lz[i][0], m[i], bound)
            if dim==4:
             E_ref[i]=ener_4d(Lt, L,L,L, Lz[i][0], m[i], bound)
            write_data_ref(Lt,L,Lz[i][0],m[i],E_ref[i],bound,dim)
    return E_ref
def write_data_2nd(Lt,L,Lz,L0,m,energy,bound):
    Name='../Energy_analytical/Energy_an2_'+str(Lt)+'_'+str(L)+'_L0_'+str(L0)+'_m_'+'%.4f'%m+'_'+bound+'.txt'
    f = open(Name, "w")
    f.write("M Lz Energy\n")
    for i in range(np.size(Lz)):
     f.write(str(m)+' '+str(Lz[i])+' '+str(energy[i])+'\n')
    f.close()
def create_2nd_norm(Lt,L,Lz,L0,m,bound):
    E=[]
    for i in range (len(m)):
        E.append([])
        for j in range (len(Lz[i])):
            E[i].append(ener_2d(Lt, L, Lz[i][j], m[i], bound)*Lz[i][j]+ener_2d(Lt, L, Lz[i][j]+2*L0, m[i], bound)*(Lz[i][j]+2*L0)-2*ener_2d(Lt, L, Lz[i][j]+L0, m[i], bound)*(Lz[i][j]+L0))
        write_data_2nd(Lt,L,Lz[i],L0,m[i],E[i],bound)
    return E

def write_data_check_ren(Lt,L,Lz,m,energy,bound):
    Name='../Energy_analytical/Energy_ren_'+str(Lt)+'_'+str(L)+'_m_'+'%.4f'%m+'_'+bound+'.txt'
    f = open(Name, "w")
    f.write("M Lz Energy\n")
    for i in range(np.size(Lz)):
     f.write(str(m)+' '+str(Lz[i])+' '+str(energy[i])+'\n')
    f.close()
def create_check_norm_per(Lt,L,Lz,m,bound):
    E=[]
    for i in range (len(m)):
        E.append([])
        for j in range (len(Lz[i])):
            E[i].append(ener_2d(Lt, L, Lz[i][j], m[i], bound))
        write_data_check_ren(Lt,L,Lz[i],m[i],E[i],bound)
    return E
import mpmath
mpmath.mp.dps=48#working precision of mpmath (mpmath.dps does nothing)
def eigenm(i,L):
    return (2*mpmath.sin(mpmath.pi*i/L))**2
def eigendm(i,L):
    return (2*mpmath.sin(mpmath.pi*i/L/2))**2
def ener_2dm(Lt,Lx,Ly,m,bound):
 aux=mpmath.mpf(0)
 m=mpmath.mpf(m)
 if bound=='dir':
  for t in range(Lt):
     for x in range(Lx):
         for y in range(1,Ly):
               a=eigenm(t,Lt)
               aux+=a/(eigenm(x,Lx)+eigendm(y,Ly)+m**2+a)
  return mpmath.mpf(1)/2*(1-mpmath.mpf(1)/Ly)-aux/Lt/Lx/Ly

 if bound=='per':
   for t in range(Lt):
     for x in range(Lx):
         for y in range(Ly):
                a=eigenm(t,Lt)
                aux+=a/(eigenm(x,Lx)+eigenm(y,Ly)+m**2+a)
   return mpmath.mpf(1)/2-aux/Lt/Lx/Ly
"""
print(ener_2d(96, 96, 22, 0.4, 'per')-ener_2d(96, 96, 96, 0.4, 'per'))
print(ener_2dm(96, 96, 22, 0.4, 'per')-ener_2dm(96, 96, 96, 0.4, 'per'))
print(ener_2d(96, 96, 22, 0.4, 'dir')*22-ener_2d(96, 96, 96, 0.4, 'dir')*96-(22-96)*ener_2d(96, 96, 96, 0.4, 'per'))
print(ener_2dm(96, 96, 22, 0.4, 'dir')*22-ener_2dm(96, 96, 96, 0.4, 'dir')*96-(22-96)*ener_2dm(96, 96, 96, 0.4, 'per'))
"""

"""
print('per 32',(ener_2d(96, 96, 23, 0.4, 'per')-ener_2d(96, 96, 32, 0.4, 'per'))/0.4**2)
print('per 96',(ener_2d(96, 96, 23, 0.4, 'per')-ener_2d(96, 96, 96, 0.4, 'per'))/0.4**2)
print('dir 32 32',(ener_2d(96, 96, 23, 0.4, 'dir')*23-ener_2d(96, 96, 32, 0.4, 'dir')*32-(23-32)*ener_2d(96, 96, 32, 0.4, 'per'))/0.4**2)
print('dir 96 32',(ener_2d(96, 96, 23, 0.4, 'dir')*23-ener_2d(96, 96, 96, 0.4, 'dir')*96-(23-96)*ener_2d(96, 96, 32, 0.4, 'per'))/0.4**2)
print('dir 32 96',(ener_2d(96, 96, 23, 0.4, 'dir')*23-ener_2d(96, 96, 32, 0.4, 'dir')*32-(23-32)*ener_2d(96, 96, 48, 0.4, 'per'))/0.4**2)
print('dir 96 96',(ener_2d(96, 96, 23, 0.4, 'dir')*23-ener_2d(96, 96, 96, 0.4, 'dir')*96-(23-96)*ener_2d(96, 96, 96, 0.4, 'per'))/0.4**2)
"""

if __name__ == "__main__":
    print(ener_2d(32, 32,8, 0, 'dir'))
