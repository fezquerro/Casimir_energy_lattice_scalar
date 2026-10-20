# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:23:52 2024

@author: fer
"""
import numpy as np


def create_Lz(Lz,jump,extra):
    A=[]
    nbeta=len(Lz)
    for i in range (nbeta):
        nL=int((Lz[i][1]-Lz[i][0])/jump)
        A.append([])
        for j in range (nL+1):
            A[i].append(Lz[i][0]+j*jump)
        if extra:
            A[i].append(Lz[i][-1])
    return np.array(A,dtype=tuple)


def read_data_an(Lt,L,Lz,m,boundary,dim):
 data=[]
 bdata=np.size(m)
 E=[]
 for j in range(bdata):
  Name='../Energy_analytical/Energy_an_'+str(Lt)+'_'
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
  Name='../Energy_analytical/Energy_an_ref_'+str(Lt)+'_'
  for i in range(1,dim):
   Name=Name+str(L)+'_'
  Name=Name+str(Lz[j][0])+'m_'+'%.4f'%m[j]+'_'+boundary+'.txt'
  data.append(np.loadtxt(Name,dtype=float,delimiter=' ',skiprows=1,ndmin=2))
 for j in range(bdata):
         E.append(data[j][0][2])
 return np.array(E,dtype=tuple)

def read_data_2nd(Lt,L,Lz,L0,m,boundary):
 data=[]
 bdata=np.size(m)
 E=[]
 for j in range(bdata):
  Name='../Energy_analytical/Energy_an2_'+str(Lt)+'_'+str(L)+'_L0_'+str(L0)+'_m_'+'%.4f'%m[j]+'_'+boundary+'.txt'
  data.append(np.loadtxt(Name,dtype=float,delimiter=' ',skiprows=1,ndmin=2))
 for j in range(bdata):
     lut={int(row[1]):row[2] for row in data[j]}#Lz -> energy
     E.append([lut[int(Lzj)] for Lzj in Lz[j]])#KeyError if a requested Lz is not in the file
 return np.array(E,dtype=tuple)

def read_check_norm(Lt,L,Lz,m,boundary):
 data=[]
 bdata=np.size(m)
 E=[]
 for j in range(bdata):
  Name='../Energy_analytical/Energy_ren_'+str(Lt)+'_'+str(L)+'_m_'+'%.4f'%m[j]+'_'+boundary+'.txt'
  data.append(np.loadtxt(Name,dtype=float,delimiter=' ',skiprows=1,ndmin=2))
 for j in range(bdata):
     lut={int(row[1]):row[2] for row in data[j]}#Lz -> energy
     E.append([lut[int(Lzj)] for Lzj in Lz[j]])#KeyError if a requested Lz is not in the file
 return np.array(E,dtype=tuple)
