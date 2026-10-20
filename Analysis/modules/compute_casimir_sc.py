# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:32:20 2024

@author: fer
"""
import numpy as np



def compute_casimir_analytic(E,Lz,m,E_ref,xref,cprint,boundary,dim):
    cas=[]
    xdata=[]
    if boundary=='per' or boundary=='anti':
     for i in range(np.size(m)):
        cas.append([])
        xdata.append([])
        a=m[i]
        for j in range(np.size(Lz[i])):
            cas[i].append((E[i][j]-E_ref[i])/a**dim*Lz[i][j])
            xdata[i].append(a*Lz[i][j])
            if cprint:
                print("Mass",m[i],"L",Lz[i][j],"x",np.round(xdata[i][j],3),"C",np.round(cas[i][j],9))
     return cas,xdata
    if boundary=='dir' or boundary=='neu' or boundary=='zar':
     for i in range(np.size(m)):
        cas.append([])
        xdata.append([])
        a=m[i]
        for j in range(np.size(Lz[i])):
            L=Lz[i][j]
            Ld=xref[i][0]-L
            cas[i].append((E[i][j]*L-E_ref[1][i]*xref[i][0]+Ld*E_ref[0][i])/a**dim)
            if boundary=='zar':
                xdata[i].append(a*(Lz[i][j]-0.5))
            else:
                xdata[i].append(a*Lz[i][j])
            if cprint:
                print("\nMass",m[i],"L",Lz[i][j],"x",np.round(xdata[i][j],3),"C",np.round(cas[i][j],9))
    return cas,xdata

def compute_casimir_renor(E,Lz,m,cprint,boundary):
    cas=[]
    xdata=[]
    if boundary=='per':
     for i in range(np.size(m)):
        cas.append([])
        xdata.append([])
        a=m[i]
        for j in range(np.size(Lz[i])):
            cas[i].append((E[i][j])/a**2*Lz[i][j])
            xdata[i].append(a*Lz[i][j])
            if cprint:
                print("Mass",m[i],"L",Lz[i][j],"x",np.round(xdata[i][j],3),"C",np.round(cas[i][j],9))
     return cas,xdata
    if boundary=='dir':
     for i in range(np.size(m)):
        cas.append([])
        xdata.append([])
        a=m[i]
        for j in range(np.size(Lz[i])):
            L=Lz[i][j]
            cas[i].append((E[i][j])/a**2)
            xdata[i].append(a*Lz[i][j])
            if cprint:
                print("\nMass",m[i],"L",Lz[i][j],"x",np.round(xdata[i][j],3),"C",np.round(cas[i][j],9))
    return cas,xdata        
                
                
        