# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:05:20 2024

@author: usuario
"""

import matplotlib.pyplot as plt

from modules.read_sc import read_check_norm,create_Lz
from modules.create_data import create_check_norm_per
import numpy as np
plt.rcParams.update({
    'text.usetex': True
})
plt.rc('text.latex', preamble=r'\usepackage{braket}')



Lt=100
L=100

m=[0.2,0.22,0.25,0.3,0.4]
Lz=[[10,100],[10,100],[10,100],[10,100],[10,100]]
Lv=create_Lz(Lz, 10,False)
print(Lv)
#E=create_check_norm_per(Lt,L,Lv,m,'per')
E=read_check_norm(Lt,L, Lv,m,'per')
print(E)

E_ad=[]
xdata=[]
for i in range(np.size(m)):
        E_ad.append([])
        xdata.append([])
        a=m[i]
        print(np.size(Lv[i]))
        for j in range(np.size(Lv[i])):
            E_ad[i].append((E[i][j])/a**3)
            xdata[i].append(a*Lv[i][j])
            print("Mass",m[i],"L",Lv[i][j],"x",np.round(xdata[i][j],3),"C",np.round(E_ad[i][j],9))

print(E_ad)
for i in range(len(m)):
    plt.scatter(Lv[i],E_ad[i],label='$M$=%g'%m[i])
plt.legend( prop={'size': 8},ncol=3,loc=1)
plt.xlabel('$N_L$',fontsize=14)
plt.ylabel('$\\braket{\mathcal E_P}/(ALm^3)$',fontsize=14)
plt.ylim(2,30)
plt.savefig('Periodic_divergence_scalar.pdf', bbox_inches='tight')
plt.show()

Lt=100
L=100

m=[0.2,0.22,0.25,0.3,0.4]
Lz=[[10,100],[10,100],[10,100],[10,100],[10,100]]
Lv=create_Lz(Lz, 10,False)
print(Lv)
#Eref=create_check_norm_per(Lt,L,Lv,m,'per')
Eref=read_check_norm(Lt,L, Lv,m,'per')

#E=create_check_norm_per(Lt,L,Lv,m,'dir')
E=read_check_norm(Lt,L, Lv,m,'dir')
print(E)

E_ad=[]
xdata=[]
for i in range(np.size(m)):
        E_ad.append([])
        xdata.append([])
        a=m[i]
        print(np.size(Lv[i]))
        for j in range(np.size(Lv[i])):
            E_ad[i].append(Lv[i][j]*(E[i][j]-Eref[i][-1])/a**2)
            xdata[i].append(a*Lv[i][j])
            print("Mass",m[i],"L",Lv[i][j],"x",np.round(xdata[i][j],3),"C",np.round(E_ad[i][j],9))

print(E_ad)
for i in range(len(m)):
    plt.scatter(Lv[i],E_ad[i],label='$M$=%g'%m[i])
plt.legend( prop={'size': 8},ncol=3,loc=1)
plt.xlabel('$N_L$',fontsize=14)
plt.ylabel('$(\\braket{\mathcal E_D}-\\frac{N_L}{100}\\braket{\mathcal E_P})/(Am^2)$',fontsize=14)
plt.ylim(-4,0.5)
plt.savefig('Dirichlet_divergence_scalar.pdf', bbox_inches='tight')
plt.show()



Lt=100
L=100

m=[0.2,0.22,0.25,0.3,0.4]
Lz=[[10,100],[10,100],[10,100],[10,100],[10,100]]
Lv=create_Lz(Lz, 10,False)
print(Lv)
#E=create_check_norm_per(Lt,L,Lv,m,'anti')
E=read_check_norm(Lt,L, Lv,m,'anti')
print(E)

E_ad=[]
xdata=[]
for i in range(np.size(m)):
        E_ad.append([])
        xdata.append([])
        a=m[i]
        print(np.size(Lv[i]))
        for j in range(np.size(Lv[i])):
            E_ad[i].append((E[i][j])/a**3)
            xdata[i].append(a*Lv[i][j])
            print("Mass",m[i],"L",Lv[i][j],"x",np.round(xdata[i][j],3),"C",np.round(E_ad[i][j],9))

print(E_ad)
for i in range(len(m)):
    plt.scatter(Lv[i],E_ad[i],label='$M$=%g'%m[i])
plt.legend( prop={'size': 8},ncol=3,loc=1)
plt.xlabel('$N_L$',fontsize=14)
plt.ylabel('$\\braket{\mathcal E_A}/(ALm^3)$',fontsize=14)
plt.ylim(2,30)
plt.savefig('AntiPeriodic_divergence_scalar.pdf', bbox_inches='tight')
plt.show()


Lt=100
L=100

m=[0.2,0.22,0.25,0.3,0.4]
Lz=[[10,100],[10,100],[10,100],[10,100],[10,100]]
Lv=create_Lz(Lz, 10,False)
print(Lv)
#Eref=create_check_norm_per(Lt,L,Lv,m,'per')
Eref=read_check_norm(Lt,L, Lv,m,'per')

#E=create_check_norm_per(Lt,L,Lv,m,'neu')
E=read_check_norm(Lt,L, Lv,m,'neu')
print(E)

E_ad=[]
xdata=[]
for i in range(np.size(m)):
        E_ad.append([])
        xdata.append([])
        a=m[i]
        print(np.size(Lv[i]))
        for j in range(np.size(Lv[i])):
            E_ad[i].append(Lv[i][j]*(E[i][j]-Eref[i][-1])/a**2)
            xdata[i].append(a*Lv[i][j])
            print("Mass",m[i],"L",Lv[i][j],"x",np.round(xdata[i][j],3),"C",np.round(E_ad[i][j],9))

print(E_ad)
for i in range(len(m)):
    plt.scatter(Lv[i],E_ad[i],label='$M$=%g'%m[i])
plt.legend( prop={'size': 8},ncol=3,loc=1)
plt.xlabel('$N_L$',fontsize=14)
plt.ylabel('$(\\braket{\mathcal E_N}-\\frac{N_L}{100}\\braket{\mathcal E_P})/(Am^2)$',fontsize=14)
plt.ylim(-4,0.5)
plt.savefig('Neumann_divergence_scalar.pdf', bbox_inches='tight')
plt.show()



Lt=100
L=100

m=[0.2,0.22,0.25,0.3,0.4]
Lz=[[10,100],[10,100],[10,100],[10,100],[10,100]]
Lv=create_Lz(Lz, 10,False)
print(Lv)
#Eref=create_check_norm_per(Lt,L,Lv,m,'per')
Eref=read_check_norm(Lt,L, Lv,m,'per')

#E=create_check_norm_per(Lt,L,Lv,m,'zar')
E=read_check_norm(Lt,L, Lv,m,'zar')
print(E)

E_ad=[]
xdata=[]
for i in range(np.size(m)):
        E_ad.append([])
        xdata.append([])
        a=m[i]
        print(np.size(Lv[i]))
        for j in range(np.size(Lv[i])):
            E_ad[i].append(Lv[i][j]*(E[i][j]-Eref[i][-1])/a**2)
            xdata[i].append(a*Lv[i][j])
            print("Mass",m[i],"L",Lv[i][j],"x",np.round(xdata[i][j],3),"C",np.round(E_ad[i][j],9))

print(E_ad)
for i in range(len(m)):
    plt.scatter(Lv[i],E_ad[i],label='$M$=%g'%m[i])
plt.legend( prop={'size': 8},ncol=3,loc=1)
plt.xlabel('$N_L$',fontsize=14)
plt.ylabel('$(\\braket{\mathcal E_Z}-\\frac{N_L}{100}\\braket{\mathcal E_P})/(Am^2)$',fontsize=14)
plt.ylim(-6,0)
plt.savefig('Zaremba_divergence_scalar.pdf', bbox_inches='tight')
plt.show()