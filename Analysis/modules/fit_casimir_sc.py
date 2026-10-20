# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:35:40 2024

@author: fer
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import mpmath
plt.rcParams.update({
    'text.usetex': True
})

def pol_sum(x,m,n):
    aux=m*x
    s=0
    for i in range(1,n+1):
        s+=aux*np.exp(-i*aux)/i**2+np.exp(-i*aux)/i**3
    return -s/np.pi/x**2/2
def pol_sum_dir(x,m,n):
    aux=2*m*x
    s=0
    for i in range(1,n+1):
        s+=aux*np.exp(-i*aux)/i**2+np.exp(-i*aux)/i**3
    return -s/np.pi/x**2/16

def Cas_Per(L,m):
    aux=m*L
    aux2=math.exp(-aux)
    return -(aux*mpmath.polylog(2,aux2)+mpmath.polylog(3,aux2))/(2*math.pi*L**2)
def Cas_Dir(L,m):
    aux=2*m*L
    aux2=math.exp(-aux)
    return -(aux*mpmath.polylog(2,aux2)+mpmath.polylog(3,aux2))/(16*math.pi*L**2)
def Cas_Pergen(mL,D,ep):
    if mL<=0:
        raise ValueError('mL must be > 0: the Bessel series diverges at mL=0 (use the massless formula)')
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
    if mL<=0:
        raise ValueError('mL must be > 0: the Bessel series diverges at mL=0 (use the massless formula)')
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

def Cas_AntiPergen(mL,D,ep):
    if mL<=0:
        raise ValueError('mL must be > 0: the Bessel series diverges at mL=0 (use the massless formula)')
    auxD=(D+1)/2
    s=mpmath.besselk(auxD,mL)
    aux2=s
    c=1
    while(abs(aux2/s)>ep):
        c=c+1
        aux2=mpmath.besselk(auxD,mL*c)/pow(c,auxD)
        #print(aux2/s)
        s=s+aux2*(-1)**(c+1)
    return s/(pow(2,auxD-1)*pow(mpmath.pi,auxD)*pow(mL,auxD-1))
def Cas_Zarembagen(mL,D,ep):
    if mL<=0:
        raise ValueError('mL must be > 0: the Bessel series diverges at mL=0 (use the massless formula)')
    aux=mL
    auxD=(D+1)/2
    s=mpmath.besselk(auxD,2*aux)
    aux2=s
    c=1
    while(abs(aux2/s)>ep):
        c=c+1
        aux2=mpmath.besselk(auxD,2*aux*c)/pow(c,auxD)
        #print(aux2/s)
        s=s+aux2*(-1)**(c+1)
    return s/(pow(2,D)*pow(mpmath.pi,auxD)*pow(mL,auxD-1))

def plot_Casimir(C,x,m,bcond,yscale,dimension,plot,fname=None):
    xdata=np.array(np.concatenate(x).flat)#fuse all the x
    E=np.array(np.concatenate(C).flat)#fuse all the Cas energy
    DP=np.linspace(min(xdata), max(xdata), num=100)#for plot space x (data are not sorted across masses)
    plt.plot(xdata,E,'o',markersize=4,label='$D=%g$ lattice'%dimension)
    Econt=[]
    if bcond=='periodic':
            for mL in DP:
                Econt.append(Cas_Pergen(mL,dimension,1e-12))
            plt.plot(DP,Econt,label='$D=%g$ contiuum '%dimension)
            plt.ylabel('$E_P/(Am^D)$',fontsize=14)
    if bcond=='Dirichlet':
            for mL in DP:
                Econt.append(Cas_Dirgen(mL,dimension,1e-12))
            plt.plot(DP,Econt,label='$D=%g$ contiuum '%dimension)
            plt.ylabel('$E_D/(Am^D)$',fontsize=14)
    if bcond=='antiperiodic':
            for mL in DP:
                Econt.append(Cas_AntiPergen(mL,dimension,1e-12))
            plt.plot(DP,Econt,label='$D=%g$ contiuum '%dimension)
            plt.ylabel('$E_A/(Am^D)$',fontsize=14)
    if bcond=='Neumann':
            for mL in DP:
                Econt.append(Cas_Dirgen(mL,dimension,1e-12))
            plt.plot(DP,Econt,label='$D=%g$ contiuum '%dimension)
            plt.ylabel('$E_N/(Am^D)$',fontsize=14)
    if bcond=='Zaremba':
            for mL in DP:
                Econt.append(Cas_Zarembagen(mL,dimension,1e-12))
            plt.plot(DP,Econt,label='$D=%g$ contiuum '%dimension)
            plt.ylabel('$E_Z/(Am^D)$',fontsize=14)
    plt.xlabel('$mL$',fontsize=14)
    
    if yscale=='log':
      if bcond=='periodic':
        plt.yscale('symlog', linthresh=0.00000001)
        plt.yticks([-0.1,-0.0001,-0.000001,-0.00000001,0])
        plt.legend( prop={'size': 8},ncol=3,loc=1)
      if bcond=='Dirichlet' or bcond=='Neumann':
        plt.yscale('symlog', linthresh=0.000000032)
        plt.yticks([-0.01,-0.0001,-0.000001,-0.00000001,0])
        plt.legend( prop={'size': 8},ncol=3,loc=1)
        #plt.ylim(-0.1,0.000000000001)
      if bcond=='antiperiodic':
        plt.yscale('log')
        plt.legend(prop={'size': 8},bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3)
      if bcond=='Zaremba':
        plt.yscale('log')
        plt.legend(prop={'size': 8},bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3)
      
        #plt.yticks([0,0.00000001,0.000001,0.0001,0.1])
     #
    
    #plt.title('V=%g %g %g' %(Lt,L,Lz[0][-1] ))
    
    #if bcond=='Dirichlet':
     #    plt.savefig('Casimir_scalar_2d_full.pdf',bbox_inches='tight')
    if plot:
     #plt.show()
     if fname is None:#default name of the thesis figures
         fname='Casimir_lattice_D'+bcond+'s.pdf'
     plt.savefig(fname,bbox_inches='tight')
     plt.show()

      
      
def cas_m_compare(C,x,m,bcond,dif_m,yscale):
    xdata=np.array(np.concatenate(x).flat)#fuse all the x
    E=np.array(np.concatenate(C).flat)#fuse all the Cas energy
    dx=(max(xdata)-min(xdata))/100
    DP=np.linspace(min(xdata)-dx, max(xdata)+dx, num=102)#for plot space x
    Econt=[]
    if dif_m:#plot m in dif colors
         for i in range (len(m)):
             plt.plot(x[i],np.array(C[i]),'o',label='$M=$ %g'%m[i])
    else:#plot all betas same
         plt.plot(xdata,E,'o',label='Lattice '+bcond)
     #plt.plot(DP,fun(DP,popt[0],popt[1]),label='Fit m=%g'%popt[1] )
     #plt.plot(DP,fun(DP,popt[0],popt[1]),label='Fit '+bcond )
    if bcond=='periodic':
        for i in range(len(DP)):
            Econt.append(Cas_Per(DP[i],1))
        
        plt.plot(DP,Econt,label='Contiuum '+bcond)
    else:
        for i in range(len(DP)):
            Econt.append(Cas_Dir(DP[i],1))
        plt.plot(DP,Econt,label='Contiuum '+bcond) 
    plt.xlabel('$mL$',fontsize=14)
    plt.ylabel('$E_U/(Am^2)$',fontsize=14)
    if yscale=='log':
      if bcond=='periodic':
        plt.yscale('symlog', linthresh=0.0000008)
        plt.yticks([-0.1,-0.0001,-0.000001,0])
      else:
        plt.yscale('symlog', linthresh=0.0000000000032)
        plt.yticks([-0.1,-0.001,-0.00001,-0.0000001,-0.000000001,-0.00000000001,0])
     #plt.yscale('log')
    plt.legend( prop={'size': 8},ncol=3,loc=2)
    #plt.title('V=%g %g %g' %(Lt,L,Lz[0][-1] ))
    #plt.savefig('Casimir_scalar_2d_lin_'+bcond+'.pdf',bbox_inches='tight')
    #if bcond=='Dirichlet':
    #plt.savefig('Casimir_scalar_2d_full.pdf',bbox_inches='tight')
    #plt.show() 
    if bcond=='Dirichlet':
     plt.savefig('Casimir_scalar_renorm.pdf',bbox_inches='tight')
