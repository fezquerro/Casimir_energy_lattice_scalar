# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 17:27:39 2026

@author: fer
"""
import numpy as np
import mpmath
import math
import time
import matplotlib.pyplot as plt

def ener_Df(L_periodic, Ly, m, bound):
    """
    Generalized function for D dimensions where D = len(L_periodic) periodic directions
    (first one is assumed to be the time direction Lt), plus one special direction Ly with given boundary condition.
    For the original case with D=2, call as ener_Df([Lt, Lx], Ly, m, bound).
    """
    D = len(L_periodic)
    m2 = m * m / 4  # mass contribution
    
    # Compute lambda for each periodic direction (time + normal spatial directiosn)
    lambda_list = []
    for L in L_periodic:
        n = np.arange(0, L)
        lambda_i = (np.sin(np.pi * n / L))**2
        lambda_list.append(lambda_i)
    
    # Compute ny and lambda_y based on boundary condition
    if bound == 'dir':  # Dirichlet bc
        ny = np.arange(1, Ly)
        lambda_y = (np.sin(np.pi * ny / (Ly * 2)))**2
    elif bound == 'per':  # Periodic bc
        ny = np.arange(0, Ly)
        lambda_y = (np.sin(np.pi * ny / Ly))**2
    elif bound == 'anti':  # Anti-periodic bc
        ny = np.arange(0, Ly)
        lambda_y = (np.sin(np.pi * (ny + 0.5) / Ly))**2
    elif bound == 'neu':  # Neumann bc
        ny = np.arange(0, Ly)
        lambda_y = (np.sin(np.pi * ny / (Ly * 2)))**2
    elif bound == 'zar':  # Zaremba bc
        ny = np.arange(0, Ly - 1)
        lambda_y = (np.sin(np.pi * (ny + 0.5) / (2 * Ly - 1)))**2
    else:
        raise ValueError(f"Unknown boundary condition: {bound}")
    
    number_ny = len(ny) #number of eigenvalues in Ly
    
    # Volume
    volume = np.prod(L_periodic) * Ly
    
    # Build the D-dimensional array of the periodic directions using broadcasting
    S = np.full(tuple(L_periodic), m2, dtype=float)#add the mass term to the denom
    for i in range(D):
        resh_shape = [1] * D
        resh_shape[i] = L_periodic[i]#shape [1]...[Li]..[1]
        S += lambda_list[i].reshape(resh_shape)#sum the eigenvalues of that direction

    # lambda_t is lambda_list[0], reshaped to have full dim in first axis
    resh_t = [L_periodic[0] if j == 0 else 1 for j in range(D)]
    lambda_t_resh = lambda_list[0].reshape(resh_t)

    # Compute aux: sum of lambda_t / denom over all modes, one ny mode at a time
    # (memory of a single D-dimensional array instead of the full (D+1)-dimensional one)
    aux = 0.0
    for k in range(number_ny):
        denom = S + lambda_y[k]
        # Handle zero mode for per or neu when m==0 (lambda_y[0]=0 for both)
        if (bound in ['per', 'neu']) and m == 0 and k == 0:
            denom[tuple(0 for _ in range(D))] = 1#lambda_t=0 there, so this term sums zero
        aux += np.sum(lambda_t_resh / denom) #sum over the energy of each eigenvector
    
    # Determine the infrarred term based on boundary and m
    if bound in ['dir', 'zar']:
        infr = 0.5 * (1 - 1 / Ly)
    elif m == 0 and bound in ['per', 'neu']:
        infr = 0.5 * (1 - 1 / volume)
    else:
        infr = 0.5
    # Final result
    result = infr - aux / volume
    return result


# ----------------------------------------------------------------------------
# Closed-form sum over the confined direction (see closed_form_note/closed_form_casimir.tex)
# Key identity: sum_{n=0}^{N-1} 1/(sin^2(pi n/N)+S) = N coth(N theta)/(sinh theta cosh theta), sinh^2 theta = S
# ----------------------------------------------------------------------------

def _transverse_modes(L_periodic, m):
    """S = sum_mu sin^2(pi n_mu/N_mu) + m^2/4 and lambda_t on the grid of the periodic directions."""
    D = len(L_periodic)
    S = np.full(tuple(L_periodic), m * m / 4, dtype=float)
    for i, L in enumerate(L_periodic):
        resh_shape = [1] * D
        resh_shape[i] = L
        S += ((np.sin(np.pi * np.arange(L) / L))**2).reshape(resh_shape)
    resh_t = [L_periodic[0] if j == 0 else 1 for j in range(D)]
    lt = np.broadcast_to(((np.sin(np.pi * np.arange(L_periodic[0]) / L_periodic[0]))**2).reshape(resh_t), S.shape)
    # S=0 only for m=0 and all n_mu=0, where lambda_t=0: that term never contributes, so it is dropped
    keep = S > 0
    return S[keep], lt[keep]


def _P(K, th, shch):
    """Periodic chain of length K: sum_k 1/(sin^2(pi k/K)+S) = K coth(K theta)/(sinh theta cosh theta)."""
    return K / np.tanh(K * th) / shch


def F_closed(N, S, bound):
    """sum over the modes of the confined direction of 1/(S+lambda_y), for every boundary condition (eqs. 9-12 of the note)."""
    th = np.arcsinh(np.sqrt(S))
    shch = np.sqrt(S * (1 + S))
    if bound == 'per':
        return _P(N, th, shch)
    if bound == 'anti':
        return _P(2 * N, th, shch) - _P(N, th, shch)
    if bound == 'dir':
        return (_P(2 * N, th, shch) - 1 / S - 1 / (1 + S)) / 2
    if bound == 'neu':
        return (_P(2 * N, th, shch) + 1 / S - 1 / (1 + S)) / 2
    if bound == 'zar':
        return (_P(4 * N - 2, th, shch) - _P(2 * N - 1, th, shch) - 1 / (1 + S)) / 2
    raise ValueError(f"Unknown boundary condition: {bound}")


def ener_fast(L_periodic, Ly, m, bound):
    """
    Same result as ener_Df (energy per site), with the sum over the Ly direction done analytically.
    Cost: one sum over prod(L_periodic) modes instead of prod(L_periodic)*Ly.
    """
    S, lt = _transverse_modes(L_periodic, m)
    volume = np.prod(L_periodic) * Ly
    aux = np.sum(lt * F_closed(Ly, S, bound))
    if bound in ['dir', 'zar']:
        infr = 0.5 * (1 - 1 / Ly)
    elif m == 0 and bound in ['per', 'neu']:
        infr = 0.5 * (1 - 1 / volume)
    else:
        infr = 0.5
    return infr - aux / volume


def bulk_C0(L_periodic, m):
    """Exact bulk energy density C0 = 1/2 - <lambda_t/sqrt(S(1+S))> (limit N_L0 -> infinity of thesis eq. 3.78)."""
    S, lt = _transverse_modes(L_periodic, m)
    return 0.5 - np.sum(lt / np.sqrt(S * (1 + S))) / np.prod(L_periodic)


def boundary_C1(L_periodic, m, bound):
    """
    Exact boundary energy per unit area C1^U (eqs. 21-24 of the note).
    Equivalently C1^D = -(e0+e1)/2, C1^N = (e0-e1)/2, C1^Z = -e1/2 with e0 = bulk energy one dimension
    less at mass m and e1 the same at mass sqrt(m^2+4).
    For m=0, 'per' and 'neu' include the term -1/(2 N0 A) from the removed constant mode (eq. 43 of the note),
    so that N_L*ener_Df = bulk_C0*L_U + boundary_C1 + casimir_exact holds exactly also at m=0.
    """
    norm = np.prod(L_periodic)
    zero_mode = -0.5 / norm if (m == 0 and bound in ['per', 'neu']) else 0.0
    if bound == 'anti':
        return 0.0
    if bound == 'per':
        return zero_mode
    S, lt = _transverse_modes(L_periodic, m)
    if bound == 'dir':
        return -0.5 + 0.5 * np.sum(lt * (1 + 2 * S) / (S * (1 + S))) / norm
    if bound == 'neu':
        return -0.5 * np.sum(lt / (S * (1 + S))) / norm + zero_mode
    if bound == 'zar':
        return -0.25 + 0.5 * np.sum(lt / (1 + S)) / norm
    raise ValueError(f"Unknown boundary condition: {bound}")


def casimir_exact(L_periodic, NL, m, bound):
    """
    Casimir energy per unit transverse area, E_U/A, in the limit N_L0 -> infinity (eq. for E_U/A in the note).
    No reference lattice is needed. Divide by m**D (D=len(L_periodic)) to compare with Cas_*gen.
    L_periodic = [Lt, Lx, ...] are the periodic directions, NL the size of the direction with boundary condition bound.
    For 'zar' the physical length is NL-1/2.
    """
    S, lt = _transverse_modes(L_periodic, m)
    th = np.arcsinh(np.sqrt(S))
    shch = np.sqrt(S * (1 + S))
    if bound == 'per':
        x = 2 * NL * th
        g = 2 * NL * np.exp(-x) / (-np.expm1(-x))
    elif bound == 'anti':
        q = np.exp(-2 * NL * th)
        g = -2 * NL * q / (1 + q)
    elif bound in ['dir', 'neu']:
        x = 4 * NL * th
        g = 2 * NL * np.exp(-x) / (-np.expm1(-x))
    elif bound == 'zar':
        Le = NL - 0.5
        q = np.exp(-4 * Le * th)
        g = -2 * Le * q / (1 + q)
    else:
        raise ValueError(f"Unknown boundary condition: {bound}")
    return -np.sum(lt / shch * g) / np.prod(L_periodic)


def casimir_finite_ref(L_periodic, NL, NL0, m, bound):
    """
    Thesis prescription (3.79)-(3.87) with a finite reference lattice NL0, evaluated with ener_fast.
    Kept to reproduce the old results; casimir_exact is its NL0 -> infinity limit.
    """
    if bound in ['per', 'anti']:
        return (ener_fast(L_periodic, NL, m, bound) - ener_fast(L_periodic, NL0, m, 'per')) * NL
    return (ener_fast(L_periodic, NL, m, bound) * NL - ener_fast(L_periodic, NL0, m, bound) * NL0
            - (NL - NL0) * ener_fast(L_periodic, NL0, m, 'per'))




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


"""DIR MASSLESS WORKS
Lx=150
Ly=8
L0=60
m=0
#a=Ly*ener_2d(Lx, Lx, Ly, 0, 'dir')+(2*L0+Ly)*ener_2d(Lx, Lx, 2*L0+Ly, 0, 'dir')-2*(L0+Ly)*ener_2d(Lx, Lx, L0+Ly, 0, 'dir')
b=Ly*ener_2df(Lx, Lx, Ly, m, 'dir')+(2*L0+Ly)*ener_2d(Lx, Lx, 2*L0+Ly, m, 'dir')-2*(L0+Ly)*ener_2d(Lx, Lx, L0+Ly, m, 'dir')
#print(a)
print(b)
#print(Cas_Dirgen(m*Ly,2,1e-12)*m**2)
print(-0.02391/Ly**2)
"""

#PER MASSLESS check
"""
a=0.1662803650
b=0.0000019971
Lx=32
Ly=8
m=0
c=ener_2df(Lx,Lx,Ly,m,'per0')
print(c)
print((a-c)/b)
"""
"""
# per massless works

Lx=200
Ly=12
L0=80
a=ener_2df(Lx,Lx,Ly,0,'per0')-ener_2df(Lx,Lx,L0,0,'per0')
print(a*Ly)
print(-0.1913/Ly**2)
"""

"""
Lx=1024
Ly=8
L0=1024
m=0
bound='dir'
a=[]
x=[]
a2=[]
b=ener_Df([Lx], L0, m, bound)*(L0)
for i in range(40):
    a.append(ener_Df([Lx], Ly+i, m, bound)*(Ly+i)-b)
    x.append(Ly+i)
    a2.append(-0.5236/(Ly+i)/4)
plt.scatter(x,a)
plt.plot(x,a2)
start=time.time()
a=ener_Df([Lx], Ly, m, bound)
end=time.time()
print(end-start)
print(a)
print(-0.5236/Ly)

"""

"""
print((ener_2df2(Lx, Lx, Ly, m, 'dir')*Ly-ener_2df2(Lx, Lx, L0, m, 'dir')*L0-(Ly-L0)*ener_2df2(Lx, Lx, L0, m, 'per')))
print((ener_2df2(Lx, Lx, Ly, m, 'neu')*Ly-ener_2df2(Lx, Lx, L0, m, 'neu')*L0-(Ly-L0)*ener_2df2(Lx, Lx, L0, m, 'per')))
print(-0.02391/Ly**2)
"""
if __name__ == "__main__":
    print(ener_Df([32,32],8,0.5,'zar'))