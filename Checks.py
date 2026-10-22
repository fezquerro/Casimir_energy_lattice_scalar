# -*- coding: utf-8 -*-
"""
Created on Fri May  9 17:49:57 2025

@author: fer

Consistency checks of modules/energy_formulas.py (replaces check_eigenvalues.py and check_precision.py):
1. the eigenvalues of the lattice operator, built as a matrix, are the analytic modes used in energy_formulas
2. the closed-form sum (ener_fast) equals the explicit sum over all the modes (ener_Df)
3. N_L*ener_Df = C0*L_U + C1 + casimir_exact exactly, and casimir_finite_ref -> casimir_exact for large N_L0
4. casimir_exact/m^D -> continuum as the lattice spacing goes to zero, with O(a^2) corrections
5. massless case: the massive continuum formula tends to the massless one, and casimir_exact at M=0 tends to it
   as N_L grows. At M=0 the finite size N_A of the periodic directions only decays as a power of N_L/N_A,
   so N_A/N_L is taken large (larger for small D).
6. no integer overflow (numpy int is int32 on Windows): lattice sizes above 2^31 given as Python ints and as numpy int32
Every line prints the largest deviation found.
"""
import numpy as np
from modules.energy_formulas import (BOUNDS, _confined_modes, _size, phys_length, ener_Df, ener_fast,
                                     bulk_C0, boundary_C1, casimir_exact, casimir_finite_ref)
from modules.continuum import casimir_continuum, casimir_continuum_massless


def lap1d(N, bound):
    """Lattice Laplacian of one direction with N sites (N >= 3) as a matrix."""
    A = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    if bound == 'per':
        A[0, -1] = A[-1, 0] = -1
    elif bound == 'anti':
        A[0, -1] = A[-1, 0] = 1
    elif bound == 'neu':
        A[0, 0] = A[-1, -1] = 1
    elif bound == 'zar':#Dirichlet at one end, Neumann at the other
        A[-1, -1] = 1
    return A#'dir': the field is zero outside, nothing to change


def operator_eigenvalues(L_periodic, NL, m, bound):
    """Eigenvalues of -Laplacian+m^2 from the matrix: periodic directions plus the confined one."""
    sites = NL - 1 if bound in ['dir', 'zar'] else NL#N_L counts the boundary links
    mats = [lap1d(L, 'per') for L in L_periodic] + [lap1d(sites, bound)]
    V = int(np.prod([A.shape[0] for A in mats]))
    M = m * m * np.eye(V)
    for i, A in enumerate(mats):#Kronecker sum
        left = int(np.prod([B.shape[0] for B in mats[:i]]))
        right = V // (left * A.shape[0])
        M += np.kron(np.kron(np.eye(left), A), np.eye(right))
    return np.sort(np.linalg.eigvalsh(M))


def analytic_eigenvalues(L_periodic, NL, m, bound):
    modes = [4 * np.sin(np.pi * np.arange(L) / L)**2 for L in L_periodic] + [4 * _confined_modes(NL, bound)]
    ev = m * m
    for v in modes:
        ev = np.add.outer(ev, v)
    return np.sort(ev.ravel())


print('1. matrix eigenvalues - analytic modes (L_periodic=[3,3,3], N_L=5, m=0.3)')
for b in BOUNDS:
    d = np.max(np.abs(operator_eigenvalues([3, 3, 3], 5, 0.3, b) - analytic_eigenvalues([3, 3, 3], 5, 0.3, b)))
    print('   %-4s max |diff| = %.1e' % (b, d))

print('2. ener_fast - ener_Df (L_periodic=[24,24], N_L=11)')
for b in BOUNDS:
    d = max(abs(ener_fast([24, 24], 11, m, b) / ener_Df([24, 24], 11, m, b) - 1) for m in [0, 0.05, 0.3, 1.0])
    print('   %-4s max relative diff (m=0, 0.05, 0.3, 1) = %.1e' % (b, d))

print('3. N_L*ener_Df - (C0*L_U + C1 + casimir_exact), and casimir_finite_ref(N_L0=400) vs casimir_exact (L_periodic=[24,24], N_L=11)')
for b in BOUNDS:
    d = max(abs(11 * ener_Df([24, 24], 11, m, b) - bulk_C0([24, 24], m) * phys_length(11, b)
                - boundary_C1([24, 24], m, b) - casimir_exact([24, 24], 11, m, b)) for m in [0, 0.3])
    r = abs(casimir_finite_ref([24, 24], 11, 400, 0.3, b) / casimir_exact([24, 24], 11, 0.3, b) - 1)
    print('   %-4s split: max |diff| (m=0, 0.3) = %.1e    finite ref: relative diff = %.1e' % (b, d, r))

print('4. casimir_exact/m^2 vs continuum at fixed mL=2.4, D=2, N_A=max(96,16 N_L): relative deviation')
for b in BOUNDS:
    dev = []
    for NL in [8, 16, 32, 64]:
        M = 2.4 / phys_length(NL, b)
        NA = max(96, 16 * NL)
        dev.append(abs(casimir_exact([NA, NA], NL, M, b) / M**2 / casimir_continuum(2.4, 2, b) - 1))
    print('   %-4s N_L=8,16,32,64: ' % b + '  '.join('%.1e' % v for v in dev))

print('5. massless case')
print('   (mL)^D * continuum(mL=1e-4) / massless continuum - 1')
for D in range(1, 5):
    print('   D=%d  ' % D + '  '.join('%s %+.1e' % (b, casimir_continuum(1e-4, D, b) * 1e-4**D / casimir_continuum_massless(D, b) - 1)
                                    for b in BOUNDS))
print('   casimir_exact(M=0)*L^D / massless continuum - 1, with N_A = r*N_L')
#   D   N_L values        r=N_A/N_L
for D, NLs, r in [(1, [4, 8, 16, 32], 4096), (2, [4, 8, 16, 32], 128), (3, [4, 8, 16], 32), (4, [4, 6, 8, 12], 16)]:
    for b in BOUNDS:
        dev = [casimir_exact([r * NL] * D, NL, 0.0, b) * phys_length(NL, b)**D / casimir_continuum_massless(D, b) - 1 for NL in NLs]
        print('   D=%d r=%-4d %-4s N_L=%s: ' % (D, r, b, ','.join(map(str, NLs))) + '  '.join('%+.1e' % v for v in dev))

print('6. integer overflow')
L32 = [np.int32(224)] * 4
print('   N_A^D of [224]^4 given as int32: _size %.0f, exact %d, np.prod %d' % (_size(L32), 224**4, np.prod(L32)))
#   L_periodic      N_L    (volume N_A^D*N_L > 2^31)
for Lp, NL in [([70000], 40000), ([2000, 2000], 600)]:
    Lp32, NL32 = [np.int32(L) for L in Lp], np.int32(NL)
    split = same = 0
    for b in BOUNDS:
        e = ener_fast(Lp, NL, 0.3, b)
        split = max(split, abs(NL * e - (bulk_C0(Lp, 0.3) * phys_length(NL, b) + boundary_C1(Lp, 0.3, b) + casimir_exact(Lp, NL, 0.3, b))))
        same = max(same, abs(ener_fast(Lp32, NL32, 0.3, b) - e),
                   abs(casimir_finite_ref(Lp32, NL32, np.int32(2 * NL), 0.3, b) - casimir_finite_ref(Lp, NL, 2 * NL, 0.3, b)))
    print('   L_periodic=%s, N_L=%d (volume %.1e): split max |diff| = %.1e, int32 vs int max |diff| = %.1e'
          % (Lp, NL, _size(Lp) * NL, split, same))
