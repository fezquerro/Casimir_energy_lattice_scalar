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
7. infinite transverse lattice: bulk_C0_inf, boundary_C1_inf and casimir_inf against the values of the article (D=2),
   against bulk_C0, boundary_C1 and casimir_exact on a large lattice, the relations between the boundary terms
   and the bulk energy in one dimension less, the exact D=1 integrals J(c) = 1 - tanh(theta_c), the limit m -> 0,
   the massless continuum at m=0, and the cases that hung or overflowed before (large m*l, huge t, int32 N_L)
Every line prints the largest deviation found.
"""
import numpy as np
from modules.energy_formulas import (BOUNDS, _confined_modes, _size, phys_length, ener_Df, ener_fast,
                                     bulk_C0, boundary_C1, casimir_exact, casimir_finite_ref,
                                     bulk_C0_inf, boundary_C1_inf, casimir_inf, _J)
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

print('7. infinite transverse lattice')
#   M: C0, C1^D, C1^N, C1^Z (article, Table tab:D2inf, D=2, 12 digits)
TAB_C = {0.2: (0.169818792371, -0.140004803720, -0.129410889348, -0.134707846534),
         0.4: (0.178378553548, -0.152951909730, -0.119756962473, -0.136354436101)}
#   (M, N_L): E_P, E_A, E_D=E_N, E_Z (same table, 7 digits)
TAB_E = {(0.2, 8): (-1.454752e-3, 1.344202e-3, -5.527501e-5, 7.184201e-5),
         (0.2, 16): (-1.105500e-4, 1.085941e-4, -9.779675e-7, 1.237286e-6),
         (0.4, 8): (-4.861834e-4, 4.778040e-4, -4.189705e-6, 6.720711e-6)}
dC = max(abs(v - r) for M, ref in TAB_C.items()
         for v, r in zip([bulk_C0_inf(2, M)] + [boundary_C1_inf(2, M, b) for b in ['dir', 'neu', 'zar']], ref))
dE = max(abs(casimir_inf(2, NL, M, b) / r - 1) for (M, NL), ref in TAB_E.items() for b, r in zip(['per', 'anti', 'dir', 'zar'], ref))
print('   article table (D=2): C0, C1 max |diff| = %.1e    E_U max relative diff = %.1e (digits printed: 12 and 7)' % (dC, dE))
print('   finite lattice N_A^D vs infinite, M=0.4, N_L=8: max relative diff of C0, C1^U, E_U')
for D, NA in [(1, 4096), (2, 1024), (3, 160), (4, 96)]:
    dC = max([abs(bulk_C0([NA] * D, 0.4) / bulk_C0_inf(D, 0.4) - 1)] +
             [abs(boundary_C1([NA] * D, 0.4, b) / boundary_C1_inf(D, 0.4, b) - 1) for b in ['dir', 'neu', 'zar']])
    dE = max(abs(casimir_exact([NA] * D, 8, 0.4, b) / casimir_inf(D, 8, 0.4, b) - 1) for b in BOUNDS)
    print('   D=%d N_A=%-4d C0, C1: %.1e    E_U: %.1e' % (D, NA, dC, dE))
dN = max(abs(boundary_C1_inf(D, M, 'neu') - boundary_C1_inf(D, M, 'dir') - bulk_C0_inf(D - 1, M))
         for D in range(2, 5) for M in [0.1, 0.4])
dZ = max(abs(-2 * boundary_C1_inf(D, M, 'zar') - bulk_C0_inf(D - 1, np.sqrt(M * M + 4)))
         for D in range(2, 5) for M in [0.1, 0.4])
print('   C1^N - C1^D - C0^(D-1)(M): %.1e    -2 C1^Z - C0^(D-1)(sqrt(M^2+4)): %.1e  (D=2..4, M=0.1, 0.4)' % (dN, dZ))
dJ = max(abs(_J(1, M, c) - (1 - np.tanh(np.arcsinh(np.sqrt(M * M / 4 + c))))) for M in [0, 1e-5, 1e-3, 0.1, 1] for c in [0, 1])
print('   D=1: J(c) - (1 - tanh theta_c), sinh^2 theta_c = M^2/4 + c, M=0..1: max |diff| = %.1e' % dJ)
dC = max(abs(bulk_C0_inf(D, 1e-5) - bulk_C0_inf(D, 0)) for D in [2, 3, 4])
dE = max(abs(casimir_inf(D, 8, 1e-5, b) / casimir_inf(D, 8, 0.0, b) - 1) for D in [2, 3, 4] for b in ['per', 'dir'])
print('   m -> 0 (D=2..4): |C0(1e-5) - C0(0)| = %.1e    E_U(1e-5)/E_U(0) - 1 (N_L=8) = %.1e' % (dC, dE))
print('   m=0: E_U^inf L^D/A / massless continuum - 1, N_L=8,16,32,64 (falls as 1/N_L^2)')
for D in range(1, 5):
    print('   D=%d  ' % D + '   '.join('%s %s' % (b, ' '.join('%+.1e' % (casimir_inf(D, NL, 0.0, b) * phys_length(NL, b)**D
                                                                        / casimir_continuum_massless(D, b) - 1) for NL in [8, 16, 32, 64]))
                                    for b in ['per', 'anti', 'dir', 'zar']))
print('   edge cases (hung or gave nan before):')
for args in [(1, 400, 2.0, 'per'), (1, 30000, 0.0005, 'dir'), (1, np.int32(30000), 0.0005, 'dir'), (2, 8, 1e-4, 'per')]:
    print('   casimir_inf%-38s = %.6e' % (str(args), casimir_inf(*args)))
