
# Analytic Casimir Energy of a Free Scalar Field on the Lattice


**Exact computation of the lattice Casimir energy of a free massive scalar field in $D+1$ dimensions with periodic, anti-periodic, Dirichlet, Neumann and Zaremba boundary conditions, and comparison with the continuum, in Python.**



## Table of Contents
- [Overview](#overview)
- [Physics](#physics)
- [Code structure](#code-structure)
- [Running](#running)
- [Figures](#figures)

## Overview

This project computes the vacuum energy of a free massive scalar field on a Euclidean lattice with $D$ periodic directions (time included) and one confined direction with boundary conditions. Since the theory is free, the energy is a sum over the lattice modes and no Monte Carlo simulation is needed. The sum over the confined direction is done in closed form, which gives the exact lattice Casimir energy without a reference lattice and without the cancellation of large energies.

The code checks the renormalization prescription with a reference lattice used for the Monte Carlo simulations, the divergent bulk and boundary contributions of the energy, the finite-size effects, and the approach to the continuum Casimir energy for $D=1,\dots,4$.

This code was developed for the PhD thesis **[Properties of the quantum vacuum in non-abelian gauge theories](https://arxiv.org/abs/2605.18220)**, where the free scalar field is the test case of the methods later used for the gauge theories. See Chapter 2 for the derivation of the analytical formulas of the continuum Casimir energy, and Chapter 3 for the lattice scalar field, the implementation of the boundary conditions on the lattice and the renormalization prescription with a reference lattice.

## Physics

- **Lattice**: $N_0 \times \dots \times N_{D-1}$ periodic directions (all of size $N_A$ in the scripts) and one direction of $N_L$ sites with boundary condition $U$. $M = ma$ is the mass in lattice units.
- **Boundary conditions** (`bound` in the code): `'per'` periodic, `'anti'` anti-periodic, `'dir'` Dirichlet, `'neu'` Neumann, `'zar'` Zaremba (Dirichlet on one side, Neumann on the other). For Zaremba the physical length is $L = N_L - 1/2$.
- **Exact decomposition**: the energy per unit transverse area splits as

  $$N_L\, e_U(N_L) = C_0\, L_U + C_1^U + \frac{E_U}{A},$$

  with $C_0$ the bulk energy density, $C_1^U$ the boundary energy and $E_U/A$ the Casimir energy, which decays exponentially with $N_L$. All three are computed in closed form.
- **Dirichlet and Neumann** have exactly the same lattice Casimir energy: they differ only in the boundary term $C_1$.
- **Boundary terms and bulk energy in one dimension less**: with $e_0$ and $e_1$ the bulk energy of the lattice with one dimension less at masses $m$ and $\sqrt{m^2+4}$,
  $C_1^D = -(e_0+e_1)/2$, $C_1^N = (e_0-e_1)/2$, $C_1^Z = -e_1/2$.
- **Renormalization prescription** of the thesis: subtraction of a reference lattice of size $N_{L_0}$. It differs from the exact result by the Casimir energy of the reference lattice, $\sim e^{-mN_{L_0}}$ (periodic) or $e^{-m(N_{L_0}-2N_L)}$ (Dirichlet, Neumann, Zaremba).
- **Continuum**: Bessel-function series for the Casimir energy $E_U/(Am^D)$ as a function of $mL$.

## Code structure

### Modules (`modules/`)

| File | Contents |
|---|---|
| `energy_formulas.py` | Lattice formulas. `ener_Df` (explicit sum over all the modes), `ener_fast` (closed-form sum over the confined direction), `bulk_C0`, `boundary_C1`, `casimir_exact` (exact Casimir energy, no reference lattice), `casimir_finite_ref` (prescription with a reference lattice), `casimir_second_diff` (second-difference renormalization), `casimir_scan` (Casimir energy and $mL$ for lists of masses and $N_L$). |
| `continuum.py` | Continuum Casimir energy for every boundary condition, `casimir_continuum(mL, D, bound)`, and its massless limit $E_U/A = c/L^D$, `casimir_continuum_massless(D, bound)`. |
| `plotting.py` | Shared plot helpers, labels, colors and default parameters (masses, $N_L$ range, $N_A$). Every figure is saved in `plots/`. |

### Scripts

| Script | What it does |
|---|---|
| `Casimir_lattice.py` | Exact lattice Casimir energy vs the continuum: all the boundary conditions together for $D=2$ and $D=3$, and $D=1..4$ together for each boundary condition. |
| `Casimir_massless.py` | Massless case: exact lattice Casimir energy at $M=0$, $E_U L^D/A$ against $N_L$, vs the massless continuum value, $D=1..4$. The periodic directions are taken much larger than $N_L$ ($N_A = rN_L$), since at $M=0$ their finite-size effects decay only as a power of $N_L/N_A$. |
| `Casimir_size_check.py` | Finite-size effects of the exact Casimir energy: scan of the size $N_A$ of the periodic directions and of the size $N_L$ of the confined direction at fixed $mL$ (continuum limit), $D=1..4$. |
| `Casimir_prescription_compare.py` | Exact Casimir energy vs the prescription with a reference lattice $N_{L_0}$, $D=1..4$. Prints the largest relative difference for $N_{L_0} = N_A/2,\ N_A,\ 2N_A$. |
| `Casimir_cont_ren.py` | Casimir energy from the second-difference renormalization $E(N_L)+E(N_L+2L_0)-2E(N_L+L_0)$, periodic and Dirichlet, $D=2$. |
| `Energy_renormalization.py` | Divergent contributions: bulk ($C_0$) and boundary ($C_1$) terms from the energies vs the exact formulas, and the bulk energy in one dimension less recovered from the boundary terms. |
| `Checks.py` | Consistency checks: eigenvalues of the lattice operator built as a matrix vs the analytic modes, `ener_fast` vs `ener_Df`, the exact bulk/boundary/Casimir decomposition, the reference prescription for large $N_{L_0}$, the $O(a^2)$ approach to the continuum, and the massless case (lattice at $M=0$ vs the massless continuum formulas). |

## Running

Requirements: Python 3 with `numpy`, `scipy` and `matplotlib`, and a LaTeX installation (the figures use `text.usetex` and the `braket` package).

Run each script from the repository folder, for example
```bash
python Casimir_lattice.py
```
The figures are saved in `plots/`. `Checks.py` only prints the deviations it finds.

Approximate running times: `Checks.py` and `Casimir_cont_ren.py` about 1 second, `Casimir_lattice.py` about 10 seconds, `Energy_renormalization.py`, `Casimir_prescription_compare.py`, `Casimir_size_check.py` and `Casimir_massless.py` about 1 minute (most of it in $D=4$). The sums run only over the distinct transverse modes: $\sin^2(\pi n/N)=\sin^2(\pi(N-n)/N)$ and the spatial periodic directions of equal size can be permuted, which reduces the number of terms from $N_A^D$ to about $(N_A/2)^D/(D-1)!$ (from $10^8$ to $1.2\cdot10^6$ for $D=4$, $N_A=100$).

## Figures

| Script | Files in `plots/` |
|---|---|
| `Casimir_lattice.py` | `Casimir_lattice_2ds.pdf`, `Casimir_lattice_3d.pdf`, `Casimir_lattice_D<bc>s.pdf` (`DDirichlets` is Dirichlet = Neumann) |
| `Casimir_massless.py` | `Casimir_massless_D<D>.pdf` |
| `Casimir_size_check.py` | `Casimir_size_A_D<D>.pdf`, `Casimir_size_L_D<D>.pdf` |
| `Casimir_prescription_compare.py` | `Casimir_prescription_D<D>.pdf` |
| `Casimir_cont_ren.py` | `Casimir_scalar_renorm.pdf` |
| `Energy_renormalization.py` | `<bc>_divergence_scalar.pdf`, `Boundary_vs_bulk_lower_dim.pdf` |
