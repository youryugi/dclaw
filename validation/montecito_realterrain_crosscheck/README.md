# Montecito real-terrain cross-check: dclaw_jax vs Fortran D-Claw

Runs the differentiable JAX re-implementation (`differentiable_dclaw/`) and
the compiled Fortran D-Claw on **identical inputs** for the 9 January 2018
Montecito Creek debris flow, and compares both with the observed
inundation.

- Terrain and observations: Prescott et al. 2023,
  [Zenodo 10.5281/zenodo.7838914](https://doi.org/10.5281/zenodo.7838914)
  (`thomas_preevent_5mresample_fill_meters.txt`,
  `montecito_inundation_5mresample.txt`), saved as
  `dataset/differentiable/montecito_preevent_5m.npz`. Montecito Creek
  sub-domain, 5 m DEM averaged to 20 m, nodata filled with the nearest
  valid elevation.
- Release: D-Claw has no inflow boundary, so both runs start from the same
  ponds at rest at the Cold Springs / Hot Springs sources (179,000 and
  52,000 m3, Prescott/Kean volumes), each filled to one level within
  150 m of the channel source cell. The JAX run reads its terrain and
  initial state back from the Fortran run's own output, so inputs are
  bit-identical.
- Parameters: Barnhart et al. 2021 Montecito ensemble median (phi 37.63
  deg, kref 3.348e-12 m2, m0 0.512, m_crit 0.64, mu 0.005, Manning 0.06),
  hydrostatic initial pore pressure, single 20 m level (no AMR), frames
  every 20 s; peak depth = max over frames on both sides.

## Run

```
source environment.sh
python build_inputs.py
cd model && make .exe && ORDER=1 python setrun.py && ./xdclaw && mkdir -p _output_o1 && mv fort.* _output_o1/
ORDER=2 python setrun.py && ./xdclaw && mkdir -p _output_o2 && mv fort.* _output_o2/ && cd ..
python run_jax.py          # ~1 min on CPU
python compare.py          # results_crosscheck.png
python diag_state.py       # speed / m / pb-to-lithostatic side by side
python grad_check.py       # autodiff vs finite differences (~10 min)
python calibrate.py        # Gauss-Newton on (phi, log10 kref), ~40 min
python plot_calibration.py # results_calibration.png
python sensitivity.py 50    # reverse-mode dL/d(bed) at damaged buildings (~7 s on GPU)
python plot_sensitivity.py  # results_sensitivity.png
```

## Results

Critical success index (CSI) against the observed extent, peak depth > 0.1 m:

| run | front y at 300 / 600 / 1200 / 1800 s (m) | CSI |
|---|---|---|
| Fortran D-Claw, order 1 | 2410 / 1390 / 410 / 90 (coast) | 0.519 |
| Fortran D-Claw, order 2 | 2770 / 2530 / 2110 / 1990 | 0.154 |
| dclaw_jax | 2390 / 1490 / 830 / 450 | 0.488 |
| dclaw_jax, calibrated (phi 31.45 deg, kref 7.43e-12 m2) | | 0.512 |

Autodiff through the full 1800 s run matches finite differences to 6
significant figures (d/dphi 0.4808602 vs 0.4808612; d/dlog10 kref
-30.81516 vs -30.81516 at eps=1e-7). Calibration from the ensemble median
reaches the calibrated point in 2 Gauss-Newton iterations and then creeps
along a flat valley (~2 min on GPU; `calibrate_alphafix.log`).
These numbers are after items 12 and 13 of `differentiable_dclaw/README.md`
(no floor on the compressibility alpha; a 2 cm floor on h in the
dilatancy forcing); before them: CSI 0.491 with
fronts 2410 / 1490 / 830 / 450, calibrated to phi 31.75 deg,
kref 7.06e-12 m2, CSI 0.517 (`calibrate_gpu.log`). What remains is structural and shared with the Fortran
run: the observed south-west lobe of the lower fan is not reached, and the
lower fan spreads too much.

This comparison is what exposed the last two bugs in dclaw_jax (a
non-well-balanced scheme, and a pore-pressure compression term missing
rho); see `differentiable_dclaw/README.md`, items 8 and 9. Before them the
same JAX run had CSI 0.163 and stopped 2.6 km short.

## Terrain sensitivity (reverse mode)

`sensitivity.py`: L = summed simulated peak depth at the 238 buildings in
this domain recorded as damaged (CalFire state != Unimpacted,
buildings.csv from the Barnhart et al. 2021 data release), at the
calibrated parameters. One `jax.grad` through `rollout_peak` with
checkpoint blocks of 50 steps gives dL/d(bed) for all 23,600 cells plus
dL/dphi and dL/dkref: 7 s and 0.21 GB on an RTX 5000 Ada (without
checkpointing: ~618 GB, out of memory). Directional check along a random
smooth bed perturbation: reverse -2.181514, forward -2.181514, finite
difference -2.181514 at eps=1e-6 (before items 12-13: -2.54499 vs
-2.57144, 1%). The map (results_sensitivity.png) is a local
linear sensitivity at cell scale; designing a protective structure would
optimize a smooth parameterized shape instead. Buildings the simulation
does not reach (the south-west lobe) get no information from it.

## High-dimensional inversion: identical-twin test

`twin_inversion.py [lam] [noise] [maxiter] [phi|bed]` sets a known field
(23,600 unknowns, one per cell), generates observations of the same kinds
as the real data (soft inundation map over the scored cells + peak depth at
the 2187 buildings), and inverts from a wrong start with L-BFGS-B (one
checkpointed reverse-mode gradient per evaluation, ~6 s on GPU),
regularized by lam * sum |grad x|^2. `plot_twin.py` draws the results.
(Run before items 12-13 of the package README; not re-run.)

| field | truth | start RMSE (flow footprint) | final RMSE | misfit start -> end |
|---|---|---|---|---|
| friction angle | smooth, 26-38 deg | 3.28 deg | 3.25 deg | 75 -> 64 |
| bed correction | +2 m mound, -1.5 m hollow on the path | 0.380 m | 0.389 m | 116 -> 30 |

Neither field is recovered. Gradients are not the problem (the full
observation-vector derivative matches finite differences, cosine 1.000);
the observations are. Measured sensitivities (`identifiability.py`): a
~100 m local change moves the observation vector by 0.006 per degree of
phi, 0.035 per 0.1 decade of kref and 1.5 per metre of bed -- phi barely
matters because the flow is liquefied almost everywhere (friction ~
sigma_e*tan(phi) with sigma_e ~ 0). With an inundation map and
building-point depths, many different fields fit the observations about
equally well (the bed run reduced the misfit 4x while moving further from
the truth), and L-BFGS stops at a non-zero misfit. Per-cell fields are not
identifiable from this kind of data; lower-dimensional parameterizations
or richer observations (a full deposit-thickness map from pre/post-event
lidar, velocities, timing) are needed.

Why order 2 does worse than order 1 here was not investigated.
