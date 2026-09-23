# Eigenstructure sanity checks

Two small, fast checks of D-Claw's characteristic (wave) structure against
closed-form results that do not depend on reproducing any specific paper's
published numerical run -- only on the model's own governing equations
(Iverson & George 2014, *Physical basis*, eq 2.1-2.28) and, for the
dam-break case, the classical shallow-water solution. `kappa` (the lateral
pressure coefficient) is hardcoded to 1.0 in `digclaw_module.f90`, matching
what the 2014 papers state they used for all of their flume comparisons, so
the eigenvalues (eq 2.24) reduce to the familiar shallow-water form
`lambda = u -/+ sqrt(gz*h)` (nonlinear fields) and `lambda = u` (linearly
degenerate/contact fields).

## Test A: contact-wave preservation

`EIGEN_TEST=contact` sets up a solid-volume-fraction (`m`) jump at `x=0` in
otherwise still water of uniform depth on a flat bed, with hydrostatic pore
pressure on both sides. With equal `h` and `kappa=1`, every term in the
momentum source (eq 2.4b) vanishes identically, so the net driving force is
zero everywhere and the `m`-interface is a pure contact discontinuity
(eigenvalue `u=0`, eigenvector `(0,0,0,1,0)` in eq 2.26): nothing should
move.

Pass criterion: after 2 s, `max|u|` over the whole domain and the
`m`-interface position both match their initial values to machine
precision.

## Test B: dry dam-break front-speed bound

`EIGEN_TEST=damfront` releases a 2 m deep reservoir onto a dry, flat,
frictionless (`phi=0`) bed from rest. The classical frictionless
shallow-water dam-break solution gives an exact front speed of
`2*sqrt(g*h_L)`. Basal friction and any residual dilatancy/compressibility
effects in the full D-Claw equations can only slow the modeled front below
this value, never exceed it, so it is a genuine upper bound implied by the
nonlinear eigenvalue `lambda = u + sqrt(gz*h)`.

Pass criterion: the modeled wet front position never exceeds the
theoretical bound (plus a small grid-scale tolerance) at any output time.

## What this does and does not verify

A pass confirms the compiled solver's wave-propagation/Riemann-solver core
correctly reproduces the model's own published characteristic structure,
independent of any particular paper's specific run configuration. It does
**not** verify grid convergence, the AMR flagging/regridding logic, or any
specific application's setrun (those are covered by the application's own
validation, e.g. `validation/usgs_flume_2017`).

## Running

```bash
source tests/eigenstructure_check/environment.sh
cd tests/eigenstructure_check
make .exe
python3 check.py
```

`check.py` runs both cases end to end (rasters, run data, the D-Claw
executable) and exits non-zero if either check fails.
