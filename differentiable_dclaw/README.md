# Differentiable D-Claw (JAX)

A reduced, fully-differentiable re-implementation of the digclaw
five-equation system (Iverson & George 2014, *A depth-averaged debris-flow
model that includes the effects of evolving dilatancy. I. Physical
basis*), for gradient-based calibration research. Not a drop-in
replacement for D-Claw -- see "What's deliberately out of scope" below.

## Status

All eight tests in `tests/` pass (CPU or GPU):

```
python3 tests/test_contact_wave.py   # still-water m-jump stays exactly still
python3 tests/test_damfront.py       # dry dam-break front stays under the frictionless bound
python3 tests/test_gradients.py      # autodiff matches finite differences, float64
python3 tests/test_lake_at_rest.py   # well-balanced: still water over uneven terrain stays still
python3 tests/test_reverse_mode.py   # reverse mode w.r.t. the whole bed field, checkpointed = plain,
                                     #   directional derivative matches finite differences
python3 tests/test_calibration.py    # Gauss-Newton recovers phi and kref exactly from a
                                     #   peak-depth profile, starting 10 deg / 10x away
python3 tests/test_gate.py           # moving flume headgate: closed gate holds a pond exactly,
                                     #   release moves, gradients through it match FD
python3 tests/test_steep_slope.py    # gravity driving force of thin layers on a 31 deg slope,
                                     #   for both hydrostatic reconstructions
```

Autodiff through a full rollout is exact: it matches converged finite
differences to 6+ significant figures, forward mode equals reverse mode,
on both a small synthetic problem and the full-scale Montecito setup
(216x8 cells, 25 m, 1800 s, ~1000-2500 steps). No multiple shooting,
checkpointing tricks or gradient special-casing are involved: plain
`jax.jacfwd` / `jax.grad` through `solver.rollout`. See "Why gradients
used to look like they exploded" below -- the short version is that it
was never an adjoint problem, it was five places where this port had
drifted from the Fortran reference.

## GPU and reverse mode

With a CUDA build of jaxlib (`pip install "jax[cuda13]==<your jax version>"`)
everything runs unchanged on GPU, in float64. On an RTX 5000 Ada the full
1800 s real-terrain Montecito run (118x200 cells) takes 0.7 s instead of
19 s on CPU, with identical results -- this stencil code is bandwidth-
bound, so the card's low FP64 rate does not matter.

For gradients with respect to many inputs (a whole field), use
`solver.rollout_peak(..., checkpoint_block=k)` with `jax.grad`: it carries
only the running peak depth and recomputes each block of k steps on the
backward pass. On the real-terrain setup, the gradient of summed peak
depth at the 238 damaged buildings with respect to all 23,600 bed
elevations (plus phi, kref) takes 7 s and 0.21 GB; without checkpointing
the same backward pass would need ~618 GB. See
`validation/montecito_realterrain_crosscheck/sensitivity.py`.

## Real Montecito data

`examples/montecito_calibration.py` fits (phi_deg, kref, cv) to the
Kean et al. 2019 peak-depth profile (20 bins, 107-3274 m downstream)
using the idealized setup of `examples/montecito_real_data.py`. It runs
end to end (~16 min CPU). The best RMS misfit is **0.81 m**, against
1.54 m at the Barnhart ensemble median on the same setup, but it is not a
meaningful calibration: kref goes to its upper bound (1e-8 m^2), where
the mixture dilates to near-pure water (m < 0.1, so Coulomb friction is
tapered off and only Manning friction resists), and phi_deg then has no
effect at all. The mismatch is structural, not an optimizer problem: an
instantaneous pile on a uniform 4 deg slope can only produce peak depths
that *decrease* downstream and stop by ~2.4 km, while the observed ones
*increase* (0.2 m near the apex to 1.2 m at 3.3 km). A real fan DEM and
an inflow hydrograph are needed before the fitted parameters mean
anything.

**On the real pre-event terrain** (Prescott et al. 2023 5 m DEM,
Zenodo 10.5281/zenodo.7838914, coarsened to 20 m), the cross-check in
`validation/montecito_realterrain_crosscheck/` runs this solver and the
compiled Fortran D-Claw on bit-identical inputs (terrain and initial
ponds read back from the Fortran run's own output; ensemble-median
parameters; Manning n=0.06). Against the observed inundation (critical
success index, CSI):

| run | front position at 300 / 600 / 1200 / 1800 s (m) | CSI |
|---|---|---|
| Fortran D-Claw, order 1 | 2410 / 1390 / 410 / 90 (reaches the coast) | 0.519 |
| Fortran D-Claw, order 2 | 2770 / 2530 / 2110 / 1990 | 0.154 |
| dclaw_jax | 2390 / 1490 / 830 / 450 | 0.488 |

This solver now tracks the first-order Fortran run closely through the
first ~10 minutes (same front positions, pore pressure at lithostatic,
same solid fraction) and falls a little behind late in the lower fan.
Gauss-Newton calibration of (phi_deg, kref) against the observed extent
(`calibrate.py` there) gets from the ensemble median to phi_deg~31.5,
kref=7.4e-12 m^2 in 2 iterations and then only creeps along a flat
valley, raising CSI from 0.488 to 0.512 (~2 min on GPU). (Before item 12:
CSI 0.491, calibrated to phi_deg=31.75, kref=7.06e-12, CSI 0.517; with
item 12 but not 13 one run stopped at phi_deg=28.66 on the same valley,
CSI 0.509.)
Getting here took two more fixes, found by this exact comparison (items
8 and 9 below); before them the same run had CSI 0.163 and stalled
2.6 km short. Autodiff through this run matches finite differences to
6-7 significant figures (`grad_check.py`).

Two things are not verified against the Fortran reference and should be
before relying on high-permeability runs: the "dilates to water" regime
itself (seen for kref >~ 1e-10.4 m^2 with this setup; at extreme values
without Manning friction it produced a 117 m column of m~0.004 fluid),
and the effect of the dry-cell and friction simplifications listed under
"What's deliberately out of scope".


## Why gradients used to look like they exploded (and what actually fixed it)

Early on, `jax.grad` of a loss at the end of a long rollout came back
absurdly large (d/dkref ~1e21 after 200 steps, ~1e308 later), and I
spent a long time treating that as the well-known "exploding adjoint"
problem of long chaotic trajectories (Least-Squares Shadowing, multiple
shooting, Gauss-Newton/Augmented-Lagrangian on shooting states). That
diagnosis was wrong, and all of that code has been removed. What
actually happened, found by checking basics instead of adding machinery:

1. **Forward-mode autodiff matched finite differences; reverse mode did
   not.** Mathematically the two are identical, so this was never
   "the true derivative is huge". Finite differences also swung wildly
   with step size (+3.8 at 1e-4 deg, -5.35 at 1e-6), i.e. the loss itself
   was rough: piecewise plateaus with jumps.
2. **Root cause 1: Coulomb friction stick-slip chattering.** Friction
   was a regularized closure inside the explicit RK2. For quasi-static
   material (e.g. the pile's tail, tan(phi+psi)=0.53 on a 10 deg slope)
   one explicit step changes u by ~0.26 m/s, far more than the 1e-3 m/s
   smoothing width, so u sawtoothed through zero every step (0.02 -> 0.13
   -> 0.22 -> 0.07 ...). Two runs 1e-4 deg apart fell one chatter period
   out of phase -- that phase flip was the loss "jump". Fix: port the
   reference's split step, `vnorm <- max(0, vnorm - dt*tau/(rho*h))`
   (src2.f90), in `model.friction_step`. The tail then sits at exactly
   u=0, and the loss became smooth to 1.7e-7 on a 2e-3 deg scan. (This
   also removed the 5.5e-3 m/s "gradient-epsilon drift" the old
   contact-wave test tolerated -- it was the same chattering; that test
   now stays at exactly zero velocity.)
3. **Root cause 2: the dilatancy (D) sources stepped explicitly.** The
   D-driven changes of h, hu, hv, hm were in the explicit RK2, with D
   computed from pore pressure *before* relaxation; in thin frontal cells
   that scales like 1/h^2 and amplified m's sensitivity by up to ~1e6 per
   step (measured per operator with power iteration). Fix: port the
   reference's default source method (mp_update_relax_Dclaw4,
   src2method=0) as `model.dilatancy_step`: exponential update of hm and
   h/hu/hv, using D from the *relaxed* pb.
4. **Root cause 3: no admissibility projection.** The reference's `qfix`
   keeps m in [0, 1] (rewriting hm) and 0 <= pb <= rho*g*h. Without it,
   hm could exceed h and pb could exceed lithostatic -- seen at full
   scale as 15 m peak-depth spikes from a 4 m release. Ported as
   `model.qfix`, called where the reference calls it. This also made two
   ad hoc caps from an earlier session (tanpsi clipped to +-0.2, a
   "can't liquefy faster than 1 s" cap on pressure generation)
   unnecessary; both were removed.
5. **Root cause 4: m transported in primitive form.** A newly wetted
   cell at a wet/dry edge got far too little solid (m=0.08 next to an
   m=0.512 pile on the first step); those fake-dilute cells sit on the
   reference's steep m~0.1 friction taper and stick-slipped. Fix: hm is
   carried conservatively with the h flux at the upwind cell's m
   (`solver._tracer_flux`) -- conserves solids and adds no diffusion
   across a stationary contact.
6. **Root cause 5: missing Manning friction.** Barnhart et al. 2021 used
   Manning n=0.06 for every Montecito D-Claw run; without it, dilute
   films had no real terminal velocity and collapsed the CFL step
   (>12,000 steps couldn't reach 1800 s). Ported as `model.manning_step`
   (`MaterialParams.manning_n`, default 0 = off).
7. **Last: the hard dry-cell reset.** The reference zeroes the whole
   state where h <= 1 mm. That is the only truly discontinuous operator;
   each time a cell crossed it a step earlier or later, the whole
   solution jumped slightly. `solver._clip_dry` instead scales the whole
   state by a smoothstep weight between 1 and 2 mm -- thin films are
   still removed, but continuously. (An intermediate version kept h and
   only ramped momentum/pressure; on the real Montecito DEM that left
   residual films of h~1e-7 m wherever flow had passed, and the 1/h,
   1/h^2 source terms there blew the gradient up to ~1e218. A
   deliberate deviation from the reference, made for differentiability.)

8. **Not well-balanced.** Rusanov on raw h plus a centred -g*h*db/dx
   source does not keep still water still over uneven terrain; on the
   real DEM it pushed material out of the channel onto its banks. Fixed
   with hydrostatic reconstruction (Audusse et al. 2004) in
   `solver._hr_fluxes` -- `tests/test_lake_at_rest.py` holds still water
   over a bump/pit/dry island to 1e-15 -- and an HLL flux instead of
   Rusanov (less diffusive; by itself it made little difference).
9. **A units error in the pb compression term.** The reference's pb
   equation carries gamma*rho*gz*h*div(u) (riemannsolvers_dclaw.f,
   del(4)); this port had gz*chi*h*div(u), missing rho (~1900x too
   small). Material piling up therefore gained almost no pore pressure,
   lost its liquefaction and stopped: on the real DEM the Fortran run
   stayed at pb/lithostatic ~1 while this one fell to ~0.2. Fixed, with
   h*div(u) taken from the actual numerical mass change as the reference
   does (a cell receiving dh gains gamma*rho*gz*dh). This one fix took
   the real-terrain CSI from 0.33 to 0.49.

10. **Explicit dilatancy forcing in the pb step.** The forcing
    S = -(3/(alpha*h))*|u|*tanpsi depends on pb itself (through sigma_e),
    and in liquefied flow that dependence is extremely stiff (3/(alpha*h)
    ~2e5 Pa, m_eq ~ sqrt(sigma_e) near 0). Frozen at the start of the
    step -- as the reference does before clipping pb in qfix -- the
    forward result stays bounded only because of that clip, while the
    per-step derivative dpb/dpb reached ~ -2e6. It showed up only once a
    target was linear in peak depth: reverse and forward mode agreed with
    each other but not with finite differences (4.8e3 vs -2.05). Now
    implicit (per-cell Newton, `model.pb_relaxation_step`); forward
    results on the real-terrain case are unchanged, and the gradient
    matches finite differences to 0.02-1% (depending on the parameter point).

11. **Dilatancy forcing in thin, nearly static deposits.** With a small
    floor on |u| (needed for finite derivatives), 5-10 cm deposits
    creeping at <2 cm/s kept a dilatancy forcing with coefficient
    3/(alpha*h) ~ 3.5e6 Pa, closing a speed -> pore pressure -> friction
    -> speed loop with enormous gain (the reference has zero forcing at
    exactly u=0). Invisible in the forward results and in targets summed
    over buildings, it showed up in a 25,787-component observation vector
    (inundation map + building depths): the tangent dh/dphi reached ~3e10
    and its direction had cosine ~0 with finite differences. The forcing
    is now tapered smoothly to zero below 5 cm/s
    (`model.DILATANCY_U_TAPER`); real-terrain results are unchanged
    (CSI 0.490 vs 0.491, identical fronts) and the full observation-vector
    derivative now matches finite differences with cosine 1.000 at every
    step from 1e-4 to 1e-8.

12. **A floor on the compressibility alpha.** `alpha` (eq 2.8) was
    clamped below at 1e-5 1/Pa, a leftover from the explicit pb stepping
    (see "The numerical stiffness in eq 2.4e" below, where a floor alone
    did not help); the reference has no floor. Realistic values are
    smaller -- 2.0e-6 for the USGS flume (George & Iverson 2014 table 2),
    ~2e-6 at Montecito -- so the floor was always active, and the
    dilatancy forcing and pb relaxation rate, both ~1/alpha, were ~5x too
    weak. Found by the flume cross-check against Fortran D-Claw
    (`gradient_particle_filter/flume_crosscheck`): behind the opening
    gate the reference's pore pressure rose (pb/lithostatic 0.53 -> 0.63
    in 1 s) while this one fell (-> 0.35). Without the floor it follows
    (-> 0.62), gauge depth errors drop by ~40%, and alpha_c, which the
    floor had made almost irrelevant (d(output)/d(alpha_c) -1.5), becomes
    a real parameter (-44.7). Gradients stay exact (every test, and the
    Montecito `grad_check.py`, `sensitivity.py`). Montecito itself, liquefied
    almost everywhere, barely moves: CSI 0.491 -> 0.488, same fronts.

13. **Positive pore-pressure feedback in thin, fast, dilating fronts.**
    Checking the full gauge-series Jacobian of the USGS flume at 16
    parameter points near the paper's values (instead of at one point)
    found forward-mode derivatives of 1e4-1e19 at half of them, where
    finite differences gave O(1-10). The growth came from
    `pb_relaxation_step` alone (+15 decades over 0.7 s) in ~1 cm frontal
    cells moving at ~8 m/s: when they dilate, the forcing
    -(3/(alpha*h))|u|tanpsi lowers pb, which raises sigma_e, lowers alpha
    and strengthens the forcing, a feedback with rate ~1e3 1/s > 1/dt. The
    clip to [0, lithostatic] keeps the forward solution bounded; the
    tangent is not. It is not an artefact of item 12 (with the old floor
    one point still reached 8e20), and neither more Newton iterations nor a
    larger DILATANCY_U_TAPER helped. Fix: h in that forcing is floored at
    `model.DILATANCY_H_FLOOR = 0.02` m. The Jacobian then matches finite
    differences (eps=1e-6) to <1e-3 at all 16 points, the flume gauge
    series move by <= 1.1 mm and 18 Pa, Montecito is unchanged (CSI 0.488,
    same fronts), and losses got smoother too (`test_gate`: finite
    differences at eps=1e-3 now agree to 6 digits, 0.4% before).

What is *not* a bug, and limits calibration regardless of method:
long, slowly creeping deposits still show a little deterministic
structure (steps of ~1e-4 m in peak depth on a 1e-5 scale in log10
kref), and a few locations are bistable (a deposit lobe forms or not,
+-0.2-0.3 m). Gauss-Newton therefore uses a *smoothed* autodiff
Jacobian (the exact Jacobian averaged over a few nearby parameter
points). More importantly, with a peak-depth profile as the only data,
**kref is essentially not identifiable** between ~1e-13 and ~1e-10.5
m^2 -- the profile barely changes across 2.5 decades except for a
regime switch to long runout near 1e-10.45 -- while phi_deg is
recovered to within a few tenths of a degree in synthetic tests.

## Moving headgate (USGS flume)

`solver.gate_bed(b_base, gate_shape, gate_t, gate_deg)` returns b(t) for
a gate release: the closed gate is a ridge `gate_shape` on `b_base`, and
it is lowered to cos(angle(t)) of its height as the gate swings open --
the reference's own representation
(`validation/usgs_flume_2010_paper2014/model/setinput.py`,
`write_gate_dtopo`). `tests/test_gate.py`:

- a pond behind the closed gate, with the gate turned 25 deg (still
  above the pond), stays still to 1e-15 m/s and nothing gets past it;
- with the paper's flume material (phi 40.7 deg, kref 5e-9, m0 0.62) on
  a 31 deg slope, a 0.85 s opening lets 8.6 of 10 m3 past x=3 m by 4 s
  (8.9 with a 0.05 s opening, 5.2 with a 3 s one);
- d/d(opening time) and d/dphi of that volume at 2.5 s: reverse = forward,
  and both match finite differences at eps=1e-6 to 6 digits.

Two things this surfaced, both properties of the solver rather than of the
gate:

- **Dry-film removal loses mass on a steep flume.** `_clip_dry` removed
  1.4% of the mixture mass in 4 s here (every other operator conserves
  it to <1e-5; the reference zeroes only h <= 1 mm, this removes films
  up to 2 mm). Much larger than on the 20 m Montecito grid; worth
  checking against the Fortran run before trusting deposit volumes.
- **The loss is rougher in the gate timing than in material parameters**
  (4e-6 vs 2e-8 m3 residual from a quadratic over +-2e-4): each step the
  lowering ridge crosses the flow surface at a slightly different point, a
  kink in the hydrostatic reconstruction's max(). The derivative is exact
  but local -- a finite difference at eps=1e-3 is 1-2% off. The opening
  time is measured (GateAngle), so this matters only if it were estimated.

The full-scale comparison with Fortran D-Claw on this flume, which found
item 12 above and the steep-slope issue below, is in
`gradient_particle_filter/flume_crosscheck/`.

## Steep slopes: which hydrostatic reconstruction

`solver.RECONSTRUCTION` selects "audusse" (default, Audusse et al. 2004)
or "chen_noelle" (Chen & Noelle 2017). They agree wherever the
interface is fully wet, so lake at rest, shorelines and positivity are the
same. On a steep slope a layer thinner than the bed step between two
cells gets only h/(2 db) of its gravity driving force from Audusse
(27% for a 2 cm layer on the 31 deg flume at 6.25 cm cells) and
1 - h/(2 db) from Chen-Noelle (73%) -- `tests/test_steep_slope.py`.
With Audusse, thin tails stay pinned to the flume slope (2 cm left at
32 and 66 m where Fortran leaves nothing); the error is first order in
dx and shrinks with refinement (7 and 4 mm at 3.1 cm cells).
Chen-Noelle fixes the tails and the near-gate hydrograph, but the thin
films it sets moving keep crossing `_clip_dry`'s 1-2 mm ramp: losses
integrated over the flow or set by a thin front become rough at small
scales (autodiff and finite differences disagree by 2-10x down to
eps=1e-7) and 13-18% of the mixture mass is removed in 35 s (Fortran 7%,
Audusse 8%). A 2-4 mm ramp makes its gradients exact again but removes
30%. For gradient-based work Audusse on a finer grid is the better
trade today; having both would need a mass-conserving, differentiable
treatment of thin films.

## Equations implemented

State `q = (h, hu, hv, hm, pb)`. Governing equations: eq 2.1(a-e).
Closures: permeability eq 2.7, compressibility eq 2.8 (alphamethod=0,
"case 0" in the Fortran reference), dilatancy eq 2.12-2.14, friction eq
2.9-2.11, source terms eq 2.4. `kappa=1` throughout (hardcoded in the
Fortran reference too, `digclaw_module.f90:113`; the paper states this is
what all of its flume comparisons used), which eliminates the direct
pb-gradient term from the momentum flux (eq B1) and gives the
shallow-water-like eigenvalues `lambda = u -/+ sqrt(gz*h)` verified in
`tests/eigenstructure_check` at the repo root.

Every closure was checked line-by-line against
`clawpack-runtime/dclaw/src/2d/dig/digclaw_module.f90` while building
this (same source read for `tests/eigenstructure_check`).

## What's deliberately out of scope

Per an explicit scope decision (not an oversight):
- **No AMR.** Single uniform rectangular grid.
- **No topography-file reading.** The bed `b(x, y)` is any array you
  build or load yourself (including a real DEM) and pass to `rollout`
  directly -- there's no `.tt3`/raster-format machinery.
- **No general moving topography.** `b` may be a function of time
  (`rollout(q0, b_of_t, ...)`), evaluated once per step with h left
  unchanged, which is how the reference's dtopo moves the bed. The one
  case built and tested is the USGS flume headgate (`solver.gate_bed`, a
  ridge lowered to cos(angle) of its height, the same representation the
  reference's flume setups use); there is no `.tt3` dtopo reader, and a
  bed moving under wet cells has not been tested.
- **Friction is the reference's split step only.** `model.friction_step`
  ports src2.f90's exact Coulomb + viscous update; the additional
  static-friction bound the reference applies inside its Riemann solver
  (eq 2.15-2.16, calc_taudir) is not ported.
- **h, hu, hv use an HLL flux with hydrostatic reconstruction**, not the
  reference's augmented Riemann solver. Well-balanced (still water stays
  still), first order, somewhat more diffusive, and on steep slopes it
  under-drives layers thinner than the bed step between cells (see
  "Steep slopes" above).
- **pb is advected in primitive (upwind, non-conservative) form**, not
  via the reference's augmented Riemann solver's exact treatment of the
  linearly-degenerate fields; hm is conservative (upwind tracer flux).
  Both first order.
- **Dry cells are handled continuously**, not hard-reset (see item 7
  above).

## The numerical stiffness in eq 2.4e, and how it's handled

This is the part worth understanding before trusting or extending this
code, because it wasn't obvious going in and cost real debugging time.

The paper itself flags that the pb source term is stiff enough to need
special treatment: "the system dQ/dt = phi^B(Q) is solved exactly as an
exponential relaxation to an equilibrium pressure" (section 3, discussing
eq 3.9) -- i.e. the reference Fortran code does **not** step this term
explicitly. Building this solver with a naive explicit RK2 for
everything (the obvious first approach) confirmed exactly why: it blows
up within single-digit steps, for two separable reasons found by
successive isolation testing (zero all sources -> stable; reintroduce
sources -> unstable; narrow down term by term):

1. **The D-relaxation term itself.** `D` (eq 2.6) has a `1/h` factor;
   at a wet/dry front, explicit stepping of the primitive (non-
   conservative) m/pb equations divided by a near-zero h amplifies noise
   without bound. Fixed by flooring h at the grain diameter `delta`
   specifically inside `dilation_rate` (a film thinner than one grain has
   no physical meaning anyway) -- see that function's docstring.
2. **The dilatancy-forcing term.** phi5's second term scales as
   `1/(alpha*h)`, and alpha (eq 2.8) shrinks as effective stress grows --
   under realistic confining stress this coefficient reached ~1e6 and
   drove a genuine positive-feedback runaway (pb rises -> sigma_e falls
   -> m_eq rises -> tanpsi more negative -> pb rises faster). An alpha
   floor alone reduced but did not fix this (mass still inflated ~180x
   over a 0.5s test).

The actual fix is Godunov splitting: the explicit RK2 stage carries no
sources except the bed slope, and `pb_relaxation_step` (applied once per
step, after the transport and friction stages) solves pb's full source
*exactly*, freezing every coefficient (rate, hydrostatic target,
dilatancy forcing) at the post-transport state -- the same "exact
exponential relaxation to an equilibrium pressure" the paper describes,
just derived and implemented here rather than ported from Fortran. See
`model.pb_relaxation_step`'s docstring for the closed form and its
numerically-stable evaluation (the `(1-exp(-x))/x -> 1` limit at x=0).

With this fix: a dam-break-style test that previously reached t=0.085s
before NaN-ing (mass inflated to ~4e7x initial) instead runs cleanly to
completion with plausible-magnitude fields, and the two physics tests
below pass.

## Gradient-safety epsilons, and their forward-accuracy cost

Two `sqrt(x)` calls -- `wave_speed`'s `sqrt(gz*h)` and
`m_equilibrium`'s `sqrt(Nden)`/`sqrt(Nnum)` -- have value 0 at (very
common) physical states: dry cells, and material at rest. `sqrt`'s
forward value there is fine, but its derivative `1/(2*sqrt(x))` is
infinite, and reverse-mode autodiff evaluates *both* branches of any
`jnp.where` built on top of these, so `0 * inf = nan` leaks through
regardless of which branch the forward pass selects. This surfaced as
`jax.grad` returning exactly `nan` even though the forward simulation was
completely clean -- confirmed by testing `rhs`/`step` calls directly
(no nan) immediately before the full gradient check (nan).

Fixed by adding a tiny epsilon (`1e-12`) inside each sqrt. An earlier
version of this README attributed a small velocity drift in
`test_contact_wave.py` to that epsilon; it was actually friction
chattering (see root cause 1 above), and the test now stays at exactly
zero velocity.

## Finite-difference gradient verification needs float64

JAX defaults to float32. `kref` (~1e-9) needs finite-difference
perturbations many orders of magnitude smaller still to resolve its
gradient; in float32 these underflow completely (confirmed directly:
central-difference estimates collapsed to exactly 0 below `eps~1e-11`).
`tests/test_gradients.py` sets
`jax.config.update("jax_enable_x64", True)` before anything else runs.
With that, gradients match finite differences to 5-6 significant figures.
This cost real time to find because the *symptom* (gradients off by a
roughly constant factor across a wide range of FD step sizes, rather than
diverging or converging) looked like a code bug rather than a precision
ceiling; the tell was that finite differences on the *same* quantity
computed after enabling float64 converged cleanly where the float32
version had plateaued at a stable-but-wrong ratio.

## A note on the dam-break bound test specifically

`tests/test_damfront.py` checks the modeled front never exceeds the
classical frictionless shallow-water speed `2*sqrt(g*h_L)`. Setting
`phi_deg=0` alone is *not* sufcient to make the material frictionless:
the dilatancy angle psi (from `m < m_crit` material contracting under
shear) adds to the effective friction angle as `tan(phi_deg + psi)`, and
psi is negative for contracting material -- i.e. it can drive the
effective friction coefficient negative, accelerating the flow rather
than resisting it. That's a real mechanism in the full model (matches
the physical picture of contraction-driven mobility that motivates this
whole paper), not a bug, but it breaks the assumption behind the
classical bound, which has no accelerating source term. The test sets
`c1=0` (disabling the dilatancy-angle contribution entirely) for a fair
like-for-like comparison. First attempt at this test also failed for an
unrelated, more mundane reason: the domain was short enough that the
front reflected off the far wall before `t_final`, which looks identical
to "exceeded the bound" if you only check the final front position
without checking domain size first.
