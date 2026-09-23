"""D-Claw runtime config for the two eigenstructure sanity checks.

See setinput.py for the physical setup and eigenstructure background
(George & Iverson 2014, eq 2.17-2.26). This is a correctness check against
closed-form results, independent of any specific published run -- it does
not depend on matching a particular paper's numbers, only on the model's
own math (Iverson & George 2014) and the classical dry dam-break solution.
"""

from __future__ import annotations

import os

TEST = os.environ.get("EIGEN_TEST", "contact")

if TEST == "contact":
    X_LOWER, X_UPPER = -5.0, 5.0
    Y_LOWER, Y_UPPER = -0.25, 0.25
    FINAL_TIME = 2.0
    PHI = 35.0
elif TEST == "damfront":
    X_LOWER, X_UPPER = 0.0, 20.0
    Y_LOWER, Y_UPPER = -0.25, 0.25
    FINAL_TIME = 1.0
    PHI = 0.0
else:
    raise ValueError(f"unknown EIGEN_TEST={TEST!r}")

DX = 0.05
DY = 0.05


def setrun(claw_pkg="dclaw"):
    from clawpack.clawutil import data

    assert claw_pkg.lower() == "dclaw"
    rundata = data.ClawRunData(claw_pkg, 2)
    clawdata = rundata.clawdata
    clawdata.lower = [X_LOWER, Y_LOWER]
    clawdata.upper = [X_UPPER, Y_UPPER]
    clawdata.num_cells = [
        int(round((X_UPPER - X_LOWER) / DX)),
        int(round((Y_UPPER - Y_LOWER) / DY)),
    ]
    clawdata.num_eqn = 7
    clawdata.num_aux = 10
    clawdata.capa_index = 0
    clawdata.t0 = 0.0
    clawdata.restart = False
    clawdata.output_style = 1
    clawdata.num_output_times = int(FINAL_TIME / 0.02)
    clawdata.tfinal = FINAL_TIME
    clawdata.output_t0 = True
    clawdata.output_format = "ascii"
    clawdata.output_q_components = "all"
    clawdata.output_aux_components = "none"
    clawdata.output_aux_onlyonce = True
    clawdata.verbosity = 0
    clawdata.dt_variable = True
    clawdata.dt_initial = 1.0e-5
    clawdata.dt_max = 0.01
    clawdata.cfl_desired = 0.4
    clawdata.cfl_max = 0.5
    clawdata.steps_max = 100000
    clawdata.order = 1
    clawdata.dimensional_split = "unsplit"
    clawdata.transverse_waves = 0
    clawdata.num_waves = 5
    clawdata.limiter = [4] * 5
    clawdata.use_fwaves = True
    clawdata.source_split = "godunov"
    clawdata.num_ghost = 2
    clawdata.bc_lower = ["wall", "wall"]
    clawdata.bc_upper = ["extrap", "wall"]
    clawdata.checkpt_style = 0

    amrdata = rundata.amrdata
    amrdata.amr_levels_max = 1
    amrdata.refinement_ratios_x = [2]
    amrdata.refinement_ratios_y = [2]
    amrdata.refinement_ratios_t = [2]
    amrdata.aux_type = [
        "center", "center", "yleft", "center", "center",
        "center", "center", "center", "center", "center",
    ]
    amrdata.flag_richardson = False
    amrdata.flag2refine = True
    amrdata.regrid_interval = 3
    amrdata.regrid_buffer_width = 2
    amrdata.clustering_cutoff = 0.7
    amrdata.verbosity_regrid = 0
    amrdata.max1d = 500

    if TEST == "contact":
        rundata.gaugedata.gauges = [
            [1, -2.0, 0.0, 0.0, FINAL_TIME],
            [2, 2.0, 0.0, 0.0, FINAL_TIME],
        ]
    else:
        rundata.gaugedata.gauges = [
            [1, 8.0, 0.0, 0.0, FINAL_TIME],
            [2, 12.0, 0.0, 0.0, FINAL_TIME],
        ]

    geo = rundata.geo_data
    geo.gravity = 9.81
    geo.coordinate_system = 1
    geo.earth_radius = 6367.5e3
    geo.coriolis_forcing = False
    geo.sea_level = -1.0e6
    geo.dry_tolerance = 1.0e-3
    geo.friction_forcing = False
    geo.manning_coefficient = 0.0
    geo.friction_depth = 1.0e6
    rundata.refinement_data.variable_dt_refinement_ratios = True
    rundata.refinement_data.wave_tolerance = 0.01
    rundata.topo_data.topofiles.append([3, "basal_topo.tt3"])

    qinit = rundata.qinitdclaw_data
    qinit.qinitfiles.append([3, 8, "surface_topo.tt3"])
    qinit.qinitfiles.append([3, 4, "solid_fraction.tt3"])

    dclaw = rundata.dclaw_data
    dclaw.rho_f = 1000.0
    dclaw.rho_s = 2700.0
    dclaw.m_crit = 0.64
    dclaw.m0 = 0.55
    dclaw.mref = 0.60
    dclaw.kref = 5.0e-9
    dclaw.phi = PHI
    dclaw.delta = 0.001
    dclaw.mu = 0.005
    dclaw.alpha_c = 0.01
    dclaw.c1 = 1.0
    dclaw.sigma_0 = 1000.0
    dclaw.src2method = 0
    dclaw.alphamethod = 0
    dclaw.segregation = 0
    dclaw.beta_seg = 0.0
    dclaw.chi0 = 0.5
    dclaw.chie = 0.5
    dclaw.bed_normal = 0
    dclaw.theta_input = 0.0
    dclaw.entrainment = 0
    dclaw.entrainment_rate = 0.0
    dclaw.entrainment_method = 1
    dclaw.me = 0.55
    dclaw.mom_autostop = False
    dclaw.curvature = 0

    pinit = rundata.pinitdclaw_data
    pinit.init_ptype = 3
    pinit.init_pratio = 1.0
    rundata.flowgrades_data.flowgrades = []
    return rundata


if __name__ == "__main__":
    import sys

    setrun(*sys.argv[1:]).write()
