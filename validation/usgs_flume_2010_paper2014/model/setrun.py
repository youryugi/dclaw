"""D-Claw runtime configuration for the 2010 SGM rough-bed reproduction."""

from __future__ import annotations

from model_config import (
    DX,
    DY,
    FINAL_TIME,
    PARAMETERS,
    X_LOWER,
    X_UPPER,
    Y_LOWER,
    Y_UPPER,
)


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
    clawdata.num_output_times = int(FINAL_TIME / 0.25)
    clawdata.tfinal = FINAL_TIME
    clawdata.output_t0 = True
    clawdata.output_format = "ascii"
    clawdata.output_q_components = "all"
    clawdata.output_aux_components = "none"
    clawdata.output_aux_onlyonce = True
    clawdata.verbosity = 0
    clawdata.dt_variable = True
    clawdata.dt_initial = 1.0e-5
    clawdata.dt_max = 0.05
    clawdata.cfl_desired = 0.4
    clawdata.cfl_max = 0.5
    clawdata.steps_max = 200000
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
    amrdata.amr_levels_max = PARAMETERS["amr_levels_max"]
    n_ratios = PARAMETERS["amr_levels_max"] - 1
    amrdata.refinement_ratios_x = [PARAMETERS["refinement_ratio"]] * n_ratios
    amrdata.refinement_ratios_y = [PARAMETERS["refinement_ratio"]] * n_ratios
    amrdata.refinement_ratios_t = [PARAMETERS["refinement_ratio"]] * n_ratios
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

    # Gauges match the three cross sections tabulated in the 2010 aggregate
    # dataset (x = 32, 66, 90 m); gauge 1 sits at x=2 m to match figure 13a
    # of the paper, the closest published validation point to the gate.
    # Gauges beyond a shortened diagnostic domain are dropped so cheap
    # near-field sensitivity runs (small X_UPPER) do not error out.
    all_gauges = [
        [1, 2.0, 0.0, 0.0, FINAL_TIME],
        [2, 32.0, 0.0, 0.0, FINAL_TIME],
        [3, 66.0, 0.0, 0.0, FINAL_TIME],
        [4, 90.0, 0.0, 0.0, FINAL_TIME],
    ]
    rundata.gaugedata.gauges = [g for g in all_gauges if X_LOWER < g[1] < X_UPPER]

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
    dclaw.rho_f = PARAMETERS["rho_f_kg_m3"]
    dclaw.rho_s = PARAMETERS["rho_s_kg_m3"]
    dclaw.m_crit = PARAMETERS["critical_solid_fraction"]
    dclaw.m0 = PARAMETERS["solid_volume_fraction"]
    dclaw.mref = PARAMETERS["reference_solid_volume_fraction"]
    dclaw.kref = PARAMETERS["reference_permeability_m2"]
    dclaw.phi = PARAMETERS["basal_friction_angle_deg"]
    dclaw.delta = PARAMETERS["characteristic_grain_size_m"]
    dclaw.mu = PARAMETERS["pore_fluid_viscosity_pa_s"]
    dclaw.alpha_c = PARAMETERS["compressibility_constant"]
    dclaw.c1 = 1.0
    dclaw.sigma_0 = PARAMETERS["reference_stress_pa"]
    dclaw.src2method = PARAMETERS["source_method"]
    dclaw.alphamethod = PARAMETERS["compressibility_method"]
    dclaw.segregation = 0
    dclaw.beta_seg = 0.0
    dclaw.chi0 = 0.5
    dclaw.chie = 0.5
    dclaw.bed_normal = 0
    dclaw.theta_input = 0.0
    dclaw.entrainment = 0
    dclaw.entrainment_rate = 0.0
    dclaw.entrainment_method = 1
    dclaw.me = PARAMETERS["solid_volume_fraction"]
    dclaw.mom_autostop = False
    dclaw.curvature = 0

    pinit = rundata.pinitdclaw_data
    pinit.init_ptype = 3
    pinit.init_pratio = PARAMETERS["initial_pore_pressure_ratio"]
    if PARAMETERS["use_finite_gate"]:
        rundata.dtopo_data.dtopofiles.append([3, "gate_dtopo.tt3"])
        rundata.dtopo_data.dt_max_dtopo = 0.01

    # D-Claw ignores wave_tolerance for refinement (see flag2refine2.f90):
    # refinement must be driven explicitly by flowgrades. Refine to the
    # finest level wherever the flow is wet. (keep_fine is left False: it
    # only maintains an already-fine grid and, per flag2refine2.f90's gate
    # `(.not. keep_fine) .or. (level >= fine_level)`, disables the ordinary
    # flowgrades check below the finest level -- exactly the coarse-to-fine
    # refinement we need here.)
    flowgrades = rundata.flowgrades_data
    flowgrades.flowgrades = [[geo.dry_tolerance, 1, 1, amrdata.amr_levels_max]]
    return rundata


if __name__ == "__main__":
    import sys

    setrun(*sys.argv[1:]).write()
