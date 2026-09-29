"""D-Claw (Fortran) run for the real-terrain cross-check against dclaw_jax.

Same inputs as ../run_jax.py: build_inputs.py's terrain and ponds, single
20 m level (no AMR), Barnhart et al. 2021 Montecito ensemble-median
parameters, hydrostatic initial pore pressure, Manning n=0.06, a frame
every 20 s. ORDER (1 or 2) selects the Fortran scheme's order.
"""

import os

import numpy as np

INPUTS = np.load("inputs_20m.npz")
X_LOWER, Y_LOWER, DX = float(INPUTS["x_lower"]), float(INPUTS["y_lower"]), float(INPUTS["dx"])
NY, NX = INPUTS["h0"].shape
T_FINAL = 1800.0
FRAME_DT = 20.0
ORDER = int(os.environ.get("ORDER", "2"))

PARAMS = dict(phi=37.63, kref=3.348e-12, m0=0.512, m_crit=0.64, mref=0.60, delta=0.01, mu=0.005,
              alpha_c=0.01, sigma_0=1000.0, rho_f=1000.0, rho_s=2700.0, manning=0.06)


def setrun(claw_pkg="dclaw"):
    from clawpack.clawutil import data

    rundata = data.ClawRunData(claw_pkg, 2)
    c = rundata.clawdata
    c.lower = [X_LOWER, Y_LOWER]
    c.upper = [X_LOWER + NX * DX, Y_LOWER + NY * DX]
    c.num_cells = [NX, NY]
    c.num_eqn = 7
    c.num_aux = 10
    c.capa_index = 0
    c.t0 = 0.0
    c.restart = False
    c.output_style = 1
    c.num_output_times = int(T_FINAL / FRAME_DT)
    c.tfinal = T_FINAL
    c.output_t0 = True
    c.output_format = "ascii"
    c.output_q_components = "all"
    c.output_aux_components = "all"
    c.output_aux_onlyonce = True
    c.verbosity = 0
    c.dt_variable = True
    c.dt_initial = 0.01
    c.dt_max = 5.0
    c.cfl_desired = 0.8 if ORDER == 2 else 0.4
    c.cfl_max = 1.0 if ORDER == 2 else 0.5
    c.steps_max = 500000
    c.order = ORDER
    c.dimensional_split = "unsplit"
    c.transverse_waves = 2 if ORDER == 2 else 0
    c.num_waves = 5
    c.limiter = [4] * 5
    c.use_fwaves = True
    c.source_split = "godunov"
    c.num_ghost = 2
    c.bc_lower = ["extrap", "extrap"]
    c.bc_upper = ["extrap", "extrap"]
    c.checkpt_style = 0

    a = rundata.amrdata
    a.amr_levels_max = 1
    a.refinement_ratios_x = [2]
    a.refinement_ratios_y = [2]
    a.refinement_ratios_t = [2]
    a.aux_type = ["center", "center", "yleft", "center", "center",
                  "center", "center", "center", "center", "center"]
    a.flag_richardson = False
    a.flag2refine = False
    a.regrid_interval = 1000000
    a.verbosity_regrid = 0
    a.max1d = 500

    geo = rundata.geo_data
    geo.gravity = 9.81
    geo.coordinate_system = 1
    geo.coriolis_forcing = False
    geo.sea_level = -1.0e6
    geo.dry_tolerance = 1.0e-3
    geo.friction_forcing = True
    geo.manning_coefficient = PARAMS["manning"]
    geo.friction_depth = 1.0e6
    rundata.refinement_data.variable_dt_refinement_ratios = True
    rundata.topo_data.topofiles.append([3, "topo_5m.asc"])

    q = rundata.qinitdclaw_data
    q.qinitfiles.append([3, 1, "h0_20m.asc"])
    q.qinitfiles.append([3, 4, "m0_20m.asc"])

    d = rundata.dclaw_data
    d.rho_f, d.rho_s = PARAMS["rho_f"], PARAMS["rho_s"]
    d.m_crit, d.m0, d.mref = PARAMS["m_crit"], PARAMS["m0"], PARAMS["mref"]
    d.kref, d.phi, d.delta = PARAMS["kref"], PARAMS["phi"], PARAMS["delta"]
    d.mu, d.alpha_c, d.c1, d.sigma_0 = PARAMS["mu"], PARAMS["alpha_c"], 1.0, PARAMS["sigma_0"]
    d.src2method = 0
    d.alphamethod = 0
    d.segregation = 0
    d.bed_normal = 0
    d.theta_input = 0.0
    d.entrainment = 0
    d.entrainment_rate = 0.0
    d.me = PARAMS["m0"]
    d.mom_autostop = False
    d.curvature = 0
    d.dd_manning = False

    rundata.pinitdclaw_data.init_ptype = 0
    return rundata


if __name__ == "__main__":
    setrun().write()
