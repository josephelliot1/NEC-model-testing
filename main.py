import numpy as np
import model
from pathlib import Path
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
# Planck 2018
cosmology = dict(ombh2=0.022383, omch2=0.12011, mnu=0.06)
H0_ref = 67.32

ref, thetastar, zstar = model.reference_lcdm(cosmology, H0_ref)

# redshifts to be considered
z = np.geomspace(1e-4, zstar, 600)

# CPL, w0=-0.7, wa=-4(1+w0)  (thin line in paper) 
w0 = -0.7
cpl_res = model.fixed_theta_model(cosmology, model.CPL(w0, -4 * (1 + w0)), thetastar)

#thawing model

w0_th, p = -0.5, 2.0
a = np.geomspace(1e-8, 1.0, 2000)
w = -1 + (1 + w0_th) * a**p
thaw_res = model.fixed_theta_model(cosmology, model.specified_w(a, w), thetastar)

# check: D_M(z*) must match LCDM for every model
for name, r in [("cpl", cpl_res), ("thaw", thaw_res)]:
    rel = r.comoving_radial_distance(zstar) / ref.comoving_radial_distance(zstar) - 1
    print(f"{name}: H0={r.hubble_parameter(0):.2f}, D_M(z*) rel diff={rel:.1e}")
    assert abs(rel) < 1e-4


z_grid = np.geomspace(1e-4, 1e8, 2000)     
a, w, E0 = model.solve_klein_gordon(phi=0.5, z=z_grid, cosmology=cosmology, H0=70)
res = model.fixed_theta_model(cosmology, model.specified_w(a, w), thetastar)
#print("res:", res)
out = dict(z=z, cpl=model.ratios(cpl_res, ref, z), thaw=model.ratios(thaw_res, ref, z), SS = model.ratios(res, ref, z))
np.save(results_dir/"ratios.npy", out, allow_pickle=True)

