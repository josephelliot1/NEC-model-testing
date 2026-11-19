import numpy as np
import model

# Planck 2018
cosmo = dict(ombh2=0.022383, omch2=0.12011, mnu=0.06)
H0_ref = 67.32

ref, thetastar, zstar = model.reference_lcdm(cosmo, H0_ref)

# redshifts to be considered
z = np.geomspace(1e-4, zstar, 600)

# CPL, w0=-0.7, wa=-4(1+w0)  (thin line in paper) 
w0 = -0.7
cpl_res = model.fixed_theta_model(cosmo, model.cpl(w0, -4 * (1 + w0)), thetastar)

#thawing model
# # The paper doesn't say which thawing model. Toy form: w = -1 + (1+w0) a^p.
w0_th, p = -0.5, 3.0
a = np.geomspace(1e-8, 1.0, 2000)
w = -1 + (1 + w0_th) * a**p
thaw_res = model.fixed_theta_model(cosmo, model.tabulated(a, w), thetastar)

# check: D_M(z*) must match LCDM for every model
for name, r in [("cpl", cpl_res), ("thaw", thaw_res)]:
    rel = r.comoving_radial_distance(zstar) / ref.comoving_radial_distance(zstar) - 1
    print(f"{name}: H0={r.hubble_parameter(0):.2f}, D_M(z*) rel diff={rel:.1e}")
    assert abs(rel) < 1e-4

out = dict(z=z, cpl=model.ratios(cpl_res, ref, z), thaw=model.ratios(thaw_res, ref, z))
np.save("ratios.npy", out, allow_pickle=True)

