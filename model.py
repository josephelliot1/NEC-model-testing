import numpy as np
import camb
from camb.dark_energy import DarkEnergyFluid, DarkEnergyEqnOfState, DarkEnergyModel, DarkEnergyPPF

ckms = camb.constants.c/ 1e3 #speed of light in kms-1

#params for different DE models: LCDM, CPL and specified (w0,wa) 

def LCDM():
    return lambda pars: pars.set_dark_energy(w=-1.0, wa=0.0) 
def CPL():
    return lambda pars: pars.set_dark_energy(w = w0, wa =wa, dark_energy_model = 'ppf')
def specified_w():
    def f(pars):
        pars.DarkEnergy = DarkEnergyFluid()
        pars.DarkEnergy.set_w_a_table(np.asarray(a), np.asarray(w))
    return f

def run_background(cosmology, dark_energy, H0 = None, thetastar = None):
    pars = camb.CAMBparams()
    dark_energy(pars)
    pars.set_cosmology(H0=H0, thetastar=thetastar, omk = 0 **cosmology)
    return camb.get_background(pars)
def reference_lcdm(cosmo, H0):
    res = run_background(cosmo, lcdm(), H0=H0)
    d = res.get_derived_params()
    return res, d["thetastar"] / 100.0, d["zstar"]

def fixed_theta_model(cosmo, dark_energy, thetastar):
    return run_background(cosmo, dark_energy, thetastar=thetastar)
def distances(res, z):
    DM = res.comoving_radial_distance(z)      # D_M = comoving distance Mpc
    DH = C_KMS / res.hubble_parameter(z)      
    DV = (z * DM**2 * DH) ** (1 / 3)
    FAP = DM / DH
    return dict(DM=DM, DH=DH, DV=DV, FAP=FAP)


def ratios(res, ref, z):
    d, dl = distances(res, z), distances(ref, z)
    return {k: d[k] / dl[k] for k in d}