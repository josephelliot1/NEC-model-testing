import numpy as np
import camb
from camb.dark_energy import DarkEnergyFluid, DarkEnergyEqnOfState, DarkEnergyModel, DarkEnergyPPF
from scipy.integrate import solve_ivp
ckms = camb.constants.c/ 1e3 #speed of light in kms-1


def LCDM():
    return lambda pars: pars.set_dark_energy(w=-1.0, wa=0.0) 
def CPL(w0,wa):
    return lambda pars: pars.set_dark_energy(w = w0, wa =wa, dark_energy_model = 'ppf')
def specified_w(a,w):
    def f(pars):
        pars.DarkEnergy = DarkEnergyFluid()
        pars.DarkEnergy.set_w_a_table(np.asarray(a), np.asarray(w))
    return f


#klein gordon stuff to solve for a specified potential:

#tunable params:
V0,M,k,m = 1.0,1.0,0.1,1.0

def V(phi):
    return V0 * (np.exp(-phi / M) - k * np.exp(phi / m))
def dV(phi):
    return V0 * (-np.exp(-phi / M) / M - k * np.exp(phi / m) / m)

def solve_klein_gordon(phi, z, cosmology, H0, V=V, dV=dV):
    h2 = (H0/100)**2 #dimensionless hubble param
    Om = (cosmology["ombh2"] + cosmology["omch2"] + cosmology["mnu"] / 93.14) / h2
    Or = 4.184e-5 / h2 
    N_eval = np.log(1.0 / (1.0 + np.unique(np.append(z, 0.0))))   #coordinate grid to evaluate on
    N_eval = np.sort(N_eval)
    def E2_of(N, phi, phiN): #calculate friedmann equation needed
        a = np.exp(N)
        rho_m, rho_r = Om * a**-3, Or * a**-4
        E2 = (rho_m + rho_r + V(phi) / h2 / 3.0) / (1.0 - phiN**2 / 6.0)
        return E2, rho_m, rho_r
    def rhs(N, y): #core part of KG equation
        phi, phiN = y
        E2, rho_m, rho_r = E2_of(N, phi, phiN)
        dlnE = -0.5 * ((3 * rho_m + 4 * rho_r) / E2 + phiN**2)
        phiNN = -(3.0 + dlnE) * phiN - dV(phi) / h2 / E2
        return [phiN, phiNN]    

    sol = solve_ivp(rhs, (N_eval[0], 0.0), [phi, 0.0], t_eval=N_eval, #solver
                    method="DOP853", rtol=1e-10, atol=1e-12)

    if not sol.success:
        raise RuntimeError(sol.message)
    #compute eq of state from solution for potential
    phi, phiN = sol.y
    E2, _, _ = E2_of(sol.t, phi, phiN)
    kin = 0.5 * E2 * phiN**2          # phi_dot^2 / 2 in H0^2 Mpl^2 units
    pot = V(phi) / h2
    w = (kin - pot) / (kin + pot)
    return np.exp(sol.t), w, np.sqrt(E2[-1])






def run_background(cosmology, dark_energy, H0 = None, thetastar = None):
    pars = camb.CAMBparams()
    dark_energy(pars)
    pars.set_cosmology(H0=H0, thetastar=thetastar, omk = 0,**cosmology)
    return camb.get_background(pars)

def reference_lcdm(cosmology, H0):
    res = run_background(cosmology, LCDM(), H0=H0)
    d = res.get_derived_params()
    return res, d["thetastar"] / 100.0, d["zstar"]

def fixed_theta_model(cosmology, dark_energy, thetastar):
    return run_background(cosmology, dark_energy, thetastar=thetastar)
def distances(res, z):
    DM = res.comoving_radial_distance(z)      # D_M = comoving distance Mpc
    DH = ckms / res.hubble_parameter(z)      
    DV = (z * DM**2 * DH) ** (1 / 3)
    FAP = DM / DH
    return dict(DM=DM, DH=DH, DV=DV, FAP=FAP)


def ratios(res, ref, z):
    d, dl = distances(res, z), distances(ref, z)
    return {k: d[k] / dl[k] for k in d}
from scipy.interpolate import CubicSpline

def ratios_from_arrays(z_q, H_q, DM_q, ref, z):
    """Same output format as ratios(), for a model given as arrays on its own z grid."""
    DM = CubicSpline(z_q, DM_q)(z)
    DH = C_KMS / CubicSpline(z_q, H_q)(z)
    d = dict(DM=DM, DH=DH, DV=(z * DM**2 * DH) ** (1 / 3), FAP=DM / DH)
    dl = distances(ref, z)
    return {k: d[k] / dl[k] for k in d}