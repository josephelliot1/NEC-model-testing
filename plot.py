import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],              
    "mathtext.fontset": "cm",             
    "axes.formatter.use_mathtext": True,  
    "axes.unicode_minus": False,
})

data = np.load("results/ratios.npy", allow_pickle=True).item()
z = data["z"]

cmap = plt.get_cmap("plasma")
norm = Normalize(vmin=0, vmax=1, clip=True)   # colour scale saturates at z = 1


def coloured_line(ax, x, y, z, lw):
    """Line coloured by redshift, drawn as a collection of short segments."""
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=lw, capstyle="round")
    lc.set_array(0.5 * (z[:-1] + z[1:]))      # colour each segment by its mid-z
    ax.add_collection(lc)
    return lc


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
grey = "0.9"

xlim, ylim = (0.971, 1.05), (0.969, 1.063)
x = np.linspace(1.0, xlim[1], 200)
axL.axvspan(xlim[0], 1.0, color=grey, zorder=0)                          # F_AP >= F_AP^L
axL.fill_between(x, ylim[0], x ** (-1 / 3), color=grey, zorder=0)       # D_V bound, z > z_c
axL.plot(x, x ** (-1 / 3), "--", color="gray", lw=1.5, zorder=1)
axL.set_xlim(xlim); axL.set_ylim(ylim)
axL.set_xlabel(r"$F_{\rm AP}/F_{\rm AP}^{\Lambda}$", fontsize = 20)
axL.set_ylabel(r"$D_V/D_V^{\Lambda}$", fontsize = 20)

lim = (0.96, 1.06)
x = np.linspace(1.0, lim[1], 200)
axR.axvspan(lim[0], 1.0, color=grey, zorder=0)                          # D_M >= D_M^L
axR.fill_between(x, x, lim[1], color=grey, zorder=0)                    # D_H/D_H^L <= D_M/D_M^L
axR.plot(x, x, "--", color="gray", lw=1.5, zorder=1)
axR.set_xlim(lim); axR.set_ylim(lim)
axR.set_xlabel(r"$D_M/D_M^{\Lambda}$", fontsize = 20)
axR.set_ylabel(r"$D_H/D_H^{\Lambda}$", fontsize = 20)

for ax in (axL, axR):
    ax.axhline(1, color="k", lw=1.5, zorder=2)
    ax.axvline(1, color="k", lw=1.5, zorder=2)

for key, lw in [("SS", 4.0), ("cpl", 1.5)]:       # thick = thawing, thin = CPL
    r = data[key]
    coloured_line(axL, r["FAP"], r["DV"], z, lw)
    lc = coloured_line(axR, r["DM"], r["DH"], z, lw)

cbar = fig.colorbar(lc, ax=[axL, axR], pad=0.02, fraction=0.04)
cbar.set_label(r"$z$", fontsize = 20)

fig.savefig("figures/fig1.pdf", dpi=200)
