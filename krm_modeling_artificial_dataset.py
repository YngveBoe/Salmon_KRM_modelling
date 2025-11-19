from echosms import KRMModel, KRMdata, KRMorganism, KRMshape
try:
    # boundary enum is exposed in recent echoSMs
    from echosms import boundary_type
except Exception:
    boundary_type = None  # we'll fall back if missing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xarray as xr
from utils import load_krmorganism_from_json, scale_krm_organism, plot_ts_with_linear_regression, _to_boundary, _to_shape

frequencies = np.linspace(34000.,380000.,int((380000-34000)/100+1))

mod = KRMModel()
lengths=np.linspace(0.2,0.8,61)
medium_c = np.linspace(1480, 1500, 3)
medium_rho = np.linspace(1020, 1030, 6)
theta = np.linspace(65,115,51)

log_lengths = np.log10(lengths)
#fish = KRMdata().model('Sardine')
fish = load_krmorganism_from_json('salmon_krmorganism_ventral.json')

ref_length = fish.length
scales = lengths/ref_length
dfs = []

fig,axs = plt.subplots(2,1)
ax1, ax2 = axs.flatten()
def plot_fish(fish, ax1, ax2m, label, color):
    ax1.axis('equal')
    ax2.axis('equal')
    line_body_z_U = ax1.plot(fish.body.x, fish.body.z_U, color, label=label)
    line_body_z_L = ax1.plot(fish.body.x, fish.body.z_L, color, label='__nolabel__')
    line_inclusion_z_U = ax1.plot(fish.inclusions[0].x, fish.inclusions[0].z_U, color)
    line_inclusion_z_L = ax1.plot(fish.inclusions[0].x, fish.inclusions[0].z_L, color)
    ax1.legend()
    line_body_w = ax2.plot(fish.body.x, fish.body.w, color, label=label)
    line_body_w_neg = ax2.plot(fish.body.x, -fish.body.w, color)
    line_inclusion_w = ax2.plot(fish.inclusions[0].x, fish.inclusions[0].w, color)
    line_inclusion_w_neg = ax2.plot(fish.inclusions[0].x, -fish.inclusions[0].w, color)
    ax2.legend()
    ax1.set_xlim([-0.4,0.40])
    ax2.set_xlim([-0.4,0.4])
    ax1.grid()
    ax2.grid()
    plt.pause(0.01)
    return line_body_z_U, line_body_z_L, line_inclusion_z_U, line_inclusion_z_L, line_body_w, line_body_w_neg, line_inclusion_w, line_inclusion_w_neg
    

datasets = []

for length, scale in zip(lengths, scales):
    print(f'Current length: {length}')
    #diameter_scale = 1.
    diameter_scale = (length/lengths.min())**1.0
    print(f'diameter_scale: {diameter_scale}')
    fish_scaled = scale_krm_organism(fish, scale=scale, diameter_scale=diameter_scale, noise_scale=0.1)
    
    if length == lengths.min():
        orig_fish =plot_fish(fish, ax1, ax2, label='Original fish', color='blue')
        current_fish = plot_fish(fish_scaled, ax1, ax2, label=f'Current scaled fish', color='red')
    else:
        current_fish[0][0].set_xdata(fish_scaled.body.x)
        current_fish[0][0].set_ydata(fish_scaled.body.z_U)
        current_fish[1][0].set_xdata(fish_scaled.body.x)
        current_fish[1][0].set_ydata(fish_scaled.body.z_L)
        current_fish[2][0].set_xdata(fish_scaled.inclusions[0].x)
        current_fish[2][0].set_ydata(fish_scaled.inclusions[0].z_U)
        current_fish[3][0].set_xdata(fish_scaled.inclusions[0].x)
        current_fish[3][0].set_ydata(fish_scaled.inclusions[0].z_L)
        current_fish[4][0].set_xdata(fish_scaled.body.x)
        current_fish[4][0].set_ydata(fish_scaled.body.w)
        current_fish[5][0].set_xdata(fish_scaled.body.x)
        current_fish[5][0].set_ydata(-fish_scaled.body.w)
        current_fish[6][0].set_xdata(fish_scaled.inclusions[0].x)
        current_fish[6][0].set_ydata(fish_scaled.inclusions[0].w)
        current_fish[7][0].set_xdata(fish_scaled.inclusions[0].x)
        current_fish[7][0].set_ydata(-fish_scaled.inclusions[0].w)
        fig.canvas.draw()
        fig.canvas.flush_events()

    p = {'medium_c': medium_c, 'medium_rho': medium_rho, 'organism': fish_scaled, 'theta': theta,
            'f': frequencies}
    df = mod.calculate_ts(p, expand=True)
    df = df.drop(columns=["organism"])
    df.rename(columns={"f": "frequency"}, inplace=True)
    df['length'] = length
    df['sigma'] = 10**(df['ts']/10)

    df.set_index(['frequency', 'medium_c', 'medium_rho','theta','length'], inplace=True)
    ds = df.to_xarray()
    datasets.append(ds)
    del df

ds = xr.concat(datasets, dim='length')
stack_dims = ['medium_c', 'medium_rho', 'theta','length']
ds = ds.stack(i=stack_dims)
ds.to_netcdf('artificial_dataset.nc')
print('done')
 
    
