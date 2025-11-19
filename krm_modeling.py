from echosms import KRMModel, KRMdata, KRMorganism, KRMshape
try:
    # boundary enum is exposed in recent echoSMs
    from echosms import boundary_type
except Exception:
    boundary_type = None  # we'll fall back if missing

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import statsmodels.api as sm

def _to_boundary(b):
    """Accepts 'fluid-filled' / 'pressure-release' strings or enum; returns enum if available."""
    if boundary_type is None:
        return b  # let KRMshape accept the string (works in many builds)
    if not isinstance(b, str):
        return b
    # normalize common spellings
    key = b.replace("-", "_").lower()
    # try name lookup then value lookup
    if hasattr(boundary_type, key):
        return getattr(boundary_type, key)
    # value-based construction (Enum("fluid-filled") etc.)
    try:
        return boundary_type(b)
    except Exception:
        # last resort: default to fluid-filled
        return getattr(boundary_type, "fluid_filled")

def _to_shape(d: dict) -> KRMshape:
    return KRMshape(
        boundary=_to_boundary(d.get("boundary", "fluid-filled")),
        x=np.asarray(d["x"], dtype=float),
        w=np.asarray(d["w"], dtype=float),
        z_U=np.asarray(d["z_U"], dtype=float),
        z_L=np.asarray(d["z_L"], dtype=float),
        c=float(d.get("c", 1570.0)),
        rho=float(d.get("rho", 1070.0)),
    )

def load_krmorganism_from_json(path: str) -> KRMorganism:
    """
    Load a KRMorganism saved by the extraction script.
    Returns: KRMorganism
    """
    with open(path, "r") as f:
        j = json.load(f)

    body = _to_shape(j["body"])
    inclusions = [_to_shape(s) for s in j.get("inclusions", [])]

    return KRMorganism(
        name=j.get("name", "Salmon"),
        source=j.get("source", ""),
        body=body,
        inclusions=inclusions,
        aphiaid=j.get("aphiaid", None),
        length=float(j.get("length", 1.0)),
        vernacular_name=j.get("vernacular_name", "Atlantic salmon"),
    )

def plot_ts_with_linear_regression(df, ts_filter, plotlabel):
    cat_col   = "length"
    value_col = "ts"

    groups = df.groupby(cat_col)[value_col]
    labels = 100*np.unique(df[cat_col])
    ts_data   = [g.dropna().values for _, g in groups]
    
    
    import matplotlib.ticker as mticker
    fig, ax = plt.subplots()
    ax.violinplot(ts_data, positions=labels, showmeans=True, showmedians=True)
    
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.set_xlabel('Length [cm]'); ax.set_ylabel(value_col)
    ax.set_title(f"{value_col} by {cat_col} for {plotlabel}")
    plt.tight_layout(); plt.pause(1)

    log_l = np.log10(labels)
    ts_data_numpy = np.stack(ts_data)
    log_l = np.repeat(log_l, ts_data_numpy.shape[1]).reshape(ts_data_numpy.shape)
    ts_filter_mask = ts_data_numpy>ts_filter
    ts_data_numpy = ts_data_numpy[ts_filter_mask]
    log_l = log_l[ts_filter_mask]
    unique_log_l, inv, counts = np.unique(log_l, return_inverse=True, return_counts=True)

    # Sum TS by group
    sum_sigma_data = np.sum(np.bincount(inv, weights=10**(ts_data_numpy/10)))
    mean_sigma_data = sum_sigma_data / counts
    
    mean_ts_data = 10*np.log10(mean_sigma_data)

    model = sm.OLS(mean_ts_data, sm.add_constant(unique_log_l))
    results = model.fit()
    #print(results.summary())
    ax.plot(labels, results.predict(sm.add_constant(unique_log_l)), 'r-', label='Ungrouped')
    groups = df.groupby("length_group")['sigma']
    ts_data_grouped = []
    length_group = df['length_group'].values
    unique_length_group = np.unique(length_group)
    for g in unique_length_group:
        mask = length_group == g
        ts_data_grouped.append(df['ts'].values[mask])
    
    ax.violinplot(ts_data_grouped, positions=unique_length_group, showmeans=True, showmedians=True, widths=5)

    mean_sigma_data_grouped = np.array([g.dropna().mean() for _, g in groups])
    l = np.unique(df['length_group'])
    log_l = np.log10(l)
    
    mean_ts_data_grouped = 10*np.log10(mean_sigma_data_grouped)
    model_grouped = sm.OLS(mean_ts_data_grouped, sm.add_constant(log_l))
    results_grouped = model_grouped.fit()
    ax.plot(l, results_grouped.predict(sm.add_constant(log_l)), 'm-',label='Grouped')


    ax.set_xscale('log')
    ax.set_xticks([30,40,50,60])
    ax.set_xticklabels([30,40,50,60], rotation=20, ha="right")
    ax.legend()
    plt.pause(1)
    print(results_grouped.summary())
    
    return fig, ax, df_complete


def scale_krm_organism(org, scale: float=None, diameter_scale: float=None, target_length: float=None, add_noise: bool=False, noise_scale: float=0.01):
    """
    Scale geometry of a KRMorganism isotropically.
    Provide either `scale` (e.g. 1.25) or `target_length` in meters.
    """
    if (scale is None) == (target_length is None):
        raise ValueError("Pass exactly one of `scale` or `target_length`.")
    s = scale if scale is not None else (target_length / float(org.length))

    out = copy.deepcopy(org)

    # helper to multiply arrays if present
    def _mul_inplace(shape, scale_x = True, scale_w = True, scale_zL = True, scale_zU = True):
        # Find Center line between z_U and z_L
        z_center_line = (shape.z_U + shape.z_L) / 2
        for name in ("x", "w", "z_U", "z_L"):
            if add_noise:
                noise = np.random.normal(1, noise_scale)
            else:
                noise = 1.
            

            if hasattr(shape, name):
                arr = getattr(shape, name)
                if arr is not None and shape.boundary.name != 'pressure_release':
                    setattr(shape, name, np.asarray(arr, float) * s * noise)
                elif arr is not None and shape.boundary.name == 'pressure_release' and name == 'x' and scale_x:
                    setattr(shape, name, np.asarray(arr, float) * s * noise)
                elif arr is not None and shape.boundary.name == 'pressure_release' and name == 'w' and scale_w:
                    if diameter_scale > 1:
                        do_nothing = []
                    setattr(shape, name, np.asarray(arr, float) * s * noise * diameter_scale) # * diameter_scale Need to fix this so it scales against the center line of the swimbladder
                elif arr is not None and shape.boundary.name == 'pressure_release' and name == 'z_L' and scale_zL:
                    _diameter_scale = 1
                    zL = np.asarray(arr, float)
                    new_zL = (zL + (zL - z_center_line) * _diameter_scale)  * noise
                    setattr(shape, name, new_zL)
                elif arr is not None and shape.boundary.name == 'pressure_release' and name == 'z_U' and scale_zU:
                    _diameter_scale = 1
                    zU = np.asarray(arr, float)
                    new_zU = (zU + (zU - z_center_line) * _diameter_scale) * noise
                    setattr(shape, name, new_zU)



    
        # keep c, rho, boundary as-is

    # body
    _mul_inplace(out.body)

    # inclusions (e.g., swimbladder)
    if getattr(out, "inclusions", None):
        for inc in out.inclusions:
            _mul_inplace(inc, scale_x = True, scale_w = True, scale_zL = False, scale_zU = False)

    # length metadata
    if hasattr(out, "length") and out.length is not None:
        out.length = float(out.length) * s

    return out

fish_data = pd.read_excel(r'C:\\Users\\Yngve\\OneDrive - University of Bergen\\Yngve\Artikler\\Paper 1 - Log-length-paper\\data\\Fish Size Data.xlsx')
lengths = fish_data['Length (mm)'].values/1000
lengths, idx = np.unique(lengths, return_index=True)
length_groups = fish_data['Length_group'].values/10
length_groups = length_groups[idx]
mask = length_groups < 55
lengths = lengths[mask]
length_groups = length_groups[mask]
# Extend lengths by factor of n to simulate varying fill levels
n = 1
lengths = np.repeat(lengths, n)
length_groups = np.repeat(length_groups, n)

frequencies = np.array([38000, 70000, 120000, 200000])
n_ts_per_size = 500
mod = KRMModel()


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
    

for length_group, length, scale in zip(length_groups, lengths, scales):
    print(f'Current length: {length}')
    diameter_scale = np.random.normal(0.75, 0.15)
    diameter_scale = (length/lengths.min())**1.5
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

        
    medium_c = 1490
    medium_rho = 1022
    theta = np.clip(np.random.normal(90,10,n_ts_per_size), 65, 115)
    
    p = {'medium_c': medium_c, 'medium_rho': medium_rho, 'organism': fish_scaled, 'theta': theta,
     'f': frequencies}
    df = mod.calculate_ts(p, expand=True)
    df['length_group'] = length_group
    df['length'] = length
    df['log_length'] = np.log10(length*100)
    df['sigma'] = 10**(df['ts']/10)
    dfs.append(df)
    del df

df_complete = pd.concat(dfs, ignore_index=True)
print(df_complete.head())



df_38 = df_complete[df_complete['f'] == 38000.]
df_70 = df_complete[df_complete['f'] == 70000.]
df_120 = df_complete[df_complete['f'] == 120000.]
df_200 = df_complete[df_complete['f'] == 200000.]

plot_ts_with_linear_regression(df_38, -60, '38 kHz')
plot_ts_with_linear_regression(df_70, -60, '70 kHz')
plot_ts_with_linear_regression(df_120, -60, '120 kHz')
plot_ts_with_linear_regression(df_200, -60, '200 kHz')

print('')
