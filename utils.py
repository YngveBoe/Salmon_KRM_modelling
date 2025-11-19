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

def plot_ts_with_linear_regression(df, plotlabel):
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
    mean_sigma_data = np.mean(10**(ts_data_numpy/10),axis=1)
    mean_ts_data = 10*np.log10(mean_sigma_data)
    model = sm.OLS(mean_ts_data, sm.add_constant(log_l))
    results = model.fit()
    #print(results.summary())
    ax.plot(labels, results.predict(sm.add_constant(log_l)), 'r-', label='Ungrouped')
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
