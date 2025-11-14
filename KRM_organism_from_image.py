# pip install numpy opencv-python pillow matplotlib
import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


# -------- settings --------
IMG_PATH = "Salmon.png"        # change if needed
OUT_JSON_DORSAL = "salmon_krmorganism_dorsal.json"
OUT_JSON_VENTRAL = "salmon_krmorganism_ventral.json"
N_SAMPLES = 400                # x-samples along the body (uniform)

# -------- helpers --------
def preprocess(img_gray):
    blur = cv2.GaussianBlur(img_gray, (5,5), 1.0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    return edges

def largest_n_contours(edge_img, n):
    cnts, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return []
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    return cnts[:n]

def ensure_two(edge_img, cnts):
    if len(cnts) >= 2:
        return cnts
    cnts2, _ = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not cnts2:
        return cnts
    cnts2 = sorted(cnts2, key=cv2.contourArea, reverse=True)
    return cnts2[:2]

def contour_xy(contour):
    pts = contour.squeeze(1)  # (N,2)
    return pts[:,0].astype(float), pts[:,1].astype(float)

def sample_upper_lower_from_contour(x, y, xgrid):
    """For a side/top outline, get upper/lower y(x) envelope."""
    upper = np.full_like(xgrid, np.nan, dtype=float)
    lower = np.full_like(xgrid, np.nan, dtype=float)
    dx = np.diff(xgrid).mean()
    half = dx/2
    for i, xc in enumerate(xgrid):
        mask = (x >= xc-half) & (x < xc+half)
        if not np.any(mask):
            mask = (x >= xc-1.5*dx) & (x < xc+1.5*dx)
        if np.any(mask):
            yseg = y[mask]
            upper[i] = np.min(yseg)  # in image coords: top = smaller y
            lower[i] = np.max(yseg)  # bottom = larger y
    for arr in (upper, lower):  # fill small gaps
        good = ~np.isnan(arr)
        if good.sum() >= 2:
            arr[np.isnan(arr)] = np.interp(np.flatnonzero(np.isnan(arr)),
                                           np.flatnonzero(good), arr[good])
        else:
            arr[:] = np.nanmedian(arr)
    return upper, lower

def sample_halfwidth_from_contour(x, y, xgrid):
    """From dorsal outline: half-width w(x) via envelope and centerline."""
    upper, lower = sample_upper_lower_from_contour(x, y, xgrid)
    center = 0.5*(upper + lower)
    w = np.maximum(lower - center, center - upper)
    return w, center

def scale_x_y(x, y, x_min, x_max, ymin, scale_x_min=-0.15, scale_x_max=0.20, shift_y=True):
    """scale x to [-0.15,0.20]; shift/scale y so units match x scaling."""
    px_per_m = (x_max-x_min) / (scale_x_max - scale_x_min)
    x_scaled = x/px_per_m + scale_x_min
    if shift_y:
        y = ymin - y
    return x_scaled, y / px_per_m 

def smooth(y, win=9):
    if y is None or win < 3:
        return y
    k = np.ones(win)/win
    return np.convolve(y, k, mode="same")

def resample(y, oldx, newx):
    return np.interp(newx, oldx, y)

# -------- load & split panels --------
img = Image.open(IMG_PATH)
img_np = np.array(img)
if img_np.ndim == 3:
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
else:
    gray = img_np

H, W = gray.shape
mid = W // 2
left = gray[50:250, 40:485]    # lateral
right = gray[50:250, 530:980]   # dorsal

# -------- edges & contours --------
edges_left = preprocess(left)
edges_right = preprocess(right)

cnts_left = ensure_two(edges_left, largest_n_contours(edges_left, n=3))
cnts_left = sorted(cnts_left, key=cv2.contourArea, reverse=True)
cnts_right = ensure_two(edges_right, largest_n_contours(edges_right, n=2))
cnts_right = sorted(cnts_right, key=cv2.contourArea, reverse=True)

# -------- LATERAL (z vs x) --------
if not cnts_left:
    raise RuntimeError("No lateral contour found.")
xL0, zL0 = contour_xy(cnts_left[0])               # swim_bladder
xL1, zL1 = contour_xy(cnts_left[1]) # body upper
xL2, zL2 = contour_xy(cnts_left[2])          # body lower
y0 = 165-50

x_min_L, x_max_L = xL1.min(), xL1.max()
xgrid_L = np.linspace(x_min_L, x_max_L, N_SAMPLES)
x_min_i, x_max_i = xL0.min(), xL0.max()
scale = (x_max_i - x_min_i) / (x_max_L - x_min_L)
xgrid_i = np.linspace(x_min_i, x_max_i, int(N_SAMPLES*scale))

zU_U_img, zU_L_img = sample_upper_lower_from_contour(xL1, zL1, xgrid_L)
z_center_U = 0.5*(zU_U_img + zU_L_img)                # centerline
zL_U_img, zL_L_img = sample_upper_lower_from_contour(xL2, zL2, xgrid_L)
z_center_L = 0.5*(zL_U_img + zL_L_img)
xL, zU = scale_x_y(xgrid_L, z_center_U, x_min_L, x_max_L, y0)
_,  zL = scale_x_y(xgrid_L, z_center_L, x_min_L, x_max_L, y0)
zU_i_img, zL_i_img = sample_upper_lower_from_contour(xL0, zL0, xgrid_i)
xL_i, zU_i = scale_x_y(xgrid_i, zU_i_img, x_min_L, x_max_L, y0)
_,  zL_i = scale_x_y(xgrid_i, zL_i_img, x_min_L, x_max_L, y0)


# -------- DORSAL (w vs x) --------
if not cnts_right:
    raise RuntimeError("No dorsal contour found.")

xR0, yR0 = contour_xy(cnts_right[0]) 

has_dor_incl = len(cnts_right) >= 2
if has_dor_incl:
    xR1, yR1 = contour_xy(cnts_right[1])          # swimbladder

x_min_R, x_max_R = xR0.min(), xR0.max()
xgrid_R = np.linspace(x_min_R, x_max_R, N_SAMPLES)
x_min_i, x_max_i = xR1.min(), xR1.max()
scale = (x_max_i - x_min_i) / (x_max_R - x_min_R)
xgridR_i = np.linspace(x_min_i, x_max_i, int(N_SAMPLES*scale))
w_body, _ = sample_halfwidth_from_contour(xR0, yR0, xgrid_R)
print('w')
xR, w = scale_x_y(xgrid_R, w_body, x_min_R, x_max_R, y0, shift_y=False)

if has_dor_incl:
    w_incl, _ = sample_halfwidth_from_contour(xR1, yR1, xgridR_i)
    xR_i, w_i = scale_x_y(xgridR_i, w_incl, x_min_R, x_max_R, y0, shift_y=False)
else:
    w_i = None

# Ensure w and w_i are always positive
w[w <= 0] = 1e-6
w_i[w_i <= 0] = 1e-6
assert np.all(w > 0)
assert np.all(w_i > 0)
#Check that zL < zU and zL_i < zU_i
for i in range(len(zL)):
    if zL[i] >= zU[i]:
        zL[i] = zU[i] - 1e-3
    assert zL[i] < zU[i]

for i in range(len(zL_i)):
    if zL_i[i] >= zU_i[i]:
        zL_i[i] = zU_i[i] - 1e-3
    assert zL_i[i] < zU_i[i]


## -------- align on a common x grid & smooth --------
#x_norm = np.linspace(-0.15, 0.20, N_SAMPLES)
#zU = smooth(resample(zU, xL, x_norm), 9)
#zL = smooth(resample(zL, xL, x_norm), 9)
#w  = smooth(resample(w, xR, x_norm), 9)
#if zU_i is not None:
#    zU_i = smooth(resample(zU_i, xL, x_norm), 9)
#    zL_i = smooth(resample(zL_i, xL, x_norm), 9)
#if w_i is not None:
#    w_i = smooth(resample(w_i,  xR, x_norm), 9)

# -------- Plot -------------------------------
plt.figure()
plt.plot(xL, zU, label="zU")
plt.plot(xL, zL, label="zL")
plt.plot(xL_i, zU_i, label="zU_i")
plt.plot(xL_i, zL_i, label="zL_i")

plt.axis('equal')
plt.legend()
plt.pause(1)

plt.figure()
plt.plot(xR, w, label="w")
plt.plot(xR, -w, label="w")
plt.plot(xR_i, w_i, label="w_i")
plt.plot(xR_i, -w_i, label="w_i")
plt.legend()
plt.axis('equal')
plt.pause(1)



# -------- build KRM-like organism dict --------
organism_dorsal = {
    "name": "Salmon",
    "source": "MAREN FORSTRØNEN RONG, The application of CW and FMsonar technology to detect a decrease in air in the swim bladder of Atlantic salmon, measurements and modeling",
    "vernacular_name": "Atlantic salmon",
    "length": 0.35,           # normalized; rescale later as needed
    "body": {
        "boundary": "fluid-filled",
        "x": xL.tolist(),
        "w": w.tolist(),
        "z_U": zU.tolist(),
        "z_L": zL.tolist(),
        "c": 1570,
        "rho": 1060
    },
    "inclusions": []
}

if (w_i is not None) and (zU_i is not None):
    organism_dorsal["inclusions"].append({
        "boundary": "pressure-release",
        "x": xL_i.tolist(),
        "w": w_i.tolist(),
        "z_U": zU_i.tolist(),
        "z_L": zL_i.tolist(),
        "c": 340,
        "rho": 1.24
    })

organism_ventral = {
    "name": "Salmon",
    "source": "MAREN FORSTRØNEN RONG, The application of CW and FMsonar technology to detect a decrease in air in the swim bladder of Atlantic salmon, measurements and modeling",
    "vernacular_name": "Atlantic salmon",
    "length": 0.35,           # normalized; rescale later as needed
    "body": {
        "boundary": "fluid-filled",
        "x": xL.tolist(),
        "w": w.tolist(),
        "z_U": (-zL).tolist(),
        "z_L": (-zU).tolist(),
        "c": 1570,
        "rho": 1060
    },
    "inclusions": []
}

if (w_i is not None) and (zU_i is not None):
    organism_ventral["inclusions"].append({
        "boundary": "pressure-release",
        "x": xL_i.tolist(),
        "w": w_i.tolist(),
        "z_U": (-zL_i).tolist(),
        "z_L": (-zU_i).tolist(),
        "c": 340,
        "rho": 1.24
    })

# -------- write JSON --------
# Dorsal
with open(OUT_JSON_DORSAL, "w") as f:
    json.dump(organism_dorsal, f, indent=2)

# Ventral

with open(OUT_JSON_VENTRAL, "w") as f:
    json.dump(organism_ventral, f, indent=2)

