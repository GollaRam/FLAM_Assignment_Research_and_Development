import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

data_path = "xy_data.csv"

def residuals(params, x, y):
    theta, M, X = params
    c, s_ = np.cos(theta), np.sin(theta)
    xt, yt = x - X, y - 42.0
    t = xt * c + yt * s_
    s = -xt * s_ + yt * c
    predicted_s = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    return s - predicted_s


def fit(x, y):
    theta_lo, theta_hi = np.deg2rad(1e-3), np.deg2rad(50 - 1e-3)
    lo = [theta_lo, -0.05, 0.0]
    hi = [theta_hi, 0.05, 100.0]

    inits = [
        [th0, M0, X0]
        for th0 in np.deg2rad(np.linspace(2, 48, 12))
        for M0 in (-0.03, -0.005, 0.0, 0.005, 0.03)
        for X0 in (10, 30, 50, 70, 90)
    ]

    best = None
    for p0 in inits:
        res = least_squares(
            residuals, p0, args=(x, y), bounds=(lo, hi),
            method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14,
        )
        if best is None or res.cost < best.cost:
            best = res
    return best


def compute_l1(x, y, theta, M, X, n_samples=5000):
    t_dense = np.linspace(6, 60, n_samples)
    s_dense = np.exp(M * np.abs(t_dense)) * np.sin(0.3 * t_dense)
    x_fit = t_dense * np.cos(theta) - s_dense * np.sin(theta) + X
    y_fit = 42 + t_dense * np.sin(theta) + s_dense * np.cos(theta)

    curve_pts = np.column_stack([x_fit, y_fit])
    tree = cKDTree(curve_pts)
    data_pts = np.column_stack([x, y])
    _, idx = tree.query(data_pts, k=1)
    l1_per_point = np.abs(data_pts - curve_pts[idx]).sum(axis=1)

    return l1_per_point, x_fit, y_fit


def main():
    df = pd.read_csv(data_path)
    x, y = df["x"].values, df["y"].values

    result = fit(x, y)
    theta, M, X = result.x
    resid = residuals(result.x, x, y)

    print(f"theta = {theta:.6f} rad = {np.rad2deg(theta):.4f} deg")
    print(f"M     = {M:.6f}")
    print(f"X     = {X:.6f}")
    print(f"fit residual RMSE = {np.sqrt(np.mean(resid**2)):.3e}")

    l1_per_point, x_fit, y_fit = compute_l1(x, y, theta, M, X)
    print(f"L1 mean  = {l1_per_point.mean():.6e}")
    print(f"L1 max   = {l1_per_point.max():.6e}")
    print(f"L1 total = {l1_per_point.sum():.6e}")

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=4, alpha=0.6, label="raw data")
    plt.plot(x_fit, y_fit, color="red", linewidth=1.2, label="fitted curve")
    plt.title(f"Fitted curve vs raw data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scatter_fit_l1.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()