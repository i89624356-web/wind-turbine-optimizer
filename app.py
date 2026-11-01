"""
Streamlit Web App
Wind Turbine Blade Shape Optimizer
ML model for instant predictions, sweep data for analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["axes.unicode_minus"] = False

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SRC_DIR
sys.path.insert(0, SRC_DIR)

FEATURES = ["wind_speed_ms", "radius_m", "thickness", "camber", "twist_root_deg"]


@st.cache_data
def load_sweep_data():
    return pd.read_csv(os.path.join(OUTPUT_DIR, "sweep_results.csv"))


@st.cache_resource
def train_model():
    """CSV data -> train model on startup"""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "sweep_results.csv"))
    X = df[FEATURES].values
    y = df["cp"].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_s, y)
    return model, scaler


def predict_cp(model, scaler, wind_speed, radius, thickness, camber, twist_root):
    """ML model prediction (instant)"""
    X = np.array([[wind_speed, radius, thickness, camber, twist_root]])
    X_s = scaler.transform(X)
    return float(model.predict(X_s)[0])


def naca4_coords(max_camber, camber_pos, thickness, n_points=100):
    """NACA 4-digit airfoil coordinates"""
    x = (1 - np.cos(np.linspace(0, np.pi, n_points))) / 2

    # Thickness distribution
    yt = 5 * thickness * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
        + 0.2843 * x**3 - 0.1015 * x**4
    )

    if max_camber == 0 or camber_pos == 0:
        xu, yu = x, yt
        xl, yl = x, -yt
    else:
        m, p = max_camber, camber_pos
        yc = np.where(x < p,
                      m / p**2 * (2 * p * x - x**2),
                      m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(x < p,
                       2 * m / p**2 * (p - x),
                       2 * m / (1 - p)**2 * (p - x))
        theta = np.arctan(dyc)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl


def plot_airfoil_shape(camber, thickness, camber_pos=0.4):
    """Airfoil cross-section plot"""
    m = camber
    p = camber_pos if camber > 0 else 0
    xu, yu, xl, yl = naca4_coords(m, p, thickness)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill(np.concatenate([xu, xl[::-1]]),
            np.concatenate([yu, yl[::-1]]),
            color="steelblue", alpha=0.4, edgecolor="steelblue", linewidth=2)
    ax.plot(xu, yu, "b-", linewidth=2)
    ax.plot(xl, yl, "b-", linewidth=2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.3, 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x/c", fontsize=12)
    ax.set_ylabel("y/c", fontsize=12)

    naca_m = int(round(m * 100))
    naca_p = int(round(p * 10))
    naca_t = int(round(thickness * 100))
    ax.set_title(f"NACA {naca_m}{naca_p}{naca_t:02d} Airfoil", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    return fig


def plot_blade_3d(radius, thickness, camber, twist_root, twist_tip=2.0, camber_pos=0.4):
    """3D blade shape with filled surface"""
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import LightSource

    n_sections = 30
    n_points = 40
    r_positions = np.linspace(radius * 0.1, radius, n_sections)

    m = camber
    p = camber_pos if camber > 0 else 0

    # Build upper and lower surface meshes
    X_upper = np.zeros((n_sections, n_points))
    Y_upper = np.zeros((n_sections, n_points))
    Z_upper = np.zeros((n_sections, n_points))
    X_lower = np.zeros((n_sections, n_points))
    Y_lower = np.zeros((n_sections, n_points))
    Z_lower = np.zeros((n_sections, n_points))

    for i, r in enumerate(r_positions):
        frac = (r - radius * 0.1) / (radius * 0.9)
        twist = twist_root + (twist_tip - twist_root) * frac
        chord = (0.8 + (0.3 - 0.8) * frac) * radius / 5.0
        twist_rad = np.radians(twist)

        xu, yu, xl, yl = naca4_coords(m, p, thickness, n_points)

        # Scale
        xu_s, yu_s = xu * chord, yu * chord
        xl_s, yl_s = xl * chord, yl * chord

        # Rotate by twist
        X_upper[i] = xu_s * np.cos(twist_rad) - yu_s * np.sin(twist_rad)
        Z_upper[i] = xu_s * np.sin(twist_rad) + yu_s * np.cos(twist_rad)
        Y_upper[i] = r

        X_lower[i] = xl_s * np.cos(twist_rad) - yl_s * np.sin(twist_rad)
        Z_lower[i] = xl_s * np.sin(twist_rad) + yl_s * np.cos(twist_rad)
        Y_lower[i] = r

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot surfaces
    ax.plot_surface(X_upper, Y_upper, Z_upper, color="steelblue", alpha=0.8,
                    edgecolor="none", shade=True)
    ax.plot_surface(X_lower, Y_lower, Z_lower, color="steelblue", alpha=0.8,
                    edgecolor="none", shade=True)

    # Section outlines at key positions
    for idx in [0, n_sections // 4, n_sections // 2, 3 * n_sections // 4, n_sections - 1]:
        x_outline = np.concatenate([X_upper[idx], X_lower[idx, ::-1]])
        y_outline = np.full_like(x_outline, Y_upper[idx, 0])
        z_outline = np.concatenate([Z_upper[idx], Z_lower[idx, ::-1]])
        ax.plot(x_outline, y_outline, z_outline, color="navy", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Chord (m)", fontsize=10)
    ax.set_ylabel("Span (m)", fontsize=10)
    ax.set_zlabel("", fontsize=10)
    ax.set_title(f"Blade Shape (R={radius}m, twist={twist_root:.0f}~{twist_tip:.0f} deg)",
                 fontsize=13, fontweight="bold")
    ax.view_init(elev=25, azim=-55)

    # Equal aspect ratio approximation
    max_range = max(
        X_upper.max() - X_upper.min(),
        Y_upper.max() - Y_upper.min(),
        max(Z_upper.max(), -Z_lower.min()) * 2
    ) / 2
    mid_x = (X_upper.max() + X_upper.min()) / 2
    mid_y = (Y_upper.max() + Y_upper.min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(-max_range * 0.3, max_range * 0.3)

    return fig


def plot_streamlines(camber, thickness, wind_speed, camber_pos=0.4, alpha_deg=5.0):
    """Simplified streamlines around airfoil using potential flow"""
    m = camber
    p = camber_pos if camber > 0 else 0
    xu, yu, xl, yl = naca4_coords(m, p, thickness, 80)

    # Grid around airfoil
    nx, ny = 120, 80
    x_grid = np.linspace(-0.5, 2.0, nx)
    y_grid = np.linspace(-0.8, 0.8, ny)
    X, Y = np.meshgrid(x_grid, y_grid)

    alpha_rad = np.radians(alpha_deg)

    # Uniform flow + approximation
    U = wind_speed * np.cos(alpha_rad)
    V = wind_speed * np.sin(alpha_rad)

    # Source/sink doublet for thickness + vortex for lift
    Vx = np.full_like(X, U)
    Vy = np.full_like(X, V)

    # Doublet at 0.25c for thickness effect
    xd, yd = 0.25, 0.0
    dx = X - xd
    dy = Y - yd
    r2 = dx**2 + dy**2 + 1e-6
    doublet_strength = thickness * wind_speed * 0.5
    Vx += doublet_strength * (dy**2 - dx**2) / r2**2
    Vy += -2 * doublet_strength * dx * dy / r2**2

    # Vortex for camber/lift
    gamma = 2 * np.pi * wind_speed * (camber * 4 + np.sin(alpha_rad)) * 0.3
    Vx += gamma * dy / (2 * np.pi * r2)
    Vy += -gamma * dx / (2 * np.pi * r2)

    # Speed magnitude
    speed = np.sqrt(Vx**2 + Vy**2)

    # Airfoil boundary
    from matplotlib.path import Path
    airfoil_x = np.concatenate([xu, xl[::-1]])
    airfoil_y = np.concatenate([yu, yl[::-1]])
    airfoil_path = Path(np.column_stack([airfoil_x, airfoil_y]))
    points = np.column_stack([X.ravel(), Y.ravel()])
    mask = airfoil_path.contains_points(points).reshape(X.shape)

    # Expand mask slightly to remove white fringe
    from scipy.ndimage import binary_dilation
    mask_expanded = binary_dilation(mask, iterations=2)

    # For streamlines: mask inside airfoil
    Vx_stream = Vx.copy()
    Vy_stream = Vy.copy()
    Vx_stream[mask_expanded] = np.nan
    Vy_stream[mask_expanded] = np.nan

    # For contour: interpolate through airfoil (no white gap)
    speed_filled = speed.copy()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Speed color map (use unmasked speed so no white region)
    cm = ax.contourf(X, Y, speed_filled / wind_speed, levels=30, cmap="coolwarm", alpha=0.85)
    fig.colorbar(cm, ax=ax, label="V / V_inf", shrink=0.8)

    # Streamlines
    ax.streamplot(X, Y, Vx_stream, Vy_stream, color="black", linewidth=0.5,
                  density=2.0, arrowsize=0.8)

    # Airfoil on top
    ax.fill(airfoil_x, airfoil_y, color="#2c3e50", alpha=1.0, edgecolor="black", linewidth=2, zorder=10)

    ax.set_xlim(-0.3, 1.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x/c", fontsize=12)
    ax.set_ylabel("y/c", fontsize=12)
    ax.set_title(f"Flow Visualization (V={wind_speed} m/s, AoA={alpha_deg} deg)",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    return fig


def estimate_power(cp, wind_speed, radius, rho=1.225):
    """P = Cp * 0.5 * rho * A * V^3"""
    A = np.pi * radius ** 2
    return cp * 0.5 * rho * A * wind_speed ** 3


def main():
    st.set_page_config(
        page_title="Wind Turbine Blade Optimizer",
        layout="wide",
    )

    st.title("Wind Turbine Blade Shape Optimizer")
    st.markdown("**BEM-based analysis with ML prediction**")

    model, scaler = train_model()
    df = load_sweep_data()

    # --- Sidebar ---
    st.sidebar.header("Design Parameters")

    wind_speed = st.sidebar.slider(
        "Wind Speed (m/s)", min_value=4.0, max_value=25.0, value=10.0, step=0.5
    )
    radius = st.sidebar.slider(
        "Blade Radius (m)", min_value=3.0, max_value=7.0, value=5.0, step=0.5
    )
    thickness = st.sidebar.slider(
        "Thickness Ratio", min_value=0.09, max_value=0.21, value=0.15, step=0.01
    )
    camber = st.sidebar.slider(
        "Camber", min_value=0.0, max_value=0.08, value=0.04, step=0.01
    )
    twist_root = st.sidebar.slider(
        "Root Twist (deg)", min_value=10.0, max_value=30.0, value=20.0, step=1.0
    )

    # --- ML prediction (instant) ---
    user_cp = predict_cp(model, scaler, wind_speed, radius, thickness, camber, twist_root)
    user_cp = np.clip(user_cp, 0, 0.593)
    user_power = estimate_power(user_cp, wind_speed, radius)

    # --- Tabs ---
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "Blade & Flow Visualization",
        "Performance Analysis",
        "Optimal Shape Search",
        "Variable Impact",
        "Data Explorer",
    ])

    # === Tab 0: Visualization ===
    with tab0:
        st.subheader("Blade Shape & Air Flow")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Airfoil Cross-Section")
            fig_af = plot_airfoil_shape(camber, thickness)
            st.pyplot(fig_af)
            plt.close()

        with col_b:
            st.markdown("#### 3D Blade Shape")
            fig_3d = plot_blade_3d(radius, thickness, camber, twist_root)
            st.pyplot(fig_3d)
            plt.close()

        st.markdown("#### Air Flow Around Airfoil")
        aoa = st.slider("Angle of Attack (deg)", min_value=-5.0, max_value=20.0, value=5.0, step=1.0)
        fig_flow = plot_streamlines(camber, thickness, wind_speed, alpha_deg=aoa)
        st.pyplot(fig_flow)
        plt.close()

    # === Tab 1 ===
    with tab1:
        st.subheader("Performance Analysis")

        col1, col2, col3 = st.columns(3)
        col1.metric("Power Coefficient (Cp)", f"{user_cp:.4f}")
        col2.metric("Power Output", f"{user_power:.0f} W")
        col3.metric("Power Output", f"{user_power/1000:.2f} kW")

        # Cp vs wind speed curve (ML, instant)
        st.markdown("#### Cp vs Wind Speed")
        v_range = np.arange(4, 26, 0.5)
        cp_vs_v = [predict_cp(model, scaler, v, radius, thickness, camber, twist_root) for v in v_range]
        cp_vs_v = np.clip(cp_vs_v, 0, 0.593)
        power_vs_v = [estimate_power(cp, v, radius) for cp, v in zip(cp_vs_v, v_range)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(v_range, cp_vs_v, "b-", linewidth=2)
        ax1.axvline(wind_speed, color="red", linestyle="--", alpha=0.7, label=f"Current: {wind_speed} m/s")
        ax1.scatter([wind_speed], [user_cp], color="red", s=100, zorder=5)
        ax1.set_xlabel("Wind Speed (m/s)", fontsize=12)
        ax1.set_ylabel("Cp", fontsize=12)
        ax1.set_title("Cp vs Wind Speed", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 0.6)

        ax2.plot(v_range, np.array(power_vs_v) / 1000, "b-", linewidth=2)
        ax2.axvline(wind_speed, color="red", linestyle="--", alpha=0.7, label=f"Current: {wind_speed} m/s")
        ax2.scatter([wind_speed], [user_power / 1000], color="red", s=100, zorder=5)
        ax2.set_xlabel("Wind Speed (m/s)", fontsize=12)
        ax2.set_ylabel("Power (kW)", fontsize=12)
        ax2.set_title("Power Curve", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Cp vs each variable (fix others, sweep one)
        st.markdown("#### Sensitivity to Each Variable")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        var_configs = [
            ("radius_m", "Blade Radius (m)", np.arange(3, 7.5, 0.25), radius),
            ("thickness", "Thickness Ratio", np.arange(0.09, 0.22, 0.01), thickness),
            ("camber", "Camber", np.arange(0.0, 0.09, 0.005), camber),
            ("twist_root_deg", "Root Twist (deg)", np.arange(10, 31, 1), twist_root),
        ]
        base = [wind_speed, radius, thickness, camber, twist_root]
        for ax, (var_name, label, var_range, current_val) in zip(axes.flat, var_configs):
            cps = []
            for val in var_range:
                inputs = base.copy()
                idx_map = {"radius_m": 1, "thickness": 2, "camber": 3, "twist_root_deg": 4}
                inputs[idx_map[var_name]] = val
                cp = predict_cp(model, scaler, *inputs)
                cps.append(np.clip(cp, 0, 0.593))
            ax.plot(var_range, cps, "b-", linewidth=2)
            ax.axvline(current_val, color="red", linestyle="--", alpha=0.7)
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel("Cp", fontsize=11)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # === Tab 2 ===
    with tab2:
        st.subheader("Optimal Shape Comparison")

        closest_ws = df["wind_speed_ms"].unique()
        ws_idx = np.argmin(np.abs(closest_ws - wind_speed))
        ws_selected = closest_ws[ws_idx]

        sub = df[df["wind_speed_ms"] == ws_selected]
        best_row = sub.loc[sub["cp"].idxmax()]

        st.markdown(f"**Nearest data wind speed: {ws_selected} m/s**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Your Design")
            st.markdown(f"""
| Parameter | Value |
|---|---|
| Radius | {radius} m |
| Thickness | {thickness} |
| Camber | {camber} |
| Root Twist | {twist_root} deg |
| **Cp** | **{user_cp:.4f}** |
| **Power** | **{user_power:.0f} W** |
""")

        with col2:
            opt_power = estimate_power(best_row['cp'], ws_selected, best_row['radius_m'])
            st.markdown("##### Optimal Design")
            st.markdown(f"""
| Parameter | Value |
|---|---|
| Radius | {best_row['radius_m']} m |
| Thickness | {best_row['thickness']} |
| Camber | {best_row['camber']} |
| Root Twist | {best_row['twist_root_deg']} deg |
| **Cp** | **{best_row['cp']:.4f}** |
| **Power** | **{best_row['power_W']:.0f} W** |
""")

        cp_diff = user_cp - best_row["cp"]
        if cp_diff >= 0:
            st.success(f"Your design Cp is {cp_diff:+.4f} compared to sweep optimal")
        else:
            st.error(f"Your design Cp is {cp_diff:+.4f} compared to sweep optimal")

        st.markdown("#### Optimal Shape by Wind Speed")
        idx = df.groupby("wind_speed_ms")["cp"].idxmax()
        optimal_df = df.loc[idx].reset_index(drop=True)
        st.dataframe(
            optimal_df[["wind_speed_ms", "radius_m", "thickness", "camber",
                        "twist_root_deg", "cp", "power_W", "optimal_tsr"]].rename(columns={
                "wind_speed_ms": "Wind Speed (m/s)",
                "radius_m": "Radius (m)",
                "thickness": "Thickness",
                "camber": "Camber",
                "twist_root_deg": "Root Twist (deg)",
                "cp": "Cp",
                "power_W": "Power (W)",
                "optimal_tsr": "Optimal TSR",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Plot optimal variables vs wind speed
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        vars_plot = [
            ("radius_m", "Radius (m)"),
            ("thickness", "Thickness"),
            ("camber", "Camber"),
            ("twist_root_deg", "Root Twist (deg)"),
        ]
        for ax, (col, label) in zip(axes.flat, vars_plot):
            ax.plot(optimal_df["wind_speed_ms"], optimal_df[col], "s-",
                    color="royalblue", markersize=8, linewidth=2)
            ax.set_xlabel("Wind Speed (m/s)", fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.grid(True, alpha=0.3)
        fig.suptitle("How Optimal Design Changes with Wind Speed", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # === Tab 3 ===
    with tab3:
        st.subheader("Variable Impact Analysis")

        var_select = st.selectbox(
            "Select variable",
            ["radius_m", "thickness", "camber", "twist_root_deg"],
            format_func=lambda x: {
                "radius_m": "Blade Radius",
                "thickness": "Thickness Ratio",
                "camber": "Camber",
                "twist_root_deg": "Root Twist Angle",
            }[x],
        )
        ws_filter = st.multiselect(
            "Wind speeds to show",
            sorted(df["wind_speed_ms"].unique().tolist()),
            default=[4, 10, 18, 25],
        )

        if ws_filter:
            grouped = df[df["wind_speed_ms"].isin(ws_filter)].groupby(
                [var_select, "wind_speed_ms"]
            )["cp"].mean().reset_index()

            fig, ax = plt.subplots(figsize=(10, 6))
            for ws in sorted(ws_filter):
                subset = grouped[grouped["wind_speed_ms"] == ws]
                ax.plot(subset[var_select], subset["cp"], "o-", label=f"{ws} m/s", markersize=6)

            ax.set_xlabel(var_select, fontsize=12)
            ax.set_ylabel("Average Cp", fontsize=12)
            ax.set_title(f"Effect of {var_select} on Cp by Wind Speed", fontsize=14, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

        # Heatmap
        st.markdown("#### Cp Heatmap")
        hm_ws = st.selectbox("Wind speed for heatmap", sorted(df["wind_speed_ms"].unique().tolist()), index=3)
        hm_vars = [v for v in ["radius_m", "thickness", "camber", "twist_root_deg"] if v != var_select]
        hm_var2 = st.selectbox("Second variable", hm_vars)

        sub_hm = df[df["wind_speed_ms"] == hm_ws]
        pivot = sub_hm.groupby([var_select, hm_var2])["cp"].mean().reset_index()
        pivot_table = pivot.pivot(index=var_select, columns=hm_var2, values="cp")

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.pcolormesh(pivot_table.columns, pivot_table.index, pivot_table.values,
                           cmap="YlOrRd", shading="auto")
        fig.colorbar(im, ax=ax, label="Cp")
        ax.set_xlabel(hm_var2, fontsize=12)
        ax.set_ylabel(var_select, fontsize=12)
        ax.set_title(f"Cp Heatmap at {hm_ws} m/s", fontsize=14, fontweight="bold")
        st.pyplot(fig)
        plt.close()

        # Feature importance
        img_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
        if os.path.exists(img_path):
            st.markdown("#### ML Feature Importance")
            st.image(img_path)

    # === Tab 4 ===
    with tab4:
        st.subheader("Raw Data Explorer")
        st.write(f"Total records: {len(df)}")

        ws_range = st.slider(
            "Wind Speed Range (m/s)",
            float(df["wind_speed_ms"].min()),
            float(df["wind_speed_ms"].max()),
            (4.0, 25.0),
        )
        cp_min = st.slider("Minimum Cp", 0.0, 0.6, 0.0, 0.01)

        filtered = df[
            (df["wind_speed_ms"] >= ws_range[0])
            & (df["wind_speed_ms"] <= ws_range[1])
            & (df["cp"] >= cp_min)
        ]
        st.write(f"Filtered records: {len(filtered)}")
        st.dataframe(filtered, use_container_width=True)

        csv = filtered.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download filtered data as CSV",
            csv,
            "filtered_results.csv",
            "text/csv",
        )


if __name__ == "__main__":
    main()
