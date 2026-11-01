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
import joblib
import os
import sys

matplotlib.rcParams["axes.unicode_minus"] = False

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SRC_DIR
sys.path.insert(0, SRC_DIR)


@st.cache_data
def load_sweep_data():
    return pd.read_csv(os.path.join(OUTPUT_DIR, "sweep_results.csv"))


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(OUTPUT_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(OUTPUT_DIR, "scaler.pkl"))
    return model, scaler


def predict_cp(model, scaler, wind_speed, radius, thickness, camber, twist_root):
    """ML model prediction (instant)"""
    X = np.array([[wind_speed, radius, thickness, camber, twist_root]])
    X_s = scaler.transform(X)
    return float(model.predict(X_s)[0])


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

    model, scaler = load_model()
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "Performance Analysis",
        "Optimal Shape Search",
        "Variable Impact",
        "Data Explorer",
    ])

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
