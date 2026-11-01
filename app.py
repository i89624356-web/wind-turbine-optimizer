"""
Streamlit Web App
풍력 터빈 날개 형상 최적화 프로그램
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import joblib
import os
import sys

# 한글 폰트 (배포 환경에서는 fallback)
try:
    matplotlib.rcParams["font.family"] = "Noto Sans KR"
except:
    pass
matplotlib.rcParams["axes.unicode_minus"] = False

# 경로 설정 (deploy 폴더에서는 모든 파일이 같은 디렉토리)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SRC_DIR
sys.path.insert(0, SRC_DIR)

from airfoil import AirfoilParams
from bem import BladeDesign, solve_bem, find_optimal_tsr


@st.cache_data
def load_sweep_data():
    return pd.read_csv(os.path.join(OUTPUT_DIR, "sweep_results.csv"))


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(OUTPUT_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(OUTPUT_DIR, "scaler.pkl"))
    return model, scaler


def main():
    st.set_page_config(
        page_title="Wind Turbine Blade Optimizer",
        page_icon="",
        layout="wide",
    )

    st.title("Wind Turbine Blade Shape Optimizer")
    st.markdown("**BEM (Blade Element Momentum) analysis**")

    # --- Sidebar: 입력 ---
    st.sidebar.header("Design Parameters")

    wind_speed = st.sidebar.slider(
        "Wind Speed (m/s)", min_value=3.0, max_value=25.0, value=10.0, step=0.5
    )
    radius = st.sidebar.slider(
        "Blade Radius (m)", min_value=2.0, max_value=10.0, value=5.0, step=0.5
    )
    thickness = st.sidebar.slider(
        "Thickness Ratio", min_value=0.06, max_value=0.24, value=0.12, step=0.01
    )
    camber = st.sidebar.slider(
        "Camber", min_value=0.0, max_value=0.09, value=0.04, step=0.01
    )
    twist_root = st.sidebar.slider(
        "Root Twist Angle (deg)", min_value=5.0, max_value=35.0, value=20.0, step=1.0
    )

    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "Performance Analysis",
        "Optimal Shape Search",
        "Variable Impact",
        "Data Explorer",
    ])

    # === Tab 1: 성능 분석 ===
    with tab1:
        st.subheader("BEM Performance Analysis")

        airfoil = AirfoilParams(
            max_camber=camber,
            camber_pos=0.4 if camber > 0 else 0.0,
            thickness=thickness,
        )
        scale = radius / 5.0
        blade = BladeDesign(
            radius=radius,
            hub_radius=radius * 0.1,
            num_blades=3,
            twist_root=twist_root,
            twist_tip=2.0,
            chord_root=0.8 * scale,
            chord_tip=0.3 * scale,
            airfoil=airfoil,
        )

        # 최적 TSR 탐색
        best = find_optimal_tsr(blade, wind_speed, n_points=40)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Power Coefficient (Cp)", f"{best.cp:.4f}")
        col2.metric("Power Output", f"{best.power:.0f} W")
        col3.metric("Optimal TSR", f"{best.tsr:.2f}")
        col4.metric("RPM", f"{best.rpm:.1f}")

        # Cp vs TSR 곡선
        st.markdown("#### Cp vs TSR Curve")
        tsrs = np.linspace(1, 15, 50)
        cps = []
        for tsr in tsrs:
            result = solve_bem(blade, wind_speed, tsr)
            cps.append(result.cp)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(tsrs, cps, "b-", linewidth=2, label="Cp(TSR)")
        ax.axvline(best.tsr, color="red", linestyle="--", alpha=0.7, label=f"Optimal TSR = {best.tsr:.2f}")
        ax.axhline(best.cp, color="red", linestyle=":", alpha=0.5)
        ax.set_xlabel("Tip Speed Ratio (TSR)", fontsize=12)
        ax.set_ylabel("Power Coefficient (Cp)", fontsize=12)
        ax.set_title(f"Cp-TSR Curve (V={wind_speed} m/s)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        st.pyplot(fig)
        plt.close()

        # 풍속별 출력 곡선
        st.markdown("#### Power Curve")
        v_range = np.arange(3, 26, 1)
        powers = []
        cps_vs_v = []
        for v in v_range:
            r = find_optimal_tsr(blade, v, n_points=30)
            powers.append(r.power)
            cps_vs_v.append(r.cp)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(v_range, np.array(powers) / 1000, "b-o", markersize=4, linewidth=2)
        ax1.axvline(wind_speed, color="red", linestyle="--", alpha=0.7)
        ax1.set_xlabel("Wind Speed (m/s)", fontsize=12)
        ax1.set_ylabel("Power (kW)", fontsize=12)
        ax1.set_title("Power Curve", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        ax2.plot(v_range, cps_vs_v, "g-o", markersize=4, linewidth=2)
        ax2.axvline(wind_speed, color="red", linestyle="--", alpha=0.7)
        ax2.set_xlabel("Wind Speed (m/s)", fontsize=12)
        ax2.set_ylabel("Cp", fontsize=12)
        ax2.set_title("Cp vs Wind Speed", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # === Tab 2: 최적 형상 탐색 ===
    with tab2:
        st.subheader("Optimal Shape Comparison")

        try:
            df = load_sweep_data()

            # 현재 풍속에 가장 가까운 데이터
            closest_ws = df["wind_speed_ms"].unique()
            ws_idx = np.argmin(np.abs(closest_ws - wind_speed))
            ws_selected = closest_ws[ws_idx]

            sub = df[df["wind_speed_ms"] == ws_selected]
            best_row = sub.loc[sub["cp"].idxmax()]

            st.markdown(f"**Wind Speed: {ws_selected} m/s**")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Your Design")
                st.write(f"- Radius: {radius} m")
                st.write(f"- Thickness: {thickness}")
                st.write(f"- Camber: {camber}")
                st.write(f"- Root Twist: {twist_root} deg")
                st.write(f"- **Cp: {best.cp:.4f}**")
                st.write(f"- **Power: {best.power:.0f} W**")

            with col2:
                st.markdown("##### Optimal Design (from sweep)")
                st.write(f"- Radius: {best_row['radius_m']} m")
                st.write(f"- Thickness: {best_row['thickness']}")
                st.write(f"- Camber: {best_row['camber']}")
                st.write(f"- Root Twist: {best_row['twist_root_deg']} deg")
                st.write(f"- **Cp: {best_row['cp']:.4f}**")
                st.write(f"- **Power: {best_row['power_W']:.0f} W**")

            # 비교 바 차트
            cp_diff = best.cp - best_row["cp"]
            color = "green" if cp_diff >= 0 else "red"
            st.markdown(f"**Cp difference: <span style='color:{color}'>{cp_diff:+.4f}</span>**",
                        unsafe_allow_html=True)

            # 풍속별 최적 형상 변화
            st.markdown("#### How Optimal Shape Changes with Wind Speed")
            idx = df.groupby("wind_speed_ms")["cp"].idxmax()
            optimal_df = df.loc[idx].reset_index(drop=True)
            st.dataframe(
                optimal_df[["wind_speed_ms", "radius_m", "thickness", "camber",
                            "twist_root_deg", "cp", "power_W", "optimal_tsr"]],
                use_container_width=True,
            )

        except FileNotFoundError:
            st.warning("Sweep data not found. Run sweep.py first.")

    # === Tab 3: 변수별 영향 ===
    with tab3:
        st.subheader("Variable Impact Analysis")

        try:
            df = load_sweep_data()

            var_select = st.selectbox(
                "Select variable",
                ["radius_m", "thickness", "camber", "twist_root_deg"],
            )
            ws_filter = st.multiselect(
                "Wind speeds to show",
                sorted(df["wind_speed_ms"].unique()),
                default=[6, 10, 15, 25],
            )

            grouped = df[df["wind_speed_ms"].isin(ws_filter)].groupby(
                [var_select, "wind_speed_ms"]
            )["cp"].mean().reset_index()

            fig, ax = plt.subplots(figsize=(10, 6))
            for ws in sorted(ws_filter):
                subset = grouped[grouped["wind_speed_ms"] == ws]
                ax.plot(subset[var_select], subset["cp"], "o-", label=f"{ws} m/s", markersize=6)

            ax.set_xlabel(var_select, fontsize=12)
            ax.set_ylabel("Average Cp", fontsize=12)
            ax.set_title(f"Effect of {var_select} on Cp", fontsize=14, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

            # ML Feature Importance
            img_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
            if os.path.exists(img_path):
                st.markdown("#### ML Feature Importance")
                st.image(img_path)

        except FileNotFoundError:
            st.warning("Sweep data not found. Run sweep.py first.")

    # === Tab 4: Data Explorer ===
    with tab4:
        st.subheader("Raw Data Explorer")

        try:
            df = load_sweep_data()
            st.write(f"Total records: {len(df)}")

            # 필터
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

            # CSV 다운로드
            csv = filtered.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download filtered data as CSV",
                csv,
                "filtered_results.csv",
                "text/csv",
            )

        except FileNotFoundError:
            st.warning("Sweep data not found. Run sweep.py first.")


if __name__ == "__main__":
    main()
