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
import plotly.graph_objects as go
import os
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["axes.unicode_minus"] = False

# Korean font setup
import matplotlib.font_manager as fm
import platform

if platform.system() == "Linux":
    # Streamlit Cloud (Linux) - use NanumGothic installed via packages.txt
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        matplotlib.rcParams["font.family"] = "NanumGothic"
elif platform.system() == "Windows":
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    matplotlib.rcParams["font.family"] = "AppleGothic"

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
    """3D blade shape with interactive plotly surface"""
    n_sections = 30
    n_points = 40
    r_positions = np.linspace(radius * 0.1, radius, n_sections)

    m = camber
    p = camber_pos if camber > 0 else 0

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

        xu_s, yu_s = xu * chord, yu * chord
        xl_s, yl_s = xl * chord, yl * chord

        X_upper[i] = xu_s * np.cos(twist_rad) - yu_s * np.sin(twist_rad)
        Z_upper[i] = xu_s * np.sin(twist_rad) + yu_s * np.cos(twist_rad)
        Y_upper[i] = r

        X_lower[i] = xl_s * np.cos(twist_rad) - yl_s * np.sin(twist_rad)
        Z_lower[i] = xl_s * np.sin(twist_rad) + yl_s * np.cos(twist_rad)
        Y_lower[i] = r

    fig = go.Figure()

    # Upper surface
    fig.add_trace(go.Surface(
        x=X_upper, y=Y_upper, z=Z_upper,
        colorscale=[[0, "steelblue"], [1, "steelblue"]],
        showscale=False, opacity=0.9,
        name="Upper Surface",
    ))
    # Lower surface
    fig.add_trace(go.Surface(
        x=X_lower, y=Y_lower, z=Z_lower,
        colorscale=[[0, "steelblue"], [1, "steelblue"]],
        showscale=False, opacity=0.9,
        name="Lower Surface",
    ))

    # Section outlines
    for idx in [0, n_sections // 4, n_sections // 2, 3 * n_sections // 4, n_sections - 1]:
        x_outline = np.concatenate([X_upper[idx], X_lower[idx, ::-1]])
        y_outline = np.full_like(x_outline, Y_upper[idx, 0])
        z_outline = np.concatenate([Z_upper[idx], Z_lower[idx, ::-1]])
        fig.add_trace(go.Scatter3d(
            x=x_outline, y=y_outline, z=z_outline,
            mode="lines", line=dict(color="navy", width=3),
            showlegend=False,
        ))

    max_range = max(
        X_upper.max() - X_upper.min(),
        Y_upper.max() - Y_upper.min(),
        max(Z_upper.max(), -Z_lower.min()) * 2
    ) / 2
    mid_x = (X_upper.max() + X_upper.min()) / 2
    mid_y = (Y_upper.max() + Y_upper.min()) / 2

    fig.update_layout(
        title=dict(text=f"블레이드 형상 (R={radius}m, 비틀림={twist_root:.0f}~{twist_tip:.0f} deg)"),
        scene=dict(
            xaxis_title="코드 (m)",
            yaxis_title="스팬 (m)",
            zaxis_title="",
            xaxis=dict(range=[mid_x - max_range, mid_x + max_range]),
            yaxis=dict(range=[mid_y - max_range, mid_y + max_range]),
            zaxis=dict(range=[-max_range * 0.3, max_range * 0.3]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=2, z=0.3),
            camera=dict(eye=dict(x=1.5, y=-1.0, z=0.6)),
        ),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
    )

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

    # Speed color map with tight range to show velocity variation near airfoil
    levels = np.linspace(0.5, 1.8, 40)
    cm = ax.contourf(X, Y, speed_filled / wind_speed, levels=levels, cmap="coolwarm",
                     alpha=0.85, extend="both")
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
    ax.set_title(f"공기 흐름 시각화 (V={wind_speed} m/s, 받음각={alpha_deg} deg)",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    return fig


def estimate_power(cp, wind_speed, radius, rho=1.225):
    """P = Cp * 0.5 * rho * A * V^3"""
    A = np.pi * radius ** 2
    return cp * 0.5 * rho * A * wind_speed ** 3


def main():
    st.set_page_config(
        page_title="풍력 터빈 날개 형상 최적화",
        layout="wide",
    )

    st.title("풍속에 따른 풍력 터빈 날개 형상별 효율 분석 및 최적 형상 추천")
    st.markdown("**BEM(Blade Element Momentum) 이론 기반 시뮬레이션 + 머신러닝 예측**")

    model, scaler = train_model()
    df = load_sweep_data()

    # --- 1. 입력 설정 (사이드바) ---
    st.sidebar.header("1. 입력 설정")
    st.sidebar.markdown("**풍속**")
    wind_speed = st.sidebar.slider(
        "풍속 (m/s)", min_value=4.0, max_value=25.0, value=10.0, step=0.5
    )
    st.sidebar.markdown("**날개 형상 설계 변수**")
    radius = st.sidebar.slider(
        "날개 길이 (m)", min_value=3.0, max_value=7.0, value=5.0, step=0.5
    )
    thickness = st.sidebar.slider(
        "두께 비율", min_value=0.09, max_value=0.21, value=0.15, step=0.01
    )
    camber = st.sidebar.slider(
        "곡률", min_value=0.0, max_value=0.08, value=0.04, step=0.01
    )
    twist_root = st.sidebar.slider(
        "비틀림 각도 (deg)", min_value=10.0, max_value=30.0, value=20.0, step=1.0
    )

    # --- ML 예측 ---
    user_cp = predict_cp(model, scaler, wind_speed, radius, thickness, camber, twist_root)
    user_cp = np.clip(user_cp, 0, 0.593)
    user_power = estimate_power(user_cp, wind_speed, radius)

    # 최적 형상 미리 계산
    closest_ws = df["wind_speed_ms"].unique()
    ws_idx = np.argmin(np.abs(closest_ws - wind_speed))
    ws_selected = closest_ws[ws_idx]
    sub = df[df["wind_speed_ms"] == ws_selected]
    best_row = sub.loc[sub["cp"].idxmax()]
    opt_power = estimate_power(best_row['cp'], ws_selected, best_row['radius_m'])

    # --- 탭 (계획서 구조도 순서) ---
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "2. 날개 형상 시각화",
        "3. 시뮬레이션 결과",
        "4. 풍속별 발전 효율 비교",
        "5. 최적 형상 탐색 및 결과 비교",
        "6. 데이터 탐색",
    ])

    # === 2. 날개 형상 시각화 ===
    with tab0:
        st.subheader("날개 형상 시각화")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 에어포일 단면")
            fig_af = plot_airfoil_shape(camber, thickness)
            st.pyplot(fig_af)
            plt.close()

        with col_b:
            st.markdown("#### 3D 블레이드 (드래그하여 회전)")
            fig_3d = plot_blade_3d(radius, thickness, camber, twist_root)
            st.plotly_chart(fig_3d, use_container_width=True, key="blade3d_tab0")

        st.markdown("#### 공기 흐름 시각화")
        aoa = st.slider("받음각 (deg)", min_value=-5.0, max_value=20.0, value=5.0, step=1.0)
        fig_flow = plot_streamlines(camber, thickness, wind_speed, alpha_deg=aoa)
        st.pyplot(fig_flow)
        plt.close()

    # === 3. 시뮬레이션 결과 ===
    with tab1:
        st.subheader("시뮬레이션 결과")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 사용자 입력 형상")
            st.metric("발전 효율 (Cp)", f"{user_cp:.4f}")
            st.metric("발전 출력", f"{user_power:.0f} W ({user_power/1000:.2f} kW)")
            st.markdown(f"""
| 설계 변수 | 값 |
|---|---|
| 날개 길이 | {radius} m |
| 두께 비율 | {thickness} |
| 곡률 | {camber} |
| 비틀림 각도 | {twist_root} deg |
""")

        with col2:
            st.markdown(f"##### 최적 형상 (풍속 {ws_selected} m/s 기준)")
            st.metric("발전 효율 (Cp)", f"{best_row['cp']:.4f}")
            st.metric("발전 출력", f"{opt_power:.0f} W ({opt_power/1000:.2f} kW)")
            st.markdown(f"""
| 설계 변수 | 값 |
|---|---|
| 날개 길이 | {best_row['radius_m']} m |
| 두께 비율 | {best_row['thickness']} |
| 곡률 | {best_row['camber']} |
| 비틀림 각도 | {best_row['twist_root_deg']} deg |
""")

        # 효율 비교
        cp_diff = user_cp - best_row["cp"]
        efficiency_pct = user_cp / best_row["cp"] * 100 if best_row["cp"] > 0 else 0
        if cp_diff >= 0:
            st.success(f"입력 형상의 Cp가 최적 형상 대비 {cp_diff:+.4f} (최적 대비 {efficiency_pct:.1f}%)")
        else:
            st.error(f"입력 형상의 Cp가 최적 형상 대비 {cp_diff:+.4f} (최적 대비 {efficiency_pct:.1f}%)")

        # 에어포일 형상 비교
        st.markdown("#### 에어포일 형상 비교")
        opt_camber = best_row["camber"]
        opt_thickness = best_row["thickness"]
        m_user = camber
        p_user = 0.4 if camber > 0 else 0
        m_opt = opt_camber
        p_opt = 0.4 if opt_camber > 0 else 0

        xu_u, yu_u, xl_u, yl_u = naca4_coords(m_user, p_user, thickness)
        xu_o, yu_o, xl_o, yl_o = naca4_coords(m_opt, p_opt, opt_thickness)

        fig_cmp, ax_cmp = plt.subplots(figsize=(10, 4))
        ax_cmp.fill(np.concatenate([xu_u, xl_u[::-1]]),
                    np.concatenate([yu_u, yl_u[::-1]]),
                    color="steelblue", alpha=0.25, label="사용자 입력 형상")
        ax_cmp.plot(xu_u, yu_u, "-", color="steelblue", linewidth=2)
        ax_cmp.plot(xl_u, yl_u, "-", color="steelblue", linewidth=2)

        ax_cmp.fill(np.concatenate([xu_o, xl_o[::-1]]),
                    np.concatenate([yu_o, yl_o[::-1]]),
                    color="crimson", alpha=0.25, label="최적 형상")
        ax_cmp.plot(xu_o, yu_o, "--", color="crimson", linewidth=2)
        ax_cmp.plot(xl_o, yl_o, "--", color="crimson", linewidth=2)

        ax_cmp.set_xlim(-0.05, 1.05)
        ax_cmp.set_ylim(-0.3, 0.3)
        ax_cmp.set_aspect("equal")
        ax_cmp.set_xlabel("x/c", fontsize=12)
        ax_cmp.set_ylabel("y/c", fontsize=12)
        ax_cmp.set_title("에어포일 단면 비교: 사용자 입력 vs 최적 형상", fontsize=14, fontweight="bold")
        ax_cmp.legend(fontsize=11)
        ax_cmp.grid(True, alpha=0.3)
        st.pyplot(fig_cmp)
        plt.close()

        # 3D 블레이드 비교
        st.markdown("#### 3D 블레이드 비교 (드래그하여 회전)")
        opt_radius = best_row["radius_m"]
        opt_twist = best_row["twist_root_deg"]

        col_3d_1, col_3d_2 = st.columns(2)
        with col_3d_1:
            st.markdown("##### 사용자 입력 형상")
            fig_3d_user = plot_blade_3d(radius, thickness, camber, twist_root)
            st.plotly_chart(fig_3d_user, use_container_width=True, key="blade3d_user")
        with col_3d_2:
            st.markdown("##### 최적 형상")
            fig_3d_opt = plot_blade_3d(opt_radius, opt_thickness, opt_camber, opt_twist)
            st.plotly_chart(fig_3d_opt, use_container_width=True, key="blade3d_opt")

    # === 4. 풍속별 발전 효율 비교 ===
    with tab2:
        st.subheader("풍속별 발전 효율 비교")

        # Cp vs 풍속, Power vs 풍속
        v_range = np.arange(4, 26, 0.5)
        cp_vs_v = [predict_cp(model, scaler, v, radius, thickness, camber, twist_root) for v in v_range]
        cp_vs_v = np.clip(cp_vs_v, 0, 0.593)
        power_vs_v = [estimate_power(cp, v, radius) for cp, v in zip(cp_vs_v, v_range)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(v_range, cp_vs_v, "b-", linewidth=2)
        ax1.axvline(wind_speed, color="red", linestyle="--", alpha=0.7, label=f"현재: {wind_speed} m/s")
        ax1.scatter([wind_speed], [user_cp], color="red", s=100, zorder=5)
        ax1.set_xlabel("풍속 (m/s)", fontsize=12)
        ax1.set_ylabel("Cp (발전 효율)", fontsize=12)
        ax1.set_title("풍속에 따른 발전 효율 (Cp)", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 0.6)

        ax2.plot(v_range, np.array(power_vs_v) / 1000, "b-", linewidth=2)
        ax2.axvline(wind_speed, color="red", linestyle="--", alpha=0.7, label=f"현재: {wind_speed} m/s")
        ax2.scatter([wind_speed], [user_power / 1000], color="red", s=100, zorder=5)
        ax2.set_xlabel("풍속 (m/s)", fontsize=12)
        ax2.set_ylabel("출력 (kW)", fontsize=12)
        ax2.set_title("풍속에 따른 발전 출력", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # 설계 변수별 Cp 민감도
        st.markdown("#### 설계 변수별 Cp 민감도")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        var_configs = [
            ("radius_m", "날개 길이 (m)", np.arange(3, 7.5, 0.25), radius),
            ("thickness", "두께 비율", np.arange(0.09, 0.22, 0.01), thickness),
            ("camber", "곡률", np.arange(0.0, 0.09, 0.005), camber),
            ("twist_root_deg", "비틀림 각도 (deg)", np.arange(10, 31, 1), twist_root),
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
            ax.axvline(current_val, color="red", linestyle="--", alpha=0.7, label="현재 설정")
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel("Cp", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"풍속 {wind_speed} m/s에서 각 변수 변화에 따른 Cp", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # === 5. 최적 형상 탐색 및 결과 비교 ===
    with tab3:
        st.subheader("최적 형상 탐색 및 결과 비교")

        # 풍속별 최적 형상 표
        st.markdown("#### 풍속별 최적 날개 형상")
        idx_opt = df.groupby("wind_speed_ms")["cp"].idxmax()
        optimal_df = df.loc[idx_opt].reset_index(drop=True)
        st.dataframe(
            optimal_df[["wind_speed_ms", "radius_m", "thickness", "camber",
                        "twist_root_deg", "cp", "power_W", "optimal_tsr"]].rename(columns={
                "wind_speed_ms": "풍속 (m/s)",
                "radius_m": "날개 길이 (m)",
                "thickness": "두께 비율",
                "camber": "곡률",
                "twist_root_deg": "비틀림 각도 (deg)",
                "cp": "Cp",
                "power_W": "출력 (W)",
                "optimal_tsr": "최적 TSR",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # 풍속에 따른 최적 설계 변수 변화
        st.markdown("#### 풍속에 따른 최적 설계 변수 변화")
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        vars_plot = [
            ("radius_m", "날개 길이 (m)"),
            ("thickness", "두께 비율"),
            ("camber", "곡률"),
            ("twist_root_deg", "비틀림 각도 (deg)"),
        ]
        for ax, (col, label) in zip(axes.flat, vars_plot):
            ax.plot(optimal_df["wind_speed_ms"], optimal_df[col], "s-",
                    color="royalblue", markersize=8, linewidth=2)
            ax.set_xlabel("풍속 (m/s)", fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.grid(True, alpha=0.3)
        fig.suptitle("풍속에 따른 최적 형상 변화", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # 변수별 풍속 의존성 분석
        st.markdown("#### 날개 형상별 풍력에 대한 시뮬레이션 결과 비교")
        var_select = st.selectbox(
            "분석할 변수 선택",
            ["radius_m", "thickness", "camber", "twist_root_deg"],
            format_func=lambda x: {
                "radius_m": "날개 길이",
                "thickness": "두께 비율",
                "camber": "곡률",
                "twist_root_deg": "비틀림 각도",
            }[x],
        )
        ws_filter = st.multiselect(
            "표시할 풍속 선택",
            sorted(df["wind_speed_ms"].unique().tolist()),
            default=[4, 10, 18, 25],
        )

        if ws_filter:
            grouped = df[df["wind_speed_ms"].isin(ws_filter)].groupby(
                [var_select, "wind_speed_ms"]
            )["cp"].mean().reset_index()

            var_label = {"radius_m": "날개 길이", "thickness": "두께 비율",
                         "camber": "곡률", "twist_root_deg": "비틀림 각도"}[var_select]

            fig, ax = plt.subplots(figsize=(10, 6))
            for ws in sorted(ws_filter):
                subset = grouped[grouped["wind_speed_ms"] == ws]
                ax.plot(subset[var_select], subset["cp"], "o-", label=f"{ws} m/s", markersize=6)

            ax.set_xlabel(var_label, fontsize=12)
            ax.set_ylabel("평균 Cp", fontsize=12)
            ax.set_title(f"{var_label} 변화에 따른 풍속별 Cp 비교", fontsize=14, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

        # 히트맵
        st.markdown("#### Cp 히트맵")
        hm_ws = st.selectbox("풍속 선택 (m/s)", sorted(df["wind_speed_ms"].unique().tolist()), index=3)
        hm_vars = [v for v in ["radius_m", "thickness", "camber", "twist_root_deg"] if v != var_select]
        hm_label_map = {"radius_m": "날개 길이", "thickness": "두께 비율",
                        "camber": "곡률", "twist_root_deg": "비틀림 각도"}
        hm_var2 = st.selectbox("두 번째 변수", hm_vars,
                               format_func=lambda x: hm_label_map[x])

        sub_hm = df[df["wind_speed_ms"] == hm_ws]
        pivot = sub_hm.groupby([var_select, hm_var2])["cp"].mean().reset_index()
        pivot_table = pivot.pivot(index=var_select, columns=hm_var2, values="cp")

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.pcolormesh(pivot_table.columns, pivot_table.index, pivot_table.values,
                           cmap="YlOrRd", shading="auto")
        fig.colorbar(im, ax=ax, label="Cp")
        ax.set_xlabel(hm_label_map[hm_var2], fontsize=12)
        ax.set_ylabel(hm_label_map.get(var_select, var_select), fontsize=12)
        ax.set_title(f"풍속 {hm_ws} m/s에서의 Cp 히트맵", fontsize=14, fontweight="bold")
        st.pyplot(fig)
        plt.close()

    # === 6. 데이터 탐색 ===
    with tab4:
        st.subheader("데이터 탐색")
        st.write(f"전체 데이터 수: {len(df)}건")

        ws_range = st.slider(
            "풍속 범위 (m/s)",
            float(df["wind_speed_ms"].min()),
            float(df["wind_speed_ms"].max()),
            (4.0, 25.0),
        )
        cp_min = st.slider("최소 Cp", 0.0, 0.6, 0.0, 0.01)

        filtered = df[
            (df["wind_speed_ms"] >= ws_range[0])
            & (df["wind_speed_ms"] <= ws_range[1])
            & (df["cp"] >= cp_min)
        ]
        st.write(f"필터링 결과: {len(filtered)}건")
        st.dataframe(filtered, use_container_width=True)

        csv = filtered.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "필터링 데이터 CSV 다운로드",
            csv,
            "filtered_results.csv",
            "text/csv",
        )


if __name__ == "__main__":
    main()
