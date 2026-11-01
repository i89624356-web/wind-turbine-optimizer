"""
데이터 분석 + ML 모델
탐색 결과 분석, 변수별 영향도, 풍속별 최적 형상, ML 회귀
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# 한글 폰트 설정
matplotlib.rcParams["font.family"] = "Noto Sans KR"
matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
DATA_PATH = os.path.join(OUTPUT_DIR, "sweep_results.csv")


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nCp range: {df['cp'].min():.4f} ~ {df['cp'].max():.4f}")
    return df


def find_optimal_per_windspeed(df):
    """풍속별 최적 설계변수 조합 추출"""
    design_vars = ["radius_m", "thickness", "camber", "twist_root_deg"]
    idx = df.groupby("wind_speed_ms")["cp"].idxmax()
    optimal = df.loc[idx, ["wind_speed_ms"] + design_vars + ["cp", "optimal_tsr", "power_W"]].reset_index(drop=True)
    print("\n=== Optimal design per wind speed ===")
    print(optimal.to_string(index=False))
    return optimal


def variable_importance_analysis(df):
    """각 설계변수가 Cp에 미치는 영향도 분석"""
    design_vars = ["radius_m", "thickness", "camber", "twist_root_deg"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Design Variable vs Power Coefficient (Cp)", fontsize=16, fontweight="bold")

    for ax, var in zip(axes.flat, design_vars):
        grouped = df.groupby([var, "wind_speed_ms"])["cp"].mean().reset_index()

        for ws in sorted(df["wind_speed_ms"].unique()):
            subset = grouped[grouped["wind_speed_ms"] == ws]
            ax.plot(subset[var], subset["cp"], "o-", label=f"{ws} m/s", markersize=3)

        ax.set_xlabel(var, fontsize=12)
        ax.set_ylabel("Cp", fontsize=12)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "variable_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: variable_importance.png")


def optimal_shape_vs_windspeed(df):
    """풍속에 따라 최적 형상이 달라지는지 시각화"""
    design_vars = ["radius_m", "thickness", "camber", "twist_root_deg"]
    var_labels = ["Blade Radius (m)", "Thickness Ratio", "Camber", "Root Twist (deg)"]

    # 풍속별 최적 조합
    idx = df.groupby("wind_speed_ms")["cp"].idxmax()
    optimal = df.loc[idx].reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Optimal Design Variables by Wind Speed", fontsize=16, fontweight="bold")

    for ax, var, label in zip(axes.flat, design_vars, var_labels):
        ax.plot(optimal["wind_speed_ms"], optimal[var], "s-", color="royalblue",
                markersize=8, linewidth=2)
        ax.set_xlabel("Wind Speed (m/s)", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "optimal_vs_windspeed.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: optimal_vs_windspeed.png")


def cp_heatmaps(df, wind_speed=10):
    """특정 풍속에서 설계변수 2개 조합별 Cp 히트맵"""
    sub = df[df["wind_speed_ms"] == wind_speed]

    pairs = [
        ("thickness", "camber"),
        ("twist_root_deg", "thickness"),
        ("twist_root_deg", "camber"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Cp Heatmap at Wind Speed = {wind_speed} m/s", fontsize=16, fontweight="bold")

    for ax, (v1, v2) in zip(axes, pairs):
        pivot = sub.groupby([v1, v2])["cp"].mean().reset_index()
        pivot_table = pivot.pivot(index=v1, columns=v2, values="cp")

        im = ax.pcolormesh(
            pivot_table.columns, pivot_table.index, pivot_table.values,
            cmap="YlOrRd", shading="auto"
        )
        fig.colorbar(im, ax=ax, label="Cp")
        ax.set_xlabel(v2, fontsize=11)
        ax.set_ylabel(v1, fontsize=11)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"cp_heatmap_{wind_speed}ms.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: cp_heatmap_{wind_speed}ms.png")


def train_ml_model(df):
    """ML 회귀 모델 학습: (풍속, 설계변수) -> Cp 예측"""
    features = ["wind_speed_ms", "radius_m", "thickness", "camber", "twist_root_deg"]
    target = "cp"

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 스케일러
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # RandomForest
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    y_pred_rf = rf.predict(X_test_s)

    # GradientBoosting
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    gb.fit(X_train_s, y_train)
    y_pred_gb = gb.predict(X_test_s)

    print("\n=== ML Model Performance ===")
    print(f"RandomForest  - R2: {r2_score(y_test, y_pred_rf):.4f}, MAE: {mean_absolute_error(y_test, y_pred_rf):.4f}")
    print(f"GradientBoost - R2: {r2_score(y_test, y_pred_gb):.4f}, MAE: {mean_absolute_error(y_test, y_pred_gb):.4f}")

    # Feature importance
    print(f"\nFeature Importance (GradientBoosting):")
    for feat, imp in sorted(zip(features, gb.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:>20s}: {imp:.4f}")

    # Feature importance 그래프
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_idx = np.argsort(gb.feature_importances_)
    ax.barh(
        [features[i] for i in sorted_idx],
        gb.feature_importances_[sorted_idx],
        color="steelblue"
    )
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("GradientBoosting Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: feature_importance.png")

    # Prediction vs Actual 그래프
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, y_pred, name in zip(axes, [y_pred_rf, y_pred_gb], ["RandomForest", "GradientBoosting"]):
        ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
        ax.plot([0, 0.6], [0, 0.6], "r--", linewidth=1)
        ax.set_xlabel("Actual Cp", fontsize=12)
        ax.set_ylabel("Predicted Cp", fontsize=12)
        r2 = r2_score(y_test, y_pred)
        ax.set_title(f"{name} (R2={r2:.4f})", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "ml_prediction.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: ml_prediction.png")

    # 모델 저장
    best_model = gb if r2_score(y_test, y_pred_gb) > r2_score(y_test, y_pred_rf) else rf
    best_name = "GradientBoosting" if best_model is gb else "RandomForest"
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    print(f"\nBest model ({best_name}) saved to best_model.pkl")

    return best_model, scaler


def main():
    df = load_data()

    print("\n" + "=" * 60)
    print("1. Optimal design per wind speed")
    print("=" * 60)
    optimal = find_optimal_per_windspeed(df)

    print("\n" + "=" * 60)
    print("2. Variable importance analysis")
    print("=" * 60)
    variable_importance_analysis(df)

    print("\n" + "=" * 60)
    print("3. Optimal shape vs wind speed")
    print("=" * 60)
    optimal_shape_vs_windspeed(df)

    print("\n" + "=" * 60)
    print("4. Cp heatmaps")
    print("=" * 60)
    for ws in [6, 10, 15, 25]:
        cp_heatmaps(df, wind_speed=ws)

    print("\n" + "=" * 60)
    print("5. ML model training")
    print("=" * 60)
    model, scaler = train_ml_model(df)

    print("\n\nAll analysis complete!")


if __name__ == "__main__":
    main()
