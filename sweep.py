"""
설계변수 전수탐색 스크립트
날개 길이, 두께 비율, 곡률(캠버), 비틀림 각도 x 풍속 그리드
"""

import numpy as np
import pandas as pd
import itertools
import time
import sys
import os

from airfoil import AirfoilParams
from bem import BladeDesign, find_optimal_tsr


def run_sweep():
    # ===== 설계변수 그리드 =====

    # 1. 날개 반지름 (m)
    radii = [3.0, 4.0, 5.0, 6.0, 7.0]

    # 2. 두께 비율 (chord 대비, NACA 4자리 마지막 2자리)
    thicknesses = [0.09, 0.12, 0.15, 0.18, 0.21]

    # 3. 곡률/캠버 (NACA 4자리 첫째 자리)
    cambers = [0.0, 0.02, 0.04, 0.06, 0.08]

    # 4. 루트 비틀림 각도 (degree, 팁은 항상 2도 고정)
    twist_roots = [10.0, 15.0, 20.0, 25.0, 30.0]

    # 5. 풍속 (m/s)
    wind_speeds = [4, 6, 8, 10, 12, 15, 18, 20, 25]

    # 캠버 위치는 캠버가 0이 아닐 때 0.4 고정 (NACA 표준)
    camber_pos = 0.4

    # 고정 파라미터
    num_blades = 3
    chord_root = 0.8  # 반지름에 비례하게 스케일링
    chord_tip = 0.3
    twist_tip = 2.0
    hub_ratio = 0.1   # 허브 = 반지름의 10%

    # 전체 조합 수 계산
    total = len(radii) * len(thicknesses) * len(cambers) * len(twist_roots) * len(wind_speeds)
    print(f"Total cases: {total}")

    results = []
    count = 0
    start_time = time.time()

    for R, t, m, tw_root in itertools.product(radii, thicknesses, cambers, twist_roots):
        # 에어포일 파라미터
        airfoil = AirfoilParams(
            max_camber=m,
            camber_pos=camber_pos if m > 0 else 0.0,
            thickness=t,
        )

        # 코드 길이를 반지름에 비례하게 스케일
        scale = R / 5.0
        c_root = chord_root * scale
        c_tip = chord_tip * scale

        blade = BladeDesign(
            radius=R,
            hub_radius=R * hub_ratio,
            num_blades=num_blades,
            twist_root=tw_root,
            twist_tip=twist_tip,
            chord_root=c_root,
            chord_tip=c_tip,
            airfoil=airfoil,
        )

        for v in wind_speeds:
            best = find_optimal_tsr(blade, wind_speed=v, n_points=30)

            results.append({
                "radius_m": R,
                "thickness": t,
                "camber": m,
                "twist_root_deg": tw_root,
                "wind_speed_ms": v,
                "optimal_tsr": best.tsr,
                "cp": best.cp,
                "power_W": best.power,
                "torque_Nm": best.torque,
                "thrust_N": best.thrust,
                "rpm": best.rpm,
                "ct": best.ct,
            })

            count += 1
            if count % 100 == 0:
                elapsed = time.time() - start_time
                rate = count / elapsed
                remaining = (total - count) / rate
                print(f"  [{count}/{total}] {elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining")

    elapsed = time.time() - start_time
    print(f"\nDone! {total} cases in {elapsed:.1f}s ({total/elapsed:.1f} cases/s)")

    # DataFrame 저장
    df = pd.DataFrame(results)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sweep_results.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved to {output_path}")
    print(f"\nShape: {df.shape}")
    print(f"\nCp stats:")
    print(df["cp"].describe())

    return df


if __name__ == "__main__":
    df = run_sweep()
