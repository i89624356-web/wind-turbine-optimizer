"""
BEM (Blade Element Momentum) 코어 엔진
풍력 터빈 블레이드 성능 계산
"""

import numpy as np
from dataclasses import dataclass, field
from airfoil import AirfoilParams, cl_alpha, cd_alpha


@dataclass
class BladeDesign:
    """블레이드 설계 파라미터"""
    radius: float               # 블레이드 반지름 (m)
    hub_radius: float           # 허브 반지름 (m), 보통 radius의 10~20%
    num_blades: int             # 블레이드 수 (보통 3)
    twist_root: float           # 루트 비틀림 각도 (degree)
    twist_tip: float            # 팁 비틀림 각도 (degree)
    chord_root: float           # 루트 코드 길이 (m)
    chord_tip: float            # 팁 코드 길이 (m)
    airfoil: AirfoilParams      # 에어포일 파라미터

    def twist_at(self, r: float) -> float:
        """반지름 위치 r에서의 비틀림 각도 (선형 보간)"""
        frac = (r - self.hub_radius) / (self.radius - self.hub_radius)
        return self.twist_root + (self.twist_tip - self.twist_root) * frac

    def chord_at(self, r: float) -> float:
        """반지름 위치 r에서의 코드 길이 (선형 보간)"""
        frac = (r - self.hub_radius) / (self.radius - self.hub_radius)
        return self.chord_root + (self.chord_tip - self.chord_root) * frac


@dataclass
class BEMResult:
    """BEM 계산 결과"""
    wind_speed: float       # 풍속 (m/s)
    rpm: float              # 회전수 (RPM)
    tsr: float              # 주속비 (Tip Speed Ratio)
    power: float            # 출력 (W)
    torque: float           # 토크 (N·m)
    thrust: float           # 추력 (N)
    cp: float               # 출력계수
    ct: float               # 추력계수

    # 구간별 상세 데이터
    r_sections: np.ndarray = field(default_factory=lambda: np.array([]))
    a_sections: np.ndarray = field(default_factory=lambda: np.array([]))
    a_prime_sections: np.ndarray = field(default_factory=lambda: np.array([]))
    cl_sections: np.ndarray = field(default_factory=lambda: np.array([]))
    cd_sections: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_sections: np.ndarray = field(default_factory=lambda: np.array([]))


def solve_bem(
    blade: BladeDesign,
    wind_speed: float,
    tsr: float,
    rho: float = 1.225,
    n_sections: int = 20,
    max_iter: int = 200,
    tol: float = 1e-6,
    nu: float = 1.5e-5,
) -> BEMResult:
    """
    BEM 계산 수행.

    Parameters
    ----------
    blade : BladeDesign
        블레이드 설계
    wind_speed : float
        자유류 풍속 (m/s)
    tsr : float
        주속비 (Tip Speed Ratio = ΩR/V)
    rho : float
        공기 밀도 (kg/m³)
    n_sections : int
        블레이드 분할 수
    max_iter : int
        유도계수 반복 최대 횟수
    tol : float
        수렴 판정 기준

    Returns
    -------
    BEMResult
        계산 결과
    """
    R = blade.radius
    r_hub = blade.hub_radius
    B = blade.num_blades
    omega = tsr * wind_speed / R   # 각속도 (rad/s)
    rpm = omega * 60 / (2 * np.pi)

    # 블레이드 구간 분할 (허브~팁)
    r_sections = np.linspace(r_hub + 0.01, R * 0.99, n_sections)
    dr = r_sections[1] - r_sections[0]

    # 구간별 결과 저장
    dT_sections = np.zeros(n_sections)  # 추력
    dQ_sections = np.zeros(n_sections)  # 토크
    a_arr = np.zeros(n_sections)
    a_prime_arr = np.zeros(n_sections)
    cl_arr = np.zeros(n_sections)
    cd_arr = np.zeros(n_sections)
    alpha_arr = np.zeros(n_sections)

    for i, r in enumerate(r_sections):
        c = blade.chord_at(r)
        theta = blade.twist_at(r)   # 비틀림 + 피치 (degree)
        local_tsr = omega * r / wind_speed  # 국소 주속비

        # 솔리디티
        sigma = B * c / (2 * np.pi * r)

        # 유도계수 초기값
        a = 0.1
        a_prime = 0.01

        for _ in range(max_iter):
            # 유입 각도 (degree)
            if (1 - a) <= 0 or local_tsr * (1 + a_prime) <= 0:
                break
            phi_rad = np.arctan2(
                wind_speed * (1 - a),
                omega * r * (1 + a_prime)
            )
            phi_deg = np.degrees(phi_rad)

            # 받음각
            alpha = phi_deg - theta

            # 상대속도 및 레이놀즈 수 계산
            V_ax = wind_speed * (1 - a)
            V_tan = omega * r * (1 + a_prime)
            V_rel = np.sqrt(V_ax**2 + V_tan**2)
            re_local = V_rel * c / nu

            # 공력계수 (Re 의존)
            cl = cl_alpha(alpha, blade.airfoil, re=re_local)
            cd = cd_alpha(alpha, blade.airfoil, re=re_local)

            # 법선/접선 방향 힘 계수
            cn = cl * np.cos(phi_rad) + cd * np.sin(phi_rad)
            ct = cl * np.sin(phi_rad) - cd * np.cos(phi_rad)

            # 새 유도계수
            sin_phi = np.sin(phi_rad)
            cos_phi = np.cos(phi_rad)

            if sin_phi < 0.01:
                break

            # Glauert 보정 포함
            a_new = 1.0 / (4.0 * sin_phi ** 2 / (sigma * cn) + 1.0)

            # 축방향 유도계수가 너무 크면 Glauert 경험 보정
            if a_new > 0.4:
                # Buhl 보정
                K = 4.0 * sin_phi ** 2 / (sigma * cn)
                discriminant = (K * (1.0 - 2.0 * 0.4) + 2.0) ** 2 + 4.0 * (K * 0.4 ** 2 - 1.0)
                if discriminant < 0:
                    a_new = 0.4
                else:
                    a_new = 0.5 * (2.0 + K * (1.0 - 2.0 * 0.4) - np.sqrt(discriminant))
                a_new = np.clip(a_new, 0, 0.95)

            denom_ap = 4.0 * sin_phi * cos_phi / (sigma * ct) - 1.0
            if abs(denom_ap) < 1e-10:
                a_prime_new = 0.0
            else:
                a_prime_new = 1.0 / denom_ap

            a_prime_new = np.clip(a_prime_new, -0.5, 0.5)

            # 수렴 확인 (언더릴랙세이션)
            relax = 0.3
            a_updated = a + relax * (a_new - a)
            a_prime_updated = a_prime + relax * (a_prime_new - a_prime)

            if abs(a_updated - a) < tol and abs(a_prime_updated - a_prime) < tol:
                a = a_updated
                a_prime = a_prime_updated
                break

            a = a_updated
            a_prime = a_prime_updated

        # 수렴된 값으로 최종 계산
        phi_rad = np.arctan2(
            wind_speed * (1 - a),
            omega * r * (1 + a_prime)
        )
        alpha = np.degrees(phi_rad) - theta
        V_ax = wind_speed * (1 - a)
        V_tan = omega * r * (1 + a_prime)
        V_rel_final = np.sqrt(V_ax**2 + V_tan**2)
        re_local = V_rel_final * c / nu
        cl = cl_alpha(alpha, blade.airfoil, re=re_local)
        cd = cd_alpha(alpha, blade.airfoil, re=re_local)

        # Prandtl 팁 손실 보정
        f_tip = (B / 2) * (R - r) / (r * np.sin(phi_rad) + 1e-10)
        F_tip = (2 / np.pi) * np.arccos(np.clip(np.exp(-f_tip), -1, 1))

        f_hub = (B / 2) * (r - r_hub) / (r * np.sin(phi_rad) + 1e-10)
        F_hub = (2 / np.pi) * np.arccos(np.clip(np.exp(-f_hub), -1, 1))

        F = F_tip * F_hub
        F = max(F, 0.01)

        # 구간별 추력, 토크
        V_rel_sq = (wind_speed * (1 - a)) ** 2 + (omega * r * (1 + a_prime)) ** 2
        cn = cl * np.cos(phi_rad) + cd * np.sin(phi_rad)
        ct_force = cl * np.sin(phi_rad) - cd * np.cos(phi_rad)

        dT = 0.5 * rho * V_rel_sq * B * c * cn * dr
        dQ = 0.5 * rho * V_rel_sq * B * c * ct_force * r * dr

        dT_sections[i] = dT
        dQ_sections[i] = dQ
        a_arr[i] = a
        a_prime_arr[i] = a_prime
        cl_arr[i] = cl
        cd_arr[i] = cd
        alpha_arr[i] = alpha

    # 전체 합산
    total_thrust = np.sum(dT_sections)
    total_torque = np.sum(dQ_sections)
    total_power = total_torque * omega

    # 이론 최대 출력 (Betz 한계 기준이 아닌, 실제 유입 에너지)
    A = np.pi * R ** 2
    P_wind = 0.5 * rho * A * wind_speed ** 3

    cp = total_power / P_wind if P_wind > 0 else 0.0
    ct_total = total_thrust / (0.5 * rho * A * wind_speed ** 2) if wind_speed > 0 else 0.0

    # Cp가 물리적으로 불합리하면 클리핑 (Betz 한계 ≈ 0.593)
    cp = np.clip(cp, 0, 0.6)

    return BEMResult(
        wind_speed=wind_speed,
        rpm=rpm,
        tsr=tsr,
        power=total_power,
        torque=total_torque,
        thrust=total_thrust,
        cp=cp,
        ct=ct_total,
        r_sections=r_sections,
        a_sections=a_arr,
        a_prime_sections=a_prime_arr,
        cl_sections=cl_arr,
        cd_sections=cd_arr,
        alpha_sections=alpha_arr,
    )


def find_optimal_tsr(
    blade: BladeDesign,
    wind_speed: float,
    tsr_range: tuple = (1, 15),
    n_points: int = 50,
    rho: float = 1.225,
) -> BEMResult:
    """
    주어진 풍속에서 Cp를 최대화하는 최적 TSR을 탐색.

    Parameters
    ----------
    blade : BladeDesign
        블레이드 설계
    wind_speed : float
        풍속 (m/s)
    tsr_range : tuple
        TSR 탐색 범위
    n_points : int
        탐색 점 수

    Returns
    -------
    BEMResult
        최적 TSR에서의 결과
    """
    tsrs = np.linspace(tsr_range[0], tsr_range[1], n_points)
    best_result = None
    best_cp = -1

    for tsr in tsrs:
        result = solve_bem(blade, wind_speed, tsr, rho=rho)
        if result.cp > best_cp:
            best_cp = result.cp
            best_result = result

    return best_result


if __name__ == "__main__":
    from airfoil import AIRFOILS

    # 테스트: 표준적인 3-블레이드 터빈
    blade = BladeDesign(
        radius=5.0,
        hub_radius=0.5,
        num_blades=3,
        twist_root=20.0,
        twist_tip=2.0,
        chord_root=0.8,
        chord_tip=0.3,
        airfoil=AIRFOILS["NACA4412"],
    )

    print("=" * 60)
    print("BEM 엔진 테스트 - NACA4412, R=5m, 3-blade")
    print("=" * 60)

    # TSR별 Cp 계산
    print(f"\n{'TSR':>6} {'Cp':>8} {'Power(W)':>12} {'Torque(Nm)':>12}")
    print("-" * 42)
    for tsr in range(2, 13):
        result = solve_bem(blade, wind_speed=10.0, tsr=tsr)
        print(f"{tsr:>6} {result.cp:>8.4f} {result.power:>12.1f} {result.torque:>12.1f}")

    # 최적 TSR 탐색
    print("\n--- 최적 TSR 탐색 ---")
    best = find_optimal_tsr(blade, wind_speed=10.0)
    print(f"최적 TSR: {best.tsr:.2f}")
    print(f"최대 Cp:  {best.cp:.4f}")
    print(f"출력:     {best.power:.1f} W")
    print(f"RPM:      {best.rpm:.1f}")

    # 풍속별 최적 Cp
    print(f"\n{'풍속(m/s)':>10} {'최적TSR':>8} {'Cp':>8} {'Power(W)':>12}")
    print("-" * 42)
    for v in [4, 6, 8, 10, 12, 15, 20, 25]:
        best = find_optimal_tsr(blade, wind_speed=v)
        print(f"{v:>10} {best.tsr:>8.2f} {best.cp:>8.4f} {best.power:>12.1f}")
