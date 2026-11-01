"""
에어포일 공력 데이터 모듈
NACA 4자리 에어포일의 Cl(alpha), Cd(alpha) 계산
레이놀즈 수(Re) 의존성 포함
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class AirfoilParams:
    """NACA 4자리 에어포일 파라미터"""
    max_camber: float       # 최대 캠버 (chord 대비 비율, 0~0.09)
    camber_pos: float       # 최대 캠버 위치 (chord 대비 비율, 0~0.9)
    thickness: float        # 최대 두께 (chord 대비 비율, 0.06~0.24)

    @classmethod
    def from_naca(cls, naca: str) -> "AirfoilParams":
        """NACA 4자리 문자열(예: '4412')에서 파라미터 추출"""
        if len(naca) != 4:
            raise ValueError(f"NACA 4-digit code required: {naca}")
        m = int(naca[0]) / 100
        p = int(naca[1]) / 10
        t = int(naca[2:]) / 100
        return cls(max_camber=m, camber_pos=p, thickness=t)

    @property
    def naca_code(self) -> str:
        m = round(self.max_camber * 100)
        p = round(self.camber_pos * 10)
        t = round(self.thickness * 100)
        return f"{m}{p}{t:02d}"


def _re_factor_cl(re: float) -> float:
    """
    레이놀즈 수에 따른 양력기울기 보정 계수.
    Re가 낮으면 경계층 분리가 빨라져 양력기울기가 감소하고 실속이 일찍 온다.
    Re ~ 1e6 기준 1.0, Re ~ 1e5에서 ~0.85, Re ~ 5e4에서 ~0.70
    """
    if re <= 0:
        return 0.7
    log_re = np.log10(max(re, 1e4))
    # 6.0 (Re=1e6) -> 1.0, 5.0 (Re=1e5) -> 0.85, 4.7 (Re=5e4) -> 0.70
    factor = np.clip(0.15 * log_re - 0.1, 0.5, 1.05)
    return factor


def _re_factor_cd(re: float) -> float:
    """
    레이놀즈 수에 따른 항력 보정 계수.
    Re가 낮으면 마찰항력이 증가한다.
    Re ~ 1e6 기준 1.0, Re ~ 1e5에서 ~1.8, Re ~ 5e4에서 ~2.5
    """
    if re <= 0:
        return 3.0
    log_re = np.log10(max(re, 1e4))
    # 6.0 -> 1.0, 5.0 -> 1.8, 4.0 -> 3.0
    factor = np.clip(-1.0 * log_re + 7.0, 0.8, 3.5)
    return factor


def _stall_angle_re(base_stall: float, re: float) -> float:
    """
    레이놀즈 수에 따른 실속 각도 보정.
    Re가 낮으면 실속이 일찍 발생한다.
    """
    if re <= 0:
        return base_stall * 0.6
    log_re = np.log10(max(re, 1e4))
    # 6.0 -> 1.0, 5.0 -> 0.85, 4.0 -> 0.65
    factor = np.clip(0.175 * log_re - 0.05, 0.5, 1.05)
    return base_stall * factor


def cl_alpha(alpha_deg: float, params: AirfoilParams, re: float = 1e6) -> float:
    """
    받음각(alpha)에서의 양력계수 Cl 계산.
    Re 의존성 포함.
    """
    # 영양력 받음각 (캠버에 의한 보정)
    if params.camber_pos > 0:
        alpha_L0_rad = -2.0 * params.max_camber
    else:
        alpha_L0_rad = 0.0

    alpha_L0_deg = np.degrees(alpha_L0_rad)

    # 양력기울기 (Re 보정 포함)
    re_cl = _re_factor_cl(re)
    cl_slope = 2.0 * np.pi * (1.0 + 0.77 * params.thickness) * re_cl

    # 실속 각도 (Re 보정 포함)
    base_stall = 12.0 + 40.0 * params.thickness
    stall_angle = _stall_angle_re(base_stall, re)

    # 선형 구간
    cl_linear = cl_slope * np.radians(alpha_deg - alpha_L0_deg)

    # 실속 모델
    if isinstance(alpha_deg, np.ndarray):
        cl = np.where(
            np.abs(alpha_deg - alpha_L0_deg) <= stall_angle,
            cl_linear,
            cl_linear * np.exp(-0.1 * (np.abs(alpha_deg - alpha_L0_deg) - stall_angle) ** 1.5)
        )
    else:
        if abs(alpha_deg - alpha_L0_deg) <= stall_angle:
            cl = cl_linear
        else:
            excess = abs(alpha_deg - alpha_L0_deg) - stall_angle
            cl = cl_linear * np.exp(-0.1 * excess ** 1.5)

    return cl


def cd_alpha(alpha_deg: float, params: AirfoilParams, re: float = 1e6) -> float:
    """
    받음각(alpha)에서의 항력계수 Cd 계산.
    Re 의존성 포함.
    """
    # 최소 항력계수 (Re 보정 포함)
    re_cd = _re_factor_cd(re)
    cd_min = (0.006 + 0.1 * params.thickness ** 2) * re_cd

    # 캠버에 의한 최소항력 받음각
    if params.camber_pos > 0:
        alpha_min_drag = np.degrees(params.max_camber * 0.5)
    else:
        alpha_min_drag = 0.0

    delta_alpha = alpha_deg - alpha_min_drag

    # 실속 각도 (Re 보정)
    base_stall = 12.0 + 40.0 * params.thickness
    stall_angle = _stall_angle_re(base_stall, re)

    # 항력 모델
    if isinstance(delta_alpha, np.ndarray):
        cd_profile = cd_min + 0.0001 * delta_alpha ** 2
        cd_stall = np.where(
            np.abs(delta_alpha) > stall_angle,
            0.02 * (np.abs(delta_alpha) - stall_angle) ** 1.5,
            0.0
        )
    else:
        cd_profile = cd_min + 0.0001 * delta_alpha ** 2
        if abs(delta_alpha) > stall_angle:
            cd_stall = 0.02 * (abs(delta_alpha) - stall_angle) ** 1.5
        else:
            cd_stall = 0.0

    return cd_profile + cd_stall


# 자주 쓰는 에어포일 프리셋
AIRFOILS = {
    "NACA0012": AirfoilParams.from_naca("0012"),
    "NACA0015": AirfoilParams.from_naca("0015"),
    "NACA0018": AirfoilParams.from_naca("0018"),
    "NACA0021": AirfoilParams.from_naca("0021"),
    "NACA2412": AirfoilParams.from_naca("2412"),
    "NACA4412": AirfoilParams.from_naca("4412"),
    "NACA4415": AirfoilParams.from_naca("4415"),
    "NACA4418": AirfoilParams.from_naca("4418"),
    "NACA6412": AirfoilParams.from_naca("6412"),
    "NACA6415": AirfoilParams.from_naca("6415"),
}


if __name__ == "__main__":
    params = AIRFOILS["NACA4412"]
    print("NACA4412 - Re dependency test")
    for re in [5e4, 1e5, 5e5, 1e6, 3e6]:
        cl5 = cl_alpha(5, params, re=re)
        cd5 = cd_alpha(5, params, re=re)
        print(f"  Re={re:.0e}: Cl(5)={cl5:.3f}, Cd(5)={cd5:.4f}, L/D={cl5/cd5:.1f}")
