# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:50:16 2026

@author: VK
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# INPUTS (problem data + updated properties)
# =========================================================
T_INF = 25.0                  # °C
H = 100.0                     # W/m^2-K (top and bottom convection)
QPP = 10000.0                 # W/m^2 (chip heat flux)
L = 10e-3                     # m (10 mm side length)

# Updated material/joint data requested
K_AL = 239.0                  # W/m-K  (aluminum)
T_AL = 8e-3                    # m      (8 mm aluminum thickness)
R_EPOXY_AREA = 0.9e-4          # (m^2-K)/W (given "joint resistance" per unit area)
T_EPOXY = 0.02e-3              # m (0.02 mm) (kept for reference; not used since R is given)

# Chip properties for advanced 1D model (edit if you know the real chip thickness)
T_SI = 0.5e-3                  # m
K_SI = 130.0                   # W/m-K (silicon)


# =========================================================
# 1) LUMPED (ISOTHERMAL CHIP) MODEL USING PER-AREA RESISTANCES
# =========================================================
def lumped_model_with_heat_split(
    qpp=QPP,
    T_inf=T_INF,
    h=H,
    R_epoxy_area=R_EPOXY_AREA,
    t_al=T_AL,
    k_al=K_AL
):
    """
    Uses per-area resistances (m^2-K/W), so geometry area cancels.
    Two parallel paths from chip node to ambient:
      - Top convection: R_top_area = 1/h
      - Bottom path:   R_bottom_area = R_epoxy_area + t_al/k_al + 1/h

    Returns:
      T_chip (°C), q_up (W/m^2), q_down (W/m^2), and resistances
    """
    R_top_area = 1.0 / h
    R_bottom_area = R_epoxy_area + (t_al / k_al) + (1.0 / h)

    # Equivalent conductance per area
    G_eq = (1.0 / R_top_area) + (1.0 / R_bottom_area)

    # Chip temperature rise:
    dT = qpp / G_eq
    T_chip = T_inf + dT

    # Heat flux split
    q_up = (T_chip - T_inf) / R_top_area      # = h*(T_chip - T_inf)
    q_down = (T_chip - T_inf) / R_bottom_area # through epoxy + Al + bottom convection

    return {
        "T_chip_C": T_chip,
        "q_up_W_m2": q_up,
        "q_down_W_m2": q_down,
        "R_top_area_m2K_W": R_top_area,
        "R_bottom_area_m2K_W": R_bottom_area,
        "G_eq_W_m2K": G_eq
    }


# =========================================================
# 2) ADVANCED 1D SILICON MODEL WITH UNIFORM HEAT GENERATION
#    (same boundary physics, but allows T(x) in silicon)
# =========================================================
def chip_1d_with_generation(
    qpp=QPP,
    T_inf=T_INF,
    h_top=H,
    h_bottom=H,
    t_si=T_SI,
    k_si=K_SI,
    R_epoxy_area=R_EPOXY_AREA,
    t_al=T_AL,
    k_al=K_AL
):
    """
    x=0 at TOP of silicon, x=t_si at BOTTOM of silicon.
    Downward heat flux qx = -k dT/dx (positive downward).

    Top BC: heat leaving upward = h_top*(T_top - T_inf) = -qx(0)
    Bottom BC: heat leaving downward into stack:
               qx(t) = (T_bottom - T_inf)/Rb_area
      where Rb_area = R_epoxy_area + t_al/k_al + 1/h_bottom
    """
    t = t_si
    k = k_si
    Tinf = T_inf

    # Bottom per-area resistance from silicon bottom surface to ambient
    Rb_area = R_epoxy_area + (t_al / k_al) + (1.0 / h_bottom)

    # Volumetric generation (uniform in silicon)
    qgen = qpp / t  # W/m^3

    # Solution: T(x) = -(qgen/(2k))x^2 + C1 x + C2
    # From top BC: k*C1 = h_top*(C2 - Tinf) => C1 = (h_top/k)*(C2 - Tinf)
    A = h_top / k

    # Solve for D = (C2 - Tinf) using bottom BC:
    # qx(t) = qgen*t - k*C1 = (T(t)-Tinf)/Rb_area
    # This yields a linear equation in D.
    const_right = (-(qgen / (2.0 * k)) * t**2) / Rb_area
    coeff_right = (A * t + 1.0) / Rb_area
    D = (const_right - qgen * t) / (-h_top - coeff_right)

    C2 = Tinf + D
    C1 = A * D

    def T_of_x(x):
        return -(qgen / (2.0 * k)) * x**2 + C1 * x + C2

    T_top = T_of_x(0.0)
    T_bot = T_of_x(t)

    # Heat leaving top and bottom (positive outward from chip)
    q_up = h_top * (T_top - Tinf)
    q_down = (T_bot - Tinf) / Rb_area
    q_sum = q_up + q_down

    # Tmax inside silicon (dT/dx = 0)
    x_star = (k * C1) / qgen
    x_star = min(max(x_star, 0.0), t)
    Tmax = max(T_top, T_bot, T_of_x(x_star))
    candidates = [(0.0, T_top), (t, T_bot), (x_star, T_of_x(x_star))]
    x_Tmax, _ = max(candidates, key=lambda p: p[1])

    return {
        "Tmax_C": Tmax,
        "T_top_C": T_top,
        "T_bot_C": T_bot,
        "x_Tmax_m": x_Tmax,
        "q_up_W_m2": q_up,
        "q_down_W_m2": q_down,
        "q_sum_W_m2": q_sum,
        "Rb_area_m2K_W": Rb_area
    }


# =========================================================
# 3) RUN + PRINT RESULTS + PLOTS
# =========================================================
def run_study_and_plot():
    # --- Single-point results at given epoxy (R is fixed) ---
    lump = lumped_model_with_heat_split()
    adv = chip_1d_with_generation()

    A_chip = L * L
    Q_total_W = QPP * A_chip

    print("===================================================")
    print("UPDATED CASE (k_al = 239 W/m-K, epoxy joint R'' = 0.9e-4 m^2K/W)")
    print(f"Chip area A                = {A_chip:.6e} m^2")
    print(f"Total chip power Q         = {Q_total_W:.4f} W  (from q''={QPP} W/m^2)")
    print(f"Ambient T_inf              = {T_INF:.2f} °C")
    print(f"h (top & bottom)           = {H:.2f} W/m^2-K")
    print(f"Al thickness               = {T_AL*1e3:.2f} mm")
    print(f"Epoxy thickness (given)    = {T_EPOXY*1e3:.3f} mm (not used; R'' is specified)")
    print(f"Epoxy joint resistance R'' = {R_EPOXY_AREA:.3e} m^2-K/W")
    print("---------------------------------------------------")

    print("LUMPED MODEL (chip isothermal):")
    print(f"  T_chip                   = {lump['T_chip_C']:.2f} °C")
    print(f"  Heat up (to air) q_up     = {lump['q_up_W_m2']:.1f} W/m^2")
    print(f"  Heat down (through stack) = {lump['q_down_W_m2']:.1f} W/m^2")
    print(f"  Check: q_up+q_down        = {lump['q_up_W_m2']+lump['q_down_W_m2']:.1f} W/m^2 (should be {QPP:.1f})")
    print(f"  Upward power              = {lump['q_up_W_m2']*A_chip:.4f} W")
    print(f"  Downward power            = {lump['q_down_W_m2']*A_chip:.4f} W")
    print("---------------------------------------------------")

    print("ADVANCED 1D SILICON MODEL (with internal generation):")
    print(f"  Tmax in silicon           = {adv['Tmax_C']:.2f} °C at x = {adv['x_Tmax_m']*1e3:.3f} mm (x=0 top)")
    print(f"  T_top surface             = {adv['T_top_C']:.2f} °C")
    print(f"  T_bottom surface          = {adv['T_bot_C']:.2f} °C")
    print(f"  Heat up (to air) q_up     = {adv['q_up_W_m2']:.1f} W/m^2")
    print(f"  Heat down (through stack) = {adv['q_down_W_m2']:.1f} W/m^2")
    print(f"  Check: q_up+q_down        = {adv['q_sum_W_m2']:.1f} W/m^2 (should be {QPP:.1f})")
    print(f"  Upward power              = {adv['q_up_W_m2']*A_chip:.4f} W")
    print(f"  Downward power            = {adv['q_down_W_m2']*A_chip:.4f} W")
    print("---------------------------------------------------")

    print("ALLOWABLE MAX TEMPERATURE: 85 °C")
    print("PASS" if adv["Tmax_C"] <= 85.0 else "FAIL")
    print("===================================================\n")

    # --- Plot vs epoxy thickness (but note: you gave epoxy resistance, not conductivity) ---
    # We'll interpret this as "vary the epoxy joint resistance per area" over a range.
    R_list = np.linspace(0.2e-4, 3.0e-4, 120)  # m^2K/W
    T_l = np.zeros_like(R_list)
    T_1 = np.zeros_like(R_list)
    dT = np.zeros_like(R_list)

    for i, R_ep in enumerate(R_list):
        T_l[i] = lumped_model_with_heat_split(R_epoxy_area=R_ep)["T_chip_C"]
        T_1[i] = chip_1d_with_generation(R_epoxy_area=R_ep)["Tmax_C"]
        dT[i] = T_1[i] - T_l[i]

    plt.figure()
    plt.plot(R_list * 1e4, T_l, label="Lumped model T_chip")
    plt.plot(R_list * 1e4, T_1, "--", label="1D model Tmax in silicon")
    plt.axhline(85.0, linestyle="--", label="Max allowable (85°C)")
    plt.axvline(R_EPOXY_AREA * 1e4, linestyle=":", label="Given epoxy R''")
    plt.xlabel("Epoxy joint resistance R''  (×10⁻⁴ m²K/W)")
    plt.ylabel("Temperature (°C)")
    plt.title("Chip temperature vs epoxy joint resistance")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(R_list * 1e4, dT)
    plt.axvline(R_EPOXY_AREA * 1e4, linestyle=":")
    plt.xlabel("Epoxy joint resistance R''  (×10⁻⁴ m²K/W)")
    plt.ylabel("ΔT = Tmax(1D) - T(lumped) (°C)")
    plt.title("Difference between 1D and lumped models")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    run_study_and_plot()