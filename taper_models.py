"""
Modelos de afilamento (taper) para Pinus pinaster.

Referências:
  Kozak 1988 — A variable-exponent taper equation.
    Can. J. For. Res. 18: 1363-1368.
  Schumacher modificado — H = 1.30 + a*(1-exp(-b*DBH_cm))^c
"""

import numpy as np


def kozak_1988(q, DBH, H, params):
    """
    Modelo de afilamento Kozak 1988 (variable-exponent taper).

    Devolve raio (m) a altura relativa q = s/H.

    Args:
        q     : float ou array — altura relativa [0, 1]  (s/H ou h/H)
        DBH   : float — diâmetro à altura do peito (m)
        H     : float — altura total do tronco (m)
        params: sequência de 7 floats (a0, a1, a2, b1, b2, b3, b4)

    Returns:
        raio em metros (mesmo shape que q)
    """
    a0, a1, a2, b1, b2, b3, b4 = params
    q = np.asarray(q, dtype=np.float64)
    q = np.clip(q, 1e-4, 0.9999)

    p = 1.30 / H  # altura relativa do DAP

    X_num = 1.0 - np.sqrt(q)
    X_den = 1.0 - np.sqrt(p)
    # Previne divisão por zero (H muito pequeno → p≈1)
    X_den = np.where(np.abs(X_den) < 1e-9, 1e-9, X_den)
    X = X_num / X_den

    c_exp = (b1 * (q ** 2)
             + b2 * np.log(q + 0.001)
             + b3 * np.sqrt(q)
             + b4 * np.exp(q)
             + b1 * (DBH / H))

    # X pode ser negativo perto de q=0; garantir base não-negativa
    X_safe = np.maximum(X, 1e-9)
    d_over_DBH = a0 * (DBH ** a1) * (H ** a2) * np.power(X_safe, c_exp)
    return d_over_DBH * DBH / 2.0  # diâmetro → raio


def potencia_simples(s, r0, H, beta):
    """
    Modelo de afilamento lei-de-potência (fallback monótono).

    r(s) = r0 * (1 - s/H)^beta

    Args:
        s    : float ou array — comprimento de arco a partir da base (m)
        r0   : float — raio na base (m)
        H    : float — altura total (m)
        beta : float — expoente de afilamento (tipicamente 0.5-1.5)

    Returns:
        raio em metros
    """
    q = np.clip(np.asarray(s, dtype=np.float64) / H, 0.0, 1.0)
    return r0 * np.power(1.0 - q, beta)


# Curva de afilamento da espécie (Pinus pinaster), calibrada em 59 árvores
# destrutivas da mesma plantação: d(h)/d0 = P(x), x = h/H, R²=0.9705.
# Normalizada pelo diâmetro na base d0 (h=0). 1 parâmetro livre: k = d0/2 (raio base).
ESPECIE_COEFS = (2.7342, -7.8921, 6.9028, -2.7450, 1.0)  # x^4, x^3, x^2, x, 1


def especie_shape(x):
    """Forma normalizada d(h)/d0 da espécie. x = h/H em [0,1]."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    c4, c3, c2, c1, c0 = ESPECIE_COEFS
    return np.maximum(c4 * x**4 + c3 * x**3 + c2 * x**2 + c1 * x + c0, 0.0)


def especie_taper(s, k, H):
    """Raio (m) a altura s: r(s) = k * P(s/H), com k = d0/2 (raio na base)."""
    return k * especie_shape(np.asarray(s, dtype=np.float64) / max(H, 1e-6))


def altura_alometrica_pn(DBH_m, params=None):
    """
    Estimativa alométrica de altura para Pinus pinaster.

    H = 1.30 + a * (1 - exp(-b * DBH_cm))^c

    Args:
        DBH_m : float — diâmetro à altura do peito (m)
        params: (a, b, c) — defaults de config.PARAMS_ALOMETRIA_Pn

    Returns:
        altura estimada em metros

    Validação (não calibrado):
        DBH=10cm → H≈9m; DBH=20cm → H≈16m; DBH=30cm → H≈22m
    """
    if params is None:
        try:
            from config import PARAMS_ALOMETRIA_Pn
            params = PARAMS_ALOMETRIA_Pn
        except ImportError:
            params = (27.0, 0.038, 1.05)
    a, b, c = params
    DBH_cm = DBH_m * 100.0
    return 1.30 + a * (1.0 - np.exp(-b * DBH_cm)) ** c


def volume_analitico(params, model, H, DBH, step=0.01):
    """
    Calcula volume de tronco por integração trapezoidal do taper.

    Args:
        params: parâmetros do modelo (Kozak ou potência)
        model : "kozak" ou "potencia"
        H     : altura total (m)
        DBH   : diâmetro à altura do peito (m)
        step  : passo de integração (m), default 1 cm

    Returns:
        volume em dm³
    """
    s_grid = np.arange(0.0, H, step)
    if model == "kozak":
        r_grid = kozak_1988(s_grid / H, DBH, H, params)
    elif model == "potencia":
        r0, beta = params[0], params[1]
        r_grid = potencia_simples(s_grid, r0, H, beta)
    elif model == "especie":
        r_grid = especie_taper(s_grid, params[0], H)
    else:
        raise ValueError(f"Modelo desconhecido: {model!r}")

    r_grid = np.maximum(r_grid, 0.0)
    # Clip de segurança: nunca mais de 2.5× o raio DAP (evita overflow com modelos instáveis)
    r_max_clip = max(2.5 * (DBH / 2.0), 1.0)
    r_grid = np.minimum(r_grid, r_max_clip)
    A = np.pi * r_grid ** 2
    V_m3 = float(np.trapezoid(A, s_grid))
    return V_m3 * 1000.0  # → dm³
