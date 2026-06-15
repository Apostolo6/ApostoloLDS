"""
PIPELINE LiDAR v8 — Fase 3: RANSAC Slice + Taper Fitting
=========================================================
Substitui o método whole-stem paramétrico (instável com cobertura baixa) por:
  1. H_leafon do meta.json (altura real da árvore, medida do leafon)
  2. RANSAC circular fatia a fatia no leafoff (toda a altura)
  3. Filtro de continuidade via esqueleto suavizado (centros das fatias)
  4. Ajuste do modelo de potência às fatias fiáveis + constraint DAP + apex
  5. Kozak apenas se zona fiável > 40% da altura e tronco cilíndrico limpo

Input : *_clean.leafoff.ply + *_meta.json  (de fase2_5/trees/)
Output: *_extended.ply        (visualização com pontos sintéticos)
        *_taper_meta.json     (parâmetros do taper para fase4_v8)
        *_perfil_v8.csv       (perfil discreto para inspeção)
"""

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import numpy as np

from config import StatusFase, PARAMS_DEFAULT_Pn
from logging_utils import setup_logger
from taper_models import kozak_1988, potencia_simples, especie_shape, especie_taper


# ---------------------------------------------------------------------------
# Resultado por árvore
# ---------------------------------------------------------------------------

@dataclass
class ResultadoFase3V8:
    status: StatusFase
    tree_id: str = ""
    extended_ply: Optional[Path] = None
    taper_meta_json: Optional[Path] = None
    perfil_csv: Optional[Path] = None
    taper_model: str = ""
    sigma_residuo_mm: float = 0.0
    H_final: float = 0.0
    DBH_cm: float = 0.0
    volume_dm3: float = 0.0
    qualidade: str = ""
    flags: List[str] = field(default_factory=list)
    erro: str = ""


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _ler_ply(path: Path) -> np.ndarray:
    from plyfile import PlyData
    ply = PlyData.read(str(path))
    v = ply['vertex'].data
    x = np.array(v['x'], dtype=np.float64)
    y = np.array(v['y'], dtype=np.float64)
    z = np.array(v['z'], dtype=np.float64)
    return np.column_stack([x, y, z])


def _escrever_ply(path: Path, pts: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement
    arr = np.zeros(len(pts), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    arr['x'] = pts[:, 0].astype(np.float32)
    arr['y'] = pts[:, 1].astype(np.float32)
    arr['z'] = pts[:, 2].astype(np.float32)
    el = PlyElement.describe(arr, 'vertex')
    PlyData([el], text=False).write(str(path))


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class TaperFitV8:
    """Ajuste de taper whole-stem para pipeline v8."""

    # --- RANSAC ---
    SLICE_STEP        = 0.10   # espessura de cada fatia (m)
    MIN_PTS_SLICE     = 10     # pontos mínimos por fatia
    MIN_QUALITY       = 0.55   # fracção inliers mínima
    MIN_ANG_COV       = 0.35   # cobertura angular mínima (0–1)
    RANSAC_N_ITER     = 200    # iterações RANSAC
    RANSAC_INLIER     = 0.10   # tolerância inlier = 10% do raio prior
    RANSAC_R_FAC      = 0.45   # r aceite em [r_prior*0.45, r_prior/0.45]
    RANSAC_CTR_MAX    = 1.5    # max dist centro candidato / r_prior
    RANSAC_BAND_LO    = 0.30   # banda anular inferior (× r_prior) — v8: 0.40→0.30
    RANSAC_BAND_HI    = 3.00   # banda anular superior (× r_prior) — v8: 2.50→3.00
    MAD_FACTOR        = 3.0    # remoção outliers: |resid| > MAD_FACTOR * MAD

    # --- Continuidade ---
    SKEL_ROLL_WIN     = 5      # janela rolling median para suavizar centros
    SKEL_DRIFT_MAX    = 0.06   # desvio máx centro ao esqueleto suavizado (m)

    # --- Modelo ---
    MIN_R             = 0.02   # raio mínimo no vértice (m)
    MIN_RELIABLE      = 3      # nº mínimo de fatias fiáveis para fit

    # Kozak só se:
    KOZAK_FRAC_MIN    = 0.40   # zona fiável > 40% da altura total
    KOZAK_SIGMA_MAX   = 15.0   # mediana sigma RANSAC < 15 mm
    KOZAK_DRIFT_MAX   = 0.05   # desvio máximo dos centros < 5 cm

    # --- Limpeza e completion ---
    TUBE_MARGIN_FRAC = 0.25   # filtrar pontos além de r*(1+FRAC) do eixo do tronco
    N_SECTORS        = 18     # sectores angulares (20° cada) para avaliação de cobertura
    MIN_SECT_PTS     = 2      # sector coberto se >= N pontos reais
    SKEL_EXT_FIT_N   = 5      # nº fatias fiáveis para extrapolação linear acima
    SYNTH_NOISE_R    = 0.003  # ruído radial nos pontos sintéticos (3 mm)
    SYNTH_NOISE_Z    = 0.008  # ruído vertical nos pontos sintéticos (8 mm)

    # --- Calibração do modelo ---
    # Excluir fatias abaixo desta altura do fit taper (chapadas/raízes inflam o RANSAC)
    TAPER_FIT_H_MIN  = 0.70   # m — exclui chapadas/raízes do fit

    # DAP reportado: usar taper model avaliado em h=1.3m (DBH_taper_raw_cm).
    # Calibração derivada de 13 medições destrutivas (ensaio 2026-05-23).
    # Substituiu DBH_REPORT_CALIB=0.838 (baseado em r_DBH directa, alta variância).
    DBH_TAPER_CALIB = 1.027

    # Beta mínimo para árvores com > 50% do tronco extrapolado — força perfil cónico
    # e evita modelos quase cilíndricos que inflam o volume na zona extrapolada.
    BETA_MIN_EXTRAP = 0.20

    # v8: refit geométrico de círculo aos inliers (reduz variância vs estimativa de 3 pts RANSAC)
    USE_ROBUST_CIRCLE = True

    # v8: estratégia de taper.
    #   'auto'             -> kozak/potência como na v8
    #   'especie'          -> curva de forma da espécie (1 param) para todas
    #   'especie_fallback' -> auto, mas troca p/ espécie quando fit livre é instável
    TAPER_STRATEGY = "especie"
    ESPECIE_SIGMA_MAX = 12.0   # mm — acima disto, fit livre considerado instável
    ESPECIE_FRAC_MIN  = 0.45   # frac_extrap acima disto, usar forma de espécie

    # v8: DBH directo da mediana das fatias medidas perto de 1.3m (desacopla DBH
    # do modelo de volume; as fatias têm baixa variância após o refit de círculo).
    DBH_FROM_SLICES = True
    # v8: DBH por ajuste de cone 3D (eixo único) — sobrepõe-se ao linear-local se OK.
    DBH_FROM_CYLINDER = True

    # -----------------------------------------------------------------------

    def __init__(self, config, logger=None):
        self.cfg = config
        self.log = logger or setup_logger("Fase3_v8", level=config.LOG_LEVEL)

    # -----------------------------------------------------------------------
    # Ponto de entrada — pasta inteira
    # -----------------------------------------------------------------------

    def processar(self, trees_dir: Path, out_dir: Path,
                  tree_stats_csv: Path = None) -> object:
        from dataclasses import dataclass as _dc

        @_dc
        class _Res:
            status: StatusFase
            n_troncos: int = 0
            erro: str = ""

        out_dir.mkdir(parents=True, exist_ok=True)
        clean_plys = sorted(trees_dir.glob("*_clean.leafoff.ply"))

        if not clean_plys:
            self.log.warning(f"Nenhum *_clean.leafoff.ply em {trees_dir}")
            return _Res(status=StatusFase.SUCESSO, n_troncos=0)

        resultados = []
        for ply_path in clean_plys:
            meta_path = ply_path.parent / ply_path.name.replace(
                '_clean.leafoff.ply', '_meta.json'
            )
            if not meta_path.exists():
                self.log.warning(f"  Meta JSON não encontrado para {ply_path.name}")
                continue
            r = self.processar_um(ply_path, meta_path, out_dir)
            resultados.append(r)

        n_ok = sum(1 for r in resultados if r.status == StatusFase.SUCESSO)
        self.log.info(f"Fase 3 v8: {n_ok}/{len(resultados)} troncos processados OK")
        return _Res(status=StatusFase.SUCESSO, n_troncos=n_ok)

    # -----------------------------------------------------------------------
    # Processar uma árvore
    # -----------------------------------------------------------------------

    def processar_um(self, clean_ply: Path, meta_input: Path,
                     out_dir: Path) -> ResultadoFase3V8:
        meta = json.loads(meta_input.read_text(encoding="utf-8"))
        nome = meta.get("tree_id", clean_ply.stem.replace('_clean.leafoff', ''))
        self.log.info(f"  Fase 3 v8: {clean_ply.name}")

        # --- A: Carregar pontos e metadados ---
        try:
            pts = _ler_ply(clean_ply)
        except Exception as e:
            return ResultadoFase3V8(status=StatusFase.ERRO, tree_id=nome,
                                    erro=f"Erro ao ler PLY: {e}")

        if len(pts) < 50:
            return ResultadoFase3V8(status=StatusFase.ERRO, tree_id=nome,
                                    erro="Pontos insuficientes")

        z_base   = float(meta["z_base"])
        r_DBH    = float(meta["r_DBH"])
        DBH      = 2.0 * r_DBH
        H_leafon = float(meta.get("H_leafon", 0.0))
        h_local  = pts[:, 2] - z_base

        if H_leafon <= 1.0:
            H_leafon = float(h_local.max()) + 0.30
            self.log.warning(f"    H_leafon indisponível — usando {H_leafon:.1f} m")

        # --- B: RANSAC slice a slice ---
        all_slices = self._medir_perfil_ransac(pts, h_local, r_DBH, H_leafon)
        n_medidas  = sum(1 for s in all_slices if s is not None)

        # --- C: Filtro de continuidade ---
        reliable = self._filtrar_continuidade(all_slices)
        n_fiav   = len(reliable)
        self.log.info(
            f"    Fatias medidas={n_medidas}  fiáveis={n_fiav}"
        )

        # --- D/E: Decidir e ajustar modelo ---
        # reliable_fit: exclui chapadas/raízes (h < TAPER_FIT_H_MIN) do ajuste do modelo
        # reliable: mantido intacto para o esqueleto e filtro tubo
        reliable_fit = [s for s in reliable if s['h'] >= self.TAPER_FIT_H_MIN]
        if len(reliable_fit) < self.MIN_RELIABLE:
            reliable_fit = reliable  # fallback: usar todas se muito poucas fatias acima

        if n_fiav == 0:
            params, model, sigma_mm, flags = self._fallback_analitico(r_DBH, H_leafon)
            self.log.warning("    Fallback analítico (sem fatias fiáveis)")
        else:
            h_max_fiav   = max(s['h'] for s in reliable)
            frac_fiav    = h_max_fiav / H_leafon
            med_sigma    = float(np.median([s['sigma_mm'] for s in reliable]))
            max_drift    = float(max(s.get('drift', 0.0) for s in reliable))

            self.log.info(
                f"    h_max_fiav={h_max_fiav:.1f} m  frac={frac_fiav:.2f}"
                f"  sigma_med={med_sigma:.1f} mm  drift_max={max_drift*100:.1f} cm"
                f"  fatias_fit={len(reliable_fit)}"
            )

            frac_extrap_fit = max(0.0, H_leafon - h_max_fiav) / max(H_leafon, 1.0)

            if self.TAPER_STRATEGY == "especie":
                params, model, sigma_mm, flags = self._ajustar_especie(
                    reliable_fit, r_DBH, H_leafon
                )
            else:
                usar_kozak = (
                    frac_fiav >= self.KOZAK_FRAC_MIN
                    and med_sigma <= self.KOZAK_SIGMA_MAX
                    and max_drift <= self.KOZAK_DRIFT_MAX
                )
                if usar_kozak:
                    params, model, sigma_mm, flags = self._ajustar_kozak(
                        reliable_fit, r_DBH, H_leafon
                    )
                else:
                    params, model, sigma_mm, flags = self._ajustar_potencia(
                        reliable_fit, r_DBH, H_leafon
                    )

                # Fallback para forma de espécie quando o fit livre é instável
                # (sigma alto ou muita extrapolação) — evita perfis descontrolados.
                if (self.TAPER_STRATEGY == "especie_fallback"
                        and (sigma_mm > self.ESPECIE_SIGMA_MAX
                             or frac_extrap_fit > self.ESPECIE_FRAC_MIN)):
                    p_e, m_e, s_e, fl_e = self._ajustar_especie(
                        reliable_fit, r_DBH, H_leafon
                    )
                    params, model, sigma_mm, flags = p_e, m_e, s_e, fl_e

        self.log.info(f"    Modelo={model}  sigma={sigma_mm:.1f} mm")

        # --- DBH via modelo de taper avaliado em h=1.3m ---
        if model == "potencia" and len(params) >= 2:
            r0_t, beta_t = float(params[0]), float(params[1])
            r_at_1p3 = r0_t * max(1.0 - 1.3 / max(H_leafon, 2.0), 1e-9) ** beta_t
            DBH_taper_cm = r_at_1p3 * 200.0
        elif model == "kozak" and len(params) >= 4:
            q_1p3 = 1.3 / max(H_leafon, 2.0)
            r_at_1p3 = float(kozak_1988(np.array([q_1p3]), DBH, H_leafon, params)[0])
            DBH_taper_cm = r_at_1p3 * 200.0
        elif model == "especie" and len(params) >= 1:
            r_at_1p3 = float(especie_taper(np.array([1.3]), params[0], H_leafon)[0])
            DBH_taper_cm = r_at_1p3 * 200.0
        else:
            DBH_taper_cm = float(DBH * 100.0)

        # v8: DBH directo, desacoplado do modelo de volume, como MEDIANA robusta de
        # até 3 estimadores independentes — cada um forte num regime, a mediana
        # descarta o que dispara (ex.: cone em troncos divididos, mediana em arcos
        # parciais): (1) ajuste linear local do perfil [0.9,2.5]m, (2) mediana das
        # fatias perto de 1.3m, (3) ajuste de cone 3D (eixo único da banda).
        if self.DBH_FROM_SLICES and reliable:
            cands = []
            r_ll = self._dbh_local_linear(reliable)
            if r_ll is not None and r_ll > 0:
                cands.append(r_ll)
            near = [s['r'] for s in reliable if 1.15 <= s['h'] <= 1.45]
            if len(near) == 0:
                near = [s['r'] for s in reliable if 1.0 <= s['h'] <= 1.6]
            if len(near) >= 1:
                cands.append(float(np.median(near)))
            if self.DBH_FROM_CYLINDER:
                cxp = float(meta.get("cx_DBH", float(pts[:, 0].mean())))
                cyp = float(meta.get("cy_DBH", float(pts[:, 1].mean())))
                r_cyl = self._dbh_cone(pts, h_local, cxp, cyp, r_DBH)
                if r_cyl is not None and 0.4 * r_DBH < r_cyl < 2.5 * r_DBH:
                    cands.append(r_cyl)
            if cands:
                DBH_taper_cm = float(np.median(cands)) * 200.0

        # --- Qualidade e flags ---
        if reliable:
            h_max_fiav   = max(s['h'] for s in reliable)
            frac_extrap  = max(0.0, H_leafon - h_max_fiav) / max(H_leafon, 1.0)
        else:
            frac_extrap  = 1.0

        if frac_extrap > 0.50:
            flags.append("extrapolacao_alta")
        if model == "potencia":
            flags.append("taper_potencia")

        cobertura_DBH = float(meta.get("cobertura_DBH", 0.0))
        qualidade = self._avaliar_qualidade(cobertura_DBH, frac_extrap, sigma_mm / 1000.0)

        cx_DBH = float(meta.get("cx_DBH", float(pts[:, 0].mean())))
        cy_DBH = float(meta.get("cy_DBH", float(pts[:, 1].mean())))

        # --- E.1: Esqueleto interpolado (linha de centro do tronco h=0..H) ---
        skel_h, skel_cx, skel_cy = self._construir_esqueleto(
            reliable, H_leafon, cx_DBH, cy_DBH
        )

        # --- E.2: Filtrar pontos fora do tubo (ramos, arbustos residuais) ---
        pts_limpos = self._filtrar_pontos_tronco(
            pts, h_local, skel_h, skel_cx, skel_cy, params, model, H_leafon, DBH
        )
        n_rem = len(pts) - len(pts_limpos)
        if n_rem > 0:
            self.log.info(f"    Filtro tubo: -{n_rem} pts ({len(pts_limpos)} mantidos)")
        h_local_limpos = pts_limpos[:, 2] - z_base

        # --- E.3: Completar secções com gaps angulares + extensão base/topo ---
        pts_sint = self._completar_cross_sections(
            pts_limpos, h_local_limpos, skel_h, skel_cx, skel_cy,
            params, model, H_leafon, DBH, z_base
        )
        self.log.info(f"    Pontos sintéticos: {len(pts_sint)}")

        pts_final = np.vstack([pts_limpos, pts_sint]) if len(pts_sint) > 0 else pts_limpos

        # --- Escrever outputs ---
        stem      = nome
        ext_path  = out_dir / f"{stem}_extended.ply"
        meta_path = out_dir / f"{stem}_taper_meta.json"
        perf_path = out_dir / f"{stem}_perfil_v8.csv"

        try:
            _escrever_ply(ext_path, pts_final)
        except Exception as e:
            self.log.error(f"    Erro ao escrever extended PLY: {e}")

        s_top_fiav = float(max((s['h'] for s in reliable), default=0.0))

        taper_meta = {
            "tree_id":              nome,
            "taper_model":          model,
            "params":               list(map(float, params)),
            "H_final":              float(H_leafon),
            "H_medido":             float(H_leafon),
            "H_alometrico":         0.0,
            "fonte_H":              "leafon",
            "DBH_inicial":          float(DBH),
            "r_DBH":                float(r_DBH),
            "DBH_taper_raw_cm":     round(DBH_taper_cm, 2),
            "sigma_residuo_mm":     float(sigma_mm),
            "s_top_fiavel":         s_top_fiav,
            "fraccao_extrapolada":  float(frac_extrap),
            "qualidade":            qualidade,
            "flags":                flags,
            "irls_history":         [],
            "z_base":               float(z_base),
            "cx_DBH":               float(cx_DBH),
            "cy_DBH":               float(cy_DBH),
            "cobertura_DBH":        float(cobertura_DBH),
            "n_centroides_validos": n_fiav,
            "L_total_spline":       float(H_leafon),
            "H_leafon":             float(H_leafon),
            "skeleton_h":           [round(float(x), 3) for x in skel_h],
            "skeleton_cx":          [round(float(x), 5) for x in skel_cx],
            "skeleton_cy":          [round(float(x), 5) for x in skel_cy],
        }
        meta_path.write_text(json.dumps(taper_meta, indent=2), encoding="utf-8")
        self._escrever_perfil_csv(perf_path, params, model, H_leafon, DBH)

        self.log.info(f"    qualidade={qualidade}  flags={flags}")

        return ResultadoFase3V8(
            status=StatusFase.SUCESSO,
            tree_id=nome,
            extended_ply=ext_path,
            taper_meta_json=meta_path,
            perfil_csv=perf_path,
            taper_model=model,
            sigma_residuo_mm=float(sigma_mm),
            H_final=float(H_leafon),
            DBH_cm=float(DBH_taper_cm * self.DBH_TAPER_CALIB),
            qualidade=qualidade,
            flags=flags,
        )

    # -----------------------------------------------------------------------
    # B — RANSAC slice a slice
    # -----------------------------------------------------------------------

    def _medir_perfil_ransac(self, pts, h_local, r_DBH, H_leafon):
        """Mede raio e centro por RANSAC em fatias de SLICE_STEP ao longo de toda a altura."""
        h_max      = min(float(h_local.max()), H_leafon * 1.05)
        h_centers  = np.arange(self.SLICE_STEP / 2, h_max, self.SLICE_STEP)

        # Prior sequencial: começa com o centróide global e r_DBH
        seq_cx = float(pts[:, 0].mean())
        seq_cy = float(pts[:, 1].mean())
        seq_r  = r_DBH

        slices = []
        for hc in h_centers:
            h0 = hc - self.SLICE_STEP / 2
            h1 = hc + self.SLICE_STEP / 2
            mask = (h_local >= h0) & (h_local < h1)
            n = int(mask.sum())

            if n < self.MIN_PTS_SLICE:
                slices.append(None)
                continue

            xi = pts[mask, 0]
            yi = pts[mask, 1]

            res = self._ajustar_circulo(xi, yi, seq_cx, seq_cy, seq_r)
            if res is None:
                slices.append(None)
                continue

            cx, cy, r, quality, sigma_mm, ang_cov = res

            slices.append({
                'h':        float(hc),
                'r':        float(r),
                'cx':       float(cx),
                'cy':       float(cy),
                'quality':  float(quality),
                'sigma_mm': float(sigma_mm),
                'ang_cov':  float(ang_cov),
                'n_pts':    int(n),
            })

            # Actualizar prior para a próxima fatia
            seq_cx, seq_cy, seq_r = cx, cy, r

        return slices

    def _ajustar_circulo(self, xi, yi, ecx, ecy, er):
        """
        Ajuste RANSAC de círculo com prior (ecx, ecy, er).
        Devolve (cx, cy, r, quality, sigma_mm, ang_cov) ou None.
        """
        n = len(xi)
        if n < 3:
            return None

        # Filtro de banda anular [BAND_LO*r, BAND_HI*r]
        d_prior = np.sqrt((xi - ecx) ** 2 + (yi - ecy) ** 2)
        band    = (d_prior >= er * self.RANSAC_BAND_LO) & (d_prior <= er * self.RANSAC_BAND_HI)
        xi_f    = xi[band]
        yi_f    = yi[band]
        nf      = len(xi_f)
        if nf < 3:
            xi_f, yi_f, nf = xi, yi, n

        inlier_tol  = er * self.RANSAC_INLIER
        r_min       = er * self.RANSAC_R_FAC
        r_max       = er / self.RANSAC_R_FAC
        ctr_max     = er * self.RANSAC_CTR_MAX
        pts_f       = np.column_stack([xi_f, yi_f])

        cx, cy, r = ecx, ecy, er
        best_score = 0
        rng = np.random.default_rng(seed=int(abs(ecx * 1000) % (2 ** 31)))

        for _ in range(self.RANSAC_N_ITER):
            idx  = rng.choice(nf, 3, replace=False)
            res3 = self._circle_from_3pts(pts_f[idx])
            if res3 is None:
                continue
            cx3, cy3, r3 = res3
            if not (r_min <= r3 <= r_max):
                continue
            if np.sqrt((cx3 - ecx) ** 2 + (cy3 - ecy) ** 2) > ctr_max:
                continue
            d3    = np.sqrt((xi_f - cx3) ** 2 + (yi_f - cy3) ** 2)
            score = int(np.sum(np.abs(d3 - r3) < inlier_tol))
            if score > best_score:
                best_score = score
                cx, cy, r  = cx3, cy3, r3

        # Remoção de outliers por MAD (usando o círculo RANSAC de 3 pontos)
        dists  = np.sqrt((xi - cx) ** 2 + (yi - cy) ** 2)
        resid  = np.abs(dists - r)
        mad    = max(float(np.median(resid)), 1e-6)
        inliers = resid <= self.MAD_FACTOR * mad
        n_in   = int(np.sum(inliers))
        if n_in < 3:
            return None

        # v8: refit geométrico (Gauss-Newton) aos inliers. O RANSAC dá só o melhor
        # círculo de 3 pontos (alta variância em arcos parciais); o refit usa todos
        # os inliers e reduz a dispersão do raio por fatia.
        if self.USE_ROBUST_CIRCLE and n_in >= 5:
            fc = self._fit_circle_geom(xi[inliers], yi[inliers], cx, cy)
            if fc is not None:
                cx_f, cy_f, r_f = fc
                if (r_min <= r_f <= r_max
                        and math.hypot(cx_f - ecx, cy_f - ecy) <= ctr_max):
                    cx, cy, r = cx_f, cy_f, r_f
                    dists = np.sqrt((xi - cx) ** 2 + (yi - cy) ** 2)
                    resid = np.abs(dists - r)
                    mad = max(float(np.median(resid)), 1e-6)
                    inliers = resid <= self.MAD_FACTOR * mad
                    n_in = int(np.sum(inliers))
                    if n_in < 3:
                        return None

        sigma_mm = float(np.std(dists[inliers] - r)) * 1000.0
        quality  = float(n_in) / float(n)

        # Cobertura angular (12 sectores de 30°)
        phi     = np.arctan2(yi[inliers] - cy, xi[inliers] - cx)
        sectors = np.floor((phi + np.pi) / (np.pi / 6)).astype(int) % 12
        counts  = np.bincount(sectors, minlength=12)
        ang_cov = float((counts >= 2).sum()) / 12.0

        return cx, cy, r, quality, sigma_mm, ang_cov

    @staticmethod
    def _fit_circle_geom(x, y, cx0, cy0, iters=10):
        """
        Ajuste geométrico de círculo (Gauss-Newton reduzido) que minimiza
        sum_i (sqrt((xi-a)^2+(yi-b)^2) - R)^2, com R = média das distâncias.
        Init no centro (cx0, cy0). Devolve (cx, cy, r) ou None.
        Muito menos enviesado/ruidoso que um círculo de 3 pontos em arcos parciais.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        a, b = float(cx0), float(cy0)
        for _ in range(iters):
            dx = x - a
            dy = y - b
            di = np.sqrt(dx * dx + dy * dy)
            di = np.where(di < 1e-9, 1e-9, di)
            R = float(di.mean())
            ux = dx / di
            uy = dy / di
            f = di - R
            # Jacobiano reduzido (R depende de a,b via média): df/da = -ux + mean(ux)
            Ja = -ux + ux.mean()
            Jb = -uy + uy.mean()
            J = np.column_stack([Ja, Jb])
            try:
                delta, *_ = np.linalg.lstsq(J, -f, rcond=None)
            except Exception:
                break
            a += float(delta[0])
            b += float(delta[1])
            if abs(delta[0]) + abs(delta[1]) < 1e-7:
                break
        dx = x - a
        dy = y - b
        R = float(np.sqrt(dx * dx + dy * dy).mean())
        if not np.isfinite(R) or R <= 0:
            return None
        return a, b, R

    @staticmethod
    def _circle_from_3pts(pts):
        """Círculo pelos 3 pontos. Devolve (cx, cy, r) ou None."""
        ax, ay = pts[0, 0], pts[0, 1]
        bx, by = pts[1, 0], pts[1, 1]
        cx_, cy_ = pts[2, 0], pts[2, 1]
        D = 2.0 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
        if abs(D) < 1e-12:
            return None
        ux = ((ax**2 + ay**2)*(by - cy_) + (bx**2 + by**2)*(cy_ - ay) + (cx_**2 + cy_**2)*(ay - by)) / D
        uy = ((ax**2 + ay**2)*(cx_ - bx) + (bx**2 + by**2)*(ax - cx_) + (cx_**2 + cy_**2)*(bx - ax)) / D
        r  = float(np.sqrt((ax - ux)**2 + (ay - uy)**2))
        return float(ux), float(uy), r

    # -----------------------------------------------------------------------
    # C — Filtro de continuidade do esqueleto
    # -----------------------------------------------------------------------

    def _filtrar_continuidade(self, all_slices):
        """
        1. Filtro geométrico: quality e cobertura angular.
        2. Filtro de continuidade: centro da fatia não pode desviar muito
           do esqueleto suavizado (rolling median dos centros).
        Devolve lista das fatias fiáveis.
        """
        # Passo 1: filtro geométrico
        geom_ok = [s for s in all_slices
                   if s is not None
                   and s['quality']  >= self.MIN_QUALITY
                   and s['ang_cov']  >= self.MIN_ANG_COV]

        if len(geom_ok) < 2:
            return geom_ok

        # Passo 2: suavizar centros via rolling median
        cx_arr = np.array([s['cx'] for s in geom_ok])
        cy_arr = np.array([s['cy'] for s in geom_ok])
        win    = self.SKEL_ROLL_WIN
        cx_sm  = np.array([np.median(cx_arr[max(0, i-win):i+win+1]) for i in range(len(cx_arr))])
        cy_sm  = np.array([np.median(cy_arr[max(0, i-win):i+win+1]) for i in range(len(cy_arr))])

        # Passo 3: rejeitar fatias com desvio > SKEL_DRIFT_MAX
        reliable = []
        for i, s in enumerate(geom_ok):
            drift = float(np.sqrt((s['cx'] - cx_sm[i])**2 + (s['cy'] - cy_sm[i])**2))
            s_copy = dict(s)
            s_copy['drift'] = drift
            if drift <= self.SKEL_DRIFT_MAX:
                reliable.append(s_copy)

        return reliable

    # -----------------------------------------------------------------------
    # E — Modelo de potência
    # -----------------------------------------------------------------------

    def _ajustar_potencia(self, reliable, r_DBH, H):
        """r(s) = r0 * (1 - s/H)^beta, com constraint DAP e apex."""
        from scipy.optimize import least_squares

        flags = []
        beta0 = self._estimar_beta(r_DBH, H)
        r0_0  = r_DBH / max((1.0 - 1.3 / H) ** beta0, 1e-9)

        if len(reliable) >= self.MIN_RELIABLE:
            h_arr = np.array([s['h'] for s in reliable], dtype=np.float64)
            r_arr = np.array([s['r'] for s in reliable], dtype=np.float64)

            frac_extrap_fit = max(0.0, H - float(h_arr.max())) / max(H, 1.0)
            beta_lower = self.BETA_MIN_EXTRAP if frac_extrap_fit > 0.50 else 0.05

            def residuals(p):
                r0, beta = p
                r_pred = r0 * np.maximum(1.0 - h_arr / H, 1e-9) ** max(beta, beta_lower)
                return r_arr - r_pred

            try:
                res  = least_squares(
                    residuals, [r0_0, beta0],
                    bounds=([0.01, beta_lower], [5.0, 5.0]),
                    loss='huber', f_scale=0.02, max_nfev=500,
                )
                r0, beta = float(res.x[0]), float(res.x[1])
            except Exception as e:
                self.log.warning(f"    Fit potência falhou ({e}) — fallback analítico")
                r0, beta = r0_0, beta0
                flags.append("fallback_analitico")

            r_pred  = r0 * np.maximum(1.0 - h_arr / H, 1e-9) ** beta
            sigma_mm = float(np.median(np.abs(r_arr - r_pred))) * 1000.0 * 1.4826
        else:
            r0, beta = r0_0, beta0
            sigma_mm = 0.0
            flags.append("fallback_analitico")

        return np.array([r0, beta]), "potencia", sigma_mm, flags

    def _dbh_cone(self, pts, h_local, cx0, cy0, r_DBH):
        """
        DBH por ajuste de cone 3D aos pontos da banda [0.9, 2.5]m.
        Modelo: eixo do tronco (x0,y0 a h=1.3 + inclinação ax,ay por metro) e raio
        linear r(h) = rr + kk*(h-1.3). Ajusta TODOS os pontos da banda em simultâneo
        (Huber), pelo que arcos parciais partilham o eixo — menos viés que círculos
        por fatia. Devolve raio a 1.3m (m) ou None.
        """
        from scipy.optimize import least_squares
        href = 1.3
        bm = (h_local >= 0.9) & (h_local <= 2.5)
        if int(bm.sum()) < 40:
            return None
        xb = pts[bm, 0]; yb = pts[bm, 1]; hb = h_local[bm]
        # pré-filtro: remover ramos/arbustos longe do eixo prior
        dpre = np.sqrt((xb - cx0) ** 2 + (yb - cy0) ** 2)
        keep = dpre < 3.0 * r_DBH
        if int(keep.sum()) < 40:
            keep = dpre < 5.0 * r_DBH
            if int(keep.sum()) < 40:
                return None
        xb, yb, hb = xb[keep], yb[keep], hb[keep]

        def resid(p):
            x0, y0, ax, ay, rr, kk = p
            cx = x0 + ax * (hb - href)
            cy = y0 + ay * (hb - href)
            d = np.sqrt((xb - cx) ** 2 + (yb - cy) ** 2)
            return d - (rr + kk * (hb - href))

        p0 = [cx0, cy0, 0.0, 0.0, float(r_DBH), -0.01]
        try:
            res = least_squares(resid, p0, loss='huber', f_scale=0.01, max_nfev=300)
        except Exception:
            return None
        rr = float(res.x[4])
        if not np.isfinite(rr) or rr <= 0:
            return None
        # 2ª passagem: refit só nos inliers (remove ramos residuais)
        r_full = resid(res.x)
        mad = max(float(np.median(np.abs(r_full))), 1e-4)
        inl = np.abs(r_full) <= 3.0 * mad
        if inl.sum() >= 40 and not inl.all():
            xb, yb, hb = xb[inl], yb[inl], hb[inl]
            try:
                res = least_squares(resid, res.x, loss='huber', f_scale=0.01, max_nfev=300)
                rr = float(res.x[4])
            except Exception:
                pass
        return rr if (np.isfinite(rr) and rr > 0) else None

    def _dbh_local_linear(self, reliable):
        """
        DBH (raio a 1.3m) por ajuste linear robusto do perfil de raio nas fatias
        fiáveis em [0.9, 2.5]m. Usa muitas fatias e o afilamento ~linear nessa zona
        para baixar a variância vs mediana de poucas fatias. Devolve raio (m) ou None.
        """
        # Preferir fatias com boa cobertura angular (arcos parciais enviesam o raio).
        band = [s for s in reliable if 0.9 <= s['h'] <= 2.5]
        bons = [s for s in band if s.get('ang_cov', 0.0) >= 0.50]
        usar = bons if len(bons) >= 4 else band
        sl = [(s['h'], s['r']) for s in usar]
        if len(sl) < 4:
            return None
        h = np.array([x[0] for x in sl], dtype=np.float64)
        r = np.array([x[1] for x in sl], dtype=np.float64)
        for _ in range(2):
            if len(h) < 4:
                break
            A = np.polyfit(h, r, 1)
            res = np.abs(r - np.polyval(A, h))
            mad = max(float(np.median(res)), 1e-6)
            keep = res <= 3.0 * mad
            if keep.sum() < 4 or keep.all():
                break
            h, r = h[keep], r[keep]
        A = np.polyfit(h, r, 1)
        return float(np.polyval(A, 1.3))

    def _ajustar_especie(self, reliable, r_DBH, H):
        """
        Ajusta a curva de forma da espécie (1 parâmetro: k = d0/2, raio na base).
        r(h) = k * P(h/H). O escalar k é estimado de forma robusta como a MEDIANA
        de r_i / P(x_i) sobre as fatias fiáveis — imune a fatias contaminadas, ao
        contrário do fit livre de potência/kozak.
        """
        if len(reliable) >= 1:
            h_arr = np.array([s['h'] for s in reliable], dtype=np.float64)
            r_arr = np.array([s['r'] for s in reliable], dtype=np.float64)
            P = especie_shape(h_arr / H)
            ok = P > 1e-3
            if ok.sum() >= 1:
                k = float(np.median(r_arr[ok] / P[ok]))
            else:
                k = float(r_DBH / max(float(especie_shape(np.array([1.3 / H]))[0]), 1e-3))
            r_pred = k * especie_shape(h_arr / H)
            sigma_mm = float(np.median(np.abs(r_arr - r_pred))) * 1000.0 * 1.4826
        else:
            k = float(r_DBH / max(float(especie_shape(np.array([1.3 / H]))[0]), 1e-3))
            sigma_mm = 0.0
        return np.array([k]), "especie", sigma_mm, ["taper_especie"]

    def _estimar_beta(self, r_DBH, H):
        """Estima beta analiticamente: r(1.3)=r_DBH, r(H)=MIN_R."""
        denom = 1.0 - 1.3 / max(H, 2.0)
        if denom <= 0:
            return 0.7
        try:
            beta = math.log(self.MIN_R / max(r_DBH, 0.01)) / math.log(max(denom, 1e-9))
            return float(np.clip(beta, 0.20, 4.0))
        except Exception:
            return 0.7

    # -----------------------------------------------------------------------
    # F — Kozak (só se critérios de cobertura satisfeitos)
    # -----------------------------------------------------------------------

    def _ajustar_kozak(self, reliable, r_DBH, H):
        """Tenta Kozak; fallback para potência se divergir ou não monótono."""
        from scipy.optimize import least_squares

        h_arr = np.array([s['h'] for s in reliable], dtype=np.float64)
        r_arr = np.array([s['r'] for s in reliable], dtype=np.float64)
        q_arr = h_arr / H

        DBH   = 2.0 * r_DBH

        params = np.array(PARAMS_DEFAULT_Pn, dtype=np.float64)

        def residuals(p):
            return r_arr - kozak_1988(q_arr, DBH, H, p)

        try:
            res      = least_squares(residuals, params,
                                     loss='huber', f_scale=0.02, max_nfev=600)
            params_k = res.x
        except Exception as e:
            self.log.warning(f"    Kozak falhou: {e} — potência")
            return self._ajustar_potencia(reliable, r_DBH, H)

        # Verificar monotonicidade
        q_test = np.linspace(0.05, 0.95, 100)
        r_test = kozak_1988(q_test, DBH, H, params_k)
        if float((np.diff(r_test) > 0).sum()) / 99.0 > 0.05:
            self.log.warning("    Kozak não monótono — potência")
            return self._ajustar_potencia(reliable, r_DBH, H)

        sigma_mm = float(np.median(np.abs(
            r_arr - kozak_1988(q_arr, DBH, H, params_k)
        ))) * 1000.0 * 1.4826

        return params_k, "kozak", sigma_mm, []

    # -----------------------------------------------------------------------
    # Fallback analítico (sem fatias fiáveis)
    # -----------------------------------------------------------------------

    def _fallback_analitico(self, r_DBH, H):
        beta   = self._estimar_beta(r_DBH, H)
        r0     = r_DBH / max((1.0 - 1.3 / H) ** beta, 1e-9)
        return np.array([r0, beta]), "potencia", 0.0, ["fallback_analitico"]

    # -----------------------------------------------------------------------
    # Qualidade
    # -----------------------------------------------------------------------

    def _avaliar_qualidade(self, cobertura_DBH, frac_extrap, sigma_m):
        crit1 = cobertura_DBH > 0.5
        crit2 = frac_extrap < 0.25
        crit3 = sigma_m < 0.015
        n_ok  = sum([crit1, crit2, crit3])
        if n_ok == 3:
            return "alta"
        elif n_ok == 2:
            return "media"
        else:
            return "baixa"

    # -----------------------------------------------------------------------
    # Utilitário: raio esperado a altura h
    # -----------------------------------------------------------------------

    def _r_at_h(self, h, params, model, H, DBH):
        """Raio esperado pelo taper a altura h (escalar ou array NumPy)."""
        h_arr = np.asarray(h, dtype=np.float64)
        if model == "kozak":
            q = np.clip(h_arr / max(H, 0.01), 1e-4, 0.9999)
            r = kozak_1988(q, DBH, H, params)
        elif model == "especie":
            r = especie_taper(h_arr, params[0], H)
        else:
            r = potencia_simples(h_arr, params[0], H, params[1])
        return np.maximum(np.asarray(r, dtype=np.float64), self.MIN_R)

    # -----------------------------------------------------------------------
    # E.1 — Construir esqueleto interpolado (linha de centro h=0..H)
    # -----------------------------------------------------------------------

    def _construir_esqueleto(self, reliable, H, cx_base, cy_base):
        """
        Interpola os centros RANSAC para toda a gama [0, H].
        Ancora em (h=0, cx_base, cy_base) e extrapola linearmente acima.
        Devolve (h_arr, cx_arr, cy_arr) para uso com np.interp.
        """
        if not reliable:
            return (np.array([0.0, H], dtype=np.float64),
                    np.array([cx_base, cx_base], dtype=np.float64),
                    np.array([cy_base, cy_base], dtype=np.float64))

        h_arr  = np.array([s['h']  for s in reliable], dtype=np.float64)
        cx_arr = np.array([s['cx'] for s in reliable], dtype=np.float64)
        cy_arr = np.array([s['cy'] for s in reliable], dtype=np.float64)

        # Âncora na base (h=0) com o centro DAP se não há dados abaixo de 0.5m
        if h_arr[0] > 0.5:
            h_arr  = np.concatenate([[0.0], h_arr])
            cx_arr = np.concatenate([[cx_base], cx_arr])
            cy_arr = np.concatenate([[cy_base], cy_arr])

        # Extrapolar acima do último slice fiável até H
        if h_arr[-1] < H - 0.5:
            n_fit = min(self.SKEL_EXT_FIT_N, len(h_arr))
            hs_fit = h_arr[-n_fit:]
            if n_fit >= 2:
                px = np.polyfit(hs_fit, cx_arr[-n_fit:], 1)
                py = np.polyfit(hs_fit, cy_arr[-n_fit:], 1)
                cx_H = float(np.polyval(px, H))
                cy_H = float(np.polyval(py, H))
            else:
                cx_H, cy_H = float(cx_arr[-1]), float(cy_arr[-1])
            h_arr  = np.concatenate([h_arr,  [H]])
            cx_arr = np.concatenate([cx_arr, [cx_H]])
            cy_arr = np.concatenate([cy_arr, [cy_H]])

        return h_arr, cx_arr, cy_arr

    # -----------------------------------------------------------------------
    # E.2 — Filtrar pontos fora do tubo do tronco
    # -----------------------------------------------------------------------

    def _filtrar_pontos_tronco(self, pts, h_local, skel_h, skel_cx, skel_cy,
                                params, model, H, DBH):
        """Remove pontos claramente fora do envelope do tronco (ramos, arbustos)."""
        cx_at_h = np.interp(h_local, skel_h, skel_cx)
        cy_at_h = np.interp(h_local, skel_h, skel_cy)
        dist_xy = np.sqrt((pts[:, 0] - cx_at_h) ** 2 + (pts[:, 1] - cy_at_h) ** 2)
        r_exp   = self._r_at_h(h_local, params, model, H, DBH)
        mask    = dist_xy <= r_exp * (1.0 + self.TUBE_MARGIN_FRAC)
        return pts[mask]

    # -----------------------------------------------------------------------
    # E.3 — Completar secções com gaps angulares e gerar extensão
    # -----------------------------------------------------------------------

    def _completar_cross_sections(self, pts_limpos, h_local, skel_h, skel_cx, skel_cy,
                                   params, model, H, DBH, z_base):
        """
        Para cada fatia de SLICE_STEP ao longo de [0, H]:
          - Se pontos reais insuficientes: gera anel completo sintético.
          - Caso contrário: completa apenas os sectores angulares vazios.
        Serve também como extensão para alturas sem cobertura real.
        """
        rng   = np.random.default_rng(42)
        N     = self.N_SECTORS
        dang  = 2.0 * np.pi / N
        pontos_sint = []

        h_centers = np.arange(self.SLICE_STEP / 2, H + 1e-6, self.SLICE_STEP)

        for hc in h_centers:
            if hc > H + 0.05:
                break
            h0, h1 = hc - self.SLICE_STEP / 2, hc + self.SLICE_STEP / 2
            mask   = (h_local >= h0) & (h_local < h1)
            n_real = int(mask.sum())

            cx_s = float(np.interp(hc, skel_h, skel_cx))
            cy_s = float(np.interp(hc, skel_h, skel_cy))
            r_s  = float(self._r_at_h(hc, params, model, H, DBH))

            if n_real < self.MIN_PTS_SLICE:
                # Sem pontos reais: gerar anel completo
                for k in range(N):
                    ang = (k + 0.5) * dang - np.pi
                    r_n = r_s + rng.uniform(-self.SYNTH_NOISE_R, self.SYNTH_NOISE_R)
                    pontos_sint.append([
                        cx_s + r_n * np.cos(ang),
                        cy_s + r_n * np.sin(ang),
                        z_base + hc + rng.uniform(-self.SYNTH_NOISE_Z, self.SYNTH_NOISE_Z),
                    ])
                continue

            # Verificar cobertura angular dos pontos reais limpos
            pts_s    = pts_limpos[mask]
            phi      = np.arctan2(pts_s[:, 1] - cy_s, pts_s[:, 0] - cx_s)
            sect_ids = np.floor((phi + np.pi) / dang).astype(int) % N
            counts   = np.bincount(sect_ids, minlength=N)

            for k in range(N):
                if counts[k] < self.MIN_SECT_PTS:
                    ang = (k + 0.5) * dang - np.pi
                    for _ in range(self.MIN_SECT_PTS):
                        da  = rng.uniform(-dang * 0.4, dang * 0.4)
                        r_n = r_s + rng.uniform(-self.SYNTH_NOISE_R, self.SYNTH_NOISE_R)
                        pontos_sint.append([
                            cx_s + r_n * np.cos(ang + da),
                            cy_s + r_n * np.sin(ang + da),
                            z_base + hc + rng.uniform(-self.SYNTH_NOISE_Z, self.SYNTH_NOISE_Z),
                        ])

        if not pontos_sint:
            return np.empty((0, 3), dtype=np.float64)
        return np.array(pontos_sint, dtype=np.float64)

    # -----------------------------------------------------------------------
    # Perfil CSV
    # -----------------------------------------------------------------------

    def _escrever_perfil_csv(self, path: Path, params, model, H, DBH):
        rows = []
        for sv in np.arange(0.0, H, 0.10):
            if model == "kozak":
                q = float(np.clip(sv / H, 1e-4, 0.9999))
                r = float(kozak_1988(q, DBH, H, params))
            elif model == "especie":
                r = float(especie_taper(np.array([sv]), params[0], H)[0])
            else:
                r = float(potencia_simples(sv, params[0], H, params[1]))
            rows.append({
                "s_m":    round(float(sv), 3),
                "h_m":    round(float(sv), 3),
                "r_m":    round(max(r, 0.0), 5),
                "source": "taper_v8",
            })
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=["s_m", "h_m", "r_m", "source"])
            w.writeheader()
            w.writerows(rows)
