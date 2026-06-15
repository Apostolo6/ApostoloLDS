"""
PIPELINE LiDAR v8 — Fase 4: Volume Analítico por Taper
=======================================================
Substitui o Alpha Shape + Trimesh da v8 por integração analítica:
  V = ∫₀ᴴ π · r(s)² ds    (via taper Kozak ou potência)

Adicionalmente:
  - Mesh watertight de contorno (visual, percentil 50)
  - Incerteza estimada por fracção extrapolada
  - CSV completo com qualidade, flags, incerteza

Input : *_taper_meta.json + *_extended.ply  (de fase3_v8/)
Output: summary_volume.csv  (compatível com _gerar_resultados do executar.py)
        summary_volume_v8.csv (formato completo v8)
        *.mesh.ply (mesh visual)
"""

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import numpy as np

from config import StatusFase
from logging_utils import setup_logger
from taper_models import kozak_1988, potencia_simples, volume_analitico


# ---------------------------------------------------------------------------
# Resultado por árvore
# ---------------------------------------------------------------------------

@dataclass
class ResultadoFase4V8:
    status: StatusFase
    tree_id: str = ""
    volume_dm3: float = 0.0
    volume_mesh_dm3: float = 0.0
    incerteza_pct: float = 0.0
    qualidade: str = ""
    flags: List[str] = field(default_factory=list)
    erro: str = ""


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class VolumeV8:
    """Cálculo de volume analítico para pipeline v8."""

    # Calibração de volume em 2 classes por DBH — derivada de 45 árvores GT v2 s/Alge2 (4 parcelas, 2026-06).
    # Classe pequena (DBH_taper_raw_cm < DBH_VOLUME_SPLIT): pipeline sobrestima → factor < 1
    # Classe grande (DBH_taper_raw_cm >= DBH_VOLUME_SPLIT): pipeline subestima → factor > 1
    # Correlação DBH vs raw_vol_ratio: Pearson r=-0.451 (p<0.001) — split justificado.
    # Re-optimizado em 2026-06-05 contra GT v2 corrigido (vol médio GT caiu 9.6% vs versão anterior).
    VOLUME_CALIB_SMALL = 0.8433  # (v8, obsoleto) DBH_taper_raw_cm < 16.0 cm
    VOLUME_CALIB_LARGE = 1.1593  # (v8, obsoleto) DBH_taper_raw_cm >= 16.0 cm
    DBH_VOLUME_SPLIT   = 16.0    # (v8, obsoleto)

    # v8: factor global único de volume (log-mean GT/raw, 45 árvores s/Alge2).
    # Substitui o degrau SMALL/LARGE. Validação honesta é por LOPO (calibra_lopo.py).
    VOLUME_CALIB_GLOBAL = 0.9659

    # DAP reportado: usar DBH_taper_raw_cm do taper_meta.json (taper@1.3m) com factor por modelo.
    # Re-optimizado em 2026-06-08 contra GT v2 corrigido (s/Alge2, n=45).
    # Kozak já é preciso na base (bias +3.5% → reduzir); potência subestima (bias -0.9% → aumentar).
    DBH_REPORT_CALIB_KOZAK    = 1.0493  # (v8, obsoleto)
    DBH_REPORT_CALIB_POTENCIA = 1.0976  # (v8, obsoleto)

    # v8: factor global único de DBH (log-mean GT/raw, 45 árvores s/Alge2).
    DBH_REPORT_CALIB_GLOBAL = 1.0549

    # v8: desligar para obter volume/DBH em CRU (para re-derivar calibração honesta com LOPO)
    APPLY_CALIB = True

    # v8: limite de sanidade por factor de forma (trava volumes patológicos)
    USE_FORM_FACTOR_CLAMP = True
    FF_LO = 0.34
    FF_HI = 0.62

    # v8: modo de volume. 'taper' = integral do modelo de taper (kozak/potencia/especie).
    # 'ff' = volume por factor de forma: V = f * (pi/4) * DBH^2 * H, com DBH robusto
    # das fatias e f constante (absorvido pela calibração). Mínima variância de modelo.
    VOLUME_MODE = "taper"
    FF_NOMINAL = 0.45
    HYBRID_FRAC = 0.60   # acima desta fracção extrapolada, usar factor de forma

    def __init__(self, config, logger=None):
        self.cfg = config
        self.log = logger or setup_logger("Fase4_v8", level=config.LOG_LEVEL)

    # -----------------------------------------------------------------------
    # Ponto de entrada — processa pasta inteira
    # -----------------------------------------------------------------------

    def processar(self, fase3_dir: Path, out_dir: Path) -> object:
        """
        Processa todos os *_taper_meta.json em fase3_dir.

        Args:
            fase3_dir : pasta com outputs da Fase 3 v8
            out_dir   : pasta de output (fase4/)

        Returns:
            Objecto com .status, .n_processados, .n_sucesso
        """
        from dataclasses import dataclass as _dc

        @_dc
        class _Res:
            status: StatusFase
            n_processados: int = 0
            n_sucesso: int = 0
            erro: str = ""

        out_dir.mkdir(parents=True, exist_ok=True)
        meta_files = sorted(fase3_dir.glob("*_taper_meta.json"))

        if not meta_files:
            self.log.warning(f"Nenhum *_taper_meta.json em {fase3_dir}")
            return _Res(status=StatusFase.SUCESSO, n_processados=0, n_sucesso=0)

        resultados = []
        for meta_path in meta_files:
            stem = meta_path.name.replace("_taper_meta.json", "")
            ext_ply = fase3_dir / f"{stem}_extended.ply"
            r = self.processar_um(meta_path, ext_ply if ext_ply.exists() else None,
                                   out_dir)
            resultados.append(r)

        n_ok = sum(1 for r in resultados if r.status == StatusFase.SUCESSO)
        self.log.info(f"Fase 4 v8: {n_ok}/{len(resultados)} árvores com volume OK")

        # --- Escrever CSVs de sumário ---
        self._escrever_summary(resultados, fase3_dir, out_dir)

        return _Res(status=StatusFase.SUCESSO,
                    n_processados=len(resultados),
                    n_sucesso=n_ok)

    # -----------------------------------------------------------------------
    # Processar uma árvore
    # -----------------------------------------------------------------------

    def processar_um(self, meta_path: Path, extended_ply: Optional[Path],
                     out_dir: Path) -> ResultadoFase4V8:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tree_id = meta.get("tree_id", meta_path.stem.replace("_taper_meta",""))
        self.log.info(f"  Fase 4 v8: {tree_id}")

        try:
            params    = np.array(meta["params"], dtype=np.float64)
            model     = meta["taper_model"]
            H         = float(meta["H_final"])
            DBH       = float(meta["DBH_inicial"])
            frac_ext  = float(meta.get("fraccao_extrapolada", 0.0))
            qualidade = meta.get("qualidade", "media")
            flags     = list(meta.get("flags", []))
        except KeyError as e:
            return ResultadoFase4V8(status=StatusFase.ERRO, tree_id=tree_id,
                                    erro=f"Campo em falta no meta JSON: {e}")

        if H <= 0 or DBH <= 0:
            return ResultadoFase4V8(status=StatusFase.ERRO, tree_id=tree_id,
                                    erro=f"H={H:.2f} ou DBH={DBH:.2f} inválido")

        dbh_taper_cm = float(meta.get("DBH_taper_raw_cm", DBH * 100))

        # --- E.2: Volume ---
        try:
            usa_ff = (self.VOLUME_MODE == "ff"
                      or (self.VOLUME_MODE == "hybrid" and frac_ext > self.HYBRID_FRAC))
            if usa_ff:
                # Volume por factor de forma com DBH robusto das fatias e H.
                dbh_m = dbh_taper_cm / 100.0
                V_dm3 = self.FF_NOMINAL * (math.pi / 4.0) * dbh_m**2 * H * 1000.0
            else:
                V_dm3 = volume_analitico(params, model, H, DBH,
                                         step=self.cfg.E_INTEGRATION_STEP)
        except Exception as e:
            return ResultadoFase4V8(status=StatusFase.ERRO, tree_id=tree_id,
                                    erro=f"Erro no integral: {e}")

        if V_dm3 <= 0 or V_dm3 > 5000:
            return ResultadoFase4V8(status=StatusFase.ERRO, tree_id=tree_id,
                                    erro=f"Volume fora do esperado: {V_dm3:.1f} dm³")

        dbh_taper_cm = float(meta.get("DBH_taper_raw_cm", DBH * 100))

        # v8: limite de sanidade por factor de forma f = V / (pi/4 * DBH^2 * H).
        # Pinus pinaster tem f tipicamente ~0.42-0.50; valores fora de [FF_LO, FF_HI]
        # indicam taper patológico (scan mau). Trava o volume ao bound mais próximo.
        if self.USE_FORM_FACTOR_CLAMP:
            dbh_m = dbh_taper_cm / 100.0
            denom = (math.pi / 4.0) * dbh_m**2 * H
            if denom > 1e-6:
                ff = (V_dm3 / 1000.0) / denom
                ff_c = min(max(ff, self.FF_LO), self.FF_HI)
                if ff > 1e-6 and ff_c != ff:
                    V_dm3 *= ff_c / ff
        # v8: calibração de volume por FACTOR GLOBAL ÚNICO (substitui o degrau
        # SMALL/LARGE da v8, indefensável). Bias residual; ver nota no topo da classe.
        calib = self.VOLUME_CALIB_GLOBAL if self.APPLY_CALIB else 1.0
        V_dm3 *= calib
        self.log.info(f"    V_perfil={V_dm3:.1f} dm³  (H={H:.1f}m, DBH={dbh_taper_cm:.1f}cm, calib={calib:.3f})")

        # --- E.4: Mesh de contorno visual ---
        V_mesh = 0.0
        if extended_ply is not None and extended_ply.exists():
            try:
                V_mesh = self._calcular_volume_mesh(extended_ply, meta, out_dir)
            except Exception as e:
                self.log.warning(f"    Mesh não calculada: {e}")

        # --- E.5: Incerteza ---
        incerteza_pct = self._estimar_incerteza(frac_ext)

        # --- Actualizar qualidade se necessário ---
        sigma_mm = float(meta.get("sigma_residuo_mm", 999))
        cob_dbh  = float(meta.get("cobertura_DBH", 0.0))
        qual_crit = (
            (cob_dbh > 0.5) +
            (frac_ext < 0.20) +
            (sigma_mm < 20.0)
        )
        qualidade = ["baixa", "baixa", "media", "alta"][qual_crit]

        self.log.info(
            f"    qualidade={qualidade}  incerteza={incerteza_pct:.0f}%  "
            f"V_mesh={V_mesh:.1f} dm³"
        )

        return ResultadoFase4V8(
            status=StatusFase.SUCESSO,
            tree_id=tree_id,
            volume_dm3=float(V_dm3),
            volume_mesh_dm3=float(V_mesh),
            incerteza_pct=float(incerteza_pct),
            qualidade=qualidade,
            flags=flags,
        )

    # -----------------------------------------------------------------------
    # E.4 — Mesh watertight visual
    # -----------------------------------------------------------------------

    def _calcular_volume_mesh(self, extended_ply: Path, meta: dict,
                               out_dir: Path) -> float:
        """
        Constrói mesh de contorno por fatia (percentil 50 radial).
        Devolve volume em dm³ (apenas para inspecção — não é o valor primário).
        """
        from plyfile import PlyData

        ply  = PlyData.read(str(extended_ply))
        v    = ply['vertex'].data
        pts  = np.column_stack([
            np.array(v['x'], dtype=np.float64),
            np.array(v['y'], dtype=np.float64),
            np.array(v['z'], dtype=np.float64),
        ])

        z_base = float(meta.get("z_base", pts[:,2].min()))
        h_pts  = pts[:, 2] - z_base
        H      = float(meta["H_final"])
        cx_dbh = float(meta.get("cx_DBH", pts[:,0].mean()))
        cy_dbh = float(meta.get("cy_DBH", pts[:,1].mean()))

        # Esqueleto (linha de centro por altura) — guardado pela fase3
        skel_h_raw  = meta.get("skeleton_h",  [])
        skel_cx_raw = meta.get("skeleton_cx", [])
        skel_cy_raw = meta.get("skeleton_cy", [])
        has_skel = len(skel_h_raw) >= 2
        if has_skel:
            skel_h  = np.array(skel_h_raw,  dtype=np.float64)
            skel_cx = np.array(skel_cx_raw, dtype=np.float64)
            skel_cy = np.array(skel_cy_raw, dtype=np.float64)

        SLICE  = 0.10
        N_ANG  = 36   # 10° por sector
        angles = np.linspace(0, 2*np.pi, N_ANG, endpoint=False)

        rings    = []
        h_arr    = []
        cx_arr_m = []   # centro X por anel (para mesh)
        cy_arr_m = []   # centro Y por anel (para mesh)

        for h0 in np.arange(0.0, H, SLICE):
            hc = h0 + SLICE / 2
            m = (h_pts >= h0) & (h_pts < h0 + SLICE)
            if m.sum() < 5:
                continue
            # Centro do esqueleto para esta altura; fallback para cx_dbh
            if has_skel:
                cx_s = float(np.interp(hc, skel_h, skel_cx))
                cy_s = float(np.interp(hc, skel_h, skel_cy))
            else:
                cx_s, cy_s = cx_dbh, cy_dbh
            xs = pts[m, 0] - cx_s
            ys = pts[m, 1] - cy_s
            phi_m = np.arctan2(ys, xs)
            rho_m = np.sqrt(xs**2 + ys**2)

            # Raio mediano por sector
            sect_ids = np.floor((phi_m + np.pi) / (2*np.pi / N_ANG)).astype(int) % N_ANG
            r_sect   = np.zeros(N_ANG)
            for s in range(N_ANG):
                ms = sect_ids == s
                r_sect[s] = float(np.median(rho_m[ms])) if ms.sum() >= 2 else 0.0

            # Interpolar sectores com r=0
            if (r_sect == 0).all():
                continue
            nonzero = np.where(r_sect > 0)[0]
            for i in range(N_ANG):
                if r_sect[i] == 0:
                    nn = nonzero[np.argmin(np.abs(nonzero - i))]
                    r_sect[i] = r_sect[nn]

            rings.append(r_sect)
            h_arr.append(hc)
            cx_arr_m.append(cx_s)
            cy_arr_m.append(cy_s)

        if len(rings) < 2:
            return 0.0

        # Volume por integração trapezoidal dos anéis
        areas = [np.pi * np.mean(r**2) for r in rings]
        V_m3  = float(np.trapezoid(areas, h_arr))
        V_dm3 = V_m3 * 1000.0

        # Mesh visual (plyfile) — apenas triângulos simples
        try:
            self._escrever_mesh_ply(
                out_dir / f"{meta['tree_id']}.mesh.ply",
                rings, h_arr, cx_arr_m, cy_arr_m, z_base, N_ANG
            )
        except Exception as e:
            self.log.warning(f"    Mesh PLY não escrita: {e}")

        return V_dm3

    def _escrever_mesh_ply(self, path, rings, h_arr, cx_arr, cy_arr, z_base, N_ANG):
        """Escreve mesh de contorno como PLY com triângulos.
        cx_arr/cy_arr são listas de centros por anel (suportam tronco inclinado).
        """
        from plyfile import PlyData, PlyElement

        verts = []
        faces = []

        for ring_i, (r_sect, h, cx, cy) in enumerate(zip(rings, h_arr, cx_arr, cy_arr)):
            z = z_base + h
            angles = np.linspace(0, 2*np.pi, N_ANG, endpoint=False)
            for ai, ang in enumerate(angles):
                verts.append((
                    float(cx + r_sect[ai] * np.cos(ang)),
                    float(cy + r_sect[ai] * np.sin(ang)),
                    float(z),
                ))

        # Faces entre anéis consecutivos
        for ri in range(len(rings) - 1):
            base0 = ri * N_ANG
            base1 = (ri + 1) * N_ANG
            for ai in range(N_ANG):
                ai2 = (ai + 1) % N_ANG
                faces.append(([base0+ai, base1+ai, base1+ai2], 0))
                faces.append(([base0+ai, base1+ai2, base0+ai2], 0))

        v_arr = np.array(verts, dtype=[('x','f4'),('y','f4'),('z','f4')])
        f_arr = np.array(
            [(np.array(f, dtype=np.int32),) for f, _ in faces],
            dtype=[('vertex_indices', 'O')]
        )
        PlyData(
            [PlyElement.describe(v_arr, 'vertex'),
             PlyElement.describe(f_arr, 'face')],
            text=False
        ).write(str(path))

    # -----------------------------------------------------------------------
    # E.5 — Incerteza
    # -----------------------------------------------------------------------

    def _estimar_incerteza(self, frac_extrapolada: float) -> float:
        return self.cfg.E_UNCERT_BASE_PCT + self.cfg.E_UNCERT_FRAC_FACTOR * frac_extrapolada

    # -----------------------------------------------------------------------
    # E.6 — CSVs de sumário
    # -----------------------------------------------------------------------

    def _escrever_summary(self, resultados: list, fase3_dir: Path, out_dir: Path):
        """
        Escreve summary_volume.csv (compatível com v8 / _gerar_resultados)
        e summary_volume_v8.csv (formato completo v8).
        """
        # Recolher meta de fase3 para completar colunas
        metas = {}
        for mf in sorted(fase3_dir.glob("*_taper_meta.json")):
            try:
                m = json.loads(mf.read_text(encoding="utf-8"))
                tid = m.get("tree_id", "")
                metas[tid] = m
            except Exception:
                pass

        # --- summary_volume.csv (compatível com _gerar_resultados) ---
        rows_compat = []
        for r in resultados:
            if r.status != StatusFase.SUCESSO:
                continue
            m = metas.get(r.tree_id, {})
            rows_compat.append({
                "ficheiro": r.tree_id + "_clean.leafoff.ply",
                "volume_perfil_dm3": round(r.volume_dm3, 2),
                "volume_mesh_dm3":   round(r.volume_mesh_dm3, 2),
                "height_m":          round(float(m.get("H_final", 0)), 2),
            })

        if rows_compat:
            _escrever_csv(out_dir / "summary_volume.csv",
                          ["ficheiro","volume_perfil_dm3","volume_mesh_dm3","height_m"],
                          rows_compat)
            self.log.info(f"  [OK] summary_volume.csv  ({len(rows_compat)} árvores)")

        # --- summary_volume_v8.csv (formato completo) ---
        rows_full = []
        for r in resultados:
            m = metas.get(r.tree_id, {})
            diff_pct = 0.0
            if r.volume_dm3 > 0 and r.volume_mesh_dm3 > 0:
                diff_pct = abs(r.volume_dm3 - r.volume_mesh_dm3) / r.volume_dm3 * 100

            rows_full.append({
                "tree_id":                r.tree_id,
                "status":                 r.status.value,
                "DBH_cm":                 round(float(m.get("DBH_taper_raw_cm", float(m.get("DBH_inicial",0))*100)) * (self.DBH_REPORT_CALIB_GLOBAL if self.APPLY_CALIB else 1.0), 1),
                "H_final_m":              round(float(m.get("H_final",0)), 2),
                "H_fonte":                m.get("fonte_H",""),
                "volume_perfil_dm3":      round(r.volume_dm3, 2),
                "volume_mesh_visual_dm3": round(r.volume_mesh_dm3, 2),
                "diff_pct":               round(diff_pct, 1),
                "s_top_fiavel_m":         round(float(m.get("s_top_fiavel",0)), 2),
                "fraccao_extrapolada":    round(float(m.get("fraccao_extrapolada",0)), 3),
                "sigma_residuo_mm":       round(float(m.get("sigma_residuo_mm",0)), 1),
                "incerteza_estimada_pct": round(r.incerteza_pct, 1),
                "taper_model":            m.get("taper_model",""),
                "qualidade":              r.qualidade,
                "cobertura_DBH":          round(float(m.get("cobertura_DBH",0)), 2),
                "n_pontos_real":          m.get("n_centroides_validos",""),
                "flags":                  "|".join(r.flags) if r.flags else "",
                "motivo_erro":            r.erro,
            })

        if rows_full:
            cols = [
                "tree_id","status","DBH_cm","H_final_m","H_fonte",
                "volume_perfil_dm3","volume_mesh_visual_dm3","diff_pct",
                "s_top_fiavel_m","fraccao_extrapolada","sigma_residuo_mm",
                "incerteza_estimada_pct","taper_model","qualidade",
                "cobertura_DBH","n_pontos_real","flags","motivo_erro",
            ]
            _escrever_csv(out_dir / "summary_volume_v8.csv", cols, rows_full)
            self.log.info(f"  [OK] summary_volume_v8.csv")

        # --- Verificações de sanidade ---
        avisos = self._verificar_sanidade(rows_compat, rows_full)
        if avisos:
            aviso_path = out_dir / "verificacao_sanidade.txt"
            aviso_path.write_text("\n".join(avisos) + "\n", encoding="utf-8")
            for a in avisos:
                self.log.warning(f"  SANIDADE: {a}")

    def _verificar_sanidade(self, rows_compat, rows_full):
        avisos = []
        if not rows_full:
            return avisos

        vols = [r["volume_perfil_dm3"] for r in rows_full
                if isinstance(r["volume_perfil_dm3"], (int, float)) and r["volume_perfil_dm3"] > 0]
        dbhs = [r["DBH_cm"] for r in rows_full
                if isinstance(r["DBH_cm"], (int, float)) and r["DBH_cm"] > 0]

        if vols:
            med_v = float(np.median(vols))
            if med_v < 50:
                avisos.append(f"WARNING: volumes medianos muito baixos ({med_v:.0f} dm³ < 50)")
            elif med_v > 2000:
                avisos.append(f"WARNING: volumes medianos muito altos ({med_v:.0f} dm³ > 2000)")

        if dbhs:
            med_d = float(np.median(dbhs))
            if med_d < 12:
                avisos.append(f"WARNING: DAP mediano baixo ({med_d:.0f} cm < 12 cm)")
            elif med_d > 60:
                avisos.append(f"WARNING: DAP mediano alto ({med_d:.0f} cm > 60 cm)")

        n_tot = len(rows_full)
        if n_tot > 0:
            n_alta = sum(1 for r in rows_full if r.get("qualidade") == "alta")
            if n_alta / n_tot < 0.30:
                avisos.append(
                    f"WARNING: fracção de alta qualidade baixa "
                    f"({n_alta}/{n_tot} = {n_alta/n_tot:.0%})"
                )
            n_pot = sum(1 for r in rows_full if r.get("taper_model") == "potencia")
            if n_pot / n_tot > 0.30:
                avisos.append(
                    f"WARNING: Kozak a falhar muito "
                    f"({n_pot}/{n_tot} = {n_pot/n_tot:.0%} com modelo potência)"
                )

        return avisos


# ---------------------------------------------------------------------------
# Utilitário CSV
# ---------------------------------------------------------------------------

def _escrever_csv(path: Path, cols: list, rows: list):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
