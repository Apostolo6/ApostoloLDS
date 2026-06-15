"""
PIPELINE LiDAR v8 — Fase 2.5: Per-Tree Setup
=============================================
Processa cada árvore individualmente antes da Fase 3:
  - Calcula DTM local e z_base da árvore
  - Estima DAP robusto (centróide + raio mediano)
  - Filtra árvores com DAP < 10 cm (arbustos)
  - Remove arbustos junto à base do tronco
  - Gera *_clean.leafoff.ply e *_meta.json

Input : *.leafoff.ply  (de fase2/trees/)
Output: *_clean.leafoff.ply + *_meta.json  (em fase2_5/trees/)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

from config import StatusFase
from logging_utils import setup_logger


# ---------------------------------------------------------------------------
# Resultado por árvore
# ---------------------------------------------------------------------------

@dataclass
class ResultadoFase25:
    status: StatusFase
    tree_id: str = ""
    motivo_rejeicao: str = ""
    z_base: float = 0.0
    r_DBH: float = 0.0
    DBH_cm: float = 0.0
    cobertura_DBH: float = 0.0
    n_pontos_DBH: int = 0
    cx_DBH: float = 0.0
    cy_DBH: float = 0.0
    clean_ply: Optional[Path] = None
    meta_json: Optional[Path] = None


# ---------------------------------------------------------------------------
# Utilitário: leitura de PLY
# ---------------------------------------------------------------------------

def _ler_ply(path: Path) -> np.ndarray:
    """Lê PLY e devolve array (N, 3) xyz em float64."""
    from plyfile import PlyData
    ply = PlyData.read(str(path))
    v = ply['vertex'].data
    x = np.array(v['x'], dtype=np.float64)
    y = np.array(v['y'], dtype=np.float64)
    z = np.array(v['z'], dtype=np.float64)
    return np.column_stack([x, y, z])


def _escrever_ply(path: Path, pts: np.ndarray) -> None:
    """Escreve array (N, 3) xyz como PLY."""
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

class PerTreeSetup:
    """Pré-processamento por árvore para pipeline v8."""

    def __init__(self, config, logger=None):
        self.cfg = config
        self.log = logger or setup_logger("Fase2_5", level=config.LOG_LEVEL)
        # v8: se definido (callable (cx,cy)->z), z_base vem do solo CSF da fase1
        # em vez dos mínimos da própria nuvem da árvore.
        self.ground_lookup = None

    # -----------------------------------------------------------------------
    # Ponto de entrada — processa todas as árvores de uma pasta
    # -----------------------------------------------------------------------

    def processar(self, trees_dir: Path, out_dir: Path) -> list:
        """
        Processa todos os *.leafoff.ply em trees_dir.

        Args:
            trees_dir : pasta com os leafoff.ply da Fase 2
            out_dir   : pasta de destino (fase2_5/trees/)

        Returns:
            Lista de ResultadoFase25
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        leafoffs = sorted(trees_dir.glob("*.leafoff.ply"))
        if not leafoffs:
            self.log.warning(f"Nenhum *.leafoff.ply em {trees_dir}")
            return []

        resultados = []
        for ply_path in leafoffs:
            r_list = self.processar_um(ply_path, out_dir)
            resultados.extend(r_list)

        n_ok  = sum(1 for r in resultados if r.status == StatusFase.SUCESSO)
        n_rej = sum(1 for r in resultados if r.status == StatusFase.REJEITADA)
        n_err = sum(1 for r in resultados if r.status == StatusFase.ERRO)
        self.log.info(
            f"Fase 2.5 concluída: {n_ok} aceites, {n_rej} rejeitadas, {n_err} erros"
        )
        return resultados

    # -----------------------------------------------------------------------
    # Processar uma árvore
    # -----------------------------------------------------------------------

    def processar_um(self, leafoff_path: Path, out_dir: Path) -> list:
        """Processa uma árvore. Devolve lista de ResultadoFase25 (normalmente 1, mas 2 se split)."""
        nome = leafoff_path.stem.replace('.leafoff', '')
        self.log.info(f"  Fase 2.5: {leafoff_path.name}")

        # --- Carregar pontos ---
        try:
            pts = _ler_ply(leafoff_path)
        except Exception as e:
            self.log.error(f"    Erro ao ler PLY: {e}")
            return [ResultadoFase25(status=StatusFase.ERRO, tree_id=nome,
                                    motivo_rejeicao=f"Erro ao ler PLY: {e}")]

        if len(pts) < 30:
            msg = f"Pontos insuficientes ({len(pts)} < 30)"
            self.log.warning(f"    REJEITADA: {msg}")
            return [ResultadoFase25(status=StatusFase.REJEITADA, tree_id=nome,
                                    motivo_rejeicao=msg)]

        # --- C.2: DTM local + z_base ---
        z_base = self._calcular_z_base(pts)

        # --- Alturas locais ---
        h_local = pts[:, 2] - z_base

        # Remover pontos muito abaixo do chão (ruído)
        mask_h = h_local >= -0.20
        if mask_h.sum() < 30:
            msg = "Pontos insuficientes após filtro de altura"
            self.log.warning(f"    REJEITADA: {msg}")
            return [ResultadoFase25(status=StatusFase.REJEITADA, tree_id=nome,
                                    motivo_rejeicao=msg)]
        pts = pts[mask_h]
        h_local = h_local[mask_h]

        # --- C.3: DAP robusto ---
        dbh_res = self._calcular_dbh(pts, h_local)
        if dbh_res is None:
            msg = "Pontos insuficientes na fatia DAP"
            self.log.warning(f"    REJEITADA: {msg}")
            return [ResultadoFase25(status=StatusFase.REJEITADA, tree_id=nome,
                                    motivo_rejeicao=msg)]

        cx, cy, r_DBH, cobertura_DBH, n_DBH = dbh_res
        DBH_cm = 2.0 * r_DBH * 100.0

        self.log.info(
            f"    DAP={DBH_cm:.1f} cm  r_DBH={r_DBH*100:.1f} cm  "
            f"cob_ang={cobertura_DBH:.0%}  n_pts={n_DBH}"
        )

        if cobertura_DBH < 0.25:
            self.log.warning(
                f"    AVISO: cobertura angular DAP baixa ({cobertura_DBH:.0%})"
            )

        # --- C.4: Filtro DAP mínimo ---
        if 2.0 * r_DBH < self.cfg.C_MIN_DBH_M:
            msg = f"DAP {DBH_cm:.1f} cm < {self.cfg.C_MIN_DBH_M*100:.0f} cm — arbusto."
            self.log.warning(f"    REJEITADA: {msg}")
            return [ResultadoFase25(status=StatusFase.REJEITADA, tree_id=nome,
                                    motivo_rejeicao=msg)]

        # --- C.4b: Detecção de múltiplos troncos ---
        troncos = self._detetar_multiplos_troncos(pts, h_local, r_DBH)
        if troncos is not None:
            self.log.info(
                f"    MULTI-TRONCO: {len(troncos)} troncos detectados — separando."
            )
            # Ler leafon para H_leafon (partilhado por todos os sub-troncos)
            H_leafon_mt = 0.0
            leafon_path_mt = leafoff_path.parent / leafoff_path.name.replace(
                '.leafoff.ply', '.leafon.ply'
            )
            if leafon_path_mt.exists():
                try:
                    pts_on = _ler_ply(leafon_path_mt)
                    H_leafon_mt = max(0.0, self._estimar_H(pts_on, z_base) * self.H_LEAFON_CORRECTION)
                except Exception:
                    pass
            return self._split_troncos(
                pts, h_local, z_base, H_leafon_mt, troncos, nome, out_dir, leafoff_path
            )

        # --- C.5: Remover arbustos junto à base ---
        pts_clean, n_removidos, frac_removida = self._remover_arbustos_base(
            pts, h_local, cx, cy, r_DBH
        )
        self.log.info(
            f"    Arbustos removidos: {n_removidos} pts ({frac_removida:.1%} dos baixos)"
        )

        # Falha se removeu demasiado (arvore mal detetada)
        if frac_removida > 0.80 and len(pts_clean) < 100:
            msg = f"Arbustos removidos excedem 80% ({frac_removida:.0%}) e < 100 pts restantes."
            self.log.warning(f"    REJEITADA: {msg}")
            return [ResultadoFase25(status=StatusFase.REJEITADA, tree_id=nome,
                                    motivo_rejeicao=msg)]

        # --- Altura real da árvore a partir do leafon ---
        H_leafon = 0.0
        leafon_path = leafoff_path.parent / leafoff_path.name.replace(
            '.leafoff.ply', '.leafon.ply'
        )
        if leafon_path.exists():
            try:
                pts_on = _ler_ply(leafon_path)
                H_leafon = self._estimar_H(pts_on, z_base) * self.H_LEAFON_CORRECTION
                H_leafon = max(H_leafon, 0.0)
                self.log.info(f"    H_leafon={H_leafon:.2f} m")
            except Exception as e_on:
                self.log.warning(f"    Leafon nao lido: {e_on}")
        else:
            self.log.warning(f"    Leafon nao encontrado: {leafon_path.name}")

        # --- Escrever clean PLY e meta JSON ---
        # Sufixo .leafoff.ply mantido para que fase3 o encontre com glob
        clean_name = leafoff_path.name.replace('.leafoff.ply', '_clean.leafoff.ply')
        clean_path = out_dir / clean_name

        try:
            _escrever_ply(clean_path, pts_clean)
        except Exception as e:
            self.log.error(f"    Erro ao escrever PLY limpo: {e}")
            return [ResultadoFase25(status=StatusFase.ERRO, tree_id=nome,
                                    motivo_rejeicao=f"Erro ao escrever PLY: {e}")]

        meta = {
            "tree_id": nome,
            "z_base": float(z_base),
            "r_DBH": float(r_DBH),
            "DBH_cm": float(DBH_cm),
            "cobertura_DBH": float(cobertura_DBH),
            "n_pontos_DBH": int(n_DBH),
            "cx_DBH": float(cx),
            "cy_DBH": float(cy),
            "H_leafon": float(H_leafon),
            "n_pontos_total": int(len(pts_clean)),
            "n_pontos_removidos_arbusto": int(n_removidos),
            "frac_arbustos_removidos": float(frac_removida),
        }
        meta_path = out_dir / clean_name.replace('_clean.leafoff.ply', '_meta.json')
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        self.log.info(f"    OK: {clean_path.name}  ({len(pts_clean):,} pts)")
        return [ResultadoFase25(
            status=StatusFase.SUCESSO,
            tree_id=nome,
            z_base=float(z_base),
            r_DBH=float(r_DBH),
            DBH_cm=float(DBH_cm),
            cobertura_DBH=float(cobertura_DBH),
            n_pontos_DBH=int(n_DBH),
            cx_DBH=float(cx),
            cy_DBH=float(cy),
            clean_ply=clean_path,
            meta_json=meta_path,
        )]

    # -----------------------------------------------------------------------
    # C.4b — Detecção de múltiplos troncos
    # -----------------------------------------------------------------------

    MULTI_TRUNK_MIN_R      = 0.20    # trigger: raio composto > 20 cm (DBH > 40 cm)
    MULTI_TRUNK_H_MIN      = 1.0     # início da fatia de detecção (m)
    MULTI_TRUNK_H_MAX      = 2.5     # fim da fatia de detecção (m)
    MULTI_TRUNK_MIN_PTS    = 30      # mínimo de pontos por metade para ser válida
    MULTI_TRUNK_MIN_R_EACH = 0.075   # raio mínimo por tronco (DBH ≥ 15 cm)
    MULTI_TRUNK_MIN_COV    = 0.30    # cobertura angular mínima (4 de 12 sectores)
    MULTI_TRUNK_VALLEY_RAT = 0.50    # valley/peak_min < 0.5 → bimodal (dois troncos)

    # --- Calibração empírica (ajustar após run de referência) ---
    # TLS subestima a altura real em ~10% (copa impede o laser de chegar ao ápice)
    H_LEAFON_CORRECTION = 1.0270  # v8: H_corr = H_raw(max-z) * 1.027 (com z_base do solo CSF; log-mean GT/raw, 45 s/Alge2)
    # v8: estimador da altura a partir do leafon. 'max' (v8), 'p99.5', 'p99', 'apex'.
    HEIGHT_METHOD = "max"
    # TLS sobrestima o DAP em ~19% (chapadas, ramos residuais a 1.3 m).
    # NÃO aplicar ao r_DBH usado no fit do taper — causa efeito inverso no volume.
    # Correcção aplicada apenas ao DAP reportado: ver DBH_REPORT_CALIB na fase3.
    DBH_CALIB           = 1.0     # sem correcção ao fit do taper (intencionalmente)

    @staticmethod
    def build_ground_lookup(las_path):
        """
        Constrói um lookup (cx,cy)->z do solo classificado por CSF (classe 2) de um
        .las da fase1. Devolve callable ou None. Usado para z_base consistente.
        """
        try:
            import laspy
            from scipy.spatial import cKDTree
            las = laspy.read(str(las_path))
            cl = (np.asarray(las.classification) == 2)
            gx = np.asarray(las.x)[cl]
            gy = np.asarray(las.y)[cl]
            gz = np.asarray(las.z)[cl]
            if len(gz) < 50:
                return None
            kt = cKDTree(np.column_stack([gx, gy]))

            def lookup(cx, cy):
                idx = kt.query_ball_point([cx, cy], r=3.0)
                if len(idx) >= 5:
                    return float(np.median(gz[idx]))
                _, i = kt.query([cx, cy], k=min(25, len(gz)))
                return float(np.median(gz[i]))

            return lookup
        except Exception:
            return None

    def _estimar_H(self, pts_on, z_base):
        """
        Estima a altura total da árvore a partir da nuvem leafon.
        'max'   : v8 (ponto mais alto — sensível a outliers).
        'pXX'   : percentil XX dos z (robusto a outliers/pássaros).
        'apex'  : extrapola o ápice geométrico ajustando um cone ao topo da copa
                  (estima onde o raio horizontal → 0), por árvore.
        Devolve H_raw (sem factor de correcção).
        """
        z = pts_on[:, 2].astype(np.float64) - float(z_base)
        keep = z > 0.5
        if keep.sum() < 10:
            return max(0.0, float(np.percentile(pts_on[:, 2], 99.5)) - float(z_base))
        P = pts_on[keep]
        zl = z[keep]
        method = self.HEIGHT_METHOD

        if method == "max":
            return float(zl.max())
        if method.startswith("p"):
            return float(np.percentile(zl, float(method[1:])))
        if method == "apex":
            z995 = float(np.percentile(zl, 99.5))
            ztop = float(np.percentile(zl, 88))
            topm = zl >= ztop
            cx = float(np.median(P[topm, 0]))
            cy = float(np.median(P[topm, 1]))
            zmax = float(zl.max())
            hs, rs = [], []
            hb = ztop
            while hb < zmax:
                mm = (zl >= hb - 0.05) & (zl < hb + 0.05)
                if mm.sum() >= 3:
                    d = np.sqrt((P[mm, 0] - cx) ** 2 + (P[mm, 1] - cy) ** 2)
                    hs.append(hb)
                    rs.append(float(np.percentile(d, 80)))
                hb += 0.10
            if len(hs) < 3:
                return z995
            a, b = np.polyfit(np.array(hs), np.array(rs), 1)
            if a >= 0:
                return z995
            h_apex = -b / a
            # saturar: nunca abaixo do p99.5 nem mais de 3m acima do topo medido
            return float(min(max(h_apex, z995), zmax + 3.0))
        return float(zl.max())

    def _detetar_multiplos_troncos(self, pts, h_local, r_DBH):
        """
        Detecta dois troncos distintos na fatia 1.0–2.5 m via bimodalidade PCA.

        Projeta os pontos na direcção de maior variância (PCA). Dois troncos separados
        produzem uma distribuição claramente bimodal (dois picos com vale profundo).
        Um único tronco grande produz uma distribuição unimodal.

        Retorna [dict_a, dict_b] ou None.
        """
        if r_DBH < self.MULTI_TRUNK_MIN_R:
            return None

        mask = (h_local >= self.MULTI_TRUNK_H_MIN) & (h_local <= self.MULTI_TRUNK_H_MAX)
        n_slice = int(mask.sum())
        if n_slice < self.MULTI_TRUNK_MIN_PTS * 2:
            return None

        pts_2d = pts[mask, :2].astype(np.float64)

        # Direcção PCA de maior variância
        pts_c = pts_2d - pts_2d.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
            pca1 = Vt[0]
        except Exception:
            return None
        proj = pts_c @ pca1

        # Histograma de 24 bins na projecção PCA
        hist, edges = np.histogram(proj, bins=24)
        mids = (edges[:-1] + edges[1:]) / 2.0
        hf = hist.astype(float)

        # Dois picos maiores (separados por ≥ 4 bins)
        i_p1 = int(np.argmax(hf))
        masked = hf.copy()
        masked[max(0, i_p1 - 3): i_p1 + 4] = 0
        if masked.max() == 0:
            return None
        i_p2 = int(np.argmax(masked))
        i_lo, i_hi = sorted([i_p1, i_p2])

        if i_hi - i_lo < 4:
            return None

        # Vale entre os dois picos
        valley_idx = i_lo + int(np.argmin(hf[i_lo: i_hi + 1]))
        valley_val = float(hf[valley_idx])
        peak_min   = float(min(hf[i_p1], hf[i_p2]))

        if peak_min == 0 or valley_val / peak_min > self.MULTI_TRUNK_VALLEY_RAT:
            return None  # não bimodal — tronco único

        # Dividir pela posição do vale na projecção PCA
        split_val  = float(mids[valley_idx])
        mask_a     = proj > split_val
        mask_b     = ~mask_a

        troncos = []
        for mk in (mask_a, mask_b):
            n_k = int(mk.sum())
            if n_k < self.MULTI_TRUNK_MIN_PTS:
                continue

            xk = pts_2d[mk, 0]
            yk = pts_2d[mk, 1]
            ck_x = float(np.median(xk))
            ck_y = float(np.median(yk))

            dk = np.sqrt((xk - ck_x) ** 2 + (yk - ck_y) ** 2)
            rk = float(np.median([np.percentile(dk, p) for p in [60, 65, 70, 75, 80, 85]]))

            if rk < self.MULTI_TRUNK_MIN_R_EACH:
                continue

            # Cobertura angular (12 sectores de 30°)
            phi = np.arctan2(yk - ck_y, xk - ck_x)
            sectors = np.floor((phi + np.pi) / (np.pi / 6)).astype(int) % 12
            counts = np.bincount(sectors, minlength=12)
            cov = float((counts >= 2).sum()) / 12.0
            if cov < self.MULTI_TRUNK_MIN_COV:
                continue

            troncos.append({'cx': ck_x, 'cy': ck_y, 'r': rk, 'cov': cov, 'n': n_k})

        if len(troncos) < 2:
            return None

        # Verificar que os centros estão suficientemente separados (não se sobrepõem)
        sep = np.sqrt(
            (troncos[0]['cx'] - troncos[1]['cx']) ** 2 +
            (troncos[0]['cy'] - troncos[1]['cy']) ** 2
        )
        if sep < troncos[0]['r'] + troncos[1]['r']:
            return None

        troncos.sort(key=lambda t: t['r'], reverse=True)
        return troncos

    def _split_troncos(self, pts, h_local, z_base, H_leafon, troncos, nome,
                       out_dir, leafoff_path):
        """
        Separa pts em N sub-nuvens (uma por tronco) e escreve PLY + meta para cada uma.

        Retorna lista de ResultadoFase25.
        """
        sufixos = ['_a', '_b', '_c', '_d']
        resultados = []

        for i, tr in enumerate(troncos):
            suf = sufixos[i] if i < len(sufixos) else f'_{i}'
            tree_id_sub = nome + suf

            # Atribuir pontos pelo tronco mais próximo (distância XY)
            # Para o tronco i: é mais próximo do que todos os outros
            cx_i, cy_i = tr['cx'], tr['cy']
            dist_i = np.sqrt((pts[:, 0] - cx_i) ** 2 + (pts[:, 1] - cy_i) ** 2)
            mask_i = np.ones(len(pts), dtype=bool)
            for j, tr2 in enumerate(troncos):
                if j == i:
                    continue
                dist_j = np.sqrt((pts[:, 0] - tr2['cx']) ** 2 + (pts[:, 1] - tr2['cy']) ** 2)
                mask_i &= dist_i <= dist_j

            pts_sub = pts[mask_i]
            h_sub   = h_local[mask_i]

            if len(pts_sub) < 30:
                self.log.warning(f"    {tree_id_sub}: pontos insuficientes após split ({len(pts_sub)})")
                resultados.append(ResultadoFase25(
                    status=StatusFase.REJEITADA,
                    tree_id=tree_id_sub,
                    motivo_rejeicao=f"Pontos insuficientes após split ({len(pts_sub)})"
                ))
                continue

            # Remover arbustos com o centro deste tronco
            pts_clean, n_rem, frac_rem = self._remover_arbustos_base(
                pts_sub, h_sub, cx_i, cy_i, tr['r']
            )
            self.log.info(
                f"    {tree_id_sub}: {len(pts_clean):,} pts  "
                f"r={tr['r']*100:.1f} cm  arbustos={n_rem} ({frac_rem:.1%})"
            )

            # Escrever PLY
            clean_name = leafoff_path.name.replace(
                '.leafoff.ply', f'{suf}_clean.leafoff.ply'
            )
            clean_path = out_dir / clean_name
            try:
                _escrever_ply(clean_path, pts_clean)
            except Exception as e:
                self.log.error(f"    {tree_id_sub}: erro ao escrever PLY: {e}")
                resultados.append(ResultadoFase25(
                    status=StatusFase.ERRO, tree_id=tree_id_sub,
                    motivo_rejeicao=str(e)
                ))
                continue

            meta = {
                "tree_id": tree_id_sub,
                "z_base": float(z_base),
                "r_DBH": float(tr['r']),
                "DBH_cm": float(tr['r'] * 200.0),
                "cobertura_DBH": float(tr['cov']),
                "n_pontos_DBH": int(tr['n']),
                "cx_DBH": float(cx_i),
                "cy_DBH": float(cy_i),
                "H_leafon": float(H_leafon),
                "n_pontos_total": int(len(pts_clean)),
                "n_pontos_removidos_arbusto": int(n_rem),
                "frac_arbustos_removidos": float(frac_rem),
            }
            meta_path = out_dir / clean_name.replace('_clean.leafoff.ply', '_meta.json')
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            resultados.append(ResultadoFase25(
                status=StatusFase.SUCESSO,
                tree_id=tree_id_sub,
                z_base=float(z_base),
                r_DBH=float(tr['r']),
                DBH_cm=float(tr['r'] * 200.0),
                cobertura_DBH=float(tr['cov']),
                n_pontos_DBH=int(tr['n']),
                cx_DBH=float(cx_i),
                cy_DBH=float(cy_i),
                clean_ply=clean_path,
                meta_json=meta_path,
            ))

        return resultados

    # -----------------------------------------------------------------------
    # C.2 — DTM local + z_base
    # -----------------------------------------------------------------------

    def _calcular_z_base(self, pts: np.ndarray) -> float:
        """
        Estima z_base da árvore por DTM local.
        Usa grelha de células e p5 dos z em cada célula.
        """
        # v8: preferir o solo CSF da fase1 no local do tronco (DTM mais limpo
        # que os mínimos da nuvem da árvore, que apanham sub-bosque/raízes).
        if self.ground_lookup is not None:
            cx0 = float(np.median(pts[:, 0]))
            cy0 = float(np.median(pts[:, 1]))
            zg = self.ground_lookup(cx0, cy0)
            if zg is not None and np.isfinite(zg):
                return float(zg)

        cell = self.cfg.C_GROUND_CELL
        pct  = self.cfg.C_GROUND_PCT

        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

        x_min, y_min = xs.min(), ys.min()
        xi = np.floor((xs - x_min) / cell).astype(int)
        yi = np.floor((ys - y_min) / cell).astype(int)

        # Construir mapa z_min por célula
        n_xi, n_yi = xi.max() + 1, yi.max() + 1
        dtm = {}
        for i in range(len(pts)):
            k = (xi[i], yi[i])
            if k not in dtm:
                dtm[k] = []
            dtm[k].append(zs[i])

        ground_vals = []
        for vals in dtm.values():
            if len(vals) >= 3:
                ground_vals.append(np.percentile(vals, pct))

        if not ground_vals:
            # Fallback: percentil 5 de todos os z
            return float(np.percentile(zs, 5))

        # DTM suavizado → pega no p5 global
        return float(np.percentile(ground_vals, 5))

    # -----------------------------------------------------------------------
    # C.3 — DAP robusto
    # -----------------------------------------------------------------------

    def _calcular_dbh(self, pts, h_local):
        """
        Estima DAP por raio mediano na fatia 1.3 m ± slab/2.

        Devolve (cx, cy, r_DBH, cobertura_angular, n_pts) ou None.
        """
        dap_h  = self.cfg.C_DBH_HEIGHT
        slab   = self.cfg.C_DBH_SLAB

        mask = (h_local >= dap_h - slab) & (h_local < dap_h + slab)
        n_pts = int(mask.sum())

        if n_pts < 30:
            # Expandir slab
            mask = (h_local >= dap_h - 0.30) & (h_local < dap_h + 0.30)
            n_pts = int(mask.sum())
            if n_pts < 30:
                return None

        xs = pts[mask, 0]
        ys = pts[mask, 1]

        # Centróide robusto: mediana inicial → ponderado por 1/dist
        cx0, cy0 = float(np.median(xs)), float(np.median(ys))
        d_to_med = np.sqrt((xs - cx0) ** 2 + (ys - cy0) ** 2)
        keep = d_to_med < np.percentile(d_to_med, 80)
        if keep.sum() >= 10:
            xs_k, ys_k = xs[keep], ys[keep]
            cx = float(np.median(xs_k))
            cy = float(np.median(ys_k))
        else:
            cx, cy = cx0, cy0

        # Raio: mediana dos percentis 60-85 (robusto a outliers e buracos)
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        r_DBH = float(np.median([np.percentile(d, p) for p in [60, 65, 70, 75, 80, 85]]))

        # Cobertura angular (12 sectores de 30°)
        phi = np.arctan2(ys - cy, xs - cx)
        sectors = np.floor((phi + np.pi) / (np.pi / 6)).astype(int) % 12
        counts = np.bincount(sectors, minlength=12)
        cobertura = float((counts >= 3).sum()) / 12.0

        return cx, cy, r_DBH, cobertura, n_pts

    # -----------------------------------------------------------------------
    # C.5 — Remover arbustos junto à base
    # -----------------------------------------------------------------------

    def _remover_arbustos_base(self, pts, h_local, cx, cy, r_DBH):
        """
        Remove pontos abaixo de C_BUSHES_MAX_HEIGHT que estejam fora do
        cilindro r_DBH × C_BUSHES_R_FACTOR centrado em (cx, cy).

        Retorna (pts_clean, n_removidos, frac_removida).
        """
        max_h   = self.cfg.C_BUSHES_MAX_HEIGHT
        r_fac   = self.cfg.C_BUSHES_R_FACTOR
        r_keep  = r_DBH * r_fac

        mask_baixo = h_local < max_h
        n_baixo = int(mask_baixo.sum())

        if n_baixo == 0:
            return pts, 0, 0.0

        # Distância ao eixo do tronco (2D)
        dist = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)

        # Manter: (acima do limiar) OU (dentro do cilindro)
        mask_manter = (~mask_baixo) | (dist < r_keep)
        n_removidos = int((~mask_manter).sum())
        frac = n_removidos / n_baixo if n_baixo > 0 else 0.0

        return pts[mask_manter], n_removidos, frac
