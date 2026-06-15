#!/usr/bin/env python3
"""
Script de execucao do Pipeline LiDAR para Florestas Plantadas v8
=================================================================
Variante com suporte a ficheiro de trajeto (MLS).

Uso:
  Colocar em input_teste_com_trajeto/:
    - Um ficheiro .las da floresta (qualquer nome)
    - Um ficheiro *trajectory.las com o trajeto percorrido
  Correr este script.

Fases:
  1. Pre-processamento PDAL
  2. Separacao de arvores (TLS2trees instance)
  2.5 Per-tree setup: DAP, z_base, filtro arbustos
  3. Taper fitting: RANSAC por fatia + esqueleto + Kozak/potencia
  4. Volume analitico por integral do taper
  + Trajeto sobreposto no mapa 2D de resultados
"""

from pathlib import Path
from datetime import datetime
import shutil
import sys
import csv
import json
import traceback
import time
import re

from config import get_preset, StatusFase
from fase1_pdal import PDALPreprocessor
from fase2_instance import InstanceSegmentation
from fase2_5_per_tree_setup import PerTreeSetup
from fase3_v8_taper import TaperFitV8
from fase4_v8_volume import VolumeV8

# Directorio base do projecto (um nivel acima deste script)
BASE_DIR = Path(__file__).resolve().parent.parent


def _sanitizar_nome(path: Path) -> Path:
    """Renomeia o ficheiro se o nome tiver espacos ou caracteres invalidos.
    Substitui espacos por '_' e remove tudo o que nao seja alfanumerico, '_', '-' ou '.'.
    """
    stem   = path.stem
    suffix = path.suffix
    novo_stem = stem.replace(' ', '_')
    novo_stem = re.sub(r'[^\w\-\.]', '', novo_stem)
    novo_nome = novo_stem + suffix
    if novo_nome == path.name:
        return path
    novo_path = path.parent / novo_nome
    path.rename(novo_path)
    print(f"  [OK] Ficheiro renomeado: '{path.name}' -> '{novo_nome}'")
    return novo_path



def _encontrar_inputs_trajeto():
    """Encontra o ficheiro de floresta e o ficheiro de trajeto em input_teste_com_trajeto/.
    Devolve (forest_path, traj_path).
    """
    input_dir = BASE_DIR / "input_teste_com_trajeto"
    if not input_dir.exists():
        print(f"[ERRO] Pasta nao encontrada: {input_dir}")
        sys.exit(1)

    las_files    = sorted(input_dir.glob("*.las"))
    traj_files   = [f for f in las_files if f.name.endswith("trajectory.las")]
    forest_files = [f for f in las_files if not f.name.endswith("trajectory.las")]

    if not traj_files:
        print("[ERRO] Nenhum ficheiro *trajectory.las encontrado em input_teste_com_trajeto/")
        sys.exit(1)
    if not forest_files:
        print("[ERRO] Nenhum ficheiro de floresta (.las) encontrado em input_teste_com_trajeto/")
        sys.exit(1)

    if len(forest_files) > 1:
        print(f"[AVISO] Multiplos ficheiros de floresta -- usando: {forest_files[0].name}")
        for f in forest_files[1:]:
            print(f"         (ignorado: {f.name})")
    if len(traj_files) > 1:
        print(f"[AVISO] Multiplos ficheiros de trajeto -- usando: {traj_files[0].name}")

    return forest_files[0], traj_files[0]


def _tree_sort_key(lbl: str):
    """Chave de ordenação para labels como T0, T10, T0_a, T0_b.
    Devolve (num, sufixo) para ordenação correcta: T0 < T0_a < T0_b < T1 < T10.
    """
    raw = lbl[1:]  # "0", "10", "0_a", "0_b"
    parts = raw.split('_', 1)
    return (int(parts[0]), parts[1] if len(parts) > 1 else "")


def _gerar_resultados(arquivo_dir: Path, input_stem: str, input_filename: str,
                      trajectory_path: Path = None) -> None:
    """Gera pasta Resultados/ com LAS etiquetado por arvore (tree_id) e resumo TXT."""
    import numpy as np
    import laspy
    from plyfile import PlyData

    resultados_dir = arquivo_dir / "Resultados"
    resultados_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("[RESULTADOS] A gerar pasta Resultados/...")
    print("=" * 60)

    VOXEL  = 0.005      # 5 mm (igual ao voxel de deduplicacao)
    OFFSET = 100_000    # cobre +/- 500 m em passos de 5 mm

    def _encode(xs_v, ys_v, zs_v):
        xi = (xs_v + OFFSET).astype(np.int64)
        yi = (ys_v + OFFSET).astype(np.int64)
        zi = (zs_v + OFFSET).astype(np.int64)
        return xi * (1 << 40) | yi * (1 << 20) | zi

    # ---- 1. Mapear voxel -> tree_id a partir dos PLY leafoff de fase2 ----
    trees_dir  = arquivo_dir / "fase2" / "trees"
    ply_files  = sorted(trees_dir.glob("*.leafoff.ply"))
    n_trees    = len(ply_files)
    print(f"  Arvores (leafoff PLY): {n_trees}")

    all_keys_list = []
    all_tids_list = []
    for ply_path in ply_files:
        tid = int(ply_path.name.rsplit("_T", 1)[-1].split(".")[0])  # igual ao T-number do mapa
        v   = PlyData.read(str(ply_path))['vertex'].data
        xs  = np.round(np.array(v['x'], dtype=np.float64) / VOXEL).astype(np.int64)
        ys  = np.round(np.array(v['y'], dtype=np.float64) / VOXEL).astype(np.int64)
        zs  = np.round(np.array(v['z'], dtype=np.float64) / VOXEL).astype(np.int64)
        all_keys_list.append(_encode(xs, ys, zs))
        all_tids_list.append(np.full(len(xs), tid, dtype=np.uint16))

    tree_keys = np.concatenate(all_keys_list)
    tree_tids = np.concatenate(all_tids_list)

    # Ordenar e desduplicar (voxel com mais de uma arvore: guarda a primeira)
    sort_idx  = np.argsort(tree_keys, kind='stable')
    tree_keys = tree_keys[sort_idx]
    tree_tids = tree_tids[sort_idx]
    _, first  = np.unique(tree_keys, return_index=True)
    tree_keys = tree_keys[first]
    tree_tids = tree_tids[first]
    print(f"  Voxeis de arvore indexados: {len(tree_keys):,}")

    # ---- 2. Ler LAS (fase1) e fazer lookup vectorizado ----
    las_path = next((arquivo_dir / "fase1").glob("*.las"), None)
    if las_path is None:
        print("  [AVISO] LAS de fase1 nao encontrado -- a saltar _com_IDs.las")
    else:
        print(f"  A ler: {las_path.name} ...", flush=True)
        las   = laspy.read(str(las_path))
        n_pts = len(las.x)
        print(f"  Pontos LAS: {n_pts:,}")

        las_keys = _encode(
            np.round(np.array(las.x, dtype=np.float64) / VOXEL).astype(np.int64),
            np.round(np.array(las.y, dtype=np.float64) / VOXEL).astype(np.int64),
            np.round(np.array(las.z, dtype=np.float64) / VOXEL).astype(np.int64),
        )

        print("  A atribuir tree_id a cada ponto...", flush=True)
        ins   = np.searchsorted(tree_keys, las_keys)
        ins   = np.clip(ins, 0, len(tree_keys) - 1)
        match = tree_keys[ins] == las_keys
        NO_TREE = np.uint16(65535)   # sentinela: pontos sem arvore (filtrar com tree_id < 65535)
        result_tids = np.where(match, tree_tids[ins], NO_TREE).astype(np.uint16)
        n_lab = int(match.sum())
        print(f"  Pontos etiquetados: {n_lab:,} ({n_lab / n_pts * 100:.1f}%)")

        print("  A escrever LAS com IDs...", flush=True)
        las.add_extra_dims([laspy.ExtraBytesParams(name="tree_id", type=np.uint16)])
        las.tree_id = result_tids
        out_las = resultados_dir / f"{input_stem}_com_IDs.las"
        las.write(str(out_las))
        print(f"  [OK] {out_las.name}")

    # ---- 3. Resumo TXT ----
    summary_csv    = arquivo_dir / "fase4" / "summary_volume.csv"
    summary_v8_csv = arquivo_dir / "fase4" / "summary_volume_v8.csv"
    vol_perfil  = 0.0
    vol_mesh    = 0.0
    heights     = []
    # Contar troncos efectivos (inclui splits como T0_a/T0_b)
    n_troncos = n_trees
    if summary_v8_csv.exists():
        with open(summary_v8_csv, newline='', encoding='utf-8') as _fv8:
            n_troncos = sum(1 for row in csv.DictReader(_fv8)
                            if row.get('status') == 'sucesso')

    if summary_csv.exists():
        with open(summary_csv, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                try:
                    raw_p = row.get('volume_perfil_dm3')
                    vp = float(raw_p) if raw_p else 0.0
                    if vp > 0:
                        vol_perfil += vp
                    raw_m = row.get('volume_mesh_dm3')
                    vm = float(raw_m) if raw_m else 0.0
                    if vm > 0:
                        vol_mesh += vm
                    h = float(row.get('height_m') or 0)
                    if h > 0:
                        heights.append(h)
                except (ValueError, KeyError):
                    pass

    h_min = min(heights) if heights else 0.0
    h_max = max(heights) if heights else 0.0

    linhas = [
        f"Ficheiro de entrada:   {input_filename}",
        f"Arvores identificadas: {n_troncos}",
        f"Volume total (perfil): {vol_perfil:,.2f} dm3",
        f"Volume total (mesh):   {vol_mesh:,.2f} dm3",
        f"Alturas: entre {h_min:.2f} m e {h_max:.2f} m",
    ]
    out_txt = resultados_dir / "sumario.txt"
    out_txt.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"  [OK] {out_txt.name}")
    for l in linhas:
        print(f"       {l}")

    # ---- 4. Mapa 2D de localização das árvores (+ trajeto) ----
    try:
        import matplotlib
        matplotlib.use('Agg')          # sem janela (modo script)
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import matplotlib.patheffects as _pe
        from matplotlib.gridspec import GridSpec

        tree_stats_csv = arquivo_dir / "fase2" / "tree_stats.csv"
        if not tree_stats_csv.exists():
            print(f"  [AVISO] {tree_stats_csv.name} nao encontrado -- mapa nao gerado")
        else:
            # Ler posicoes das arvores
            xs, ys, labels, h_arr = [], [], [], []
            with open(tree_stats_csv, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    try:
                        xs.append(float(row['centro_x']))
                        ys.append(float(row['centro_y']))
                        tnum_stats = row['ficheiro'].rsplit('_T', 1)[-1].split('.')[0]
                        labels.append(f"T{int(tnum_stats)}")
                        h_arr.append(float(row['altura']))
                    except (ValueError, KeyError):
                        pass

            # Ler volumes para tamanho do ponto e lista lateral
            vol_by_t = {}   # "T{N}" -> volume_perfil dm3
            _sv8 = summary_v8_csv if summary_v8_csv.exists() else summary_csv
            if _sv8.exists():
                with open(_sv8, newline='', encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        try:
                            key      = row.get('tree_id') or row.get('ficheiro', '')
                            lbl_full = "T" + key.rsplit('_T', 1)[-1].split('.')[0]  # "T0_a"/"T10"
                            lbl_map  = "T" + lbl_full[1:].split('_')[0]             # "T0"/"T10"
                            v = float(row.get('volume_perfil_dm3') or 0)
                            vol_by_t[lbl_map] = vol_by_t.get(lbl_map, 0) + v
                        except (ValueError, IndexError):
                            pass

            if not xs:
                print("  [AVISO] tree_stats.csv sem dados -- mapa nao gerado")
            else:
                norm   = mcolors.Normalize(vmin=min(h_arr), vmax=max(h_arr))
                cmap   = cm.viridis
                colors = [cmap(norm(h)) for h in h_arr]

                # Tamanho proporcional a sqrt(volume); fallback=200
                sizes = []
                for lbl in labels:
                    v = vol_by_t.get(lbl, 0)
                    sizes.append(max(80, min(600, v ** 0.5 * 8)) if v > 0 else 200)

                # Numero de colunas na lista lateral (max 50 arvores por coluna)
                MAX_LIST_COL = 50
                n_cols_list  = max(1, (len(labels) + MAX_LIST_COL - 1) // MAX_LIST_COL)

                # Tamanho das labels no mapa
                lbl_fontsize = max(3, 7 - max(0, len(labels) - 20) // 20)

                # Bounds dos dados com padding
                _x_lo, _x_hi = min(xs), max(xs)
                _y_lo, _y_hi = min(ys), max(ys)
                _xr = _x_hi - _x_lo or 10.0
                _yr = _y_hi - _y_lo or 10.0
                _px, _py = _xr * 0.07, _yr * 0.07

                # Altura da figura proporcional ao aspect ratio dos dados
                map_w  = 9.0
                col_w  = 4.0
                map_h  = max(5.0, min(13.0, map_w * (_yr + 2*_py) / (_xr + 2*_px)))
                fig_h  = map_h + 1.1
                fig_w  = map_w + col_w * n_cols_list

                fig = plt.figure(figsize=(fig_w, fig_h))
                gs  = GridSpec(1, 2,
                               width_ratios=[map_w, col_w * n_cols_list],
                               figure=fig,
                               left=0.07, right=0.99,
                               top=0.94, bottom=0.08,
                               wspace=0.15)
                ax  = fig.add_subplot(gs[0])
                ax2 = fig.add_subplot(gs[1])

                # --- Trajeto (desenhado antes das arvores para ficar por baixo) ---
                traj_plotado = False
                if trajectory_path is not None and trajectory_path.exists():
                    try:
                        import laspy as _laspy
                        import numpy as _np
                        _tlas = _laspy.read(str(trajectory_path))
                        tx = _np.array(_tlas.x, dtype=_np.float64)
                        ty = _np.array(_tlas.y, dtype=_np.float64)
                        ax.plot(tx, ty, color='#e05c00', linewidth=0.8,
                                alpha=0.8, zorder=3, label='Trajeto')
                        ax.plot(tx[0],  ty[0],  's', color='#e05c00', ms=7,
                                zorder=3, label='_nolegend_')
                        ax.plot(tx[-1], ty[-1], 'D', color='#e05c00', ms=7,
                                zorder=3, label='_nolegend_')
                        traj_plotado = True
                        print(f"  [OK] Trajeto adicionado ao mapa ({len(tx):,} pontos)")
                    except Exception as e_traj:
                        print(f"  [AVISO] Trajeto nao desenhado: {e_traj}")

                # --- Arvores ---
                ax.scatter(xs, ys, c=colors, s=sizes, zorder=2,
                           edgecolors='k', linewidths=0.5)
                _off = _xr * 0.015
                for x, y, lbl in zip(xs, ys, labels):
                    ax.text(x, y + _off, lbl, fontsize=lbl_fontsize, ha='center',
                            va='bottom', fontweight='bold', zorder=5,
                            path_effects=[_pe.withStroke(linewidth=0.8, foreground='white')])

                # Limites explícitos; adjustable='datalim' preenche axes sem encolher
                ax.set_xlim(_x_lo - _px, _x_hi + _px)
                ax.set_ylim(_y_lo - _py, _y_hi + _py)
                ax.set_aspect('equal', adjustable='datalim')
                ax.grid(True, alpha=0.3, color='gray')
                ax.set_xlabel('X (m)', fontsize=8)
                ax.set_ylabel('Y (m)', fontsize=8)
                ax.tick_params(labelsize=7)
                ax.set_title(f"{input_filename} — {n_troncos} arvores", fontsize=10)

                # Colorbar anexada ao mapa (label por cima para nao transbordar)
                _cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                                     ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
                _cbar.ax.tick_params(labelsize=7)
                _cbar.ax.set_title('Altura\n(m)', fontsize=7, pad=3)

                # Legenda de volume (+ trajeto se presente)
                if vol_by_t:
                    for v_ref, lbl_ref in [(200, '0.200 m³'), (500, '0.500 m³'), (1000, '1.000 m³')]:
                        s_ref = max(80, min(600, v_ref ** 0.5 * 8))
                        ax.scatter([], [], c='gray', s=s_ref, edgecolors='k',
                                   linewidths=0.5, label=lbl_ref)
                if vol_by_t or traj_plotado:
                    ax.legend(title='Volume (perfil)' if vol_by_t else None,
                              loc='lower right', fontsize=8)

                # --- Painel direito: lista por colunas de MAX_LIST_COL arvores ---
                ax2.axis('off')
                # Linha fina de separacao entre mapa e lista
                ax2.plot([0.0, 0.0], [0.0, 1.0], color='#aaaaaa', linewidth=0.7,
                         transform=ax2.transAxes, clip_on=False)
                header_l = f"{'ID':<8} {'Vol (m3)':>9}"
                sep_l    = '-' * 18
                tree_list = sorted(
                    zip(labels, h_arr, [vol_by_t.get(lbl, 0) for lbl in labels]),
                    key=lambda r: _tree_sort_key(r[0])
                )
                for ci in range(n_cols_list):
                    chunk = tree_list[ci * MAX_LIST_COL : (ci + 1) * MAX_LIST_COL]
                    col_lines = [header_l, sep_l]
                    for lbl, _h, v in chunk:
                        vol_str = f"{v/1000:.3f}" if v > 0 else "-"
                        col_lines.append(f"{lbl:<8} {vol_str:>9}")
                    if ci == n_cols_list - 1:
                        col_lines.append(sep_l)
                        col_lines.append(f"{'TOTAL':8} {vol_perfil/1000:>9.3f}  (perfil)")
                        if vol_mesh > 0:
                            col_lines.append(f"{'':8} {vol_mesh/1000:>9.3f}  (mesh)")
                    x_pos = ci / n_cols_list
                    ax2.text(x_pos + 0.06, 0.97, '\n'.join(col_lines),
                             transform=ax2.transAxes,
                             va='top', ha='left',
                             fontsize=6,
                             fontfamily='monospace')

                out_map = resultados_dir / "mapa_arvores.png"
                fig.savefig(str(out_map), dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  [OK] {out_map.name}")

    except Exception as e_map:
        print(f"  [AVISO] Mapa nao gerado: {e_map}")
        traceback.print_exc()

    # ---- 5. Lista de arvores em texto (lista_arvores.txt) ----
    try:
        tree_stats_csv_5 = arquivo_dir / "fase2" / "tree_stats.csv"

        # T-label -> (centro_x, centro_y)
        coords_by_t = {}
        if tree_stats_csv_5.exists():
            with open(tree_stats_csv_5, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    try:
                        tnum = row['ficheiro'].rsplit('_T', 1)[-1].split('.')[0]
                        lbl  = f"T{int(tnum)}"
                        coords_by_t[lbl] = (float(row['centro_x']), float(row['centro_y']))
                    except (ValueError, KeyError):
                        pass

        # Enriquecer com troncos separados na fase2_5 (ex: T0_a, T0_b)
        fase25_trees = arquivo_dir / "fase2_5" / "trees"
        if fase25_trees.exists():
            for meta_path in sorted(fase25_trees.glob("*_meta.json")):
                try:
                    meta     = json.loads(meta_path.read_text(encoding='utf-8'))
                    tid      = meta.get("tree_id", "")
                    if not tid:
                        continue
                    lbl_full = "T" + tid.rsplit('_T', 1)[-1]   # "T0_a"
                    raw      = lbl_full[1:]                      # "0_a"
                    if '_' in raw:
                        lbl_parent = "T" + raw.split('_')[0]    # "T0"
                        coords_by_t.pop(lbl_parent, None)       # remover entrada original
                        coords_by_t[lbl_full] = (
                            float(meta.get("cx_DBH", 0)),
                            float(meta.get("cy_DBH", 0)),
                        )
                except Exception:
                    pass

        # T-label -> volume_perfil_dm3, height_m, DBH_cm (summary_volume_v8.csv)
        vol_by_t_5  = {}
        height_by_t = {}
        dbh_by_t    = {}
        _sv8_5 = summary_v8_csv if summary_v8_csv.exists() else summary_csv
        if _sv8_5.exists():
            with open(_sv8_5, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    try:
                        key  = row.get('tree_id') or row.get('ficheiro', '')
                        lbl5 = "T" + key.rsplit('_T', 1)[-1].split('.')[0]  # "T0_a" ou "T10"
                        vol_by_t_5[lbl5]  = float(row.get('volume_perfil_dm3') or 0)
                        height_by_t[lbl5] = float(row.get('H_final_m') or row.get('height_m') or 0)
                        dbh_by_t[lbl5]    = float(row.get('DBH_cm') or 0)
                    except (ValueError, IndexError):
                        pass

        if coords_by_t:
            all_lbls = sorted(coords_by_t.keys(), key=_tree_sort_key)

            W_ID  = max(6, max(len(lb) for lb in all_lbls))
            W_VOL = 10
            W_X   = 13
            W_Y   = 13
            W_ALT = 8
            W_DBH = 9

            sep = f"  {'-'*W_ID}  {'-'*W_VOL}  {'-'*W_X}  {'-'*W_Y}  {'-'*W_ALT}  {'-'*W_DBH}"
            hdr = f"  {'ID':<{W_ID}}  {'Vol (m3)':>{W_VOL}}  {'X (m)':>{W_X}}  {'Y (m)':>{W_Y}}  {'Alt (m)':>{W_ALT}}  {'DBH (cm)':>{W_DBH}}"

            list_lines = [
                f"Lista de arvores — {input_filename}",
                "",
                hdr,
                sep,
            ]
            for lbl in all_lbls:
                x, y  = coords_by_t[lbl]
                v     = vol_by_t_5.get(lbl, 0)
                h_val = height_by_t.get(lbl, 0)
                dbh   = dbh_by_t.get(lbl, 0)
                vol_s = f"{v/1000:.3f}" if v > 0 else "-"
                alt_s = f"{h_val:.2f}"  if h_val > 0 else "-"
                dbh_s = f"{dbh:.1f}"    if dbh   > 0 else "-"
                list_lines.append(
                    f"  {lbl:<{W_ID}}  {vol_s:>{W_VOL}}  {x:>{W_X}.2f}  {y:>{W_Y}.2f}  {alt_s:>{W_ALT}}  {dbh_s:>{W_DBH}}"
                )
            list_lines.append(sep)
            list_lines.append(
                f"  {'TOTAL':<{W_ID}}  {vol_perfil/1000:>{W_VOL}.3f}"
            )
            out_lista = resultados_dir / "lista_arvores.txt"
            out_lista.write_text('\n'.join(list_lines) + '\n', encoding='utf-8')
            print(f"  [OK] {out_lista.name}")
        else:
            print("  [AVISO] Sem dados de coordenadas -- lista_arvores.txt nao gerado")

    except Exception as e_lst:
        print(f"  [AVISO] lista_arvores.txt nao gerado: {e_lst}")


def main():
    # =========================================================================
    # ARGUMENTOS
    # =========================================================================

    input_file, traj_file = _encontrar_inputs_trajeto()
    input_file = _sanitizar_nome(input_file)
    traj_file  = _sanitizar_nome(traj_file)
    output_dir = BASE_DIR / "output"
    especie = 'pinheiro'

    # =========================================================================
    # VALIDACOES INICIAIS
    # =========================================================================

    modo_str = "v8 (RANSAC + esqueleto + Kozak/potencia)"
    print("=" * 60)
    print(f"PIPELINE LiDAR - FLORESTAS PLANTADAS v8 com trajeto  [{modo_str}]")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    FASES_OUTPUT = ["fase1", "fase2", "fase2_5", "fase3", "fase4"]
    for fase_dir in FASES_OUTPUT:
        d = output_dir / fase_dir
        if d.exists():
            try:
                shutil.rmtree(d)
            except PermissionError as e:
                print(f"[ERRO] Nao foi possivel limpar {d}: {e.filename}")
                print(f"       Fechar o ficheiro noutras aplicacoes e tentar novamente.")
                return 1
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nInput (floresta): {input_file}")
    print(f"Input (trajeto):  {traj_file}")
    print(f"Output:           {output_dir}")
    print(f"Especie:          {especie.upper()}")
    print(f"Tamanho floresta: {input_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Tamanho trajeto:  {traj_file.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 60)

    # =========================================================================
    # CARREGAR CONFIGURACAO
    # =========================================================================

    try:
        config = get_preset(especie)
        print(f"\n[OK] Preset '{especie}' carregado")
    except Exception as e:
        print(f"\n[ERRO] Falha ao carregar preset: {e}")
        return 1

    config.BARK_R_LOW  = 0.000
    config.BARK_R_HIGH = 0.000

    # =========================================================================
    # FASE 1: PRE-PROCESSAMENTO PDAL
    # =========================================================================

    _t_fase: dict = {}
    _t_fase[1] = time.time()
    print("\n" + "=" * 60)
    print("[FASE 1] PRE-PROCESSAMENTO PDAL")
    print("=" * 60)
    print("  - Deduplicacao por voxel (5mm)")
    print("  - Remocao de ruido/outliers")
    print("  - Classificacao ground (CSF)")
    print("  - Calculo HAG (Height Above Ground)")
    print()

    try:
        fase1 = PDALPreprocessor(config)
        resultado1 = fase1.processar(input_file, output_dir / "fase1")

        if resultado1.status != StatusFase.SUCESSO:
            print(f"\n[ERRO] Fase 1 falhou: {resultado1.erro}")
            return 1

        print(f"\n[OK] Fase 1 completa! Output: {resultado1.ficheiro_output}")

    except Exception as e:
        print(f"\n[ERRO] Excecao na Fase 1: {e}")
        traceback.print_exc()
        return 1
    _t_fase[1] = time.time() - _t_fase[1]
    print(f"  [Tempo Fase 1: {_t_fase[1]:.0f}s]")

    # =========================================================================
    # FASE 2: SEPARACAO DE ARVORES (Instance Segmentation)
    # =========================================================================

    _t_fase[2] = time.time()
    print("\n" + "=" * 60)
    print("[FASE 2] SEPARACAO DE ARVORES (Instance Segmentation)")
    print("=" * 60)
    print("  - TLS2trees instance.py")
    print()

    try:
        fase2 = InstanceSegmentation(config)
        resultado2 = fase2.processar(resultado1.ficheiro_output, output_dir / "fase2")

        if resultado2.status == StatusFase.ERRO:
            print(f"\n[ERRO] Fase 2 falhou: {resultado2.erro}")
            return 1

        trees_dir = output_dir / "fase2" / "trees"
        print(f"\n[OK] Fase 2 completa! {resultado2.n_arvores} arvores detectadas")
        print(f"     Output: {trees_dir}")

    except Exception as e:
        print(f"\n[ERRO] Excecao na Fase 2: {e}")
        traceback.print_exc()
        return 1
    _t_fase[2] = time.time() - _t_fase[2]
    print(f"  [Tempo Fase 2: {_t_fase[2]:.0f}s]")

    # =========================================================================
    # FASE 2.5: PER-TREE SETUP
    # =========================================================================

    _t_fase[25] = time.time()
    trees_dir_fase3 = output_dir / "fase2" / "trees"
    print("\n" + "=" * 60)
    print("[FASE 2.5] PER-TREE SETUP")
    print("=" * 60)
    trees_dir_clean = output_dir / "fase2_5" / "trees"
    try:
        setup25 = PerTreeSetup(config)
        resultados_25 = setup25.processar(
            output_dir / "fase2" / "trees", trees_dir_clean
        )
        n_aceites = sum(1 for r in resultados_25 if r.status == StatusFase.SUCESSO)
        n_rejeit  = sum(1 for r in resultados_25 if r.status == StatusFase.REJEITADA)
        print(f"\n[OK] Fase 2.5 completa! Aceites: {n_aceites} | Rejeitadas: {n_rejeit}")
        trees_dir_fase3 = trees_dir_clean
    except Exception as e:
        print(f"\n[AVISO] Fase 2.5 falhou: {e} — a continuar com pasta original da Fase 2")
        traceback.print_exc()
    _t_fase[25] = time.time() - _t_fase[25]
    print(f"  [Tempo Fase 2.5: {_t_fase[25]:.0f}s]")

    # =========================================================================
    # FASE 3: TRONCO
    # =========================================================================

    _t_fase[3] = time.time()

    print("\n" + "=" * 60)
    print("[FASE 3] TAPER FITTING (RANSAC + esqueleto + Kozak/potencia)")
    print("=" * 60)
    try:
        fase3_v8 = TaperFitV8(config)
        tree_stats_csv = output_dir / "fase2" / "tree_stats.csv"
        resultado3 = fase3_v8.processar(trees_dir_fase3, output_dir / "fase3",
                                         tree_stats_csv)
        print(f"\n[OK] Fase 3 completa! Troncos: {resultado3.n_troncos}")
    except Exception as e:
        print(f"\n[ERRO] Excecao na Fase 3: {e}")
        traceback.print_exc()
        return 1

    _t_fase[3] = time.time() - _t_fase[3]
    print(f"  [Tempo Fase 3: {_t_fase[3]:.0f}s]")

    # =========================================================================
    # FASE 4: VOLUME DO TRONCO
    # =========================================================================

    _t_fase[4] = time.time()

    print("\n" + "=" * 60)
    print("[FASE 4] VOLUME ANALITICO (integral do taper)")
    print("=" * 60)
    try:
        fase4_v8 = VolumeV8(config)
        resultado4 = fase4_v8.processar(output_dir / "fase3", output_dir / "fase4")
        print(f"\n[OK] Fase 4 completa!")
        print(f"     Processadas: {resultado4.n_processados} | Sucesso: {resultado4.n_sucesso}")
    except Exception as e:
        print(f"\n[ERRO] Excecao na Fase 4: {e}")
        traceback.print_exc()
        return 1
    _t_fase[4] = time.time() - _t_fase[4]
    print(f"  [Tempo Fase 4: {_t_fase[4]:.0f}s]")

    # =========================================================================
    # ARQUIVO: mover outputs das fases para pasta com nome do input + timestamp
    # =========================================================================

    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    input_stem = input_file.stem
    arquivo_dir = output_dir / f"{input_stem}_{timestamp}"

    print("\n" + "=" * 60)
    print("[ARQUIVO] A guardar outputs...")
    print("=" * 60)

    arquivo_ok = False
    try:
        arquivo_dir.mkdir(parents=True, exist_ok=True)
        for fase_dir in FASES_OUTPUT:
            src = output_dir / fase_dir
            dst = arquivo_dir / fase_dir
            if src.exists():
                shutil.move(str(src), str(dst))
        (output_dir / "_last_run_path.txt").write_text(str(arquivo_dir), encoding="utf-8")
        arquivo_ok = True
        print(f"  Outputs movidos para: {arquivo_dir}")
    except Exception as e:
        print(f"  [AVISO] Nao foi possivel arquivar outputs: {e}")
        print(f"  Os resultados permanecem em: {output_dir}/fase*/")
        (output_dir / "_last_run_path.txt").write_text(str(output_dir), encoding="utf-8")

    target_dir = arquivo_dir if arquivo_ok else output_dir

    # =========================================================================
    # RESULTADOS: LAS com IDs + resumo TXT + mapa (com trajeto)
    # =========================================================================

    try:
        _gerar_resultados(target_dir, input_stem, input_file.name,
                          trajectory_path=traj_file)
    except Exception as e:
        print(f"\n  [AVISO] Nao foi possivel gerar Resultados/: {e}")
        traceback.print_exc()

    # =========================================================================
    # RESULTADO FINAL
    # =========================================================================

    print("\n" + "=" * 60)
    print("[OK] PIPELINE COMPLETO!")
    print("=" * 60)

    print(f"\nResultados em: {arquivo_dir}")
    print("\nEstrutura de output:")
    print(f"  {arquivo_dir.name}/")
    print(f"  |-- fase1/                     <- Nuvem pre-processada (.las)")
    print(f"  |-- fase2/                     <- Arvores individuais (.ply)")
    print(f"  |-- fase3/                     <- Troncos extendidos")
    print(f"  |   |-- *_extended.ply         <- Tronco + completion + extensoes")
    print(f"  |   |-- *_tronco.ply           <- Tronco limpo + esqueleto")
    print(f"  |   \\-- *_perfil.csv           <- Perfil de diametro por fatia")
    print(f"  |-- fase4/                     <- Volume do tronco")
    print(f"  |   |-- *.mesh.ply             <- Mesh reconstruida")
    print(f"  |   \\-- summary_volume.csv     <- volumes por arvore")
    print(f"  \\-- Resultados/                <- OUTPUTS FINAIS")
    print(f"      |-- *_com_IDs.las          <- LAS com campo tree_id por ponto")
    print(f"      |-- sumario.txt            <- resumo: arvores, volumes, alturas")
    print(f"      |-- mapa_arvores.png        <- mapa 2D com arvores + trajeto")
    print(f"      \\-- lista_arvores.txt      <- lista: ID, volume, coordenadas")

    metrics_file = arquivo_dir / "fase4" / "summary_volume.csv"
    if metrics_file.exists():
        print(f"\n[OK] Sumario de volumes: {metrics_file}")
        print("     Abrir no Excel para ver volumes por arvore.")
    else:
        print(f"\n[AVISO] summary_volume.csv nao encontrado.")

    # =========================================================================
    # LIMPEZA: apagar APENAS o ficheiro de floresta (trajeto fica em input_teste_com_trajeto/)
    # =========================================================================

    if arquivo_ok:
        try:
            input_file.unlink()
            print(f"\n[OK] Ficheiro de floresta apagado: {input_file.name}")
        except Exception as e:
            print(f"\n[AVISO] Nao foi possivel apagar o ficheiro de floresta: {e}")
        try:
            traj_file.unlink()
            print(f"[OK] Ficheiro de trajeto apagado: {traj_file.name}")
        except Exception as e:
            print(f"\n[AVISO] Nao foi possivel apagar o ficheiro de trajeto: {e}")
    else:
        print(f"\n[AVISO] Arquivo incompleto -- ficheiro de input NAO apagado: {input_file.name}")

    _t_total = sum(_t_fase.values())
    print(f"\n  Tempos: Fase 1={_t_fase.get(1,0):.0f}s | Fase 2={_t_fase.get(2,0):.0f}s | "
          f"Fase 3={_t_fase.get(3,0):.0f}s | Fase 4={_t_fase.get(4,0):.0f}s")
    print(f"  Total pipeline: {_t_total:.0f}s ({_t_total/60:.1f} min)")
    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[INTERROMPIDO] Pipeline cancelado pelo utilizador.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO FATAL] Excecao nao tratada: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
