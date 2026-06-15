"""
FASE 2: INSTANCE SEGMENTATION (Separação de Árvores)
=====================================================
Versão Corrigida v4.0

Separação de árvores individuais a partir da point cloud pré-processada.
NOTA: Esta fase agora é executada ANTES da segmentação semântica.

CORREÇÕES CRÍTICAS:
1. Parâmetros do TLS2trees instance.py corrigidos
2. --find-stems-boundary requer 2 valores (não 1)
3. Usar repositório tls-tools-ucl (não philwilkes)
4. Fallback DBSCAN com parâmetros ajustados

Output:
- Ficheiros individuais por árvore (PLY ou LAS)
- Estatísticas das árvores detetadas
"""

import os
import sys
import subprocess
import threading
import time
import numpy as np
import laspy
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False

try:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import pdist
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from config import PipelineConfig, Config, StatusFase, LABEL_DEBRIS


@dataclass
class InfoArvore:
    """Informação de uma árvore individual"""
    id: int
    n_pontos: int
    centro_x: float
    centro_y: float
    z_min: float
    z_max: float
    altura: float
    ficheiro: Optional[Path] = None


@dataclass
class ResultadoFase2:
    """Resultado da Fase 2 (Instance Segmentation)"""
    status: StatusFase
    metodo_usado: str = ""  # 'tls2trees', 'dbscan_fallback'
    ficheiros_arvores: List[Path] = field(default_factory=list)
    n_arvores: int = 0
    arvores_info: List[InfoArvore] = field(default_factory=list)
    distancia_media: float = 0.0
    distancia_min: float = 0.0
    tempo_execucao: float = 0.0
    erro: Optional[str] = None
    avisos: List[str] = field(default_factory=list)


class InstanceSegmentation:
    """
    Instance segmentation para separar árvores individuais
    
    Métodos:
    1. TLS2trees instance.py (preferido)
    2. DBSCAN clustering (fallback)
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or Config
        self._verificar_backend()
    
    def _verificar_backend(self):
        """Verifica se TLS2trees instance.py está disponível e é a versão correta"""
        self.tem_tls2trees = False
        self.tls2trees_versao = None
        
        if self.config.TLS2TREES_PATH:
            instance_py = self.config.TLS2TREES_PATH / "tls2trees" / "instance.py"
            if not instance_py.exists():
                instance_py = self.config.TLS2TREES_PATH / "instance.py"
            
            if instance_py.exists():
                self.tls2trees_instance = instance_py
                
                # Verificar versão do TLS2trees
                versao_ok, versao = self._verificar_versao_tls2trees()
                
                if versao_ok:
                    self.tem_tls2trees = True
                    self.tls2trees_versao = versao
                    print(f"  [OK] TLS2trees instance.py: {instance_py}")
                    print(f"    Versão: {versao}")
                else:
                    print(f"  [AVISO] TLS2trees versão incorreta ou não detetada")
                    print(f"    Usar fallback DBSCAN")
            else:
                print(f"  [AVISO] TLS2trees instance.py não encontrado")
        
        if not self.tem_tls2trees:
            if HAS_SKLEARN:
                print("  [OK] Fallback DBSCAN disponível")
            else:
                print("  [ERRO] sklearn não instalado - fallback indisponível")
    
    def _verificar_versao_tls2trees(self) -> Tuple[bool, str]:
        """
        Verifica se TLS2trees é a versão correta (tls-tools-ucl)
        
        A versão CORRETA (tls-tools-ucl) usa: --find-stems-boundary <min> <max>
        A versão ANTIGA (philwilkes) usa: --find-stems-height <h> --find-stems-thickness <t>
        
        Returns:
            (versao_correta, nome_versao)
        """
        try:
            result = subprocess.run(
                [sys.executable, str(self.tls2trees_instance), '--help'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            help_text = result.stdout + result.stderr
            
            if '--find-stems-boundary' in help_text:
                return True, "tls-tools-ucl (correta)"
            elif '--find-stems-height' in help_text:
                print("  [ERRO] TLS2trees versão ERRADA (philwilkes/TLS2trees)!")
                print("    Esta versão usa parâmetros diferentes.")
                print("    Instalar versão correta:")
                print("    git clone https://github.com/tls-tools-ucl/TLS2trees.git")
                return False, "philwilkes (incompatível)"
            else:
                # Versão desconhecida - tentar usar
                return True, "desconhecida (tentando usar)"
                
        except Exception as e:
            print(f"    [AVISO] Erro ao verificar TLS2trees: {e}")
            return True, "não verificada"
    
    def _preparar_ficheiros_tls2trees(self, input_file: Path, work_dir: Path) -> Tuple[Path, Path]:
        """
        Prepara ficheiros no formato esperado pelo TLS2trees

        TLS2trees espera:
        - Ficheiro PLY com pontos segmentados (nome.segmented.ply)
        - Tile index com 5 colunas: "TILE X Y Z PATH" (space-delimited)

        CRÍTICO: O nome do tile deve coincidir EXATAMENTE com o que instance.py
        extrai do nome do ficheiro: params.n = params.fn.split('.')[0]
        """
        # Converter para PLY se necessário
        if input_file.suffix.lower() == '.las':
            las = laspy.read(str(input_file))

            # Obter HAG (Height Above Ground) - necessário para criar pré-labels
            n_z = None
            if hasattr(las, 'HeightAboveGround'):
                n_z = np.array(las.HeightAboveGround)
            elif hasattr(las, 'n_z'):
                n_z = np.array(las.n_z)
            else:
                # Calcular n_z como altura acima do mínimo Z
                z_array = np.array(las.z)
                n_z = (z_array - np.min(z_array)).astype(np.float32)

            # Remover debris marcado na Fase 1 (LABEL_DEBRIS=5)
            if 'label' in las.point_format.dimension_names:
                debris_mask = np.array(las['label']) == LABEL_DEBRIS
                n_debris_pts = int(np.sum(debris_mask))
                if n_debris_pts > 0:
                    keep = ~debris_mask
                    las.points = las.points[keep]
                    n_z = n_z[keep]
                    print(f"    Removidos {n_debris_pts:,} pontos de debris (label={LABEL_DEBRIS})")

            # Verificar se já existem labels semânticos válidos
            tem_labels_semanticos = False
            if hasattr(las, 'label'):
                labels_raw = np.array(las.label)
                # Verificar se há labels válidos (mais que só 0 ou 2)
                unique_labels = np.unique(labels_raw)
                if len(unique_labels) > 2 or self.config.LABEL_WOOD in unique_labels:
                    tem_labels_semanticos = True
                    labels = labels_raw

            if tem_labels_semanticos:
                # Remapear labels do pipeline para esquema TLS2trees instance.py
                # instance.py espera: label 3 = stem/wood, label 1 = leaves/vegetation
                # Pipeline usa: label 4 = wood, label 2 = vegetation, label 1 = terrain
                labels_remapped = np.zeros_like(labels)
                labels_remapped[labels == self.config.LABEL_TERRAIN] = 0     # terrain -> 0
                labels_remapped[labels == self.config.LABEL_VEGETATION] = 1  # vegetation -> 1 (leaf)
                labels_remapped[labels == self.config.LABEL_CWD] = 2         # cwd -> 2
                labels_remapped[labels == self.config.LABEL_WOOD] = 3        # wood -> 3 (stem)
                labels = labels_remapped
                print(f"    Usando labels semânticos existentes")
            else:
                # PRÉ-LABELS POR SKELETON-FOLLOWING (DAP → topo)
                # ─────────────────────────────────────────────────────────────
                # Estratégia:
                #   1. Detetar centro(s) do tronco à altura do peito (1.2–1.4m)
                #      com DBSCAN (eps=0.1m, min_samples=20).
                #   2. A partir de 1.3m, traçar cada tronco para CIMA em slices
                #      de 0.25m, seguindo o centróide real a cada nível
                #      (skeleton-following). Resolve o problema de trunk drift
                #      onde o tronco deriva horizontalmente acima de 3m.
                #   3. Labeling:
                #      - HAG < 0.3m                          → terrain=0
                #      - 0.3m ≤ HAG < 1.3m, dentro do raio   → stem=3
                #      - 0.3m ≤ HAG < 1.3m, fora do raio     → terrain=0
                #      - HAG ≥ 1.3m, dentro do skeleton       → stem=3
                #      - HAG ≥ 1.3m, fora do skeleton         → leaf=1
                # ─────────────────────────────────────────────────────────────
                print(f"    Criando pre-labels por skeleton-following (DAP->topo)")

                BREAST_HEIGHT    = 1.3    # altura do peito (m)
                BREAST_HALF      = 0.1    # fatia ±0.1m para deteção inicial
                DBSCAN_EPS_SK    = 0.1    # raio DBSCAN para detetar troncos no peito
                DBSCAN_MIN_SK    = 20     # mínimo de pontos por cluster
                SEARCH_RADIUS_UP = 0.5    # raio de busca ao subir slice a slice
                MIN_PTS_SLICE_SK = 5      # mínimo pontos para atualizar centro
                SLICE_STEP_UP    = 0.25   # passo de subida (m)
                trunk_radius = getattr(self.config, 'INSTANCE_TRUNK_RADIUS', 0.15)

                x_arr   = np.array(las.x, dtype=np.float32)
                y_arr   = np.array(las.y, dtype=np.float32)
                n_z_f32 = n_z.astype(np.float32)

                mask_terrain = n_z_f32 < 0.3

                # 1. Detetar troncos na fatia do peito (1.2–1.4m)
                bh_mask = (~mask_terrain) & \
                          (n_z_f32 >= BREAST_HEIGHT - BREAST_HALF) & \
                          (n_z_f32 <= BREAST_HEIGHT + BREAST_HALF)
                trunk_starts = []
                if np.sum(bh_mask) >= DBSCAN_MIN_SK:
                    try:
                        from sklearn.cluster import DBSCAN as _DBSCAN
                        bh_pts = np.column_stack([x_arr[bh_mask], y_arr[bh_mask]])
                        db = _DBSCAN(eps=DBSCAN_EPS_SK, min_samples=DBSCAN_MIN_SK).fit(bh_pts)
                        min_r_dap = getattr(self.config, 'MIN_STEM_RADIUS_AT_DAP', 0.025)
                        n_rej_r = 0
                        for lbl in sorted(set(db.labels_)):
                            if lbl == -1:
                                continue
                            m = db.labels_ == lbl
                            cx = float(np.mean(bh_pts[m, 0]))
                            cy = float(np.mean(bh_pts[m, 1]))
                            d_c = np.sqrt((bh_pts[m, 0] - cx)**2 + (bh_pts[m, 1] - cy)**2)
                            r_p75 = float(np.percentile(d_c, 75))
                            if r_p75 < min_r_dap:
                                n_rej_r += 1
                                continue  # cluster demasiado fino — provavel arbusto
                            trunk_starts.append((cx, cy))
                        if n_rej_r:
                            print(f"    Clusters rejeitados por raio < {min_r_dap*100:.0f}cm: {n_rej_r}")
                    except Exception as e:
                        print(f"\n[ERRO] DBSCAN falhou ao detectar troncos na altura do peito ({BREAST_HEIGHT}m): {e}")
                        print(f"  Pontos na fatia {BREAST_HEIGHT}m: {int(np.sum(bh_mask))}")
                        print(f"  Parametros DBSCAN: eps={DBSCAN_EPS_SK}, min_samples={DBSCAN_MIN_SK}")
                        print(f"  Verificar que sklearn esta instalado: pip install scikit-learn")
                        import traceback
                        traceback.print_exc()
                        raise

                if not trunk_starts:
                    n_bh = int(np.sum(bh_mask))
                    raise RuntimeError(
                        f"Nenhum tronco detectado na altura do peito ({BREAST_HEIGHT}m).\n"
                        f"  Pontos na fatia DAP ({BREAST_HEIGHT-BREAST_HALF:.1f}–{BREAST_HEIGHT+BREAST_HALF:.1f}m): {n_bh}\n"
                        f"  Minimo DBSCAN: {DBSCAN_MIN_SK} pontos\n"
                        f"  Possivel causa: nuvem de pontos muito esparsa ou HAG incorreto.\n"
                        f"  Verificar que a Fase 1 (PDAL/CSF/HAG) correu correctamente."
                    )
                else:
                    print(f"    Troncos detetados a {BREAST_HEIGHT}m: {len(trunk_starts)}")

                # 2. Skeleton-following: traçar para CIMA desde 1.3m até ao topo
                max_hag = float(np.max(n_z_f32))
                slice_heights_up = np.arange(BREAST_HEIGHT,
                                             max_hag + SLICE_STEP_UP,
                                             SLICE_STEP_UP)
                n_slices_up = len(slice_heights_up)

                skel_cx    = np.zeros((len(trunk_starts), n_slices_up), dtype=np.float32)
                skel_cy    = np.zeros((len(trunk_starts), n_slices_up), dtype=np.float32)
                skel_max_h = np.full(len(trunk_starts), BREAST_HEIGHT, dtype=np.float32)

                for t_idx, (cx0, cy0) in enumerate(trunk_starts):
                    cx, cy = cx0, cy0
                    for s_idx, h in enumerate(slice_heights_up):
                        slice_mask = (~mask_terrain) & \
                                     (n_z_f32 >= h - SLICE_STEP_UP / 2) & \
                                     (n_z_f32 <  h + SLICE_STEP_UP / 2)
                        if np.sum(slice_mask) >= MIN_PTS_SLICE_SK:
                            sx = x_arr[slice_mask]
                            sy = y_arr[slice_mask]
                            d  = np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2)
                            near = d <= SEARCH_RADIUS_UP
                            if np.sum(near) >= MIN_PTS_SLICE_SK:
                                cx = float(np.mean(sx[near]))
                                cy = float(np.mean(sy[near]))
                                skel_max_h[t_idx] = max(float(skel_max_h[t_idx]), h)
                        skel_cx[t_idx, s_idx] = cx
                        skel_cy[t_idx, s_idx] = cy

                print(f"    Skeleton: {n_slices_up} slices x {len(trunk_starts)} tronco(s) "
                      f"(step={SLICE_STEP_UP}m, search={SEARCH_RADIUS_UP}m)")

                # Filtrar arbustos: rejeitar troncos cujo skeleton nao chega a MIN_TRUNK_TRACK_HEIGHT
                MIN_TRACK_H = getattr(self.config, 'MIN_TRUNK_TRACK_HEIGHT', 3.5)
                valid_trunks = [i for i, h in enumerate(skel_max_h) if h >= MIN_TRACK_H]
                n_arb = len(trunk_starts) - len(valid_trunks)
                if n_arb > 0:
                    print(f"    Arbustos rejeitados (track < {MIN_TRACK_H}m): {n_arb}")
                    for i, (cx0r, cy0r) in enumerate(trunk_starts):
                        if i not in valid_trunks:
                            print(f"      Cluster {i+1} @ ({cx0r:.1f},{cy0r:.1f}): "
                                  f"max_h={skel_max_h[i]:.1f}m")
                if not valid_trunks:
                    print(f"    [AVISO] Todos os clusters rejeitados — usando fallback sem filtro")
                    valid_trunks = list(range(len(trunk_starts)))
                else:
                    print(f"    Troncos validos apos filtro: {len(valid_trunks)}")

                # 3. Labeling vetorial
                n_pts_arr = len(las.points)
                labels = np.zeros(n_pts_arr, dtype=np.int32)  # default: terrain=0

                # Pontos acima do peito, fora do skeleton → copa=1 (default)
                mask_above = (~mask_terrain) & (n_z_f32 >= BREAST_HEIGHT)
                labels[mask_above] = 1

                # Aplicar skeleton para cada tronco (apenas troncos válidos — arbustos excluídos)
                for t_idx in valid_trunks:
                    cx0, cy0 = trunk_starts[t_idx]
                    # Abaixo do peito: cilindro fixo (troncos são verticais aqui)
                    dist_low = np.sqrt((x_arr - cx0) ** 2 + (y_arr - cy0) ** 2)
                    mask_stem_low = (~mask_terrain) & \
                                    (n_z_f32 >= 0.3) & \
                                    (n_z_f32 <  BREAST_HEIGHT) & \
                                    (dist_low <= trunk_radius)
                    labels[mask_stem_low] = 3

                    # Acima do peito: skeleton-following
                    above_idx = np.where(mask_above)[0]
                    if len(above_idx) == 0:
                        continue
                    above_hag = n_z_f32[above_idx]
                    raw_si = np.round(
                        (above_hag - BREAST_HEIGHT) / SLICE_STEP_UP
                    ).astype(int)
                    si     = np.clip(raw_si, 0, n_slices_up - 1)
                    cx_pt  = skel_cx[t_idx][si]
                    cy_pt  = skel_cy[t_idx][si]
                    dist_up = np.sqrt((x_arr[above_idx] - cx_pt) ** 2 +
                                      (y_arr[above_idx] - cy_pt) ** 2)
                    labels[above_idx[dist_up <= trunk_radius]] = 3

                n_terrain = int(np.sum(labels == 0))
                n_stem    = int(np.sum(labels == 3))
                n_veg     = int(np.sum(labels == 1))
                print(f"    Pré-labels: terrain={n_terrain}, stem={n_stem}, vegetation={n_veg}")
                print(f"    Skeleton: raio={trunk_radius}m, {n_slices_up} slices, step={SLICE_STEP_UP}m")

            # Criar PLY com campos necessários
            n_points = len(las.points)

            # TLS2trees instance.py espera: x, y, z, n_z, label
            # CRÍTICO: usar 'i4' (int32) porque ply_io.py só suporta 'int'
            # Se usar 'i2' (int16), o reader falha com KeyError
            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('n_z', 'f4'), ('label', 'i4')]
            vertices = np.zeros(n_points, dtype=dtype)
            vertices['n_z'] = n_z.astype(np.float32)

            vertices['x'] = np.array(las.x).astype(np.float32)
            vertices['y'] = np.array(las.y).astype(np.float32)
            vertices['z'] = np.array(las.z).astype(np.float32)
            vertices['label'] = labels.astype(np.int32)

            ply_element = PlyElement.describe(vertices, 'vertex')
            # Nome: stem.segmented.ply (TLS2trees adiciona sufixo automaticamente)
            ply_file = work_dir / f"{input_file.stem}.segmented.ply"
            PlyData([ply_element]).write(str(ply_file))

            x_min, x_max = np.min(las.x), np.max(las.x)
            y_min, y_max = np.min(las.y), np.max(las.y)
            z_min = np.min(las.z)
        else:
            ply_file = input_file
            plydata = PlyData.read(str(ply_file))
            x = plydata['vertex']['x']
            y = plydata['vertex']['y']
            z = plydata['vertex']['z']
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            z_min = np.min(z)

        # Calcular CENTRO do tile
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # CRÍTICO: O nome do tile DEVE coincidir com o que instance.py espera!
        # instance.py faz: params.n = params.fn.split('.')[0]
        # Se ply_file.name = "abc.segmented.ply", params.n = "abc"
        # Portanto, tile_name deve ser: ply_file.name.split('.')[0]
        tile_name = ply_file.name.split('.')[0]

        tile_index = work_dir / "tile_index.dat"
        with open(tile_index, 'w') as f:
            # CRÍTICO: instance.py lê com names=['tile', 'x', 'y', 'z', 'path']
            # Devemos fornecer TODAS as 5 colunas para evitar problemas
            ply_path = str(ply_file.resolve())
            f.write(f"{tile_name} {x_center:.3f} {y_center:.3f} {z_min:.3f} {ply_path}\n")

        print(f"    Tile index: {tile_name} @ ({x_center:.1f}, {y_center:.1f})")

        return ply_file, tile_index
    
    def _executar_tls2trees_instance(self, input_file: Path, output_dir: Path) -> Tuple[bool, List[Path], str]:
        """
        Executa TLS2trees instance.py
        
        INTERFACE CORRETA (da documentação):
        python instance.py -t <tile_name> --tindex <tile_index.dat> 
            -o <output_dir> --n-tiles <N> 
            --slice-thickness <float>
            --find-stems-boundary <min> <max>  # DOIS valores!
            --find-stems-min-radius <float>
            --find-stems-min-points <int>
            --graph-edge-length <float>
            --graph-maximum-cumulative-gap <float>
            --min-points-per-tree <int>
            --add-leaves --add-leaves-voxel-length <float>
            --pandarallel --verbose
        """
        print("\n  --> Tentando TLS2trees instance.py...")
        
        # Preparar diretórios
        work_dir = output_dir / "tls2trees_work"
        work_dir.mkdir(exist_ok=True)

        trees_dir = output_dir / "trees"
        trees_dir.mkdir(exist_ok=True)
        # Limpar ficheiros anteriores para evitar resultados obsoletos de runs anteriores
        for _old_ply in trees_dir.glob("*.ply"):
            _old_ply.unlink()

        # Preparar ficheiros
        try:
            ply_file, tile_index = self._preparar_ficheiros_tls2trees(input_file, work_dir)
        except Exception as e:
            return False, [], f"Erro ao preparar ficheiros: {e}"
        
        # CORREÇÃO: -t espera o nome COMPLETO do ficheiro PLY (com extensão)
        # Da documentação: python instance.py -t 001.downsample.segmented.ply
        tile_arg = ply_file.name  # Nome completo incluindo .ply
        
        # Construir comando com parâmetros CORRETOS
        cmd = [
            sys.executable,
            str(self.tls2trees_instance),
            '-t', tile_arg,  # Nome COMPLETO do ficheiro PLY
            '--tindex', str(tile_index),
            '-o', str(trees_dir),
            '--n-tiles', str(self.config.INSTANCE_N_TILES),
            '--slice-thickness', str(self.config.INSTANCE_SLICE_THICKNESS),
            # CORREÇÃO: --find-stems-boundary precisa de 2 valores!
            '--find-stems-boundary', 
            str(self.config.INSTANCE_FIND_STEMS_BOUNDARY[0]),
            str(self.config.INSTANCE_FIND_STEMS_BOUNDARY[1]),
            '--find-stems-min-radius', str(self.config.INSTANCE_FIND_STEMS_MIN_RADIUS),
            '--find-stems-min-points', str(self.config.INSTANCE_FIND_STEMS_MIN_POINTS),
            '--graph-edge-length', str(self.config.INSTANCE_GRAPH_EDGE_LENGTH),
            '--graph-maximum-cumulative-gap', str(self.config.INSTANCE_GRAPH_MAX_CUM_GAP),
            '--min-points-per-tree', str(self.config.INSTANCE_MIN_POINTS_PER_TREE),
            '--ignore-missing-tiles',  # Importante para single tile
            '--verbose'
        ]
        
        # Adicionar folhas se configurado
        if self.config.INSTANCE_ADD_LEAVES:
            cmd.extend([
                '--add-leaves',
                '--add-leaves-voxel-length', str(self.config.INSTANCE_ADD_LEAVES_VOXEL),
                '--add-leaves-edge-length', str(self.config.INSTANCE_ADD_LEAVES_EDGE)
            ])
        
        print(f"    Comando: {' '.join(cmd[:8])}...", flush=True)
        print(f"    A iniciar segment por instancias (pode demorar varios minutos)...", flush=True)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(work_dir),
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'KMP_DUPLICATE_LIB_OK': 'TRUE'}
            )

            # Heartbeat: imprimir progresso a cada 20s enquanto o TLS2trees trabalha
            t0_tls = time.time()
            def _heartbeat_tls():
                while proc.poll() is None:
                    time.sleep(20)
                    if proc.poll() is None:
                        print(f"    A pensar... ({int(time.time() - t0_tls)}s)", flush=True)
            hb = threading.Thread(target=_heartbeat_tls, daemon=True)
            hb.start()

            try:
                stdout_b, stderr_b = proc.communicate(timeout=self.config.TIMEOUT_INSTANCE)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                return False, [], f"TLS2trees timeout (>{self.config.TIMEOUT_INSTANCE}s)"

            elapsed_tls = int(time.time() - t0_tls)
            stdout_str = (stdout_b or b'').decode('utf-8', errors='replace')
            stderr_str = (stderr_b or b'').decode('utf-8', errors='replace')
            print(f" Segmentacao concluida em {elapsed_tls}s", flush=True)

            if proc.returncode != 0:
                # Mostrar as últimas linhas do stderr (o erro real, não progress bars)
                stderr_lines = stderr_str.strip().split('\n')
                # Filtrar progress bars do tqdm
                error_lines = [l for l in stderr_lines if not any(x in l for x in ['|', 'it/s]', 'it ['])]
                if error_lines:
                    erro = '\n'.join(error_lines[-10:])
                else:
                    erro = '\n'.join(stderr_lines[-10:])
                if stdout_str.strip():
                    stdout_tail = '\n'.join(stdout_str.strip().split('\n')[-5:])
                    erro = f"{erro}\n[stdout]: {stdout_tail}"
                return False, [], f"TLS2trees falhou (returncode={proc.returncode}):\n{erro}"
            
            # Apenas ficheiros leafoff (stem); leafon contém folhagem e não entra no pipeline
            tree_files = sorted(set(trees_dir.glob("*.leafoff.ply")) |
                                set(trees_dir.glob("*/*.leafoff.ply")))  # subpastas por diâmetro

            if not tree_files:
                for subdir in trees_dir.iterdir():
                    if subdir.is_dir():
                        tree_files.extend(list(subdir.glob("*.leafoff.ply")))

            if tree_files:
                return True, tree_files, "OK"
            else:
                return False, [], "TLS2trees não produziu ficheiros de árvores"
            
        except subprocess.TimeoutExpired:
            return False, [], f"TLS2trees timeout (>{self.config.TIMEOUT_INSTANCE}s)"
        except Exception as e:
            return False, [], f"Erro: {str(e)}"
    
    def _analisar_arvores(self, tree_files: List[Path]) -> List[InfoArvore]:
        """Analisa estatísticas de cada árvore"""
        arvores = []
        
        for i, tree_file in enumerate(tree_files):
            try:
                if tree_file.suffix.lower() == '.ply':
                    plydata = PlyData.read(str(tree_file))
                    x = np.array(plydata['vertex']['x'])
                    y = np.array(plydata['vertex']['y'])
                    z = np.array(plydata['vertex']['z'])
                else:
                    las = laspy.read(str(tree_file))
                    x, y, z = np.array(las.x), np.array(las.y), np.array(las.z)
                
                # Altura real da árvore: usar o leafon.ply correspondente se existir
                # (TLS2trees pode não capturar o tronco completo, mas o leafon tem a copa)
                altura_total = float(np.max(z) - np.min(z))
                leafon_path = tree_file.parent / tree_file.name.replace('.leafoff.', '.leafon.')
                if leafon_path.exists():
                    try:
                        lon = PlyData.read(str(leafon_path))
                        lon_v = lon['vertex']
                        if 'n_z' in lon_v.data.dtype.names:
                            altura_total = max(altura_total, float(np.max(lon_v['n_z'])))
                        else:
                            lon_z = np.array(lon_v['z'])
                            altura_total = max(altura_total, float(lon_z.max() - lon_z.min()))
                        del lon, lon_v
                    except Exception:
                        pass

                try:
                    tnum = int(tree_file.name.rsplit('_T', 1)[-1].split('.')[0])
                except (ValueError, IndexError):
                    tnum = i
                info = InfoArvore(
                    id=tnum,
                    n_pontos=len(x),
                    centro_x=float(np.mean(x)),
                    centro_y=float(np.mean(y)),
                    z_min=float(np.min(z)),
                    z_max=float(np.max(z)),
                    altura=altura_total,
                    ficheiro=tree_file
                )
                arvores.append(info)
                
            except Exception as e:
                print(f"    [AVISO] Erro ao analisar {tree_file.name}: {e}")
        
        return arvores
    
    def _validar_resultado(self, arvores: List[InfoArvore]) -> Tuple[bool, Dict[str, Any]]:
        """
        Valida o resultado da instance segmentation
        
        CHECKPOINT C: Verificações de consistência
        """
        print("\n  --> Checkpoint C: Validando instance segmentation...")
        
        stats = {'n_arvores': len(arvores)}
        avisos = []
        
        if len(arvores) == 0:
            print("    [ERRO] FALHOU: Nenhuma árvore detetada!")
            return False, stats
        
        print(f"    Árvores detetadas: {len(arvores)}")
        
        # Estatísticas de altura
        alturas = [a.altura for a in arvores]
        stats['altura_min'] = min(alturas)
        stats['altura_max'] = max(alturas)
        stats['altura_media'] = np.mean(alturas)
        
        print(f"    Alturas: min={stats['altura_min']:.1f}m, "
              f"max={stats['altura_max']:.1f}m, média={stats['altura_media']:.1f}m")
        
        # Verificar alturas
        arvores_baixas = sum(1 for a in arvores if a.altura < self.config.MIN_TREE_HEIGHT)
        if arvores_baixas > 0:
            avisos.append(f"{arvores_baixas} árvores abaixo de {self.config.MIN_TREE_HEIGHT}m")
        
        arvores_altas = sum(1 for a in arvores if a.altura > self.config.MAX_TREE_HEIGHT)
        if arvores_altas > 0:
            avisos.append(f"{arvores_altas} árvores acima de {self.config.MAX_TREE_HEIGHT}m (improvável)")
        
        # Calcular distâncias entre árvores
        if len(arvores) > 1:
            centros = np.array([[a.centro_x, a.centro_y] for a in arvores])
            distancias = pdist(centros)
            
            stats['distancia_min'] = float(np.min(distancias))
            stats['distancia_media'] = float(np.mean(distancias))
            
            print(f"    Distância mínima entre árvores: {stats['distancia_min']:.2f}m")
            print(f"    Distância média: {stats['distancia_media']:.2f}m")
            
            # Verificar espaçamento
            expected = self.config.EXPECTED_TREE_SPACING
            tolerance = self.config.SPACING_TOLERANCE
            
            if stats['distancia_min'] < expected * (1 - tolerance) * 0.3:
                avisos.append(f"Algumas árvores muito próximas ({stats['distancia_min']:.1f}m) - possível over-segmentation")
        
        # Pontos por árvore
        pontos = [a.n_pontos for a in arvores]
        stats['pontos_min'] = min(pontos)
        stats['pontos_max'] = max(pontos)
        stats['pontos_medio'] = int(np.mean(pontos))
        
        print(f"    Pontos/árvore: min={stats['pontos_min']}, "
              f"max={stats['pontos_max']}, média={stats['pontos_medio']}")
        
        # Mostrar avisos
        for aviso in avisos:
            print(f"    [AVISO] {aviso}")
        
        stats['avisos'] = avisos
        
        print("    [OK] Checkpoint C: PASSOU")
        return True, stats
    
    def _guardar_estatisticas(self, arvores: List[InfoArvore], output_dir: Path):
        """Guarda estatísticas das árvores em CSV"""
        stats_file = output_dir / "tree_stats.csv"
        
        with open(stats_file, 'w') as f:
            f.write("tree_id,n_pontos,centro_x,centro_y,z_min,z_max,altura,ficheiro\n")
            for a in arvores:
                f.write(f"{a.id},{a.n_pontos},{a.centro_x:.3f},{a.centro_y:.3f},"
                       f"{a.z_min:.3f},{a.z_max:.3f},{a.altura:.2f},{a.ficheiro.name}\n")
        
        print(f"    Estatísticas guardadas: {stats_file}")

    def _remapear_labels_para_pipeline(self, tree_files: List[Path]) -> List[Path]:
        """
        Remapeia labels dos ficheiros TLS2trees de volta para o esquema do pipeline

        TLS2trees instance.py usa:
        - 0 = terrain
        - 1 = leaf/vegetation
        - 3 = stem/wood

        Pipeline usa:
        - 1 = terrain (LABEL_TERRAIN)
        - 2 = vegetation (LABEL_VEGETATION)
        - 3 = CWD (LABEL_CWD)
        - 4 = wood (LABEL_WOOD)

        Mapeamento inverso:
        - 0 -> 1 (terrain)
        - 1 -> 2 (leaf -> vegetation)
        - 3 -> 4 (stem -> wood)
        """
        import gc

        remapped_files = []

        for ply_file in tree_files:
            try:
                # Ler ficheiro e extrair dados
                plydata = PlyData.read(str(ply_file))

                if 'label' not in plydata['vertex'].data.dtype.names:
                    del plydata
                    gc.collect()
                    remapped_files.append(ply_file)
                    continue

                # Extrair todos os dados necessários
                vertex_data = plydata['vertex'].data
                labels = np.array(vertex_data['label'])

                # Copiar todos os campos para um novo array
                new_dtype = []
                for name in vertex_data.dtype.names:
                    if name == 'label':
                        new_dtype.append(('label', '<i4'))
                    else:
                        new_dtype.append((name, vertex_data.dtype[name]))

                new_vertices = np.zeros(len(vertex_data), dtype=new_dtype)
                for name in vertex_data.dtype.names:
                    if name != 'label':
                        new_vertices[name] = vertex_data[name]

                # Libertar plydata antes de escrever
                del plydata
                del vertex_data
                gc.collect()

                # Remapear: TLS2trees -> Pipeline
                labels_remapped = np.zeros_like(labels, dtype=np.int32)
                labels_remapped[labels == 0] = self.config.LABEL_TERRAIN      # 0 -> 1
                labels_remapped[labels == 1] = self.config.LABEL_VEGETATION   # 1 -> 2
                labels_remapped[labels == 2] = self.config.LABEL_CWD          # 2 -> 3
                labels_remapped[labels == 3] = self.config.LABEL_WOOD         # 3 -> 4
                new_vertices['label'] = labels_remapped

                # Guardar para ficheiro temporário
                temp_file = ply_file.parent / f".tmp_{ply_file.name}"
                new_element = PlyElement.describe(new_vertices, 'vertex')
                PlyData([new_element]).write(str(temp_file))

                # Substituir ficheiro original
                import time
                time.sleep(0.1)  # Pequena pausa para Windows libertar handles
                ply_file.unlink()
                temp_file.rename(ply_file)

                remapped_files.append(ply_file)

            except Exception as e:
                print(f"    [AVISO] Erro ao remapear {ply_file.name}: {e}")
                remapped_files.append(ply_file)
                # Limpar ficheiro temporário se existir
                temp_file = ply_file.parent / f".tmp_{ply_file.name}"
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError:
                        pass

        return remapped_files

    def _filtrar_pontos_remotos(self, tree_files: List[Path]) -> List[Path]:
        """
        Remove pontos geometricamente afastados do eixo do tronco de cada leafoff.ply.

        Aplicado após TLS2trees (e fallback) para garantir que ramos residuais /
        folhagem no ar não contaminam as fases seguintes.

        Algoritmo:
          1. Lê x, y, n_z (HAG) do PLY
          2. Tracker sequencial a partir do DAP (h=1.3m): centro=mediana, r=p75
          3. Smooth rolling-median ±2 fatias
          4. Mantém pontos com dist <= r_smooth * 1.40
        """
        SLICE_H    = 0.10
        SMOOTH_WIN = 5
        CYL_F      = 1.40
        MIN_PTS_SL = 5
        SEARCH_F   = 2.5
        MAX_JUMP   = 0.30
        MIN_R      = 0.03

        def rolling_median(a, w):
            out = np.empty_like(a, dtype=np.float64)
            half = w // 2
            for i in range(len(a)):
                lo = max(0, i - half); hi = min(len(a), i + half + 1)
                out[i] = float(np.median(a[lo:hi]))
            return out

        def track(px, py, ph, hc0, direction, cx0, cy0, r0, h_lo, h_hi):
            slices, pcx, pcy, pr = [], cx0, cy0, r0
            hc = hc0 + direction * SLICE_H
            while h_lo - SLICE_H/2 <= hc <= h_hi + SLICE_H/2:
                in_band  = np.abs(ph - hc) < SLICE_H / 2
                search_r = max(pr * SEARCH_F, 0.15)
                in_s = in_band & (np.sqrt((px - pcx)**2 + (py - pcy)**2) <= search_r)
                if in_s.sum() >= MIN_PTS_SL:
                    mx = float(np.median(px[in_s]))
                    my = float(np.median(py[in_s]))
                    jump = np.sqrt((mx - pcx)**2 + (my - pcy)**2)
                    if jump > MAX_JUMP:
                        mx = pcx + (mx - pcx) * MAX_JUMP / jump
                        my = pcy + (my - pcy) * MAX_JUMP / jump
                    d_c = np.sqrt((px[in_s] - mx)**2 + (py[in_s] - my)**2)
                    pr  = max(float(np.percentile(d_c, 75)), MIN_R)
                    pcx, pcy = mx, my
                slices.append({'h': hc, 'cx': pcx, 'cy': pcy, 'r': pr})
                hc += direction * SLICE_H
            return slices

        cleaned = []
        for tf in tree_files:
            try:
                plydata = PlyData.read(str(tf))
                vd = plydata['vertex']
                px = np.array(vd['x'], dtype=np.float64)
                py = np.array(vd['y'], dtype=np.float64)
                pz = np.array(vd['z'], dtype=np.float64)
                ph = np.array(vd['n_z'], dtype=np.float64) if 'n_z' in vd.data.dtype.names \
                     else pz - pz.min()

                h_min, h_max = float(ph.min()), float(ph.max())
                h_anc = min(1.3, (h_min + h_max) / 2)
                in_anc = np.abs(ph - h_anc) < 0.30
                if in_anc.sum() < MIN_PTS_SL:
                    in_anc = np.ones(len(ph), dtype=bool)
                cx0 = float(np.median(px[in_anc]))
                cy0 = float(np.median(py[in_anc]))
                d0  = np.sqrt((px[in_anc] - cx0)**2 + (py[in_anc] - cy0)**2)
                r0  = max(float(np.percentile(d0, 75)), MIN_R)

                up_sl   = track(px, py, ph, h_anc, +1, cx0, cy0, r0, h_min, h_max)
                down_sl = track(px, py, ph, h_anc, -1, cx0, cy0, r0, h_min, h_max)
                all_sl  = list(reversed(down_sl)) + \
                          [{'h': h_anc, 'cx': cx0, 'cy': cy0, 'r': r0}] + up_sl

                ph_a   = np.array([s['h']  for s in all_sl])
                pcx_sm = rolling_median(np.array([s['cx'] for s in all_sl]), SMOOTH_WIN)
                pcy_sm = rolling_median(np.array([s['cy'] for s in all_sl]), SMOOTH_WIN)
                pr_sm  = rolling_median(np.array([s['r']  for s in all_sl]), SMOOTH_WIN)

                idx = np.clip(np.searchsorted(ph_a, ph, side='left'), 0, len(ph_a)-1)
                idx_lo = np.clip(idx - 1, 0, len(ph_a)-1)
                use_lo = np.abs(ph_a[idx_lo] - ph) <= np.abs(ph_a[idx] - ph)
                near   = np.where(use_lo, idx_lo, idx)

                out_rng = (ph < ph_a[0] - SLICE_H) | (ph > ph_a[-1] + SLICE_H)
                d_pt    = np.sqrt((px - pcx_sm[near])**2 + (py - pcy_sm[near])**2)
                keep    = out_rng | (d_pt <= pr_sm[near] * CYL_F)

                n_rem = int((~keep).sum())
                if n_rem > 0:
                    # Reconstruir PLY só com pontos filtrados
                    new_verts = vd.data[keep].copy()  # array contíguo
                    del plydata, vd                   # libertar handle do ficheiro
                    new_el = PlyElement.describe(new_verts, 'vertex')
                    PlyData([new_el]).write(str(tf))
                    print(f"    Filtro remoto ({tf.name}): -{n_rem:,} pts afastados removidos")

                cleaned.append(tf)
            except Exception as e:
                print(f"    [AVISO] Filtro remoto falhou em {tf.name}: {e}")
                cleaned.append(tf)

        return cleaned

    def processar(self, input_file: Path, output_dir: Path) -> ResultadoFase2:
        """
        Processa instance segmentation via TLS2trees instance.py.

        TLS2trees e o unico metodo suportado. Se nao estiver instalado
        ou falhar, o pipeline para com mensagem de diagnostico detalhada.
        """
        inicio = datetime.now()
        resultado = ResultadoFase2(status=StatusFase.EM_EXECUCAO)

        print(f"\n{'='*60}")
        print(f"FASE 2: INSTANCE SEGMENTATION")
        print(f"{'='*60}")
        print(f"Input: {input_file.name}")

        input_file = Path(input_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # TLS2trees e obrigatorio
        if not self.tem_tls2trees:
            resultado.status = StatusFase.ERRO
            resultado.erro = "TLS2trees nao encontrado"
            print(f"\n[ERRO] TLS2trees nao esta instalado ou nao e acessivel.")
            print(f"  Instalar: pip install tls2trees")
            print(f"  Ou verificar que o comando 'instance' do TLS2trees esta no PATH.")
            return resultado

        sucesso, tree_files, msg = self._executar_tls2trees_instance(input_file, output_dir)
        if not sucesso:
            resultado.status = StatusFase.ERRO
            resultado.erro = msg
            print(f"\n[ERRO] TLS2trees falhou para {input_file.name}:")
            print(f"  {msg}")
            print(f"  Causas comuns:")
            print(f"    1. Nuvem de pontos sem campo 'HeightAboveGround' (Fase 1 nao correu)")
            print(f"    2. Poucos pontos para segmentacao (< 1000 pontos)")
            print(f"    3. Erro de GPU/CUDA — verificar 'nvidia-smi' ou desactivar GPU")
            print(f"    4. Timeout ({self.config.TIMEOUT_INSTANCE}s) — aumentar config.TIMEOUT_INSTANCE")
            print(f"  Ficheiro de input: {input_file}")
            return resultado

        resultado.metodo_usado = "tls2trees"
        print(f"    [OK] TLS2trees sucesso: {len(tree_files)} arvores")

        # CRÍTICO: Remapear labels de TLS2trees (3=stem) para pipeline (4=wood)
        tree_files = self._remapear_labels_para_pipeline(tree_files)
        # Remover pontos geometricamente afastados do tronco
        tree_files = self._filtrar_pontos_remotos(tree_files)

        # Analisar árvores
        print("\n  --> Analisando árvores...")
        arvores = self._analisar_arvores(tree_files)

        # Filtrar instâncias demasiado pequenas (abaixo de MIN_TREE_HEIGHT)
        n_antes = len(arvores)
        arvores_ok = [a for a in arvores if a.altura >= self.config.MIN_TREE_HEIGHT]
        rejeitadas  = [a for a in arvores if a.altura <  self.config.MIN_TREE_HEIGHT]
        if rejeitadas:
            for a in rejeitadas:
                print(f"    Instancia {a.id} rejeitada: altura={a.altura:.2f}m "
                      f"(< {self.config.MIN_TREE_HEIGHT}m minimo)")
                # Apagar ficheiros PLY da instancia rejeitada (fase3 le a pasta directamente)
                for ply in (a.ficheiro.parent.glob(f"{a.ficheiro.stem.replace('.leafoff','')}.*")
                            if a.ficheiro else []):
                    try:
                        ply.unlink()
                    except Exception:
                        pass
                if a.ficheiro and a.ficheiro.exists():
                    a.ficheiro.unlink()
            ficheiros_ok = {a.ficheiro for a in arvores_ok}
            tree_files = [f for f in tree_files if f in ficheiros_ok]
            arvores = arvores_ok
            print(f"    Filtro altura: {n_antes} -> {len(arvores)} arvore(s) valida(s)")

        # Validar
        valido, stats = self._validar_resultado(arvores)
        
        # Guardar estatísticas
        if arvores:
            self._guardar_estatisticas(arvores, output_dir)
        
        # Preencher resultado
        resultado.ficheiros_arvores = tree_files
        resultado.n_arvores = len(arvores)
        resultado.arvores_info = arvores
        resultado.distancia_media = stats.get('distancia_media', 0)
        resultado.distancia_min = stats.get('distancia_min', 0)
        resultado.tempo_execucao = (datetime.now() - inicio).total_seconds()
        resultado.avisos.extend(stats.get('avisos', []))
        
        resultado.status = StatusFase.SUCESSO
        
        print(f"\n{'='*60}")
        print(f"[OK] FASE 2 CONCLUÍDA em {resultado.tempo_execucao:.1f}s")
        print(f"  Método: {resultado.metodo_usado}")
        print(f"  Árvores: {resultado.n_arvores}")
        print(f"{'='*60}")
        
        return resultado

# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fase 2: Instance Segmentation")
    parser.add_argument("input", help="Ficheiro .las/.ply segmentado")
    parser.add_argument("-o", "--output", default="output/fase3", help="Pasta de output")
    args = parser.parse_args()

    segmentador = InstanceSegmentation()

    resultado = segmentador.processar(
        Path(args.input),
        Path(args.output)
    )
    
    exit(0 if resultado.status == StatusFase.SUCESSO else 1)
