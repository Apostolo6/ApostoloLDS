"""
FASE 1: PRÉ-PROCESSAMENTO COM PDAL
==================================
Versão Corrigida v8.0

Funções:
- Remoção de ruído/outliers
- Classificação ground com CSF
- Cálculo de HAG (Height Above Ground)
- Centralização de coordenadas UTM (com preservação de offset)

CORREÇÕES v8.0:
- CRÍTICO: filters.outlier apenas MARCA pontos, não remove!
  Adicionado filters.range para efectivamente remover outliers (class=7)

CORREÇÕES v8.0:
- Melhor gestão de erros
- Checkpoints detalhados
- Preservação de offset para georreferenciação posterior
"""

import json
import subprocess
import threading
import time
import numpy as np
import laspy
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from config import PipelineConfig, Config, StatusFase, LABEL_DEBRIS


@dataclass
class ResultadoFase1:
    """Resultado da Fase 1"""
    status: StatusFase
    ficheiro_output: Optional[Path] = None
    offset: Optional[Dict[str, float]] = None
    n_pontos_input: int = 0
    n_pontos_output: int = 0
    pct_ground: float = 0.0
    hag_stats: Optional[Dict[str, float]] = None
    tempo_execucao: float = 0.0
    erro: Optional[str] = None


class PDALPreprocessor:
    """Pré-processamento de point clouds com PDAL - CORRIGIDO"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or Config
        self.offset = None
    
    def _verificar_pdal(self) -> bool:
        """Verifica se PDAL está instalado e funcional"""
        try:
            result = subprocess.run(
                ['pdal', '--version'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                versao = result.stdout.strip().split('\n')[0]
                print(f"  [OK] PDAL instalado: {versao}")
                return True
            else:
                print(f"  [ERRO] PDAL erro: {result.stderr}")
                return False
        except FileNotFoundError:
            print("  [ERRO] PDAL não encontrado!")
            print("    Instalar: conda install -c conda-forge pdal python-pdal")
            return False
        except Exception as e:
            print(f"  [ERRO] Erro ao verificar PDAL: {e}")
            return False
    
    def _detetar_filtro_hag(self) -> str:
        """
        Deteta qual filtro HAG está disponível no PDAL instalado
        
        - PDAL >= 2.4: filters.hag_nn (preferido)
        - PDAL 2.0-2.3: filters.hag_delaunay  
        - PDAL < 2.0: filters.hag
        
        Returns:
            Nome do filtro HAG a usar
        """
        try:
            # Verificar filtros disponíveis
            result = subprocess.run(
                ['pdal', '--drivers'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            drivers = result.stdout.lower()
            
            if 'filters.hag_nn' in drivers:
                print("    Filtro HAG: filters.hag_nn (PDAL 2.4+)")
                return 'filters.hag_nn'
            elif 'filters.hag_delaunay' in drivers:
                print("    Filtro HAG: filters.hag_delaunay (PDAL 2.0-2.3)")
                return 'filters.hag_delaunay'
            elif 'filters.hag' in drivers:
                print("    Filtro HAG: filters.hag (PDAL < 2.0)")
                return 'filters.hag'
            else:
                raise RuntimeError(
                    "Nenhum filtro HAG encontrado no PDAL instalado.\n"
                    "  Filtros esperados: filters.hag_nn (PDAL>=2.4), filters.hag_delaunay (2.0-2.3), filters.hag (<2.0)\n"
                    "  Verificar versao PDAL: pdal --version\n"
                    "  Reinstalar: conda install -c conda-forge pdal python-pdal"
                )

        except RuntimeError:
            raise  # propagar erros ja formatados
        except Exception as e:
            raise RuntimeError(
                f"Erro ao executar 'pdal --drivers' para detetar filtro HAG: {e}\n"
                "  Verificar que o PDAL esta instalado e acessivel no PATH.\n"
                "  Testar manualmente: pdal --drivers | grep hag"
            ) from e
    
    def _analisar_ficheiro(self, las_file: Path) -> Dict[str, Any]:
        """Analisa ficheiro LAS para determinar características"""
        las = laspy.read(str(las_file))
        
        info = {
            'n_pontos': len(las.points),
            'x_min': float(np.min(las.x)),
            'x_max': float(np.max(las.x)),
            'y_min': float(np.min(las.y)),
            'y_max': float(np.max(las.y)),
            'z_min': float(np.min(las.z)),
            'z_max': float(np.max(las.z)),
            'x_mean': float(np.mean(las.x)),
            'y_mean': float(np.mean(las.y)),
            'is_utm': False,
            'tem_intensidade': hasattr(las, 'intensity'),
            'tem_classificacao': hasattr(las, 'classification'),
            'point_format': las.point_format.id,
        }
        
        # Verificar se coordenadas UTM (valores muito grandes)
        if abs(info['x_mean']) > self.config.UTM_THRESHOLD or \
           abs(info['y_mean']) > self.config.UTM_THRESHOLD:
            info['is_utm'] = True
        
        return info
    
    def _centralizar_coordenadas(self, input_file: Path, output_file: Path) -> Dict[str, float]:
        """
        Centraliza coordenadas ANTES do processamento PDAL
        
        IMPORTANTE: Guarda o offset para poder georreferenciar depois
        """
        print("  --> Centralizando coordenadas UTM...")
        
        las = laspy.read(str(input_file))
        
        # Calcular centro
        x_mean = float(np.mean(las.x))
        y_mean = float(np.mean(las.y))
        z_min = float(np.min(las.z))
        
        offset = {
            'x_offset': x_mean,
            'y_offset': y_mean,
            'z_offset': z_min,
            'ficheiro_original': str(input_file),
            'timestamp': datetime.now().isoformat()
        }
        
        # Aplicar offset
        las.x = las.x - x_mean
        las.y = las.y - y_mean
        las.z = las.z - z_min
        
        # Guardar
        las.write(str(output_file))
        
        print(f"    Offset aplicado: X={x_mean:.2f}, Y={y_mean:.2f}, Z={z_min:.2f}")
        
        return offset
    
    def _criar_pipeline_pdal(self, input_file: str, output_file: str) -> dict:
        """
        Cria pipeline JSON do PDAL otimizado para MLS pedestre

        Pipeline SIMPLIFICADO (v8.0):
        1. Ler ficheiro
        2. Filtro estatístico de outliers (rápido)
        3. Remover outliers marcados
        4. CSF para classificação ground
        5. Calcular HAG
        6. Guardar

        NOTA: O filtro radius foi REMOVIDO porque é O(n²) e extremamente lento.
        Para remoção de clusters isolados, usar DBSCAN pós-processamento (opcional).
        """

        # Detetar filtro HAG disponível
        filtro_hag = self._detetar_filtro_hag()

        pipeline = {
            "pipeline": [
                # 1. Ler ficheiro
                {
                    "type": "readers.las",
                    "filename": input_file
                },

                # 2. Marcar outliers estatísticos (classification = 7)
                # Este filtro é rápido (usa KD-tree internamente)
                {
                    "type": "filters.outlier",
                    "method": "statistical",
                    "mean_k": self.config.NOISE_KNN,
                    "multiplier": self.config.NOISE_MULTIPLIER
                },

                # 3. REMOVER outliers estatísticos (classification != 7)
                {
                    "type": "filters.range",
                    "limits": "Classification![7:7]"
                },

                # 4. CSF para classificação ground
                {
                    "type": "filters.csf",
                    "resolution": self.config.CSF_RESOLUTION,
                    "rigidness": self.config.CSF_RIGIDNESS,
                    "iterations": self.config.CSF_ITERATIONS,
                    "smooth": self.config.CSF_SMOOTH
                },

                # 5. Calcular HAG (Height Above Ground)
                {
                    "type": filtro_hag
                },

                # 6. Guardar resultado
                {
                    "type": "writers.las",
                    "filename": output_file,
                    "extra_dims": "HeightAboveGround=float32",
                    "minor_version": 4  # LAS 1.4 para suportar mais pontos
                }
            ]
        }
        
        # Deduplicação: remover pontos duplicados/quase-sobrepostos (voxel)
        # Inserido antes de tudo (posição 1, logo após o reader)
        if getattr(self.config, 'DEDUPLICATE_VOXEL', 0.0) > 0.0:
            pipeline["pipeline"].insert(1, {
                "type": "filters.voxelcenternearestneighbor",
                "cell": self.config.DEDUPLICATE_VOXEL
            })

        # Adicionar subsampling se configurado
        if self.config.SUBSAMPLE_RESOLUTION is not None:
            pipeline["pipeline"].insert(2, {
                "type": "filters.sample",
                "radius": self.config.SUBSAMPLE_RESOLUTION
            })
        
        return pipeline
    
    def _executar_pdal(self, pipeline: dict, descricao: str = "") -> Tuple[bool, str]:
        """Executa pipeline PDAL com gestão de erros melhorada"""
        
        # Guardar pipeline temporário
        pipeline_file = self.config.TEMP_DIR / f"pdal_pipeline_{datetime.now().strftime('%H%M%S')}.json"
        
        try:
            with open(pipeline_file, 'w') as f:
                json.dump(pipeline, f, indent=2)
            
            print(f"  --> Executando PDAL {descricao}...", flush=True)

            proc = subprocess.Popen(
                ['pdal', 'pipeline', str(pipeline_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Heartbeat: imprimir progresso enquanto o PDAL trabalha
            t0_pdal = time.time()
            def _heartbeat_pdal():
                while proc.poll() is None:
                    time.sleep(10)
                    if proc.poll() is None:
                        print(f"    A pensar... ({int(time.time() - t0_pdal)}s)", flush=True)
            hb = threading.Thread(target=_heartbeat_pdal, daemon=True)
            hb.start()

            try:
                stdout_b, stderr_b = proc.communicate(timeout=self.config.TIMEOUT_PDAL)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                return False, f"PDAL timeout (>{self.config.TIMEOUT_PDAL}s)"

            elapsed_pdal = int(time.time() - t0_pdal)
            if proc.returncode != 0:
                erro = (stderr_b or stdout_b or b'').decode('utf-8', errors='replace')
                return False, f"PDAL erro: {erro[:500]}"

            print(f"    Concluido em {elapsed_pdal}s", flush=True)
            return True, "OK"

        except subprocess.TimeoutExpired:
            return False, f"PDAL timeout (>{self.config.TIMEOUT_PDAL}s)"
        except Exception as e:
            return False, f"Erro: {str(e)}"
        finally:
            # Limpar ficheiro temporário
            if pipeline_file.exists():
                pipeline_file.unlink()
    
    def _filtrar_clusters_isolados(self, las_file: Path, eps: float = 0.5, min_samples: int = 10) -> int:
        """
        Remove clusters isolados que não estão conectados à estrutura principal.

        Usa DBSCAN para identificar clusters e mantém apenas o maior (a árvore).
        Clusters pequenos no ar são removidos.

        Args:
            las_file: Ficheiro LAS a processar (modificado in-place)
            eps: Distância máxima entre pontos do mesmo cluster (metros)
            min_samples: Mínimo de pontos para formar um cluster

        Returns:
            Número de pontos removidos
        """
        from sklearn.cluster import DBSCAN

        print("  --> Filtrando clusters isolados (DBSCAN)...")

        las = laspy.read(str(las_file))
        coords = np.column_stack([
            np.array(las.x),
            np.array(las.y),
            np.array(las.z)
        ])

        n_original = len(coords)

        # DBSCAN clustering
        # eps=0.5m significa que pontos a menos de 50cm são considerados vizinhos
        clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(coords)
        labels = clustering.labels_

        # Encontrar o maior cluster (excluindo noise label=-1)
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remover noise

        if len(unique_labels) == 0:
            print("    [AVISO] Nenhum cluster encontrado!")
            return 0

        # Contar pontos por cluster
        cluster_sizes = {}
        for label in unique_labels:
            cluster_sizes[label] = np.sum(labels == label)

        # Maior cluster
        maior_cluster = max(cluster_sizes, key=cluster_sizes.get)
        n_maior = cluster_sizes[maior_cluster]

        # Máscara para manter: maior cluster
        mask_manter = labels == maior_cluster

        n_removidos = n_original - np.sum(mask_manter)
        n_clusters_removidos = len(unique_labels) - 1

        print(f"    Clusters encontrados: {len(unique_labels)}")
        print(f"    Maior cluster: {n_maior:,} pontos")
        print(f"    Clusters removidos: {n_clusters_removidos}")
        print(f"    Pontos removidos: {n_removidos:,} ({n_removidos/n_original*100:.2f}%)")

        if n_removidos > 0:
            # Filtrar e guardar
            las.points = las.points[mask_manter]
            las.write(str(las_file))
            print(f"    [OK] Ficheiro actualizado")

        return n_removidos

    def _marcar_debris_geometrico(self, output_file: Path) -> int:
        """
        Marca pontos de lixo do chão (debris) abaixo da altura do peito (1.3m).

        Algoritmo:
          1. Detetar troncos na fatia 1.2–1.4m via DBSCAN (raio mínimo 0.03m = 6cm DAP).
          2. Para cada tronco, traçar para baixo de 0.1m em 0.1m até 0.1m:
             - Buscar pontos não-solo dentro de 0.4m do centro anterior.
             - Se ≥5 pontos: atualizar centróide e raio (percentil 90 das distâncias).
             - Se <5 pontos: manter último centróide e raio conhecidos ("hold last").
          3. Tudo com 0.1m ≤ HAG < 1.3m, fora do cone de qualquer tronco, não-solo
             → LABEL_DEBRIS (5). Escrito no campo extra 'label' do ficheiro .las.

        Returns:
            Número de pontos marcados como debris.
        """
        # Parâmetros (fixos – baseados em regras do domínio)
        BREAST_HEIGHT    = 1.3    # altura do peito (m)
        BREAST_HALF      = 0.1    # fatia ±0.1m em torno de 1.3m para deteção inicial
        DBSCAN_EPS       = 0.1    # raio DBSCAN para separar troncos individuais
        DBSCAN_MIN_PTS   = 20     # mínimo de pontos por cluster de tronco
        MIN_TRUNK_RADIUS = 0.03   # raio mínimo de tronco válido (6cm DAP)
        SEARCH_RADIUS    = 0.4    # raio de busca ao descer slice a slice
        MIN_PTS_SLICE    = 5      # mínimo de pontos para atualizar centro/raio
        SLICE_STEP       = 0.1    # passo de descida (m)
        BUFFER           = 0.03   # margem extra no raio (ruído do scanner)
        HAG_DEBRIS_MIN   = 0.1    # limite inferior de debris (m)

        try:
            from sklearn.cluster import DBSCAN as _DBSCAN
        except ImportError:
            print("    [AVISO] sklearn não disponível – debris marking ignorado")
            return 0

        las = laspy.read(str(output_file))
        hag = np.array(las.HeightAboveGround, dtype=np.float32)
        x_arr = np.array(las.x, dtype=np.float32)
        y_arr = np.array(las.y, dtype=np.float32)
        n_pts = len(las.points)

        # Solo: classification==2 (CSF) OU HAG < HAG_DEBRIS_MIN
        ground_mask = (np.array(las.classification) == 2) | (hag < HAG_DEBRIS_MIN)

        # ── Passo 1: detetar troncos a 1.3m ────────────────────────────────────
        bh_mask = (hag >= BREAST_HEIGHT - BREAST_HALF) & \
                  (hag <= BREAST_HEIGHT + BREAST_HALF) & \
                  (~ground_mask)

        trunk_list = []  # lista de dicts com {cx, cy, r} indexados por slice

        if np.sum(bh_mask) >= DBSCAN_MIN_PTS:
            bh_pts = np.column_stack([x_arr[bh_mask], y_arr[bh_mask]])
            db = _DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_PTS).fit(bh_pts)
            for lbl in sorted(set(db.labels_)):
                if lbl == -1:
                    continue
                m = db.labels_ == lbl
                pts_c = bh_pts[m]
                cx = float(np.mean(pts_c[:, 0]))
                cy = float(np.mean(pts_c[:, 1]))
                dists = np.sqrt((pts_c[:, 0] - cx) ** 2 + (pts_c[:, 1] - cy) ** 2)
                r = float(np.percentile(dists, 90))
                if r >= MIN_TRUNK_RADIUS:
                    trunk_list.append({'cx_top': cx, 'cy_top': cy, 'r_top': r})

        if not trunk_list:
            print("    [AVISO] Nenhum tronco detetado a 1.3m – debris marking ignorado")
            return 0

        print(f"    Troncos detetados a {BREAST_HEIGHT}m: {len(trunk_list)}")

        # ── Passo 2: traçar cada tronco para baixo, slice a slice ───────────────
        # Slices de h=1.3 até h=0.1 (descendo)
        slice_heights = np.arange(BREAST_HEIGHT, HAG_DEBRIS_MIN - SLICE_STEP / 2, -SLICE_STEP)
        n_slices = len(slice_heights)

        # Para cada tronco guardar (cx, cy, r) por slice
        trunk_cx = np.zeros((len(trunk_list), n_slices), dtype=np.float32)
        trunk_cy = np.zeros((len(trunk_list), n_slices), dtype=np.float32)
        trunk_r  = np.zeros((len(trunk_list), n_slices), dtype=np.float32)

        for t_idx, trunk in enumerate(trunk_list):
            cx, cy, r = trunk['cx_top'], trunk['cy_top'], trunk['r_top']
            for s_idx, h in enumerate(slice_heights):
                # Buscar pontos não-solo nesta fatia perto do centro atual
                slice_mask = (~ground_mask) & \
                             (hag >= h - SLICE_STEP / 2) & \
                             (hag <  h + SLICE_STEP / 2)
                if np.sum(slice_mask) >= MIN_PTS_SLICE:
                    sx = x_arr[slice_mask]
                    sy = y_arr[slice_mask]
                    d  = np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2)
                    near = d <= SEARCH_RADIUS
                    if np.sum(near) >= MIN_PTS_SLICE:
                        cx = float(np.mean(sx[near]))
                        cy = float(np.mean(sy[near]))
                        r  = max(float(np.percentile(
                            np.sqrt((sx[near] - cx) ** 2 + (sy[near] - cy) ** 2),
                            90)), MIN_TRUNK_RADIUS)
                # Guardar estado atual (mesmo que "hold last")
                trunk_cx[t_idx, s_idx] = cx
                trunk_cy[t_idx, s_idx] = cy
                trunk_r [t_idx, s_idx] = r

        # ── Passo 3: labeling vetorial ───────────────────────────────────────────
        # Candidatos a debris: não-solo, 0.1m ≤ HAG < 1.3m
        candidate_mask = (~ground_mask) & \
                         (hag >= HAG_DEBRIS_MIN) & \
                         (hag <  BREAST_HEIGHT)
        n_candidates = int(np.sum(candidate_mask))

        if n_candidates == 0:
            print("    Nenhum candidato a debris encontrado")
            return 0

        cand_idx = np.where(candidate_mask)[0]
        cand_hag = hag[cand_idx]
        cand_x   = x_arr[cand_idx]
        cand_y   = y_arr[cand_idx]

        # Índice do slice mais próximo para cada ponto candidato
        # slice_heights desce de 1.3 → 0.1; slice_idx=0 corresponde a 1.3m
        raw_idx = np.round(
            (BREAST_HEIGHT - cand_hag) / SLICE_STEP
        ).astype(int)
        slice_idx = np.clip(raw_idx, 0, n_slices - 1)

        # Para cada ponto, verificar se está dentro do cone de QUALQUER tronco
        is_trunk = np.zeros(n_candidates, dtype=bool)
        for t_idx in range(len(trunk_list)):
            cx_pt = trunk_cx[t_idx][slice_idx]
            cy_pt = trunk_cy[t_idx][slice_idx]
            r_pt  = trunk_r [t_idx][slice_idx]
            dist  = np.sqrt((cand_x - cx_pt) ** 2 + (cand_y - cy_pt) ** 2)
            is_trunk |= (dist <= r_pt + BUFFER)

        # Pontos fora de qualquer cone → debris
        debris_among_candidates = ~is_trunk
        n_debris = int(np.sum(debris_among_candidates))

        # ── Escrever campo 'label' no ficheiro .las ─────────────────────────────
        label_arr = np.zeros(n_pts, dtype=np.int32)
        label_arr[cand_idx[debris_among_candidates]] = LABEL_DEBRIS

        if 'label' not in las.point_format.dimension_names:
            las.add_extra_dims([laspy.ExtraBytesParams(name='label', type=np.int32)])
        las['label'] = label_arr
        las.write(str(output_file))

        n_trunk_low = int(np.sum(is_trunk))
        print(f"    Debris marcado: {n_debris:,} pontos (label={LABEL_DEBRIS})")
        print(f"    Tronco preservado abaixo de {BREAST_HEIGHT}m: {n_trunk_low:,} pontos")

        return n_debris

    def _validar_resultado(self, output_file: Path) -> Tuple[bool, Dict[str, Any]]:
        """
        Valida o resultado do pré-processamento

        CHECKPOINT A: Verificações de sanidade
        """
        print("\n  --> Checkpoint A: Validando pré-processamento...")
        
        stats = {}
        
        try:
            las = laspy.read(str(output_file))
            n_points = len(las.points)
            stats['n_pontos'] = n_points
            
            print(f"    Total pontos: {n_points:,}")
            
            # Verificar se HAG foi calculado
            if 'HeightAboveGround' not in las.point_format.dimension_names:
                print("    [ERRO] FALHOU: HeightAboveGround não encontrado!")
                return False, stats
            
            hag = np.array(las.HeightAboveGround)
            stats['hag_min'] = float(np.min(hag))
            stats['hag_max'] = float(np.max(hag))
            stats['hag_mean'] = float(np.mean(hag))
            stats['hag_std'] = float(np.std(hag))
            
            print(f"    HAG: min={stats['hag_min']:.2f}m, max={stats['hag_max']:.2f}m, "
                  f"mean={stats['hag_mean']:.2f}m")
            
            # Verificar classificação ground
            if hasattr(las, 'classification'):
                ground_points = np.sum(las.classification == 2)
                ground_pct = (ground_points / n_points) * 100
                stats['pct_ground'] = ground_pct
                stats['n_ground'] = int(ground_points)
                
                print(f"    Ground: {ground_points:,} pontos ({ground_pct:.1f}%)")
                
                # Warnings
                if ground_pct < 5:
                    print("    [AVISO] AVISO: Poucos pontos ground (<5%) - verificar CSF")
                elif ground_pct > 60:
                    print("    [AVISO] AVISO: Muitos pontos ground (>60%) - pode haver problema")
            
            # Verificar pontos com HAG negativo
            negative_hag = np.sum(hag < -0.5)
            if negative_hag > n_points * 0.01:
                stats['pontos_hag_negativo'] = int(negative_hag)
                print(f"    [AVISO] AVISO: {negative_hag:,} pontos com HAG < -0.5m")
            
            # Verificar range de alturas
            altura_range = stats['hag_max'] - stats['hag_min']
            if altura_range < 2:
                print(f"    [AVISO] AVISO: Range de alturas muito pequeno ({altura_range:.1f}m)")
            
            print("    [OK] Checkpoint A: PASSOU")
            return True, stats
            
        except Exception as e:
            print(f"    [ERRO] ERRO na validação: {e}")
            stats['erro'] = str(e)
            return False, stats
    
    def processar(self, input_file: Path, output_dir: Path) -> ResultadoFase1:
        """
        Processa um ficheiro LAS completo
        
        Returns:
            ResultadoFase1 com todos os detalhes
        """
        inicio = datetime.now()
        resultado = ResultadoFase1(status=StatusFase.EM_EXECUCAO)
        
        print(f"\n{'='*60}")
        print(f"FASE 1: PRÉ-PROCESSAMENTO PDAL")
        print(f"{'='*60}")
        print(f"Input: {input_file.name}")
        
        input_file = Path(input_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verificar PDAL
        if not self._verificar_pdal():
            resultado.status = StatusFase.ERRO
            resultado.erro = "PDAL não disponível"
            return resultado
        
        # Analisar ficheiro
        print("\n  [1/6] Analisando ficheiro...")
        try:
            info = self._analisar_ficheiro(input_file)
            resultado.n_pontos_input = info['n_pontos']
            print(f"    Pontos: {info['n_pontos']:,}")
            print(f"    Bounds X: [{info['x_min']:.2f}, {info['x_max']:.2f}]")
            print(f"    Bounds Y: [{info['y_min']:.2f}, {info['y_max']:.2f}]")
            print(f"    Bounds Z: [{info['z_min']:.2f}, {info['z_max']:.2f}]")
            print(f"    UTM detectado: {'Sim' if info['is_utm'] else 'Não'}")
        except Exception as e:
            resultado.status = StatusFase.ERRO
            resultado.erro = f"Erro ao analisar ficheiro: {e}"
            print(f"    [ERRO] ERRO: {e}")
            return resultado
        
        # Centralizar coordenadas se necessário
        print("\n  [2/6] Preparando coordenadas...")
        if info['is_utm'] and self.config.AUTO_CENTER_COORDS:
            centered_file = self.config.TEMP_DIR / f"{input_file.stem}_centered.las"
            try:
                resultado.offset = self._centralizar_coordenadas(input_file, centered_file)
                pdal_input = str(centered_file)
                
                # Guardar offset
                if self.config.GUARDAR_OFFSET:
                    offset_file = output_dir / f"{input_file.stem}_offset.json"
                    with open(offset_file, 'w') as f:
                        json.dump(resultado.offset, f, indent=2)
                    print(f"    Offset guardado: {offset_file.name}")
            except Exception as e:
                resultado.status = StatusFase.ERRO
                resultado.erro = f"Erro ao centralizar: {e}"
                return resultado
        else:
            pdal_input = str(input_file)
            resultado.offset = {'x_offset': 0, 'y_offset': 0, 'z_offset': 0}
            print("    Coordenadas OK (sem necessidade de centralização)")
        
        # Executar PDAL
        print("\n  [3/6] Processando com PDAL...")
        output_file = output_dir / f"{input_file.stem}_preprocessed.las"
        
        pipeline = self._criar_pipeline_pdal(pdal_input, str(output_file))
        
        sucesso, msg = self._executar_pdal(pipeline, "(CSF + HAG)")
        
        if not sucesso:
            resultado.status = StatusFase.ERRO
            resultado.erro = msg
            print(f"    [ERRO] {msg}")
            return resultado
        
        print("    [OK] PDAL concluído")

        # Filtrar clusters isolados (ruído no ar) - OPCIONAL
        if self.config.FILTRAR_CLUSTERS_ISOLADOS:
            print("\n  [4/6] Removendo clusters isolados (DBSCAN)...")
            try:
                n_removidos = self._filtrar_clusters_isolados(
                    output_file,
                    eps=self.config.DBSCAN_EPS,
                    min_samples=10
                )
            except Exception as e:
                print(f"\n[ERRO] Falha no filtro de clusters isolados: {e}")
                print(f"  Ficheiro LAS: {output_file}")
                print(f"  Parametros: eps={self.config.DBSCAN_EPS}, min_samples=10")
                print(f"  Para desactivar este filtro: config.FILTRAR_CLUSTERS_ISOLADOS = False")
                import traceback
                traceback.print_exc()
                raise
        else:
            print("\n  [4/6] Filtro de clusters desactivado (config)")
            n_removidos = 0

        # Marcar debris geométrico abaixo da altura do peito
        print("\n  [5/6] Marcando debris geométrico (cone de tronco descendente)...")
        try:
            n_debris = self._marcar_debris_geometrico(output_file)
        except Exception as e:
            print(f"\n[ERRO] Falha no debris marking geometrico: {e}")
            print(f"  Ficheiro LAS: {output_file}")
            print(f"  O debris marking requer que o campo 'HeightAboveGround' exista no LAS.")
            print(f"  Verificar que a fase CSF+HAG do PDAL correu com sucesso.")
            import traceback
            traceback.print_exc()
            raise

        # Validar resultado
        print("\n  [6/6] Validando resultado...")
        valido, stats = self._validar_resultado(output_file)
        
        if not valido:
            resultado.status = StatusFase.ERRO
            resultado.erro = "Validação falhou"
            return resultado
        
        # Sucesso!
        resultado.status = StatusFase.SUCESSO
        resultado.ficheiro_output = output_file
        resultado.n_pontos_output = stats.get('n_pontos', 0)
        resultado.pct_ground = stats.get('pct_ground', 0)
        resultado.hag_stats = {
            'min': stats.get('hag_min'),
            'max': stats.get('hag_max'),
            'mean': stats.get('hag_mean')
        }
        resultado.tempo_execucao = (datetime.now() - inicio).total_seconds()
        
        # Limpar ficheiro temporário
        if info['is_utm'] and self.config.AUTO_CENTER_COORDS:
            centered_file = self.config.TEMP_DIR / f"{input_file.stem}_centered.las"
            if centered_file.exists():
                centered_file.unlink()
        
        print(f"\n{'='*60}")
        print(f"[OK] FASE 1 CONCLUÍDA em {resultado.tempo_execucao:.1f}s")
        print(f"  Output: {output_file}")
        print(f"  Pontos: {resultado.n_pontos_input:,} --> {resultado.n_pontos_output:,}")
        if n_debris > 0:
            print(f"  Debris marcado: {n_debris:,} pontos (label={LABEL_DEBRIS})")
        print(f"{'='*60}")
        
        return resultado
    
    def processar_batch(self, input_dir: Path, output_dir: Path) -> list:
        """Processa múltiplos ficheiros"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        las_files = list(input_dir.glob("*.las")) + list(input_dir.glob("*.LAS"))
        las_files += list(input_dir.glob("*.laz")) + list(input_dir.glob("*.LAZ"))
        
        if not las_files:
            print(f"Nenhum ficheiro .las/.laz encontrado em {input_dir}")
            return []
        
        print(f"\nEncontrados {len(las_files)} ficheiros para processar")
        
        resultados = []
        for i, las_file in enumerate(las_files, 1):
            print(f"\n[{i}/{len(las_files)}] {las_file.name}")
            resultado = self.processar(las_file, output_dir)
            resultados.append(resultado)
            
            # Status
            status_emoji = "[OK]" if resultado.status == StatusFase.SUCESSO else "[ERRO]"
            print(f"  {status_emoji} Status: {resultado.status.value}")
        
        # Resumo
        sucessos = sum(1 for r in resultados if r.status == StatusFase.SUCESSO)
        print(f"\n{'='*60}")
        print(f"RESUMO FASE 1: {sucessos}/{len(las_files)} ficheiros processados")
        print(f"{'='*60}")
        
        return resultados


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fase 1: Pré-processamento PDAL")
    parser.add_argument("input", help="Ficheiro .las ou pasta com ficheiros")
    parser.add_argument("-o", "--output", default="output/fase1", help="Pasta de output")
    parser.add_argument("--no-center", action="store_true", help="Não centralizar coordenadas UTM")
    
    args = parser.parse_args()
    
    # Configurar
    config = Config
    if args.no_center:
        config.AUTO_CENTER_COORDS = False
    
    preprocessor = PDALPreprocessor(config)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        resultado = preprocessor.processar(input_path, output_path)
        exit(0 if resultado.status == StatusFase.SUCESSO else 1)
    elif input_path.is_dir():
        resultados = preprocessor.processar_batch(input_path, output_path)
        sucessos = sum(1 for r in resultados if r.status == StatusFase.SUCESSO)
        exit(0 if sucessos > 0 else 1)
    else:
        print(f"Erro: {input_path} não existe")
        exit(1)
