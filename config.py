"""
PIPELINE LiDAR PARA FLORESTAS PLANTADAS - CONFIGURAÇÃO
======================================================
Versão Corrigida v8.0 - Janeiro 2026

CORREÇÕES APLICADAS:
- Paths configuráveis (não hardcoded)
- Parâmetros validados contra documentação oficial
- Suporte para TLS2trees do repositório tls-tools-ucl
- Configurações específicas para MLS pedestre
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
import json


# =============================================================================
# ENUMS E CONSTANTES
# =============================================================================

class StatusFase(Enum):
    PENDENTE = "pendente"
    EM_EXECUCAO = "a_pensar"
    SUCESSO = "sucesso"
    ERRO = "erro"
    IGNORADO = "ignorado"
    REJEITADA = "rejeitada"


# =============================================================================
# LABELS SEMÂNTICOS
# =============================================================================
# 1 = Terrain (chão)
# 2 = Vegetation (vegetação baixa)
# 3 = Branches and Leaves (ramos e folhas)
# 4 = Wood (tronco principal)

LABEL_TERRAIN = 1
LABEL_VEGETATION = 2
LABEL_BRANCHES_LEAVES = 3
LABEL_WOOD = 4
LABEL_DEBRIS = 5       # Lixo (<1.3m, não-tronco) → eliminado na Fase 2
LABEL_RAMOS_BASE = 6   # Ramos junto à base rejeitados na Fase 2.5

LABEL_CWD = LABEL_BRANCHES_LEAVES  # Alias usado em fase2_instance.py

# =============================================================================
# PARÂMETROS ALOMÉTRICOS v8 (Pinus pinaster, Portugal)
# Calibrar com Etapa G.5/G.6 usando dados destrutivos
# =============================================================================

# H = 1.30 + a × (1 - exp(-b × DBH_cm))^c  (Schumacher modificado)
# Validação esperada (não calibrado): DBH=20cm→H≈16m; DBH=30cm→H≈22m
PARAMS_ALOMETRIA_Pn = (27.0, 0.038, 1.05)

# Kozak 1988 variable-exponent taper: params = (a0, a1, a2, b1, b2, b3, b4)
# Parâmetros iniciais não calibrados — substituir após Etapa G.5
PARAMS_DEFAULT_Pn = (0.98, 0.01, 0.01, 0.70, -0.50, 0.30, -0.20)


# =============================================================================
# VERIFICAÇÕES DO SISTEMA
# =============================================================================

def verificar_python_version() -> bool:
    """Verifica se a versão Python é compatível (3.8-3.11)"""
    major, minor = sys.version_info[:2]
    if major != 3 or not (8 <= minor <= 11):
        print(f"⚠ AVISO: Python {major}.{minor} pode ter problemas de compatibilidade!")
        print(f"  PyTLidar/Open3D funcionam melhor com Python 3.8-3.11")
        print(f"  Recomendado: conda create -n lidar python=3.11")
        return False
    print(f"✓ Python {major}.{minor} - OK")
    return True


def verificar_ambiente_critico(modo_estrito: bool = False) -> bool:
    """
    Verifica requisitos críticos do ambiente.
    
    Args:
        modo_estrito: Se True, levanta exceção para Python 3.12+
                     Se False, apenas mostra aviso
    
    Returns:
        True se ambiente OK, False se problemas detectados
        
    Raises:
        RuntimeError: Se modo_estrito e Python 3.12+
    """
    major, minor = sys.version_info[:2]
    
    # Python 3.12+ não suportado pelo Open3D
    if major != 3 or minor >= 12:
        msg = (
            f"\n{'='*60}\n"
            f"⚠ AVISO CRÍTICO: Python {major}.{minor} detectado!\n"
            f"{'='*60}\n"
            f"Open3D e PyTLidar NÃO funcionam em Python 3.12+.\n"
            f"O pipeline pode falhar em várias fases.\n\n"
            f"SOLUÇÃO: Criar ambiente com Python 3.11:\n"
            f"  conda create -n lidar python=3.11\n"
            f"  conda activate lidar\n"
            f"  pip install -r requirements.txt\n"
            f"{'='*60}\n"
        )
        print(msg)
        
        if modo_estrito:
            raise RuntimeError(f"Python {major}.{minor} não suportado. Usar Python 3.8-3.11.")
        return False
    
    return True


def verificar_cuda() -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Verifica se CUDA está disponível
    
    Returns:
        (cuda_disponivel, nome_gpu, memoria_gb)
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            cuda_version = torch.version.cuda
            print(f"✓ CUDA {cuda_version} - GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            return True, gpu_name, gpu_memory
        else:
            print("⚠ CUDA não disponível - processamento será mais lento")
            return False, None, None
    except ImportError:
        print("⚠ PyTorch não instalado")
        return False, None, None


def verificar_dependencia(nome: str, modulo: str) -> bool:
    """Verifica se uma dependência está instalada"""
    try:
        __import__(modulo)
        return True
    except ImportError:
        return False


def verificar_todas_dependencias() -> Dict[str, bool]:
    """Verifica todas as dependências necessárias"""
    deps = {
        'numpy': 'numpy',
        'scipy': 'scipy', 
        'pandas': 'pandas',
        'laspy': 'laspy',
        'plyfile': 'plyfile',
        'sklearn': 'sklearn',
        'open3d': 'open3d',
        'torch': 'torch',
        'yaml': 'yaml',
    }
    
    resultados = {}
    for nome, modulo in deps.items():
        ok = verificar_dependencia(nome, modulo)
        resultados[nome] = ok
        status = "✓" if ok else "✗"
        print(f"  {status} {nome}")
    
    return resultados


# =============================================================================
# CONFIGURAÇÃO PRINCIPAL
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuração central do pipeline LiDAR - CORRIGIDA"""
    
    # -------------------------------------------------------------------------
    # PATHS - CONFIGURÁVEIS
    # -------------------------------------------------------------------------
    
    # Diretório base (detectado automaticamente ou configurado)
    BASE_DIR: Path = field(default_factory=lambda: Path.cwd())
    
    # Diretórios de dados
    INPUT_DIR: Path = field(default_factory=lambda: Path.cwd() / "input")
    OUTPUT_DIR: Path = field(default_factory=lambda: Path.cwd() / "output")
    TEMP_DIR: Path = field(default_factory=lambda: Path.cwd() / "temp")
    
    # Path TLS2trees
    TLS2TREES_PATH: Optional[Path] = Path(r"C:\projetoTransformLidar\TLS2trees")
    
    # -------------------------------------------------------------------------
    # PARÂMETROS PDAL (Fase 1) - OTIMIZADOS PARA MLS PEDESTRE
    # -------------------------------------------------------------------------
    
    # CSF (Cloth Simulation Filter)
    CSF_RESOLUTION: float = 0.3   # metros - menor = mais detalhe
    CSF_RIGIDNESS: int = 2        # 1=flexível, 2=médio, 3=rígido
    CSF_ITERATIONS: int = 500
    CSF_SMOOTH: bool = True

    # Filtro de ruído estatístico
    NOISE_KNN: int = 8           # vizinhos para outlier removal
    NOISE_MULTIPLIER: float = 1.2

    # Filtro DBSCAN para clusters isolados no ar (desactivado por defeito)
    FILTRAR_CLUSTERS_ISOLADOS: bool = False
    DBSCAN_EPS: float = 0.1
   
    # Subsampling (opcional, para ficheiros muito grandes)
    SUBSAMPLE_RESOLUTION: Optional[float] = None  # None = sem subsampling

    # Deduplicação: remove pontos duplicados/quase-sobrepostos por voxel
    # 0.005 = voxel 5mm → mantém 1 ponto por célula 5×5×5mm (remove duplicados TLS)
    # Usar 0.0 para desactivar
    DEDUPLICATE_VOXEL: float = 0.005
    
    # Labels de output (usados em fase2_instance.py)
    # 1=Terrain, 2=Vegetation, 3=Branches&Leaves, 4=Wood(Trunk)
    LABEL_TERRAIN: int = 1
    LABEL_VEGETATION: int = 2
    LABEL_BRANCHES_LEAVES: int = 3
    LABEL_WOOD: int = 4
    LABEL_CWD: int = 3  # Alias para LABEL_BRANCHES_LEAVES

    # -------------------------------------------------------------------------
    # PARÂMETROS INSTANCE SEGMENTATION (Fase 2) - TLS2trees instance.py
    # -------------------------------------------------------------------------
    
    # Número de tiles vizinhos (3 = 3x3, 5 = 5x5)
    INSTANCE_N_TILES: int = 3
    
    # Espessura das slices para construir o grafo
    INSTANCE_SLICE_THICKNESS: float = 0.3  # metros
    
    # Boundary para deteção de stems (altura min, altura max)
    # CORREÇÃO: São 2 valores, não 1!
    INSTANCE_FIND_STEMS_BOUNDARY: Tuple[float, float] = (1.0, 2.0)
    
    # Raio mínimo de stems detetados
    INSTANCE_FIND_STEMS_MIN_RADIUS: float = 0.025  # metros (2.5cm)
    
    # Pontos mínimos para um stem
    INSTANCE_FIND_STEMS_MIN_POINTS: int = 50
    
    # Distância máxima para conectar pontos no grafo
    # CORRIGIDO: 0.15m era demasiado pequeno para ligar clusters entre slices.
    # O default do TLS2trees é 1.0m. Usar 0.5m: mais conservador mas funcional.
    INSTANCE_GRAPH_EDGE_LENGTH: float = 0.5  # metros

    # Gap máximo cumulativo entre base e cluster
    # CORRIGIDO: 3.0m cortava o tronco a ~3m acima da zona de deteção de stems.
    # O default do TLS2trees é inf. Usar 35.0m (cobre pinheiros até 35m).
    INSTANCE_GRAPH_MAX_CUM_GAP: float = 35.0  # metros

    # Raio do cilindro de tronco nos pré-labels (metros)
    # Pontos não-terreno dentro deste raio de cada tronco → label 3 (stem)
    # Pontos fora do cilindro e abaixo da altura do peito → label 0 (lixo)
    # Pontos fora do cilindro e acima da altura do peito → label 1 (copa)
    # Regra: diâmetro mínimo real é 0.075m; usar raio 0.15m (margem de segurança)
    INSTANCE_TRUNK_RADIUS: float = 0.15  # metros

    # Pontos mínimos por árvore final
    INSTANCE_MIN_POINTS_PER_TREE: int = 100

    # Adicionar folhas às árvores segmentadas
    INSTANCE_ADD_LEAVES: bool = True
    INSTANCE_ADD_LEAVES_VOXEL: float = 0.5  # metros
    INSTANCE_ADD_LEAVES_EDGE: float = 1.0   # metros
    
    # -------------------------------------------------------------------------
    # CORRECÇÃO DE CASCA / INCLINAÇÃO (Fase 3 — RANSAC profile)
    # -------------------------------------------------------------------------
    # Se BARK_R_LOW = BARK_R_HIGH = 0.0 (default), a fase3 calcula
    # automaticamente a partir da deriva horizontal do esqueleto:
    #   bark_r_low  = 0.20 × drift_rate  (calibrado em pinheiros inclinados)
    #   bark_r_high = 0.12 × drift_rate
    # Para override manual (ex: scan parcial, espécie não calibrada), definir
    # valores não-zero (ex: 0.020/0.012 para pinheiro muito inclinado).
    #   h <= BARK_H_LOW  : subtrai BARK_R_LOW  do raio RANSAC
    #   h >= BARK_H_HIGH : subtrai BARK_R_HIGH do raio RANSAC
    #   entre os dois    : interpolação linear
    #   h < BARK_H_MIN   : sem correcção (zona da base)
    BARK_R_LOW:  float = 0.000   # override manual (0 = automático)
    BARK_H_LOW:  float = 1.3     # altura (m) com correcção BARK_R_LOW
    BARK_R_HIGH: float = 0.000   # override manual (0 = automático)
    BARK_H_HIGH: float = 3.5     # altura (m) a partir da qual correcção = BARK_R_HIGH
    BARK_H_MIN:  float = 0.30    # abaixo desta altura: sem correcção

    # -------------------------------------------------------------------------
    # PARÂMETROS DE VALIDAÇÃO
    # -------------------------------------------------------------------------

    EXPECTED_TREE_SPACING: float = 3.0  # metros
    SPACING_TOLERANCE: float = 0.40     # 40% tolerância
    MIN_TREE_HEIGHT: float = 3.0        # metros (< 3m → arbusto rejeitado)
    MAX_TREE_HEIGHT: float = 50.0       # metros

    # Filtros anti-arbusto na pré-labelização (fase2_instance)
    # Raio mínimo do cluster DBSCAN à altura do peito — arbustos finos têm r < 2.5cm
    MIN_STEM_RADIUS_AT_DAP: float = 0.025  # metros (2.5 cm)
    # Altura mínima até onde o skeleton deve ser seguível — arbustos não chegam a 3.5m
    MIN_TRUNK_TRACK_HEIGHT: float = 3.5    # metros
    
    # -------------------------------------------------------------------------
    # COORDENADAS UTM
    # -------------------------------------------------------------------------
    
    UTM_THRESHOLD: float = 100000.0
    AUTO_CENTER_COORDS: bool = True
    GUARDAR_OFFSET: bool = True  # Guardar offset para georreferenciação
    
    # -------------------------------------------------------------------------
    # PROCESSAMENTO
    # -------------------------------------------------------------------------
    
    # Número de cores CPU (None = auto)
    NUM_CORES: Optional[int] = None
    
    # Timeouts (segundos)
    TIMEOUT_PDAL: int = 3600      # 1 hora
    TIMEOUT_INSTANCE: int = 18000  # 5 horas
    
    # -------------------------------------------------------------------------
    # PARÂMETROS FASE 2.5 — Per-Tree Setup (v8)
    # -------------------------------------------------------------------------
    C_GROUND_CELL: float = 0.5        # Tamanho célula DTM local (m)
    C_GROUND_PCT: float = 5.0         # Percentil para z_ground na célula
    C_DBH_HEIGHT: float = 1.30        # Altura DAP (m)
    C_DBH_SLAB: float = 0.15          # Espessura da fatia DAP (± m)
    C_MIN_DBH_M: float = 0.10         # DAP mínimo para aceitar árvore (10 cm)
    C_BUSHES_MAX_HEIGHT: float = 1.5  # Arbustos abaixo desta altura filtrados
    C_BUSHES_R_FACTOR: float = 2.5    # Raio de manutenção = r_DBH × factor

    # -------------------------------------------------------------------------
    # PARÂMETROS FASE 3 v8 — Taper fitting (Kozak + B-spline + IRLS)
    # -------------------------------------------------------------------------
    D_SLICE_STEP_CENTROID: float = 0.20         # Passo das fatias para centróides (m)
    D_MIN_PTS_PER_SLICE: int = 5                # Pontos mínimos por fatia
    D_MIN_ANG_COVERAGE_FOR_CENTROID: float = 0.20  # Cobertura angular mínima
    D_SPLINE_SMOOTH: float = 0.05               # Suavização B-spline
    D_SPLINE_DEGREE: int = 3                    # Grau da spline cúbica
    D_IRLS_N_ITER: int = 5                      # Iterações IRLS
    D_IRLS_HUBER_FSCALE: float = 0.02          # Escala Huber (2 cm)
    D_DAP_CONSTRAINT_WEIGHT: float = 50.0       # Peso do constraint DAP no IRLS
    D_BIN_S_FOR_TOP_FIAVEL: float = 0.5         # Largura bin para s_top_fiavel (m)
    D_TOP_FIAVEL_MIN_PTS: int = 20              # Pontos mínimos por bin
    D_TOP_FIAVEL_MIN_ANG: float = 0.30          # Cobertura angular mínima para top fiável
    D_TOP_COVERAGE_BLEND_HIGH: float = 0.30     # Acima disto → usa H medida
    D_TOP_COVERAGE_BLEND_LOW: float = 0.10      # Abaixo disto → usa H alométrica

    # -------------------------------------------------------------------------
    # PARÂMETROS FASE 4 v8 — Volume analítico
    # -------------------------------------------------------------------------
    E_INTEGRATION_STEP: float = 0.01    # Passo de integração (1 cm)
    E_UNCERT_BASE_PCT: float = 5.0      # Incerteza base (%)
    E_UNCERT_FRAC_FACTOR: float = 30.0  # Factor para fracção extrapolada

    # -------------------------------------------------------------------------
    # SALVAGUARDAS v8
    # -------------------------------------------------------------------------
    F_TAPER_MONOTONIC_TOLERANCE: float = 0.05   # Fracção violações monotonicidade tolerada
    F_DBH_SUSPECT_THRESHOLD_M: float = 0.45     # DAP > 45 cm → cluster duplo suspeito

    # -------------------------------------------------------------------------
    # LOGGING E DEBUG
    # -------------------------------------------------------------------------

    VERBOSE: bool = True
    GUARDAR_INTERMEDIOS: bool = True  # Guardar ficheiros intermédios
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    def __post_init__(self):
        """Inicialização após criação"""
        # Criar diretórios
        for dir_path in [self.INPUT_DIR, self.OUTPUT_DIR, self.TEMP_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Detetar número de cores se não especificado
        if self.NUM_CORES is None:
            self.NUM_CORES = max(1, os.cpu_count() - 2)
        
        # Procurar TLS2trees se não especificado
        if self.TLS2TREES_PATH is None:
            self.TLS2TREES_PATH = self._procurar_ferramenta(
                "TLS2trees", 
                ["tls2trees/semantic.py", "semantic.py", "instance.py"]
            )
    
    def _procurar_ferramenta(self, nome: str, ficheiros_check: List[str]) -> Optional[Path]:
        """Procura uma ferramenta em localizações comuns"""
        possiveis = [
            self.BASE_DIR / "tools" / nome,
            self.BASE_DIR / nome,
            Path.home() / nome,
            Path(f"C:/LiDAR_Pipeline/tools/{nome}"),
            Path(f"/opt/{nome}"),
        ]
        
        for path in possiveis:
            if path.exists():
                for check in ficheiros_check:
                    if (path / check).exists():
                        return path
        
        return None
    
    def validar(self) -> Tuple[bool, List[str]]:
        """Valida a configuração"""
        erros = []
        
        # Verificar diretórios
        if not self.INPUT_DIR.exists():
            erros.append(f"INPUT_DIR não existe: {self.INPUT_DIR}")
        
        # Verificar ferramentas
        if self.TLS2TREES_PATH is None:
            erros.append("TLS2trees não encontrado")

        # Verificar parâmetros
        if self.INSTANCE_FIND_STEMS_BOUNDARY[0] >= self.INSTANCE_FIND_STEMS_BOUNDARY[1]:
            erros.append("INSTANCE_FIND_STEMS_BOUNDARY: min deve ser < max")
        
        return len(erros) == 0, erros
    
    def to_dict(self) -> dict:
        """Converte config para dicionário (para guardar)"""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, (list, tuple)):
                d[k] = list(v)
            else:
                d[k] = v
        return d
    
    def guardar(self, filepath: Path):
        """Guarda configuração em JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def carregar(cls, filepath: Path) -> 'PipelineConfig':
        """Carrega configuração de JSON"""
        with open(filepath, 'r') as f:
            d = json.load(f)
        
        # Converter strings para Paths
        for k in ['BASE_DIR', 'INPUT_DIR', 'OUTPUT_DIR', 'TEMP_DIR', 'TLS2TREES_PATH']:
            if k in d and d[k] is not None:
                d[k] = Path(d[k])
        
        # Converter listas para tuplas onde necessário
        if 'INSTANCE_FIND_STEMS_BOUNDARY' in d:
            d['INSTANCE_FIND_STEMS_BOUNDARY'] = tuple(d['INSTANCE_FIND_STEMS_BOUNDARY'])
        
        return cls(**d)


# =============================================================================
# PRESETS POR TIPO DE FLORESTA
# =============================================================================

def criar_config_eucalipto(base_dir: Optional[Path] = None) -> PipelineConfig:
    """Configuração otimizada para eucalipto"""
    config = PipelineConfig()
    if base_dir:
        config.BASE_DIR = base_dir

    config.EXPECTED_TREE_SPACING = 3.0
    config.INSTANCE_FIND_STEMS_MIN_RADIUS = 0.02
    config.INSTANCE_FIND_STEMS_BOUNDARY = (1.0, 1.8)

    print("[OK] Preset EUCALIPTO aplicado")
    return config


def criar_config_pinheiro(base_dir: Optional[Path] = None) -> PipelineConfig:
    """Configuração otimizada para pinheiro"""
    config = PipelineConfig()
    if base_dir:
        config.BASE_DIR = base_dir

    config.EXPECTED_TREE_SPACING = 3.5

    # Stems mais grossos
    # Janela de deteção de stems: acima de 1.6m evita arbustos; até 3.0m cobre DAP
    config.INSTANCE_FIND_STEMS_BOUNDARY = (1.6, 3.0)

    # Raio mínimo 5cm — filtra caules finos de arbustos
    config.INSTANCE_FIND_STEMS_MIN_RADIUS = 0.05

    # Inliers RANSAC mínimos para aceitar um stem (default TLS2trees = 200)
    config.INSTANCE_FIND_STEMS_MIN_POINTS = 200

    # Aresta do grafo: 0.7m cobre inclinações até ~65°, mais restrito que 1.0m
    config.INSTANCE_GRAPH_EDGE_LENGTH = 0.7
    config.INSTANCE_GRAPH_MAX_CUM_GAP = 35.0

    # Filtro final: instâncias com menos de 8000 pontos são arbustos/ruído
    config.INSTANCE_MIN_POINTS_PER_TREE = 8000

    # Raio do cilindro de pré-labels: 0.20m — mais largo que 0.15m (capta tronco inclinado)
    # mas mais apertado que 0.30m (evita ramos próximos inflacionarem o perfil RANSAC)
    config.INSTANCE_TRUNK_RADIUS = 0.20
    
    print("[OK] Preset PINHEIRO aplicado")
    return config


def criar_config_misto(base_dir: Optional[Path] = None) -> PipelineConfig:
    """Configuração para floresta mista"""
    config = PipelineConfig()
    if base_dir:
        config.BASE_DIR = base_dir

    config.EXPECTED_TREE_SPACING = 3.0

    print("[OK] Preset MISTO aplicado")
    return config


def get_preset(especie: str, base_dir: Optional[Path] = None) -> PipelineConfig:
    """
    Função auxiliar para obter preset por nome de espécie.

    Args:
        especie: Nome da espécie ('eucalipto', 'pinheiro', 'misto')
        base_dir: Diretório base opcional

    Returns:
        PipelineConfig configurado para a espécie

    Raises:
        ValueError: Se espécie não for reconhecida
    """
    especie_lower = especie.lower()

    if especie_lower == 'eucalipto':
        return criar_config_eucalipto(base_dir)
    elif especie_lower == 'pinheiro':
        return criar_config_pinheiro(base_dir)
    elif especie_lower == 'misto':
        return criar_config_misto(base_dir)
    else:
        raise ValueError(
            f"Espécie '{especie}' não reconhecida. "
            f"Use 'eucalipto', 'pinheiro' ou 'misto'"
        )


# =============================================================================
# INSTÂNCIA GLOBAL
# =============================================================================

# Configuração por defeito
Config = PipelineConfig()


# =============================================================================
# EXECUÇÃO DIRETA - VERIFICAÇÃO DO SISTEMA
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VERIFICAÇÃO DO SISTEMA - Pipeline LiDAR v8")
    print("=" * 60)
    
    print("\n[1] Versão Python:")
    verificar_python_version()
    
    print("\n[2] CUDA/GPU:")
    verificar_cuda()
    
    print("\n[3] Dependências:")
    deps = verificar_todas_dependencias()
    
    print("\n[4] Configuração:")
    config = PipelineConfig()
    valido, erros = config.validar()
    
    if valido:
        print("  ✓ Configuração válida")
    else:
        print("  ✗ Problemas na configuração:")
        for e in erros:
            print(f"    - {e}")
    
    print("\n[5] Ferramentas externas:")
    print(f"  TLS2trees: {config.TLS2TREES_PATH or 'Não encontrado'}")
    
    print("\n" + "=" * 60)
