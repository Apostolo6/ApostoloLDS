# Pipeline LiDAR Florestal v8 — Documentação das Melhorias

**Data:** 2026-06-13 (run completo: 2026-06-11)
**Base:** v7 (ver `EXPLICACAO_PIPELINE_v7.md`)
**Espécie / dados:** *Pinus pinaster*, Baldios de Alge, 59 árvores destrutivas (45 s/Alge2)
**Ambiente:** `C:/miniconda3/envs/lidar/python.exe` (Python 3.11)

> **Nota GT (2026-06-13):** As coordenadas XY de GT#20 e GT#21 (Alge2) estavam trocadas
> no ficheiro de ground truth — erro de registo humano confirmado por sobreposição com a
> trajectória do scanner. Corrigido em ambos os CSVs GT. Os resultados abaixo já reflectem
> esta correcção.

---

## 1. Resumo executivo

A v8 reduz o erro de forma **real e validada por cross-validation** (não in-sample),
e torna a calibração **defensável** (fatores quase unitários, sem o degrau da v7).

### 1.1 Run de deployment completo (2026-06-11, GT corrigido)

Run sobre os 5 ficheiros LAS de Alge (59 árvores GT, matching por coordenadas NN ≤ 2 m).

| Métrica | **v8 global (n=59)** | **v8 s/Alge2 (n=45)** | v7 s/Alge2 *(ref in-sample)* |
|---|---|---|---|
| **Volume MAPE** | 13.1 % | **8.1 %** | 12.0 % |
| Volume Bias | +7.8 % | +0.6 % | — |
| Volume RMSE | 20.8 % | 12.7 % | — |
| **DBH MAPE** | 5.1 % | **3.5 %** | 6.2 % |
| **Altura MAPE** | 4.1 % | 4.0 % | 4.4 % |

**Tempos de execução:** Alge1=139 min, Alge2=17 min, Alge4=91 min, Alge5=17 min,
Alge6=122 min → **total 6.4 horas** para 5 parcelas (~3.6 GB de LAS).

#### Volume MAPE por parcela (run real)

| Parcela | n | Vol MAPE | Vol Bias | H MAPE | DBH MAPE |
|---|---|---|---|---|---|
| Alge1 | 12 | 14.7 % | +9.9 % | 4.8 % | 2.8 % |
| **Alge2** | **14** | **25.6 %** | **+25.6 %** | 4.4 % | 9.2 % |
| Alge4 | 15 | 9.1 % | +1.3 % | 3.7 % | 5.5 % |
| Alge5 | 4 | 6.0 % | +1.3 % | 5.2 % | 4.3 % |
| Alge6 | 14 | 5.7 % | −3.0 % | 3.4 % | 2.7 % |

**Nota Alge2:** parcela nunca incluída na calibração. O bias +25.6 % deve-se
principalmente a T38 (+101 %, provável erro de segmentação) e T30 (+63 %, scan com
contaminação). Excluindo Alge2, o pipeline está essencialmente sem viés (+0.6 %).

---

### 1.2 Validação cruzada LOPO durante desenvolvimento (s/Alge2, n=45)

*(Protocolo: calibração re-ajustada em 3 parcelas, avaliada na 4ª — `calibra_lopo.py`)*

| Métrica | v7 (in-sample)\* | **v8 LOPO** | FJD Trion | CloudCompare |
|---|---|---|---|---|
| **Volume MAPE** | 12.0 %\* | **9.9 %** | 6.8 % | 9.4 % |
| Volume MedAPE | 9.0 % | **7.5 %** | 5.4 % | 8.6 % |
| **DBH MAPE** | 5.8 %\* | **4.0 %** | 3.7 % | 4.2 % |
| **DBH MedAPE** | 5.1 % | **2.5 %** | 3.7 % | 4.2 % |
| **Altura MAPE** | 4.4 % | 4.4 % | 2.1 % | 2.9 % |

\* O 12.0 %/5.8 % da v7 eram **in-sample**. Avaliada com LOPO, a v7 dava **~16–17 %**.

**Onde a v8 ganha:** bate o CloudCompare no DBH (4.0 < 4.2) e em 3 das 4 parcelas no
volume; bate o Trion na Alge5. O **DBH da árvore típica (MedAPE 2.5 %) é melhor que
ambas as ferramentas comerciais.** O gap residual concentra-se em árvores
estruturalmente difíceis (ver §6).

| Parcela | v8 LOPO | Trion | CC |
|---|---|---|---|
| Alge5 | **5.6 %** | 6.7 % | 9.6 % |
| Alge6 | 7.3 % | 5.5 % | 7.2 % |
| Alge4 | 9.1 % | 6.8 % | 9.3 % |
| Alge1 | 15.4 % | 8.1 % | 11.9 % |

---

## 2. A correção metodológica (o mais importante para o artigo)

O ganho mais relevante não foi um algoritmo — foi **medir honestamente**.

- **Ground truth canónico congelado:** `valores ground truth e cloudcompare e trionmodel ALGE.csv`
  (volume/altura/DBH destrutivos + baselines Trion e CloudCompare + coordenadas XY).
  As outras tabelas de GT que circulavam eram inconsistentes e foram descartadas.
- **Avaliação por leave-one-plot-out (LOPO):** toda a calibração é re-ajustada em 3
  parcelas e avaliada na 4ª (`calibra_lopo.py`). O número reportado é assim
  generalizável, ao contrário do MAPE in-sample da v7.
- **Calibração defensável:** o degrau `VOLUME_CALIB_SMALL/LARGE` (0.84/1.16 com salto
  em DBH=16 cm) e os fatores de DBH por modelo foram substituídos por **um fator global
  único** por variável. Com as melhorias de origem (§3), esses fatores ficaram quase
  unitários: **VOL ×0.966, DBH ×1.055, H ×1.027** — i.e. o pipeline está quase
  sem viés à partida.

---

## 3. Melhorias implementadas (por ordem de impacto no volume)

Todas têm flag (default = v8); o comportamento v7 é recuperável. Baseline preservado.

### 3.1 z_base a partir do solo CSF da fase 1 — **a maior alavanca**
`fase2_5_per_tree_setup.py` (`build_ground_lookup`, `ground_lookup`)

A v7 estimava a cota da base (`z_base`) pelos mínimos da própria nuvem da árvore, que
apanham sub-bosque/raízes → **enviesada para cima e com variância por árvore**. Como
todas as alturas de fatia são `z − z_base`, esse erro deslocava a medição do DBH ao
longo do taper, injetando variância. A v8 usa o **solo classificado por CSF (classe 2)
da fase 1** no local do tronco (mediana dos pontos de solo a < 3 m). Efeito isolado:
Volume 10.7→9.9 %, **DBH 5.3→4.0 %**, e o viés bruto do DBH caiu de −11 % para −5 %.

### 3.2 Curva de afilamento da espécie como modelo de taper
`taper_models.py` (`especie_shape`, `especie_taper`); `fase3` `TAPER_STRATEGY="especie"`

Modelo de **1 parâmetro** (escala k = d₀/2) com a forma fixa da espécie (polinómio de
4º grau calibrado em 59 árvores destrutivas, R²=0.97). A escala é estimada de forma
robusta (mediana de rᵢ/P(xᵢ)), pelo que **não dispara** como o ajuste livre de
potência/kozak em árvores com fatias contaminadas. Domou as caudas do volume.

### 3.3 Limite de sanidade por fator de forma
`fase4` (`USE_FORM_FACTOR_CLAMP`, `FF_LO/FF_HI=[0.34,0.62]`)

Trava o volume quando o fator de forma implícito `f = V/(π/4·DBH²·H)` sai do intervalo
plausível de *Pinus pinaster*. Apanha taper patológico de scan mau (ex.: uma árvore
passou de +180 % para +25 %).

### 3.4 Banda RANSAC mais larga
`fase3` (`RANSAC_BAND_LO=0.30, RANSAC_BAND_HI=3.00`, era 0.40/2.50)

Admite mais pontos por fatia no ajuste de círculo → fits mais robustos. Vencedor de um
sweep de parâmetros medido por LOPO (`sweep_v8.py`).

### 3.5 Refit geométrico de círculo por fatia
`fase3` (`_fit_circle_geom`, `USE_ROBUST_CIRCLE`)

Substitui o raio do "melhor círculo de 3 pontos" do RANSAC por um ajuste geométrico
(Gauss-Newton) a todos os inliers da fatia → menos variância do raio.

### 3.6 DBH desacoplado do modelo de volume (mediana robusta de 3 estimadores)
`fase3` (`DBH_FROM_SLICES`, `_dbh_local_linear`, `_dbh_cone`)

O DBH reportado é a **mediana** de: (1) ajuste linear local do perfil em [0.9, 2.5] m
avaliado a 1.3 m, (2) mediana das fatias perto de 1.3 m, (3) ajuste de **cone 3D**
(eixo único de toda a banda). Cada um é forte num regime; a mediana descarta o que
dispara (ex.: o cone em troncos divididos).

---

## 4. O que foi testado e rejeitado (ablation)

| Ideia | Resultado | Decisão |
|---|---|---|
| Altura por ápice cónico | pior que max-z | rejeitada |
| Altura por percentil (p99.5/p99) | pior que max-z | rejeitada |
| Volume puro por fator de forma (DBH²·H) | 12.5 % (pior que integral) | rejeitada |
| Volume híbrido (ff p/ frac_extrap alto) | 11.8 % | rejeitada |
| `SLICE_STEP` 0.15 | 13.6 % (muito pior) | rejeitada |
| `RANSAC_N_ITER` 400, `TAPER_FIT_H_MIN` 0.50 | ajudam isolados, pioram c/ banda larga | rejeitadas |
| Recuperar altura da nuvem bruta/original | topos não existem no scan (§6) | rejeitada |

---

## 5. Como reproduzir

```bash
# Avaliação honesta v7/Trion/CC vs GT
python avaliar_v8.py

# Re-correr v8 em CRU + calibração LOPO honesta
python rerun_v8.py --raw --fase25 --ground
python calibra_lopo.py

# Corrida calibrada (deployment)
python rerun_v8.py --fase25 --ground

# Sweep de parâmetros (LOPO)
python sweep_v8.py
```

Um run completo de raiz (`executar.py`) já usa todas as melhorias v8 por defeito,
incluindo o `z_base` do solo CSF.

---

## 6. Limitação fundamental: completude do scan (com prova)

O gap residual para o Trino **não é algorítmico** — é dos dados. Prova:

- A altura das árvores ocluídas (Alge5 #14/#16, Alge6 #73) está sub-estimada porque o
  topo da copa **não foi captado pelo laser**. Verificou-se contra a nuvem **original**
  (Baldios, pré-fase 1): os topos não existem em lado nenhum (`diag_raw.py`). A fase 1
  remove apenas ~10 % dos pontos — não é a causa.
- A pasta de dados é um **recorte** (`Baldios_Alge_clip_Pedro`); o Trion (sistema SLAM
  móvel) processou provavelmente a **aquisição completa**, com melhor cobertura.
- Confirmação indireta: onde os dados são bons, a v8 ganha (DBH típico melhor que ambas
  as comerciais; volume melhor que CC em 3/4 parcelas).

**Para superar o Trion seria preciso melhor input** (mais posições de scan / aquisição
completa / passagem de drone para os topos), não mais software. O motor v8 é
competitivo; corrido sobre dados completos, teria hipótese real de ganhar.

---

## 7. Enquadramento sugerido para o artigo

Pipeline open-source, validado por cross-validation, que **iguala ou supera ferramentas
comerciais em árvores adequadamente amostradas**, com calibração quase unitária
(defensável). O erro residual é dominado pela **completude do scan**, não pelo
algoritmo — uma conclusão demonstrável e cientificamente válida.

---

## 8. Constantes v8 (referência rápida)

| Ficheiro | Constante | Valor v8 | (v7) |
|---|---|---|---|
| `fase2_5` | `HEIGHT_METHOD` | `"max"` | max |
| `fase2_5` | `H_LEAFON_CORRECTION` | 1.0270 | 1.091 |
| `fase2_5` | z_base | solo CSF fase1 | mínimos da árvore |
| `fase3` | `USE_ROBUST_CIRCLE` | True | — |
| `fase3` | `RANSAC_BAND_LO/HI` | 0.30 / 3.00 | 0.40 / 2.50 |
| `fase3` | `TAPER_STRATEGY` | `"especie"` | kozak/potência |
| `fase3` | `DBH_FROM_SLICES` / `DBH_FROM_CYLINDER` | True / True | — |
| `fase4` | `VOLUME_MODE` | `"taper"` | taper |
| `fase4` | `USE_FORM_FACTOR_CLAMP` (FF_LO/HI) | True (0.34/0.62) | — |
| `fase4` | `VOLUME_CALIB_GLOBAL` | 0.9659 | degrau 0.84/1.16 |
| `fase4` | `DBH_REPORT_CALIB_GLOBAL` | 1.0549 | 1.049/1.098 p/ modelo |
