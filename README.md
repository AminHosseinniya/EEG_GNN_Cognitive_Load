# EEG-Based Cognitive Load Detection Using Heterogeneous Graph Neural Networks

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.3+-orange.svg)](https://pyg.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Master's Thesis** — Mohammad Amin Hosseinnia  
Supervisor: Dr. Elham Akhoundzadeh  
Tarbiat Modares University, Faculty of Industrial Engineering  
Department of Information Technology Engineering — 2025

---
## How This Repository Is Organized

This project follows a sequential research pipeline. If you are reading 
this for the first time, the sections below are ordered to match the 
actual flow of the work:

1. **Datasets** — two public EEG datasets are loaded and harmonized into 
   one unified corpus
2. **Signal Representation** — raw signals are segmented into 30-second 
   windows, then further into 1-second sub-windows
3. **Graph Representations** — sub-windows and channels are converted into 
   two types of graphs: a simple channel graph and a richer heterogeneous 
   multi-relational graph
4. **Models** — four GNN architectures and one LSTM baseline are trained, 
   each on the appropriate graph type
5. **Results** — models are evaluated and compared using six metrics, 
   training curves, and confusion matrices
6. **Fairness Analysis** — the two best models are evaluated separately 
   on male and female subgroups to assess demographic robustness

The notebook `notebooks/eeg_gnn_thesis.ipynb` implements all of these 
steps end-to-end in the same order.


## Overview

This repository contains the full implementation of a graph neural network 
framework for EEG-based cognitive load detection during mental arithmetic tasks.

The key contribution is a **heterogeneous multi-relational graph representation** 
of EEG signals that simultaneously encodes three types of relationships between 
EEG sub-windows: temporal continuity, spatial channel connectivity, and signal 
similarity across time.

---

## Pipeline

![Research Flowchart](assets/flowchart.png)
> **Figure 1** — Full research pipeline: from raw EEG to graph construction, 
> model training, evaluation, and fairness analysis.

---

## Datasets

This project uses two publicly available EEG datasets. The two datasets were selected because they share a common set of frontal 
EEG channels and the same cognitive task (mental arithmetic), but differ 
in recording hardware, subject demographics, and number of cognitive load 
levels. Combining them increases the diversity of the training data. Both 
are publicly available and require no registration to download.

| Dataset | Subjects | Recordings | Task | Sampling Rate |
|---------|----------|------------|------|---------------|
| [EEGMAT](https://physionet.org/content/eegmat/1.0.0/) — Zyma et al. (2019) | 36 | 72 | Rest vs. Mental Arithmetic | 500 Hz → 250 Hz |
| [Cognitive Load EEG](https://data.mendeley.com/datasets/kt38js3jv7/1) — Nirabi et al. (2025) | 15 | 60 | Arithmetic (4 levels) | 250 Hz |
| **Combined** | **51** | **132** | Binary: Rest vs. Load | **250 Hz** |
| **After Windowing** | — | **447 hyper-windows** | 30s segments | 250 Hz |

> ⚠️ Datasets are not included due to size. Download from the links above 
> and place in a `data/` folder as described in [Usage](#usage).

---

## Signal Representation

Raw EEG recordings vary in length and come from different devices. Before 
any modeling can happen, signals need to be brought to a common format. 
This section shows what the raw signal looks like, and how it is cut into 
fixed-length pieces. The 30-second hyper-window is the unit of 
classification — one label per window. The 1-second sub-window is the 
unit of graph nodes — one node per sub-window per channel.

### Raw EEG Signal
![Raw EEG Signal](assets/raw_eeg_signal.png)
> **Figure 4** — Sample raw EEG recording across 7 frontal channels 
> (Fp1, Fp2, F7, F3, Fz, F4, F8) before any processing.

### Windowing Strategy
![Windowing Scheme](assets/windowing_scheme.png)
> **Figure 5** — Each recording is divided into 30-second hyper-windows, 
> then each hyper-window is subdivided into 1-second sub-windows. 
> The shaded regions show the first few sub-windows within one hyper-window.

![Sub-window Sample](assets/subwindow_sample.png)
> **Figure 6** — A single 1-second sub-window across all 7 channels. 
> Each sub-window becomes one node in the heterogeneous graph.

---

## Graph Representations

Rather than feeding raw signals or flat feature vectors into a neural 
network, this project converts each 30-second window into a graph. 
Two graph designs are compared. The channel graph is simpler — it only 
captures which brain regions are correlated with each other. The 
heterogeneous graph is richer — it additionally captures how activity 
evolves over time within each channel and which sub-windows share 
similar signal patterns. The hypothesis is that the richer relational 
structure of the heterogeneous graph leads to better classification, 
and the results confirm this.

### 1. Channel Graph
![Channel Graph Diagram](assets/channel_graph_diagram.png)
> **Figure 2** — In the channel graph, each EEG channel is one node (7 nodes total). 
> Edges connect channels whose Pearson correlation exceeds 0.3.

![Channel Graph — Both Classes](assets/channel_graph_classes.png)
> **Figure 8** — Channel graph instances for Rest (left) and Arithmetic (right) classes. 
> Edge density and connectivity patterns differ between cognitive states.

### 2. Heterogeneous Multi-Relational Graph
![Heterogeneous Graph Diagram](assets/hetero_graph_diagram.png)
> **Figure 3** — The heterogeneous graph has 210 nodes (7 channels × 30 sub-windows). 
> Three edge types are defined: **temporal** (gray, t→t+1 within a channel), 
> **channel** (blue, correlated channels at the same timestep), and 
> **similarity** (orange, recurring signal patterns across time).

![Heterogeneous Graph Sample](assets/hetero_graph_sample.png)
> **Figure 7** — A rendered instance of the heterogeneous graph for one 
> 30-second window. Rows represent channels, columns represent time steps.

---

## Models

Four GNN architectures are evaluated: GCN and GAT are applied to the 
simpler channel graph, while RGCN and GAT-Hetero are applied to the 
heterogeneous graph. RGCN is specifically chosen for the heterogeneous 
graph because it learns separate weight matrices for each edge type, 
allowing it to treat temporal, channel, and similarity edges differently. 
An LSTM trained directly on raw signal sequences serves as the non-graph 
baseline, representing the conventional deep learning approach to EEG 
classification.

| Model | Graph Type | Architecture |
|-------|-----------|--------------|
| GCN | Channel Graph | 2-layer Graph Convolutional Network |
| GAT | Channel Graph | 2-layer Graph Attention Network (4 heads) |
| RGCN | Hetero Graph | 2-layer Relational GCN (3 relation types) |
| GAT-Hetero | Hetero Graph | 2-layer Graph Attention Network |
| LSTM | Raw Signals | 2-layer LSTM (baseline) |

---

## Results

All models were evaluated on the same held-out test set (15% of the data) 
that was never seen during training or validation. Six metrics are reported: 
accuracy and F1 score reflect threshold-dependent classification quality, 
while ROC-AUC and PR-AUC are threshold-independent and better reflect 
performance under class imbalance. The results are presented as a table, 
followed by training curves, confusion matrices, and ROC/PR curves.

### Overall Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----------|---------|--------|
| GCN — Channel | 0.8261 | 0.8696 | 0.6897 | 0.7692 | 0.8724 | 0.8700 |
| GAT — Channel | 0.7971 | 0.7778 | 0.7241 | 0.7500 | 0.9060 | 0.8998 |
| RGCN — Hetero | **0.8551** | 0.8800 | 0.7586 | **0.8148** | 0.9112 | 0.8830 |
| GAT — Hetero  | 0.8406 | **0.9091** | 0.6897 | 0.7843 | **0.9267** | **0.9093** |
| LSTM — Baseline | 0.7353 | 0.6786 | 0.6786 | 0.6786 | 0.7259 | 0.6472 |

![Model Comparison Bar Chart](assets/model_comparison_bar.png)
> **Figure 9** — Validation accuracy comparison across all models. 
> Blue = best val accuracy, Orange = final val accuracy. 
> Hetero graph models consistently outperform channel graph and LSTM baseline.

### Training Dynamics

The plots below show the full training history of each model. Although training continued until early stopping triggered, the model weights were automatically saved at the epoch with the best validation performance — just before overfitting began to develop. The reported test results therefore reflect the best generalizing checkpoint, not the final epoch.

<table>
<tr>
<td><img src="assets/training_gcn.png" alt="GCN Training"/></td>
<td><img src="assets/training_gat_channel.png" alt="GAT Channel Training"/></td>
</tr>
<tr>
<td align="center"><b>Figure 10</b> — GCN on Channel Graph</td>
<td align="center"><b>Figure 11</b> — GAT on Channel Graph</td>
</tr>
<tr>
<td><img src="assets/training_rgcn.png" alt="RGCN Training"/></td>
<td><img src="assets/training_gat_hetero.png" alt="GAT Hetero Training"/></td>
</tr>
<tr>
<td align="center"><b>Figure 12</b> — RGCN on Hetero Graph</td>
<td align="center"><b>Figure 13</b> — GAT on Hetero Graph</td>
</tr>
</table>

![LSTM Training](assets/training_lstm.png)
> **Figure 14** — LSTM baseline training on raw EEG signals. 
> Val accuracy plateaus around 65%, well below graph-based models.

![All Models Training Comparison](assets/training_all_comparison.png)
> **Figure 15** — Validation loss and accuracy for all 5 models overlaid. 
> Hetero graph models converge faster and to better values.

### Confusion Matrices

![Confusion Matrices](assets/confusion_matrices.png)
> **Figure 16** — Confusion matrices on the test set for all five models. 
> RGCN-Hetero achieves the best balance between true positives and false negatives.

### ROC and Precision-Recall Curves

![ROC Curves](assets/roc_curves.png)
> **Figure 17** — ROC curves for all models. GAT-Hetero achieves the highest 
> AUC of 0.927, followed closely by RGCN-Hetero at 0.911. 
> LSTM (AUC=0.726) is far below all graph-based models.

![Precision-Recall Curves](assets/pr_curves.png)
> **Figure 18** — Precision-Recall curves. GAT-Hetero leads with AP=0.909. 
> The steep drop in LSTM's PR curve highlights its poor robustness under 
> threshold variation.

---

## Fairness Analysis

Most EEG classification studies report only aggregate accuracy without 
examining whether performance is consistent across demographic groups. 
Since EEG signals are known to vary across individuals, a model that 
works well on average may still underperform for specific subgroups. 
Here, the two best-performing models — RGCN-Hetero and GAT-Hetero — 
are evaluated separately on male (n=209) and female (n=74) subgroups 
from the EEGMAT dataset. Note that the TXT dataset has no embedded 
demographic metadata and is therefore excluded from this analysis.

![Fairness Confusion Matrices](assets/fairness_confusion.png)
> **Figure 19** — Confusion matrices for GAT-Hetero on female (n=74, acc=0.797) 
> and male (n=209, acc=0.823) subgroups. Error patterns are consistent 
> across both groups, indicating no gender-specific bias.

![Fairness Bar Chart](assets/fairness_bar.png)
> **Figure 20** — Accuracy comparison by sex for RGCN-Hetero and GAT-Hetero. 
> The accuracy gap between male and female subgroups is 3–6%, 
> attributable to dataset imbalance rather than model bias.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/eeg-gnn-cognitive-load.git
cd eeg-gnn-cognitive-load
pip install -r requirements.txt
```

## Usage

Open `notebooks/eeg_gnn_thesis.ipynb` in Google Colab.  
Update the dataset paths in Section 2:

```python
edf_path = "/your/path/to/EEGMAT"
txt_path = "/your/path/to/CognitiveLoadEEG/Arithmetic_Data"
```

---

## Citation

```bibtex
@mastersthesis{hosseinnia2025eeg,
  author  = {Hosseinnia, Mohammad Amin},
  title   = {EEG-Based Cognitive Load Detection Using Heterogeneous 
             Graph Neural Networks},
  school  = {Tarbiat Modares University},
  year    = {2025},
  advisor = {Akhoundzadeh, Elham}
}
```

```bibtex
@article{zyma2019electroencephalograms,
  author  = {Zyma, Igor and Tukaev, Sergii and Seleznov, Ivan and 
             Kiyono, Ken and Popov, Anton and Chernykh, Mariia and 
             Shpenkov, Oleksii},
  title   = {Electroencephalograms during mental arithmetic task performance},
  journal = {Data},
  volume  = {4},
  number  = {1},
  pages   = {14},
  year    = {2019}
}
```

```bibtex
@article{nirabi2025cognitive,
  author  = {Nirabi, Ali and others},
  title   = {Cognitive load assessment through EEG: A dataset from 
             arithmetic and Stroop tasks},
  journal = {Data in Brief},
  volume  = {60},
  pages   = {111477},
  year    = {2025}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
