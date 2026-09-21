# Multimodal Emotion Recognition for ASD

Projet de Master 2 en Intelligence Artificielle consacré à la reconnaissance multimodale des émotions, avec étude du transfert vers des comportements observés chez des enfants présentant un trouble du spectre de l'autisme (TSA).

## Pipeline

```text
Vidéo → 40 images → EfficientNet-B0 → 40×1280 → Projection 1280→192 → Mamba vidéo
Audio → WAV mono 16 kHz → OpenSMILE/eGeMAPSv02 → 40×25 → Projection 25→192 → Mamba audio
                                      ↓
                              concaténation 40×384
                                      ↓
                                  Mamba fusion
                                      ↓
                                  LayerNorm
                                      ↓
                             mean temporal pooling
                                      ↓
                                classifieur MLP
                                      ↓
                                  8 classes
```

**Important :** 40 désigne le nombre de pas temporels, pas le nombre de caractéristiques. Pour l'audio, chaque séquence contient 40 pas temporels et 25 caractéristiques par pas.

## Dimensions

| Élément | Dimension |
|---|---:|
| Vidéo | 40 × 1280 |
| Audio | 40 × 25 |
| Projection vidéo | 40 × 192 |
| Projection audio | 40 × 192 |
| Fusion | 40 × 384 |
| Pooling | 384 |
| Classifieur caché | 128 |
| Classes RAVDESS | 8 |

## Jeux de données

### RAVDESS

Classes : Neutre, Calme, Heureux, Triste, Colère, Peur, Dégoût, Surprise.

Préparation finale : `X_video=(2160,40,1280)`, `X_audio=(2160,40,25)`, `y=(2160,)`.

### SSBD

Trois catégories utilisées : Excitation positive, Détresse négative, Auto-régulation apaisante.

Préparation finale : `X_video=(183,40,1280)`, `X_audio=(183,40,25)`, `y=(183,)`.

### IEMOCAP

Utilisé comme test externe. Le corpus n'est pas inclus dans ce dépôt.

## Modèles

- **Mamba final** : deux branches Mamba indépendantes puis Mamba de fusion.
- **Transformer** : branches temporelles séparées puis fusion Transformer.
- **BiLSTM** : branches BiLSTM séparées puis BiLSTM de fusion.

## Résultats principaux

### RAVDESS — Mamba

- Accuracy : **97,92 %**
- F1 macro : **0,9870**

### SSBD — transfert Mamba

- Accuracy : **92,86 %**
- Precision macro : **94,44 %**
- Recall macro : **91,67 %**
- F1 macro : **92,21 %**

Matrice de confusion :

```text
[[6, 2, 0],
 [0,10, 0],
 [0, 0,10]]
```

Ces résultats doivent être interprétés en tenant compte de la taille limitée de SSBD et du décalage de domaine entre les jeux de données.

## Arborescence

```text
multimodal-emotion-recognition-ASD/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── configs/config.yaml
├── notebooks/
│   ├── 01_data_preparation_RAVDESS.ipynb
│   ├── 02_feature_extraction_RAVDESS.ipynb
│   ├── 03_mamba_RAVDESS.ipynb
│   ├── 04_transformer_RAVDESS.ipynb
│   ├── 05_bilstm_RAVDESS.ipynb
│   ├── 06_SSBD_transfer.ipynb
│   └── 07_IEMOCAP_external_test.ipynb
├── src/
│   ├── models/{mamba.py,transformer.py,bilstm.py}
│   ├── features/{video_features.py,audio_features.py}
│   ├── data/{ravdess.py,ssbd.py,iemocap.py}
│   └── evaluation/{metrics.py,confusion_matrix.py}
├── results/{ravdess,ssbd,iemocap}/
├── figures/
└── docs/methodology.md
```

## Installation

```bash
git clone <URL_DU_DEPOT>
cd multimodal-emotion-recognition-ASD
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sous Windows : `.venv\\Scripts\\activate`.

## Confidentialité

Ne jamais publier de mots de passe, identifiants IEMOCAP, clés API, fichiers `.env`, données personnelles ou jeux de données soumis à restriction. Les checkpoints peuvent être distribués séparément via GitHub Releases si leur redistribution est autorisée.

## Citation

```text
Mouhous, Rayane. Reconnaissance multimodale des émotions à l'aide du Deep Learning dans le contexte du Trouble du Spectre de l'Autisme. Master 2 Intelligence Artificielle.
```
