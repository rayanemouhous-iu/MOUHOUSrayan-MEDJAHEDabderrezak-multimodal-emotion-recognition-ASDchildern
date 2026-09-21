# Méthodologie

Le pipeline suit les étapes : préparation → extraction vidéo/audio → séquencement à 40 pas → normalisation → branches temporelles → fusion → pooling → classification → évaluation.

Vidéo : 40 images, EfficientNet-B0, 1280 caractéristiques par image.

Audio : mono 16 kHz, OpenSMILE/eGeMAPSv02, 25 caractéristiques par pas, ré-échantillonnées vers 40 pas.

Modèle final : projection vidéo/audio vers 192, Mamba par modalité, concaténation en 384, Mamba de fusion, LayerNorm, mean pooling, classifieur.
