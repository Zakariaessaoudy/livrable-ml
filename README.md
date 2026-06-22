# Deep Learning - Livrable strict des quatre rapports

Ce dossier contient uniquement le README et les quatre notebooks demandés. Chaque notebook reprend le dataset, les variables, le preprocessing, l'architecture, les hyperparamètres, les visualisations, les métriques, les limites et les pistes d'amélioration de son rapport associé.

## Structure

```text
livrable_notebooks_zakaria/
├── tp1_transfer_learning.ipynb  # FashionMNIST : AlexNet / ResNet18 / VGG16
├── tp2_stacked_lstm.ipynb       # Consommation électrique minute par minute
├── tp3_fuel_lstm.ipynb          # Consommation carburant et export TinyML
├── tp4_transformers.ipynb       # ViT / TAPAS / wav2vec 2.0 / CLIP
└── README.md
```

## Correspondance exacte

| Notebook | Rapport | Configuration reproduite |
|---|---|---|
| `tp1_transfer_learning.ipynb` | `Rapport_Transfer_Learning.pdf` | FashionMNIST 60k/10k, RGB 3x224x224, normalisation ImageNet, batch 64, AlexNet/VGG16/ResNet18, Feature Extraction lr 1e-3, ResNet18 Fine-Tuning `layer4` lr 1e-4, 3 epochs, CrossEntropyLoss, matrice de confusion |
| `tp2_stacked_lstm.ipynb` | `Rapport_Stacked_LSTM.pdf` | UCI minute par minute, nettoyage complet, 100 000 dernières mesures, 10 features, MinMax global, fenêtre 60, split 80/20, LSTM 3x128, dropout 0.3, 336 001 paramètres PyTorch, MSE, Adam 1e-3, batch 32, 10 epochs |
| `tp3_fuel_lstm.ipynb` | `rapport_LSTM.pdf` | Kaggle FuelConsumption, 639 puis 638 lignes, 3 features, split 70/30, reshape 3x1, LSTM 12+12, Dense 32, Adamax, MAE, validation 20 %, batch 1, 300 epochs, résidus et export TinyML compatible Keras 3 |
| `tp4_transformers.ipynb` | `Rapport_Transformers.pdf` | Chronologie log, lois d'échelle, ViT, iGPT, TAPAS, wav2vec 2.0/SUPERB, CLIP, DALL-E, wav2vec-U, Mixture of Experts et synthèse |

## Exécution dans Google Colab

1. Ouvrir [Google Colab](https://colab.research.google.com/).
2. Importer un notebook.
3. Pour TP1 et TP4, sélectionner un GPU : `Exécution > Modifier le type d'exécution > T4 GPU`.
4. Exécuter toutes les cellules dans l'ordre.

Les dépendances et datasets sont téléchargés automatiquement. Les notebooks utilisent directement les valeurs complètes des rapports : aucun mode réduit ou `SMOKE_TEST` n'est activé.

## Détails par notebook

### TP1 - Transfer Learning FashionMNIST

- Classes : T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
- Transformation `[1,28,28] -> [3,224,224]` et normalisation ImageNet.
- Poids `IMAGENET1K_V1` pour AlexNet, VGG16 et ResNet18.
- Feature Extraction des trois architectures pendant 3 epochs.
- Fine-Tuning ResNet18 : `layer4` et `fc` dégelés pendant 3 epochs.
- Évaluation : accuracy, rapport par classe et matrice de confusion.
- Tableau indicatif du rapport conservé : AlexNet 82/88 %, ResNet18 85/93 %, VGG16 84/91 %.
- Limite et amélioration : décalage RGB/grayscale, Vision Transformer ou pré-entraînement grayscale.

### TP2 - Stacked LSTM électrique

- Fichier officiel UCI `household_power_consumption.txt` téléchargé automatiquement.
- Encodage de l'heure avec `sin` et `cos` utilisant exactement l'heure entière du rapport.
- Normalisation globale avant extraction des 100 000 dernières observations, comme dans l'annexe.
- 99 940 séquences : 79 952 train et 19 988 test.
- Architecture et assertion du total PyTorch réel de 336 001 paramètres. Le rapport annonçait 334 465 parce qu'il ne comptait qu'un biais par couche, alors que `nn.LSTM` en stocke deux (`bias_ih` et `bias_hh`).
- Résultats de référence du rapport : MAE 0,1912 kW et RMSE 0,2898 kW.
- Améliorations conservées : 50 epochs, scheduler, warm-up/lr réduit, seq-to-seq, validation/early stopping et MAE/Huber.

### TP3 - LSTM carburant / TinyML

- Endpoint public Kaggle utilisé uniquement pour placer automatiquement le CSV sous le nom attendu `data/FuelConsumption.csv`.
- Aucune normalisation des features dans le pipeline principal, conformément au rapport.
- Entrée fixe batch 1 et entraînement explicite avec `batch_size=1`, cohérent avec la justification TinyML.
- Pairplot, Spearman, courbes, prédictions, scatter et histogrammes des résidus.
- Export vers `LSTM/model.h` : l'API historique `eloquent_tensorflow` est utilisée si disponible ; sinon un fallback TensorFlow Lite compatible Keras 3 produit automatiquement `model.tflite` et le header C.
- Limites conservées : LSTM suboptimal sur tabulaire, année 2000, absence de scaling et besoin de benchmarks MLP/Random Forest/régression.

### TP4 - Transformers multimodaux

- Les cinq architectures étudiées dans le rapport sont expliquées, y compris iGPT.
- Les trois défis de scaling sont conservés : complexité quadratique, environnement, biais/sécurité.
- ViT utilise le fallback rouge 100x100 du rapport.
- TAPAS reprend exactement le tableau, les quatre questions et les agrégations.
- wav2vec reprend `superb/asr`, y compris la gestion de l'erreur de compatibilité décrite dans le rapport.
- CLIP reprend les trois textes. Comme `optimusprime.jpg` n'était pas fourni, le notebook télécharge automatiquement une image de robots SO-100 depuis le dépôt officiel Hugging Face Documentation Images.
- La valeur Switch-C est `1.6e12`, cohérente avec le tableau « 1.6 T » et le facteur x7500 ; l'exposant `e13` de l'annexe est signalé comme une coquille.

## Résultats et reproductibilité

Les valeurs numériques peuvent varier légèrement selon le GPU, les versions et l'initialisation. Les valeurs imprimées par l'exécution sont les résultats à transmettre ; les valeurs historiques des rapports restent clairement identifiées comme « rapportées » ou « indicatives ».

Aucun fichier externe manuel n'est nécessaire. L'image CLIP provient de `huggingface/documentation-images` (`lerobot/so100-lerobot.jpg`, licence CC BY-NC-SA 4.0) et est enregistrée automatiquement sous `images/optimusprime.jpg` pour conserver le chemin du rapport.
