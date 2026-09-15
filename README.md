# Cats vs Dogs — CNN from Scratch vs Transfer Learning

Comparaison de deux approches de classification d'images (chats vs chiens) : un CNN construit manuellement et un modèle en transfert learning (ResNet18).

## Environnement
- Google Colab (GPU T4)
- Python 3, PyTorch, torchvision, scikit-learn, matplotlib, seaborn

## Données
Jeu de données Cats vs Dogs (25 000 images), téléchargé depuis Kaggle (dataset `salader/dogsvscats`), organisé en :
data/catsvsdogs/
  train/cats, train/dogs   (20 000 images)
  test/cats, test/dogs     (5 000 images)


## Modèles
1. **CNN from scratch** : 3 blocs convolutifs (32→64→128 filtres), chacun avec BatchNorm (stabilise l'apprentissage après chaque convolution) et Dropout (0.5) avant la couche dense finale (là où le risque de sur-apprentissage est le plus fort).
2. **Transfer Learning (ResNet18)** : couches convolutives gelées (pré-entraînées sur ImageNet), seule la couche finale (remplacée) est ré-entraînée.

## Entraînement
- Seed fixé (42) pour la reproductibilité
- Data augmentation : flip horizontal, rotation aléatoire (train uniquement)
- Optimiseurs testés : Adam (lr=0.001) et SGD (momentum=0.9)
  - SGD avec lr=0.01 : échec (le modèle n'apprend pas, accuracy bloquée à 50%)
  - SGD avec lr=0.001 : fonctionne correctement (83.8% accuracy)
- Métriques suivies à chaque époque : loss, accuracy, precision, recall

## Résultats

| Modèle | Optimiseur | Accuracy (val) | Precision | Recall |
|---|---|---|---|---|
| CNN from scratch | Adam (lr=0.001) | 82.5% | 0.93 | 0.70 |
| CNN from scratch | SGD (lr=0.001) | 83.8% | 0.84 | 0.84 |
| Transfer Learning (ResNet18) | Adam (lr=0.001) | ~90-91% | ~0.95 | ~0.86 |

Le transfert learning surpasse nettement le CNN from scratch, avec une loss de validation plus basse et plus stable au fil des époques. La matrice de confusion du modèle final montre une légère tendance à confondre des chiens avec des chats (340 erreurs) plus que l'inverse (115 erreurs).

## Limites
- CNN from scratch entraîné seulement 10 époques (accuracy encore en progression)
- Pas de recherche systématique d'hyperparamètres (grid/random search)
- Pas de scheduler de learning rate utilisé

## Utilisation
Ouvrir `notebook.ipynb` dans Google Colab, exécuter les cellules dans l'ordre (connexion à Google Drive requise pour les données).
