# CIFAR-10 Classification

## Description

Bibliothèque pour classifier les données de CIFAR-10.

## Installation

1. Cloner le répôt :

    ```bash
    git clone https://github.com/Lugdum/cifar10-classifier.git
    cd cifar10-classifier
    ```

2. Assurez-vous d'avoir `make` installé. Sur Ubuntu/Debian, vous pouvez l'installer avec :

    ```bash
    sudo apt-get install build-essential
    ```

3. Créer et activer l'environnement virtuel et installer les dépendances :

    ```bash
    make requirements
    source venv/bin/activate
    ```

    **Remarque :** Le Makefile utilise `python3` comme interpréteur. Si votre système utilise `python` au lieu de `python3`, modifiez le Makefile en conséquence :

    ```
    PROJECT_NAME = cifar10_classification
    PYTHON_VERSION = 3.10
    PYTHON_INTERPRETER = python
    ```

Cette commande va :
- Installer les dépendances listées dans `requirements.txt`.
- Télécharger et extraire les données et les modèles nécessaires.

## Utilisation

### Scripts

Vous pouvez lancer les scripts présents dans le dossier scripts/ uns par uns, ou lancer le script `run_all.py` afin de lancer le projet entier.

### Notebook

Pour utiliser les fonctions du projet dans un Notebook ou dans un fichier python, il est conseillé d'aller voir le notebook `example_notebook.ipynb` qui se trouve dans le dossier notebooks/ car il utilise les fonctions les plus utiles du projet.

### Make

Vous pouvez également lancer les scripts grâce aux commandes make, pour voir les différents scripts lançables, faites la commande :

```bash
make
```
