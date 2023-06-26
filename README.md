# Yellowbrick


[![Build Status](https://github.com/DistrictDataLabs/yellowbrick/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/DistrictDataLabs/yellowbrick/actions/workflows/ci.yml)
[![Coverage Status](https://codecov.io/gh/DistrictDataLabs/yellowbrick/branch/develop/graph/badge.svg?token=BnaSECZz2r)](https://codecov.io/gh/DistrictDataLabs/yellowbrick)
[![Total Alerts](https://img.shields.io/lgtm/alerts/g/DistrictDataLabs/yellowbrick.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/DistrictDataLabs/yellowbrick/alerts/)
[![Language Grade: Python](https://img.shields.io/lgtm/grade/python/g/DistrictDataLabs/yellowbrick.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/DistrictDataLabs/yellowbrick/context:python)
[![PyPI version](https://badge.fury.io/py/yellowbrick.svg)](https://badge.fury.io/py/yellowbrick)
[![Documentation Status](https://readthedocs.org/projects/yellowbrick/badge/?version=latest)](http://yellowbrick.readthedocs.io/en/latest/?badge=latest)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1206239.svg)](https://doi.org/10.5281/zenodo.1206239)
[![JOSS](http://joss.theoj.org/papers/10.21105/joss.01075/status.svg)](https://doi.org/10.21105/joss.01075)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/DistrictDataLabs/yellowbrick/develop?filepath=examples%2Fexamples.ipynb)



**Analyse visuelle et outils de diagnostic pour faciliter la sélection de modèles d'apprentissage automatique**.

[ ![Banner](docs/images/readme/banner.png)](https://www.scikit-yb.org/en/latest/gallery.html)

## Qu'est-ce que Yellowbrick ?

Yellowbrick est une suite d'outils de diagnostic visuel appelés "Visualizers" qui étendent l'API scikit-learn pour permettre un pilotage humain du processus de sélection de modèle. En bref, Yellowbrick combine scikit-learn et matplotlib dans la meilleure tradition de la documentation scikit-learn, mais pour produire des visualisations pour votre flux de travail d'apprentissage automatique !

Pour une documentation complète sur l'API Yellowbrick, une galerie de visualisateurs disponibles, le guide du contributeur, des tutoriels et des ressources pédagogiques, une foire aux questions, et plus encore, veuillez consulter notre documentation à l'adresse [www.scikit-yb.org](https://www.scikit-yb.org/).

## Installation de Yellowbrick

Yellowbrick est compatible avec Python 3.4 ou plus récent et dépend également de scikit-learn et matplotlib. La manière la plus simple d'installer Yellowbrick et ses dépendances est de le faire à partir de PyPI avec pip, l'installateur de paquets préféré de Python.

   $ pip install yellowbrick

Notez que Yellowbrick est un projet actif et qu'il publie régulièrement de nouvelles versions avec plus de visualiseurs et de mises à jour. Afin de mettre à jour Yellowbrick vers la dernière version, utilisez pip comme suit.

    $ pip install -U yellowbrick

Vous pouvez également utiliser le drapeau `-U` pour mettre à jour scikit-learn, matplotlib, ou tout autre utilitaire tiers qui fonctionne bien avec Yellowbrick vers leurs dernières versions.

Si vous utilisez Anaconda (recommandé pour les utilisateurs de Windows), vous pouvez utiliser l'utilitaire conda pour installer Yellowbrick :

    conda install -c districtdatalabs yellowbrick

## Utilisation de Yellowbrick

L'API Yellowbrick est spécialement conçue pour fonctionner avec scikit-learn. Voici un exemple de séquence de travail typique avec scikit-learn et Yellowbrick :

### Visualisation des caractéristiques

Dans cet exemple, nous voyons comment Rank2D effectue des comparaisons par paire de chaque caractéristique de l'ensemble de données avec une métrique ou un algorithme spécifique et les renvoie ensuite classées dans un diagramme triangulaire en bas à gauche.

```python
from yellowbrick.features import Rank2D

visualizer = Rank2D(
    features=features, algorithm='covariance'
)
visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.show()                   # Finalize and render the figure
```

### Visualisation du modèle

Dans cet exemple, nous instancions un classificateur scikit-learn et utilisons ensuite la classe ROCAUC de Yellowbrick pour visualiser le compromis entre la sensibilité et la spécificité du classificateur.

```python
from sklearn.svm import LinearSVC
from yellowbrick.classifier import ROCAUC

model = LinearSVC()
visualizer = ROCAUC(model)
visualizer.fit(X,y)
visualizer.score(X,y)
visualizer.show()
```


Pour plus d'informations sur le démarrage de Yellowbrick, consultez le [Guide de démarrage rapide](https://www.scikit-yb.org/en/latest/quickstart.html) dans la [documentation](https://www.scikit-yb.org/en/latest/) et consultez notre [cahier d'exemples](https://github.com/DistrictDataLabs/yellowbrick/blob/develop/examples/examples.ipynb).

## Contribuer à Yellowbrick

Yellowbrick est un projet open source soutenu par une communauté qui acceptera avec gratitude et humilité toutes les contributions que vous pourriez apporter au projet. Grande ou petite, toute contribution fait une grande différence ; et si vous n'avez jamais contribué à un projet open source auparavant, nous espérons que vous commencerez avec Yellowbrick !

Si vous souhaitez contribuer, consultez notre [guide du contributeur] (https://www.scikit-yb.org/en/latest/contributing/index.html). Au-delà de la création de visualiseurs, il y a de nombreuses façons de contribuer :

- Soumettre un rapport de bogue ou une demande de fonctionnalité sur [GitHub Issues] (https://github.com/DistrictDataLabs/yellowbrick/issues).
- Ajoutez un carnet Jupyter à notre [galerie] d'exemples (https://github.com/DistrictDataLabs/yellowbrick/tree/develop/examples).
- Aidez-nous avec [user testing](https://www.scikit-yb.org/en/latest/evaluation.html).
- Ajoutez à la documentation ou aidez-nous avec notre site web, [scikit-yb.org](https://www.scikit-yb.org).
- Écrire des [tests unitaires ou d'intégration](https://www.scikit-yb.org/en/latest/contributing/developing_visualizers.html#integration-tests) pour notre projet.
- Répondre aux questions sur nos problèmes, notre liste de diffusion, Stack Overflow, et ailleurs.
- Traduire notre documentation dans une autre langue.
- Écrire un billet de blog, tweeter ou partager notre projet avec d'autres.
- Enseigner](https://www.scikit-yb.org/en/latest/teaching.html) à quelqu'un comment utiliser Yellowbrick.

Comme vous pouvez le voir, il y a de nombreuses façons de s'impliquer et nous serions très heureux que vous nous rejoigniez ! La seule chose que nous vous demandons est de respecter les principes d'ouverture, de respect et de considération des autres tels que décrits dans le [Python Software Foundation Code of Conduct](https://www.python.org/psf/codeofconduct/).

Pour plus d'informations, consultez le fichier `CONTRIBUTING.md` à la racine du dépôt ou la documentation détaillée sur [Contributing to Yellowbrick](https://www.scikit-yb.org/en/latest/contributing/index.html).

## Jeux de données Yellowbrick

Yellowbrick donne un accès facile à plusieurs ensembles de données qui sont utilisés pour les exemples dans la documentation et les tests. Ces jeux de données sont hébergés dans notre CDN et doivent être téléchargés pour être utilisés. Généralement, lorsqu'un utilisateur appelle l'une des fonctions de chargement de données, par exemple `load_bikeshare()`, les données sont automatiquement téléchargées si elles ne se trouvent pas déjà sur l'ordinateur de l'utilisateur. Cependant, pour le développement et les tests, ou si vous savez que vous travaillerez sans accès à Internet, il peut être plus facile de télécharger toutes les données en une seule fois.

Le script de téléchargement des données peut être exécuté comme suit :

    $ python -m yellowbrick.download


Ceci téléchargera les données dans le répertoire fixtures à l'intérieur des paquets du site Yellowbrick. Vous pouvez spécifier l'emplacement du téléchargement soit comme argument au script de téléchargement (utilisez `--help` pour plus de détails) ou en définissant la variable d'environnement `$YELLOWBRICK_DATA`. C'est le mécanisme préféré car il influencera également la façon dont les données sont chargées dans Yellowbrick.

Note : Les développeurs qui ont téléchargé des données à partir de versions de Yellowbrick antérieures à v1.0 peuvent rencontrer des problèmes avec l'ancien format de données. Si cela se produit, vous pouvez vider votre cache de données comme suit :

    $ python -m yellowbrick.download --cleanup

Cela supprimera les anciens jeux de données et téléchargera les nouveaux. Vous pouvez également utiliser le drapeau `--no-download` pour simplement vider le cache sans retélécharger les données. Les utilisateurs qui ont des difficultés avec les jeux de données peuvent également utiliser ceci ou ils peuvent désinstaller et réinstaller Yellowbrick en utilisant `pip`.

## Citer Yellowbrick

Nous serions heureux que vous utilisiez Yellowbrick dans vos publications scientifiques ! Si vous le faites, veuillez nous citer en utilisant les [directives de citation] (https://www.scikit-yb.org/en/latest/about.html#citing-yellowbrick).

## Affiliations

[ ![District Data Labs](docs/images/readme/affiliates_ddl.png)](https://districtdatalabs.com/) [ ![Projet affilié NumFOCUS](docs/images/readme/affiliates_numfocus.png)](https://numfocus.org)