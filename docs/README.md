# Documentation Yellowbrick

*Bienvenue dans la documentation Yellowbrick !

Si vous cherchez des informations sur l'utilisation de Yellowbrick, notre guide du contributeur, des exemples et des ressources pédagogiques, des réponses aux questions fréquemment posées, et plus encore, veuillez visiter la dernière version de notre documentation à [www.scikit-yb.org](https://www.scikit-yb.org/).

## Construire les documents

Pour construire les documents localement, installez d'abord les exigences spécifiques à la documentation avec `pip` en utilisant le fichier `requirements.txt` dans le répertoire `docs` :

``bash
$ pip install -r docs/requirements.txt
```

Vous pourrez alors construire la documentation depuis le répertoire `docs` en lançant `make html` ; la documentation sera construite et rendue dans le répertoire `_build/html`. Vous pouvez la visualiser en ouvrant `_build/html/index.html` puis en naviguant vers votre documentation dans le navigateur.

## reStructuredText

Yellowbrick utilise [Sphinx](http://www.sphinx-doc.org/en/master/index.html) pour construire sa documentation. Les avantages de l'utilisation de Sphinx sont nombreux ; nous pouvons établir un lien plus direct avec la documentation et le code source d'autres projets tels que Matplotlib et scikit-learn en utilisant [intersphinx](http://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html). En outre, les chaînes de documentation utilisées pour décrire les visualisateurs Yellowbrick peuvent être automatiquement incluses lorsque la documentation est construite via [autodoc](http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#sphinx.ext.autodoc).

Pour profiter de ces fonctionnalités, notre documentation doit être écrite en reStructuredText (ou "rst"). reStructuredText est similaire à markdown, mais pas identique, et il faut un peu de temps pour s'y habituer. Par exemple, le style pour des éléments tels que les codes, les hyperliens externes, les références croisées internes, les notes et le texte à largeur fixe sont tous uniques dans rst.

Si vous souhaitez contribuer à notre documentation et que vous n'avez pas d'expérience préalable avec rst, nous vous recommandons d'utiliser ces ressources :

- [A reStructuredText Primer] (http://docutils.sourceforge.net/docs/user/rst/quickstart.html)
- [Notes sur rst et feuille de contrôle](https://cheat.readthedocs.io/en/latest/rst.html)
- [Utilisation de la directive plot](https://matplotlib.org/devel/plot_directive.html)

## Ajouter de nouveaux visualiseurs à la documentation

Si vous ajoutez un nouveau visualiseur à la documentation, celle-ci contient de nombreux exemples sur lesquels vous pouvez baser vos fichiers de type similaire.

Le format principal de la section API est le suivant :

```
.. -*- mode : rst -*-

Mon visualiseur
=============

Une brève introduction à mon visualiseur et à son utilité dans le processus d'apprentissage automatique.

.. plot: :
    :context : close-figs
    :include-source : False
    :alt : Exemple utilisant MyVisualizer

    visualizer = MyVisualizer(LinearRegression())

    visualizer.fit(X, y)
    g = visualizer.show()

Discussion sur mon visualiseur et interprétation du graphique ci-dessus.

API Reference
-------------

.. automodule:: yellowbrick.regressor.mymodule
    :members: MyVisualizer
    :undoc-members:
    :show-inheritance:
```

Il s'agit d'une structure assez bonne pour une page de documentation ; une brève introduction suivie d'un exemple de code avec une visualisation incluse en utilisant [la directive plot](https://matplotlib.org/devel/plot_directive.html). Cela rendra l'image «MyVisualizer» dans le document avec les liens pour le code source complet, le png, et les versions pdf de l'image. Il aura également le « texte de remplacement » (pour les lecteurs d'écran) et n'affichera pas la source en raison de l'option `:include-source:`. Si `:include-source:` est omis, la source sera également incluse.

La section principale se termine par une discussion sur l'interprétation du visualiseur et son utilisation dans la pratique. Enfin, la section `API Reference` utilisera `automodule` pour inclure la documentation de votre docstring.

Il existe plusieurs autres endroits où vous pouvez lister votre visualiseur, mais pour vous assurer qu'il est inclus dans la documentation, il *doit être listé dans la table des matières de l'index local*. Recherchez le fichier `index.rst` dans votre sous-répertoire et ajoutez votre fichier rst (sans l'extension `.rst`) à la directive `.toctree::`. Cela garantira que votre documentation est incluse lors de sa création.

## Génération de la Galerie

Dans la version 1.0, nous avons adopté la [directive de traçage](https://matplotlib.org/devel/plot_directive.html) de Matplotlib, ce qui signifie que la majorité des images générées pour la documentation sont générées automatiquement. Une exception est la galerie ; les images de la galerie doivent toujours être générées manuellement.

Si vous avez contribué à un nouveau visualiseur comme décrit dans la section ci-dessus, veuillez également l'ajouter à la galerie, à la fois à `docs/gallery.py` et à `docs/gallery.rst`. (Vérifiez que vous avez déjà installé Yellowbrick en mode modifiable, à partir du répertoire de niveau supérieur : `pip install -e` .)

Si vous souhaitez régénérer une seule image (par exemple le tracé de la courbe du coude), vous pouvez procéder comme suit :

```bash
$ python docs/gallery.py elbow
```

Si vous voulez tous les régénérer (remarque : cela prend beaucoup de temps !)

```bash
$ python docs/gallery.py all
```
