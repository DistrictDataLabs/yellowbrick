# Contribuer à Yellowbrick


**Pour lire le guide complet du contributeur, veuillez visiter la [page de contribution] (http://www.scikit-yb.org/en/latest/contributing/index.html) dans la documentation. Veillez à lire attentivement cette page afin de vous assurer que le processus de révision se déroule de la manière la plus fluide possible et que votre contribution ait le plus de chances d'être fusionnée.


Pour en savoir plus sur le développement, les objectifs et les motivations de Yellowbrick, consultez notre présentation aux développeurs : [Visualisation de la sélection de modèles avec Scikit-Yellowbrick : An Introduction to Developing Visualizers] (http://www.slideshare.net/BenjaminBengfort/visualizing-model-selection-with-scikityellowbrick-an-introduction-to-developing-visualizers).


## Comment contribuer


Yellowbrick est un projet open source soutenu par une communauté qui acceptera avec gratitude et humilité toutes les contributions que vous pourriez apporter au projet. Grande ou petite, toute contribution fait une grande différence ; et si vous n'avez jamais contribué à un projet open source auparavant, nous espérons que vous commencerez avec Yellowbrick !


Principalement, le développement de Yellowbrick concerne l'ajout et la création de *visualiseurs* &mdash ; des objets qui apprennent des données et créent une représentation visuelle des données ou du modèle. Les visualiseurs s'intègrent aux estimateurs, transformateurs et pipelines de scikit-learn pour des objectifs spécifiques et, par conséquent, peuvent être simples à construire et à déployer. La contribution la plus courante est donc un nouveau visualiseur pour un modèle ou une famille de modèles spécifique. Nous verrons plus tard en détail comment construire des visualiseurs.


Au-delà de la création de visualiseurs, il existe de nombreuses façons de contribuer :


- Soumettre un rapport de bogue ou une demande de fonctionnalité sur [GitHub Issues] (https://github.com/DistrictDataLabs/yellowbrick/issues).
- Ajoutez un carnet Jupyter à notre [galerie d'exemples] (https://github.com/DistrictDataLabs/yellowbrick/tree/develop/examples).
- Aidez-nous avec les [tests utilisateurs] (http://www.scikit-yb.org/en/latest/evaluation.html).
- Ajoutez à la documentation ou aidez-nous avec notre site web, [scikit-yb.org](http://www.scikit-yb.org).
- Écrire des [tests unitaires ou d'intégration](https://www.scikit-yb.org/en/latest/contributing/developing_visualizers.html#integration-tests) pour notre projet.
- Répondre aux questions sur nos problèmes, notre liste de diffusion, Stack Overflow, et ailleurs.
- Traduire notre documentation dans une autre langue.
- Écrire un billet de blog, tweeter ou partager notre projet avec d'autres.
- Enseigner](https://www.scikit-yb.org/en/latest/teaching.html) à quelqu'un comment utiliser Yellowbrick.


Comme vous pouvez le voir, il y a de nombreuses façons de s'impliquer et nous serions très heureux que vous nous rejoigniez ! La seule chose que nous vous demandons est de respecter les principes d'ouverture, de respect et de considération des autres tels que décrits dans le [Code de conduite de la Python Software Foundation](https://www.python.org/psf/codeofconduct/).


## Démarrer sur GitHub
Yellowbrick est hébergé sur GitHub à https://github.com/DistrictDataLabs/yellowbrick.

Le flux de travail typique d'un contributeur à la base de code est le suivant :

1. **Découvrez** un bug ou une fonctionnalité en utilisant Yellowbrick.
2. **Discuter** avec les contributeurs principaux en [ajoutant un problème](https://github.com/DistrictDataLabs/yellowbrick/issues).
3. **Fork** le dépôt dans votre propre compte GitHub.
4. Créez une **Demande d'extraction** en premier lieu pour [nous contacter](https://github.com/DistrictDataLabs/yellowbrick/pulls) sur votre tâche.
5. **Codez** la fonction, rédigez la documentation, ajoutez votre contribution.
6. **Examinez** le code avec les principaux contributeurs qui vous guideront vers une soumission de haute qualité.
7. **Fusionnez** votre contribution dans la base de codes Yellowbrick.

Nous pensons que *la contribution est la collaboration* et mettons donc l'accent sur *la communication* tout au long du processus open source. Nous nous appuyons fortement sur les outils de codage social de GitHub pour nous permettre de le faire. Par exemple, nous utilisons la fonctionnalité [jalestone](https://help.github.com/en/articles/about-milestones) de GitHub pour concentrer nos efforts de développement pour chaque semestre Yellowbrick, alors assurez-vous de consulter les problèmes associés à notre [jalon actuel](https://github.com/districtdatalabs/yellowbrick/milestones) !

Une fois que vous avez une bonne idée de la façon dont vous allez mettre en œuvre la nouvelle fonctionnalité (ou corriger le bogue !), vous pouvez contacter les responsables de la maintenance en créant une [demande d'extraction](https://github.com/DistrictDataLabs/yellowbrick/pulls). Veuillez noter que si nous pensons que votre solution n'a pas été sérieusement pensée, ou si la RP n'est pas alignée avec nos objectifs [étape actuelle](https://github.com/districtdatalabs/yellowbrick/milestones), nous pouvons vous demander de fermer la RP afin que nous puissions examiner en priorité les demandes de fonctionnalités les plus critiques et les corrections de bogues.

Idéalement, toute demande d'extraction devrait pouvoir être résolue dans les 6 semaines suivant son ouverture. Cette chronologie aide à garder notre file d'attente de demandes d'extraction petite et permet à Yellowbrick de maintenir un calendrier de sortie robuste pour offrir à nos utilisateurs la meilleure expérience possible. Cependant, le plus important est de maintenir le dialogue ! Et si vous n'êtes pas sûr de pouvoir terminer votre idée dans les 6 semaines, vous devriez quand même ouvrir une RP et nous serons heureux de vous aider à la délimiter au besoin.

Si nous avons des commentaires ou des questions lorsque nous évaluons votre demande d'extraction et que nous ne recevons aucune réponse, nous fermerons également la communication après cette période. Sachez que cela ne signifie pas que nous n'accordons pas de valeur à votre contribution, mais simplement que les choses se gâtent. Si, à l'avenir, vous souhaitez le récupérer, n'hésitez pas à répondre à nos commentaires d'origine et à référencer le RP d'origine dans une nouvelle demande d'extraction.

### Création du référentiel

La première étape consiste à créer le référentiel dans votre propre compte. Cela créera une copie de la base de code que vous pourrez modifier et écrire. Pour ce faire, cliquez sur le bouton **« fork »** dans le coin supérieur droit de la page Yellowbrick GitHub.

Une fois forké, suivez les étapes suivantes pour mettre en place votre environnement de développement sur votre ordinateur :

1. Clonez le dépôt.

    Après avoir cliqué sur le bouton fork, vous devriez être redirigé vers la page GitHub du dépôt de votre compte utilisateur. Vous pouvez alors cloner une copie du code sur votre machine locale.

    ```
    $ git clone https://github.com/[VOTRE NOM D'UTILISATEUR]/yellowbrick
    $ cd yellowbrick
    ```

    En option, vous pouvez également [ajouter le remote amont] (https://help.github.com/articles/configuring-a-remote-for-a-fork/) pour vous synchroniser avec les changements effectués par d'autres contributeurs :

    ```
    $ git remote add upstream https://github.com/DistrictDataLabs/yellowbrick
    ```

    Voir "Conventions de branchement" ci-dessous pour plus d'informations à ce sujet.

2. Créez un environnement virtuel.

    Les développeurs Yellowbrick utilisent généralement [virtualenv](https://virtualenv.pypa.io/en/stable/) (et [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/), [pyenv](https://github.com/pyenv/pyenv-virtualenv) ou [conda envs](https://conda.io/docs/using/envs.html) afin de gérer leur version de Python et leurs dépendances. En utilisant l'outil d'environnement virtuel de votre choix, créez-en un pour Yellowbrick. Voici comment faire avec virtualenv :

    ```
    $ virtualenv venv
    ```

3. Installer les dépendances.

    Les dépendances de Yellowbrick sont dans le document `requirements.txt` à la racine du dépôt. Ouvrez ce fichier et décommentez les dépendances qui ne sont destinées qu'au développement. Ensuite, installez les dépendances avec `pip` :

    ```
    $ pip install -r requirements.txt
    ```

    Notez qu'il peut y avoir d'autres dépendances requises pour le développement et les tests, vous pouvez simplement les installer avec `pip`. Par exemple, pour installer
    les dépendances supplémentaires pour construire la documentation ou pour exécuter la suite de
    test, utilisez les fichiers `requirements.txt` dans ces répertoires :

    ```
    $ pip install -r tests/requirements.txt
    $ pip install -r docs/requirements.txt
    ```

4) (Optionnel) Mettre en place des hooks de pré-commission.

    Lorsque vous ouvrez un PR dans le dépôt Yellowbrick, une série de vérifications sera exécutée sur votre contribution, certaines d'entre elles étant des vérifications de formatage de votre code. Ces vérifications peuvent indiquer des changements qui doivent être faits avant que votre contribution ne soit examinée. Vous pouvez mettre en place des hooks de pré-commission pour lancer ces vérifications localement lors de l'exécution de `git commit` afin de s'assurer que votre contribution passera les vérifications de formatage et de linting. Pour mettre cela en place, vous devrez décommenter la ligne pre-commit dans `requirements.txt` et ensuite lancer les commandes suivantes :

    ```
    $ pip install -r requirements.txt
    $ pre-commit install
    ```

    La prochaine fois que vous lancerez `git commit` dans le dépôt Yellowbrick, les vérifications seront automatiquement exécutées.

5. Passer à la branche develop.

    Le dépôt Yellowbrick a une branche `develop` qui est la branche de travail principale pour les contributions. C'est probablement déjà la branche sur laquelle vous êtes, mais vous pouvez vous en assurer et y basculer comme suit : :

    ```
    $ git fetch
    $ git checkout develop
    ```

A ce stade, vous êtes prêt à commencer à écrire du code !

### Conventions de branchement

Le dépôt Yellowbrick est configuré selon un cycle typique de production/diffusion/développement tel que décrit dans "[A Successful Git Branching Model](http://nvie.com/posts/a-successful-git-branching-model/)." La branche de travail principale est la branche `develop`. C'est la branche sur laquelle et à partir de laquelle vous travaillez, puisqu'elle contient tout le code le plus récent. La branche `master` contient la dernière version stable, _qui est poussée sur PyPI_. Personne d'autre que les mainteneurs ne poussera sur la branche master.

**NOTE:** Toutes les demandes de téléchargement doivent être faites dans la branche `yellowbrick/develop` de votre dépôt forké.

Vous devriez travailler directement dans votre fork et créer une pull request depuis la branche develop de votre fork vers la nôtre. Nous recommandons également la mise en place d'un remote `upstream` afin que vous puissiez facilement récupérer les derniers changements de développement depuis le dépôt principal de Yellowbrick (voir [configurer un remote pour un fork](https://help.github.com/articles/configuring-a-remote-for-a-fork/)). Vous pouvez le faire comme suit :

```
$ git remote add upstream https://github.com/DistrictDataLabs/yellowbrick.git
$ git remote -v
origin https://github.com/YOUR_USERNAME/YOUR_FORK.git (fetch)
origine https://github.com/YOUR_USERNAME/YOUR_FORK.git (push)
upstream https://github.com/DistrictDataLabs/yellowbrick.git (fetch)
upstream https://github.com/DistrictDataLabs/yellowbrick.git (push)
```

Lorsque vous êtes prêt, demandez une revue de code pour votre pull request. Ensuite, une fois la revue et l'approbation obtenues, vous pouvez fusionner votre fork dans notre branche principale. Assurez-vous d'utiliser l'option "Squash and Merge" afin de créer un historique Git compréhensible.

**NOTE aux mainteneurs** : Lorsque vous fusionnez une pull request, utilisez l'option "squash and merge" et assurez-vous d'éditer à la fois le sujet et le corps du message de commit afin que nous sachions ce qui s'est passé dans la PR lorsque nous compilons le changelog. Je recommande la lecture de [Chris Beams' _How to Write a Git Commit Message_] (https://chris.beams.io/posts/git-commit/) pour que nous soyons tous sur la même longueur d'onde !

Les contributeurs principaux et ceux qui prévoient de contribuer à plusieurs PRs peuvent envisager d'utiliser des branches de fonctionnalités pour réduire le nombre de fusions (et de conflits de fusions). Créez une branche de fonctionnalité comme suit :

```
$ git checkout -b feature-myfeature develop
$ git push --set-upstream origin feature-myfeature
```

Une fois que vous avez fini de travailler (et que tout est testé), vous pouvez soumettre un PR depuis votre branche de fonctionnalité. Synchronisez avec `upstream` une fois que la PR a été fusionnée et supprimez la branche de fonctionnalité :

```
$ git checkout develop
$ git pull upstream develop
$ git push origin develop
$ git branch -d feature-myfeature
$ git push origin --delete feature-myfeature
```

Retournez sur Github et consultez un autre problème !

## Développer des visualiseurs

Dans cette section, nous allons discuter des bases du développement de visualiseurs. Il s'agit bien sûr d'un vaste sujet, mais nous espérons que ces conseils et astuces simples vous aideront à y voir plus clair.

Il est nécessaire de bien comprendre scikit-learn et Matplotlib. Étant donné que notre API est destinée à s'intégrer à scikit-learn, un bon début consiste à examiner ["APIs of scikit-learn objects"] (http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects) et ["rolling your own estimator"] (http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator). En ce qui concerne matplotlib, consultez [le tutoriel Matplotlib de Nicolas P. Rougier] (https://www.labri.fr/perso/nrougier/teaching/matplotlib/).

### API de visualisation

Il existe deux types de visualiseurs :

- **Les Visualiseurs de caractéristiques** sont des visualisations de données à haute dimension qui sont essentiellement des transformateurs.
- Les **visualiseurs de score** intègrent un régresseur, un classificateur ou un clusterer scikit-learn et visualisent le comportement ou les performances du modèle sur des données de test.

Ces deux types de visualisateurs de base correspondent bien aux deux objets estimateurs de base de scikit-learn :

- Les **transformateurs** prennent des données en entrée et renvoient un nouvel ensemble de données.
- Les **modèles** sont ajustés aux données d'apprentissage et peuvent faire des prédictions.

L'API de scikit-learn est orientée objet, et les estimateurs sont initialisés avec des paramètres en instanciant leur classe. Les hyperparamètres peuvent également être définis en utilisant la méthode `set_attrs()` et récupérés avec la méthode correspondante `get_attrs()`. Tous les estimateurs scikit-learn ont une méthode `fit(X, y=None)` qui accepte un tableau de données à deux dimensions, `X`, et optionnellement un vecteur `y` de valeurs cibles. La méthode `fit()` entraîne l'estimateur, le rendant prêt à transformer les données ou à faire des prédictions. Les transformateurs ont une méthode associée `transform(X)` qui retourne un nouvel ensemble de données, `Xprime` et les modèles ont une méthode `predict(X)` qui retourne un vecteur de prédictions, `yhat`. Les modèles peuvent aussi avoir une méthode `score(X, y)` qui évalue la performance du modèle.

Les visualiseurs interagissent avec les objets scikit-learn en les croisant avec les méthodes définies ci-dessus. Plus précisément, les visualiseurs effectuent des actions liées à `fit()`, `transform()`, `predict()`, et `score()` puis appellent une méthode `draw()` qui initialise la figure sous-jacente associée au visualiseur. L'utilisateur appelle la méthode `show()` du visualiseur, qui à son tour appelle une méthode `finalize()` sur le visualiseur pour dessiner les légendes, les titres, etc. et ensuite `show()` rend la figure. L'API du visualiseur est donc :

- `draw()` : ajouter des éléments visuels à l'objet axes sous-jacent
- `finalize()` : prépare la figure pour le rendu, en ajoutant les touches finales telles que les légendes, les titres, les étiquettes des axes, etc.
- `show()` : rend la figure pour l'utilisateur.

Pour créer un visualiseur, il faut définir une classe qui étend `Visualizer` ou l'une de ses sous-classes, puis implémenter plusieurs des méthodes décrites ci-dessus. L'implémentation la plus simple est la suivante: :

``python
import matplotlib.pyplot as plot

from yellowbrick.base import Visualizer

class MyVisualizer(Visualizer) :

    def __init__(self, ax=None, **kwargs) :
        super(MyVisualizer, self).__init__(ax, **kwargs)

    def fit(self, X, y=None) :
        super(MyVisualizer, self).fit(X, y)
        self.draw(X)
        return self

    def draw(self, X) :
        self.ax.plot(X)
        return self.ax

    def finalize(self) :
        self.set_title("Mon Visualiseur")
```

Ce visualiseur simple dessine simplement un graphique linéaire pour un jeu de données d'entrée X, se croisant avec l'API scikit-learn au niveau de la méthode `fit()`. Un utilisateur utiliserait ce visualiseur dans le style typique: :

``python
visualizer = MyVisualizer()
visualizer.fit(X)
visualizer.show()
```

Les visualisateurs de score fonctionnent sur le même principe mais acceptent un argument supplémentaire, le `modèle`. Les visualiseurs de score enveloppent le modèle (qui peut être instancié ou non) et transmettent tous les attributs et méthodes au modèle sous-jacent, en dessinant si nécessaire.

### Test

Le package de test reflète le package `yellowbrick` dans sa structure et contient également plusieurs méthodes d'aide et des fonctionnalités de base. Pour ajouter un test à votre visualiseur, trouvez le fichier correspondant pour ajouter le cas de test, ou créez un nouveau fichier de test au même endroit que vous avez ajouté votre code.

Les tests visuels sont notoirement difficiles à créer - comment tester une visualisation ou une figure ? De plus, tester des modèles scikit-learn avec des données réelles peut consommer beaucoup de mémoire. Par conséquent, le premier test que vous devriez créer est simplement de tester votre visualiseur de bout en bout et de s'assurer qu'aucune exception ne se produit. Pour vous aider, nous avons une aide, `VisualTestCase`. Créez votre test unitaire comme suit: :

``python
import pytest

from yellowbrick.datasets import load_occupancy

from tests.base import VisualTestCase

class MyVisualizerTests(VisualTestCase) :

    def test_my_visualizer(self) :
        """
        Test de MyVisualizer sur un jeu de données réel
        """
        # Chargement des données
        X,y = load_occupancy()

        try :
            visualizer = MyVisualizer()
            visualizer.fit(X)
            visualizer.show()
        except Exception as e :
            pytest.fail("mon visualiseur n'a pas fonctionné")
```

La suite de tests complète peut être exécutée comme suit : :

```
$ pytest
```

Vous pouvez également exécuter votre propre fichier de test comme suit: :

```
$ pytest tests/test_votre_visualiseur.py
```

Le Makefile utilise le programme d'exécution et la suite de tests pytest ainsi que la bibliothèque de couverture, donc assurez-vous que ces dépendances sont installées !

**Note** : Les développeurs avancés peuvent utiliser nos _tests de comparaison d'images_ pour affirmer qu'une image générée correspond à une image de référence. Pour en savoir plus, consultez notre [documentation sur les tests] (https://www.scikit-yb.org/en/latest/contributing/developing_visualizers.html#image-comparison-tests).

### Documentation

La documentation initiale de votre visualiseur sera une docstring bien structurée. Yellowbrick utilise Sphinx pour construire la documentation, donc les docstrings doivent être écrits en reStructuredText dans le format numpydoc (similaire à scikit-learn). L'emplacement principal de votre docstring devrait être juste en dessous de la définition de la classe, voici un exemple: :

``python
class MyVisualizer(Visualizer) :
    """
    Cette section initiale doit décrire le visualiseur et ce dont il s'agit, y compris comment l'utiliser.
    de quoi il s'agit, y compris comment l'utiliser. Prenez autant de paragraphes
    autant de paragraphes que nécessaire pour obtenir autant de détails que possible.

    Dans la section suivante, décrivez les paramètres de __init__.

    Paramètres
    ----------

    model : un régresseur scikit-learn
        Doit être une instance d'un régresseur, et plus particulièrement un régresseur dont le nom se termine par "CV".
        dont le nom se termine par "CV", sinon a lèvera une exception YellowbrickTypeError
        lors de l'instanciation. Pour utiliser des régresseurs non CV, voir :
        ``ManualAlphaSelection``.

    ax : matplotlib Axes, default : None
        Les axes sur lesquels la figure doit être tracée. Si None est fourni, les axes actuels seront utilisés (ou générés si nécessaire).
        seront utilisés (ou générés si nécessaire).

    kwargs : dict
        Arguments de type mot-clé qui sont transmis à la classe de base et qui peuvent influencer la visualisation telle que définie dans d'autres Visualiseurs.
        la visualisation telle qu'elle est définie dans d'autres Visualiseurs.

    Exemples
    --------

    >>> model = MyVisualizer()
    >>> model.fit(X)
    >>> model.show()

    Notes
    -----

    Dans la section des notes, indiquez les éventuels problèmes ou autres informations.
    """
```

C'est un très bon début pour produire un visualiseur de haute qualité, mais à moins qu'il ne fasse partie de la documentation sur notre site web, il ne sera pas visible. Pour plus de détails sur l'inclusion de la documentation dans le répertoire `docs`, voir la section [Contributing Documentation](https://www.scikit-yb.org/en/latest/contributing/index.html) dans le grand guide de contribution.