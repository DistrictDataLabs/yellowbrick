# Yellowbrick

[ ![Visualiseurs](https://github.com/DistrictDataLabs/yellowbrick/raw/develop/docs/images/readme/banner.png)](https://www.scikit-yb.org/)

Yellowbrick est une suite d'outils d'analyse visuelle et de diagnostic conçus pour faciliter l'apprentissage automatique avec scikit-learn. La bibliothèque implémente un nouvel objet de l'API de base, le `Visualizer` qui est un estimateur scikit-learn &mdash ; un objet qui apprend à partir des données. Comme les transformateurs ou les modèles, les visualiseurs apprennent à partir des données en créant une représentation visuelle du flux de travail de sélection du modèle.

Les visualisateurs permettent aux utilisateurs de diriger le processus de sélection de modèles, en développant l'intuition autour de l'ingénierie des caractéristiques, de la sélection des algorithmes et du réglage des hyperparamètres. Par exemple, ils peuvent aider à diagnostiquer les problèmes courants liés à la complexité et au biais des modèles, à l'hétéroscédasticité, à l'inadaptation et au surentraînement, ou aux problèmes d'équilibre entre les classes. En appliquant des visualiseurs au flux de travail de sélection des modèles, Yellowbrick vous permet d'orienter les modèles prédictifs vers des résultats plus fructueux, plus rapidement.

La documentation complète est disponible sur [scikit-yb.org](https://scikit-yb.org/) et comprend un [Guide de démarrage rapide](https://www.scikit-yb.org/en/latest/quickstart.html) pour les nouveaux utilisateurs.

## Visualiseurs

Les visualisateurs sont des estimateurs &mdash ; des objets qui apprennent à partir des données &mdash ; dont l'objectif principal est de créer des visualisations qui permettent de comprendre le processus de sélection du modèle. En termes de scikit-learn, ils peuvent être similaires aux transformateurs lors de la visualisation de l'espace de données ou envelopper un estimateur de modèle similaire à la façon dont les méthodes `ModelCV` (par exemple [`RidgeCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html), [`LassoCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)) fonctionnent. L'objectif principal de Yellowbrick est de créer une API sensorielle similaire à scikit-learn. Voici quelques-uns de nos visualiseurs les plus populaires :

### Visualisation de la classification

- Rapport de classification** : un rapport de classification visuel qui affiche la précision, le rappel et les scores F1 par classe d'un modèle sous la forme d'une carte thermique.
- Matrice de confusion** : une carte thermique de la matrice de confusion des paires de classes dans la classification multi-classes.
- Seuil de discrimination** : visualisation de la précision, du rappel, du score F1 et du taux de file d'attente par rapport au seuil de discrimination d'un classificateur binaire.
- Courbe de précision-rappel** : représentation graphique des scores de précision et de rappel pour différents seuils de probabilité.
- ROCAUC** : graphique de la caractéristique de l'opérateur récepteur (ROC) et de l'aire sous la courbe (AUC).

### Visualisation du regroupement

- Cartes de distance inter-clusters** : visualisation de la distance relative et de la taille des clusters.
- Visualisateur de coude** : visualise les grappes selon la fonction de notation spécifiée, en recherchant le "coude" dans la courbe.
- Visualisateur de silhouette** : sélectionne `k` en visualisant les scores du coefficient de silhouette de chaque groupe dans un modèle unique.


### Visualisation des caractéristiques

- Visualisation des manifolds** : visualisation en haute dimension avec apprentissage des manifolds
- Coordonnées parallèles** : visualisation horizontale des instances
- Projection ACP** : projection des instances basée sur les composantes principales
- Visualiseur RadViz** : séparation des instances autour d'un graphe circulaire
- Rank Features** : classement simple ou par paire des caractéristiques pour détecter les relations.

### Visualisation de la sélection de modèles

- Scores de validation croisée** : affichage des scores de validation croisée sous forme de diagramme à barres avec le score moyen représenté par une ligne horizontale.
- Importance des caractéristiques** : classez les caractéristiques en fonction de leur performance dans le modèle.
- Courbe d'apprentissage** : montre si un modèle pourrait bénéficier de plus de données ou de moins de complexité.
- Élimination récursive des caractéristiques** : trouver le meilleur sous-ensemble de caractéristiques en fonction de leur importance.
- Courbe de validation** : ajuster un modèle en fonction d'un seul hyperparamètre.

### Visualisation de la régression

- Sélection de l'alpha** : montrer comment le choix de l'alpha influence la régularisation.
- Distance de Cook** : montre l'influence des instances sur la régression linéaire
- Plots d'erreur de prédiction** : trouver des ruptures de modèle le long du domaine de la cible.
- Graphique des résidus** : montre la différence entre les résidus des données d'entraînement et des données de test.

### Visualisation de la cible

- Référence de regroupement équilibré** : génère un histogramme avec des lignes verticales montrant le point de valeur recommandé pour regrouper les données dans des catégories uniformément réparties.
- Équilibre des classes** : montrez la relation entre le support de chaque classe dans les données d'apprentissage et de test en affichant la fréquence d'apparition de chaque classe sous la forme d'un diagramme à barres la fréquence de représentation des classes dans l'ensemble de données.
- Corrélation des caractéristiques** : visualisez la corrélation entre les variables dépendantes et la cible.

### Visualisation du texte

- Graphique de dispersion** : visualisez la façon dont les termes clés sont dispersés dans un corpus.
- Visualiseur PosTag** : représente le nombre de différentes parties du discours dans un corpus étiqueté.
- Distribution de la fréquence des termes** : visualisez la distribution de la fréquence des termes dans le corpus.
- Visualisation du corpus **t-SNE** : utilise l'intégration stochastique des voisins pour projeter des documents
- Visualisation du corpus UMAP** : rapproche les documents similaires pour découvrir les grappes.

... et plus encore ! Yellowbrick ajoute de nouveaux visualiseurs en permanence, alors n'hésitez pas à consulter notre [galerie d'exemples]https://github.com/DistrictDataLabs/yellowbrick/tree/develop/examples) &mdash ; ou même la branche [develop](https://github.com/districtdatalabs/yellowbrick/tree/develop) &mdash ; et n'hésitez pas à nous faire part de vos idées pour de nouveaux visualiseurs !

## Affiliations
[ ![District Data Labs](https://github.com/DistrictDataLabs/yellowbrick/raw/develop/docs/images/readme/affiliates_ddl.png)](https://www.districtdatalabs.com/) [ ![Projet affilié à NumFOCUS](https://github.com/DistrictDataLabs/yellowbrick/raw/develop/docs/images/readme/affiliates_numfocus.png)](https://numfocus.org/)