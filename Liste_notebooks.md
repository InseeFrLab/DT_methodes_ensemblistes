# Liste des _notebooks_ envisagés

## Objectifs généraux des _notebooks_

Les _notebooks_ devront être conçus pour être indépendants, reproductibles et pédagogiques, en offrant une transition progressive des bases aux approches plus avancées.

- **Illustration des méthodes**: Illustrer concrètement les méthodes ensemblistes présentées dans le document méthodologique, en mettant en lumière les étapes clés, de la préparation des données à l'entraînement des modèles, en passant par l'optimisation des performance et l’interprétation des résultats. Ces notebooks mettront en oeuvre la procédure d'entraînement proposée dans le document méthodologique.

- **Reproductibilité et accessibilité**: les _notebooks_ seront déployés sur le SSPCloud et utiliseront des données publiques pour une reproductibilité optimale.

- **Diffusion des bonnes pratiques**: utilisation de _pipelines_ `scikit-learn`, choix de librairies performantes et robustes (`ranger`, `scikit-learn`), gestion rigoureuse de l'environnement, des données.

- **Applications concrètes**: Illustrer l'application des méthodes sur des cas d'usage simples basés sur des données ouvertes.


## Liste des _notebooks_

### _Notebooks_ de base

Deux applications très simples sont envisagées: un problème de **régression** (prédire des prix immobiliers) et un problème de **classification binaire** (prédire le fait d'être diplômé du baccalauréat à partir des données individuelles du recensement de la population).

- Deux _notebooks_ en `R` : 
    - régression et classification par **forêt aléatoire** avec la librairie `ranger`.
- Quatre _notebooks_ en Python : 
    - régression et classification par **forêt aléatoire** avec la librairie `scikit-learn`;
    - régression et classification par **_gradient boosting_** avec la librairie `lightgbm`.

### _Notebooks_ avancés

Selon le temps disponible, des _notebooks_ plus avancés pourront être proposés:

- classification multi-classe (exemple: niveau de diplôme dans le recensement);
- classification déséquilibrée (exemple: être sans aucun diplôme dans le recensement);
- interpolation spatiale (exemple: prédiction des prix immobiliers utilisant les coordonnées géographiques).


