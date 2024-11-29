# Questions pour les participants de la réunion de cadrage

## Le document méthodologique

### Démarche


- Que pensez-vous de l'approche à trois niveaux de lecture : aperçu intuitif (Section 1), présentation formalisée (Section 2), guide d'entraînement pratique (Section 3) ? 

- Est-il préférable de séparer complètement la présentation des méthodes (Section 2) de leur guide d'utilisation pratique (Section 3) ? 
Ou une approche plus intégrée serait-elle plus efficace ? Un plan alternatif pourrait être: aperçu intuitif (Section 1), forêts aléatoires (fonctionnement et guide pratique, Section 2), gradient boosting (fonctionnement et guide pratique, Section 3).


### Contenu des parties rédigées
Les sections 3.3 et 4.3 présentant les forêts aléatoires sont déjà en partie rédigées:
- Sont-elles suffisamment claires ? Y a-t-il des points à éclaircir ?  Le niveau de détail est-il approprié ? 
- Le niveau de détail sur l'optimisation des hyperparamètres est-il suffisant ? 

### Présentation des algorithmes

-  Est-il pertinent d'inclure des présentations des algorithmes en pseudocode ? => proposition: ne pas inclure de pseudocode.

### Interprétabilité

- Faut-il dédier une section à part entière à l'interprétabilité des modèles ensemblistes ? Ou les approches d'interprétation doivent-elles plutôt être incluses dans les sections déjà dédiées à chaque algorithme? => proposition: faire une section à part, éventuellement en insistant sur les spécificités liées aux forêts aléatoires ou au _gradient boosting_ si besoin.

- La recherche sur l'interprétabilité des modèles est foisonnante, et il ne nous semble pas qu'aucune méthode ne fait consensus. Quelles approches d'interprétabilité doivent-être présentées?

- Les recommandations sur les méthodes d'interprétation sont-elles adaptées? Comment gérer le fait que certaines sont implémentées en R mais pas en Python ?

### Recommandations

- Les implémentations recommandées (`ranger`, `scikit-learn`, `XGBoost` et `lightgbm`) sont-elles bien choisies? Jusqu'à quel point faut-il justifier ces choix?

- Probst et al 2019 et et le cours de S. Da Veiga recommandent d'utiliser l'erreur _out-of-bag_ (plutôt que la validation croisée) pour optimiser les hyperparamètres des forêts aléatoires, car c'est plus rapide. Toutefois, cette approche ne peut pas être implémentée de façon standard en Python avec `scikit-learn` (qui ne propose que la validation croisée). Il est possible de trouver un moyen de contournement ([ici](https://stackoverflow.com/questions/34624978/is-there-easy-way-to-grid-search-without-cross-validation-in-python) par exemple), mais ce n'est pas standard. Que voulons-nous recommander: OOB ou CV? => proposition: on présente les deux approches, en insistant sur le fait que l'approche OOB est spécifique aux forêts aléatoires.

## Les _notebooks_

- Faut-il proposer des notebooks qui utilisent les _pipelines_ `scikit-learn` afin de sensibiliser/former à leur usage ? Cela ajoute-t-il une couche d'abstraction qui peut nuire à la bonne compréhension des étapes de construction d'un modèle ? => proposition: utiliser les _pipelines_.

- Y a-t-il des cas d'usages majeurs qui ne seraient pas couverts par la liste des notebooks proposés?

- Faut-il couvrir tous les cas d'usage en `R` et en Python?  => proposition: les _notebooks_ `R` ne porteront que sur les forêts aléatoires. Les _notebooks_ de _gradient boosting_ seront uniquement en Python.

