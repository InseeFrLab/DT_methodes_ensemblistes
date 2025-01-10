---
title: "Introduction aux méthodes ensemblistes"
subtitle: "Plan détaillé"
preview:
  port: 4200
  browser: true
format:
  ctk-article-typst:
    include-in-header: customisation_plan.typ
    keep-typ: true
author:
  - name: Mélina Hillion
    affiliations:
      - name: Insee
        department: Unité SSP-Lab
    email: melina.hillion@insee.fr
  - name: Olivier Meslin
    affiliations:
      - name: Insee
        department: Unité SSP-Lab
    email: olivier.meslin@insee.fr
fig-cap-location: top
date: today
abstract: |
  A compléter
keywords:
  - machine learning
  - méthodes ensemblistes
  - formation
# thanks: "Nous remercions Daffy Duck et Mickey Mouse pour leur contribution."
papersize: a4
margins: 1.25in
mainfont: New Computer Modern
codefont: New Computer Modern Math
title-page: false
toc: false
toc-depth: 3
toc-title: "Sommaire"
blind: false
linkcolor: "#000000"
bibliography: /references.bib
bibliographystyle: ieee
functions:
  - "titled-raw-block"
  - "text"
---

Cette introduction aux méthodes ensemblistes s'adresse aux agents de la statistique publique. Elle prendra deux formes:

- Un document méthodologique comprenant entre 80 et 100 pages, figures et annexes comprises;
- Un site internet reprenant le même contenu.


# Introduction
Cette section comporterait environ 1 page.

#  Aperçu des méthodes ensemblistes

Introduction intuitive aux méthodes ensemblistes, accessible sans prérequis. Cette section comporterait une dizaine de pages, figures comprises.

__Angles éditoriaux__:

- Les méthodes ensemblistes sont des approches de _machine learning_ très performantes sur données tabulaires et relativement simples à comprendre et prendre en main.
- Elles sont particulièrement bien adaptées à de nombreux cas d'usage de la statistique publique car elles sont conçues pour s'appliquer à des données tabulaires.
- Les forêts aléatoires sont d'un usage plus simple que le _gradient boosting_.

##  Que sont les méthodes ensemblistes ?

Définition des méthodes ensemblistes, leur principe de combinaison de modèles simples ("apprenants faibles") pour créer un modèle plus performant ("apprenant fort"). Focus sur les méthodes basées sur les arbres de décision.

##  Pourquoi utiliser des méthodes ensemblistes ?

Avantages des méthodes ensemblistes par rapport aux méthodes économétriques traditionnelles (régression linéaire et logistique) : puissance prédictive supérieure, moins de préparation des données, robustesse aux valeurs extrêmes et à l'hétéroscédasticité. 

Limites : interprétabilité moindre et complexité accrue de l'entraînement et de l'optimisation des hyperparamètres. Comparaison avec le deep learning.

##  Comment fonctionnent les méthodes ensemblistes ?

Présentation du modèle de base (arbre CART) et des deux grandes familles de méthodes ensemblistes : _bagging_ (dont forêts aléatoires) et _gradient boosting_.

###  Le modèle de base : l'arbre de classification et de régression (CART)

Explication intuitive du fonctionnement d'un arbre CART : partitionnement de l'espace des variables explicatives, règles de décision, prédictions. 

Avantages et limites des arbres CART (faible pouvoir prédictif et instabilité).

####  Qu'est-ce qu'un arbre CART ?

Définition formelle d'un arbre CART. Caractéristiques essentielles: partitionnement de l'espace des variables explicatives en régions homogènes, règles de décision, prédiction pour chaque région.

####  Avantages et limites des arbres CART

Points forts: simplicité, interprétabilité. 

Points Faibles: faible pouvoir prédictif, instabilité, variance élevée.

###  Le _bagging_ (Bootstrap Aggregating) et les forêts aléatoires

Introduction au _bagging_ : principe de l'échantillonnage bootstrap, entraînement parallèle, agrégation des prédictions. 
Avantages du _bagging_ : meilleure capacité prédictive et stabilité. 
Limites : corrélation entre les arbres.

####  Le _bagging_

Détail du processus de _bagging_ : création de sous-échantillons aléatoires avec remise, entraînement parallèle des modèles de base, agrégation des prédictions (moyenne ou vote majoritaire). 

llustration graphique.

####  Les _random forests_

Amélioration du _bagging_ : ajout de la sélection aléatoire des variables. Réduction de la corrélation entre les arbres, amélioration de l'efficacité de l'agrégation. 

###  Le _gradient boosting_

Présentation du _gradient boosting_ : entraînement séquentiel des arbres, correction des erreurs des arbres précédents. 
Avantages : performances élevées.

Limites : complexité, sensibilité aux hyper-paramètres, risque de surapprentissage. 

Illustration graphique.


#  Présentation formelle des méthodes ensemblistes

Objectif : Présentation formelle des méthodes ensemblistes, pour une compréhension approfondie des algorithmes.
Cette section comporterait environ 25 pages.

__Angles éditoriaux__:

- Le contenu mathématique est limité à ce qui est indispensable pour des praticiens. En revanche, le document contiendra les références aux articles fondateurs.
- L'exposé insiste sur les propriétés des algorithmes qui ont un impact sur l'utilisation des algorithmes (exemple: la performance d'une forêt aléatoire finit par plafonner avec le nombre d'arbres, et c'est commode en pratique).


##  La brique élémentaire : l'arbre de décision

Présentation formelle des arbres de décision : principe du partitionnement pour prédire, défis du partitionnement optimal. Solution apportée par les arbres de décision (simplification du partitionnement et optimisation gloutonne). Terminologie et structure d'un arbre de décision.
Cette section comporterait environ 5 pages.

###  Le principe fondamental : partitionner pour prédire

Explication du principe de base des arbres de décision : partitionner l'espace des caractéristiques en zones plus homogènes.

###  Les défis du partitionnement optimal

Difficultés liées à la recherche du partitionnement optimal.

###  Les solutions apportées par les arbres de décision

Approche de simplification du partitionnement de l'espace et optimisation gloutonne.

###  Terminologie et structure d'un arbre de décision

Définition des termes clé: Noeud racine, noeuds internes, branches, feuilles.

###  L'algorithme CART : un partitionnement binaire récursif

Détail de l'algorithme CART : fonction d'impureté (indice de Gini, entropie, somme des erreurs quadratiques), choix de la meilleure division binaire, critère d'arrêt, élagage, prédiction.

###  Avantages et limites des arbres de décision

Avantages: interprétabilité, simplicité, flexibilité, gestion des interactions. 
Limites : surapprentissage, optimatimisation locale, instabilité.

##  Le _bagging_

Présentation détaillée du _bagging_ : échantillonnage bootstrap, entraînement parallèle, agrégation des prédictions. Explication de son fonctionnement pour la régression (réduction de la variance) et la classification (classificateur presque optimal). Limites du _bagging_ en pratique.
Cette section comporterait environ 5 pages.

###  Principe du _bagging_

Détail des étapes du _bagging_.

###  Pourquoi le _bagging_ fonctionne

Explication du fonctionnement du _bagging_ avec la décomposition biais-variance et l'inégalité de @breiman1996bagging.

###  L'échantillonnage par bootstrap peut détériorer les performances théoriques du modèle agrégé

Présentation des limites de l'échantillonnage bootstrap : taille effective réduite, dépendance entre échantillons, couverture incomplète.


##  La forêt aléatoire

Présentation détaillée des forêts aléatoires : principe, construction (échantillonnage bootstrap, sélection aléatoire de variables, agrégation des prédictions). Explication de leur performance (réduction de la variance, convergence, facteurs influençant l'erreur de généralisation). Hyperparamètres clés, erreur OOB.

Cette section comporterait environ 5 pages.

###  Principe de la forêt aléatoire

Détail des éléments constitutifs d'une forêt aléatoire (arbres CART, bootstrapping, sélection aléatoire de variables, agrégation).

###  Comment construit-on une forêt aléatoire ?

Description des étapes de la construction d'une forêt aléatoire.

###  Pourquoi les forêts aléatoires sont-elles performantes ?

Explication des mécanismes de réduction de variance et des limites théoriques au surapprentissage.

###  Les hyperparamètres clés des forêts aléatoires

Détail des principaux hyperparamètres, leur rôle et leur influence sur les performances. Tableau récapitulatif.

###  Evaluation des performances par l'erreur Out-of-Bag (OOB)

Détail de la procédure d'estimation OOB, ses avantages et ses limites.

###  Interprétation et importance des variables

Discussion des méthodes d'importance des variables (MDI, MDA, valeurs de Shapley, CIF, Sobol-MDA), leurs biais et leurs avantages.


##  Le _gradient boosting_

Introduction au boosting, ses principes fondamentaux (approximation d'une fonction inconnue, somme pondérée de modèles simples, modélisation additive par étapes). Présentation des premières approches (AdaBoost, Gradient Boosting Machine). Détail de la mécanique du _gradient boosting_ (fonction objectif, gradient, hessienne, construction de l'arbre, calculs des poids optimaux). Moyens de lutter contre le surapprentissage.
Cette section comporterait environ 10 pages.

###  Introduction

Présentation générale du boosting.

###  Les premières approches du _boosting_

Présentation d'AdaBoost et de la Gradient Boosting Machine.

###  La mécanique du _gradient boosting_

Détail de la mécanique du _gradient boosting_. Fonction objectif, calcul des poids optimaux, construction de l'arbre, choix des splits. Cette partie reprendrait les équations de @chen2016xgboost.

###  Les hyperparamètres
Présentation rapide des hyperparamètres.

###  Les fonctions de perte
Présentation des fonctions de perte usuelles.

###  Les moyens de lutter contre l'overfitting

Techniques pour éviter le surapprentissage dans les modèles de _boosting_ (_shrinkage_, _subsampling_, régularisation).



#  Mise en pratique

Objectif : Guide pratique pour l'utilisation des forêts aléatoires et du _gradient boosting_.
Cette section comporterait environ 15 pages.

__Angle éditorial__: cette partie  vise à aider les praticiens à s'approprier rapidement les méthodes. Elle est donc prescriptive:

- Elle recommande certaines implémentations plutôt que d'autres;
- Elle propose une procédure d'entraînement pas à pas pour la forêt aléatoire et le _gradient boosting_, qui est facile à suivre mais pas nécessairement optimale.

##  Préparation des données

Recommandations pour la préparation des données : gestion des valeurs manquantes, encodage des variables catégorielles, utilisation de pipelines 
`scikit-learn`. Division des données en échantillons _train_ et _test_.

##  Evaluation des performances du modèle et optimisation des hyperparamètres

Méthodes d'évaluation : validation croisée, estimation OOB. Stratégies d'optimisation des hyperparamètres : recherche exhaustive (grid search, random search), optimisation basée sur modèle séquentiel (SMBO).

##  Guide d'entraînement des forêts aléatoires

Recommandations pratiques pour l'entraînement des forêts aléatoires: implémentations à privilégier, hyperparamètres clés, procédures d'entraînement.
Cette section comporterait environ 5 pages.

###  Quelles implémentations utiliser ?

Recommandations sur les implémentations (ranger en R, scikit-learn en Python).

###  Les hyperparamètres clés des forêts aléatoires

Présentation détaillée des hyperparamètres: effet sur les performances, et valeurs par défaut dans les implémentations usuelles.

###  Comment entraîner une forêt aléatoire ?

Procédure d'entraînement étape par étape.

Recommandations pour l'optimisation des hyper-paramètres: approche exploratoire (grid search et random search) et approches avancées (SMBO).

###  Mesurer l'importance des variables

Recommandations sur les méthodes d'importance des variables. Privilégier celles tenant compte des biais potentiels (algorithme CIF, Sobol-MDA et SHAFF).

##  Guide d'entraînement des algorithmes de _gradient boosting_

Recommandations pratiques pour l'entraînement des algorithmes de _gradient boosting_ : implémentations à privilégier, hyperparamètres clés, procédures d'entraînement.
Cette section comporterait environ 5 pages.

#  Cas d’usage des méthodes ensemblistes dans la statistique publique

Objectif: présenter des travaux du SSP qui mobilisent, ou qui pourraient mobiliser, des méthodes ensemblistes.

__Angle éditorial__: cette partie  vise à montrer la diversité des usages possibles des méthodes ensemblistes.

Résumés de ces travaux et liens vers les publications de référence.

Cette section comporterait entre 3 et 5 pages.

##  Les travaux déjà publiés mobilisant des méthodes ensemblistes

##  Les travaux et processus actuels qui pourraient mobiliser des méthodes ensemblistes