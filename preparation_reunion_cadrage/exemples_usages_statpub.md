# Les travaux publiés mobilisant des méthodes ensemblistes


_Deroyon, T. (2022). Enquête santé européenne (EHIS) 2019 : Calcul des pondérations. Note méthodologique. Direction de la recherche, des études, de l'évaluation et des statistiques (DREES)_

[Enquête santé européenne (EHIS) 2019 : Calcul des pondérations](https://drees.solidarites-sante.gouv.fr/sites/default/files/2022-11/Note%201%20EHIS%202019%20Calcul%20des%20pond%C3%A9rations%20-%20Thomas%20Deroyon%20%28DREES%29.pdf)

Ce document détaille le processus de (re)pondération de l'Enquête Santé Européenne (EHIS) 2019 en France métropolitaine. 
L'objectif est de produire un échantillon représentatif de la population, malgré les biais introduits par la non-réponse. 
Plusieurs approches sont comparées afin de construire les groupes de réponses homogènes (GRH): des modèles logit régularisés, des arbres de régression CART et des forêts aléatoires.
La méthode des "groupes de réponse homogène (GRH)" vise à regrouper les individus avec des probabilités de réponse similaires afin de leur attribuer un facteur de correction des poids de sondage similaire. 
L'évaluation de ces modèles se fait sur un échantillon test, en utilisant comme métriques l'erreur quadratique moyenne (EQM), l'erreur absolue moyenne (EAM), la sensibilité, la précision et la F-statistique. 
Le choix du meilleur modèle pour chaque étape se base sur les performances globales sur l'échantillon test.




# Les travaux et processus actuels qui pourraient mobiliser des méthodes ensemblistes

## La repondérations d'enquêtes

### Enquêtes auprès des entreprises:


_Le redressement de la non-réponse totale dans l'Enquete Sectorielle Annuelle_

L'Enquête sectorielle annuelle (ESA) a pour objectif d'identifier les diverses activités exercées par les entreprises en détaillant leur chiffre d'affaires par branches d'activité. 
Cette ventilation sectorielle permet notamment de déterminer l'activité principale exercée (APE) de chaque entreprise.

La correction de la non-réponse totale dans les enquêtes sectorielles annuelles (ESA) repose sur la méthode des groupes de réponse homogène (GRH). 
Les variables les plus corrélées au fait de répondre sont sélectionnées à l’aide d’un modèle logistique. Ces variables sont ensuite hiérarchisées selon leur importance, mesurée par la variation du critère d’information d’Akaike.
Les GRH sont formés en appliquant une méthode par croisements. Cela consiste à regrouper les entreprises selon les modalités des variables auxiliaires corrélées au comportement de réponse, en s’assurant qu’à l’intérieur de chaque groupe, il n’y ait plus de corrélation détectable entre ces variables et la probabilité de réponse. 


### Enquêtes auprès des ménages:


_Le redressement de la non-réponse totale dans l'Enquête Histoire de Vie et Patrimoine_

[Enquête  Histoire de Vie et Patrimoine: Calcul des pondérations](https://www.insee.fr/fr/statistiques/fichier/1381170/f1503.pdf)

L'objectif est de corriger les biais de représentativité liés à la non-réponse totale de certaines personnes enquêtées. 
Cette correction s’appuie sur une méthode basée sur les scores de réponse.
La probabilité qu’un ménage réponde à l’enquête est estimée à l’aide d’un modèle logit. 
Les ménages sont ensuite regroupés en ensembles de réponses homogènes (GRH), caractérisés par des probabilités de réponse sont similaires. 
Pour chaque groupe, on calcule le taux de réponse observé.
Les poids des répondants sont ajustés en les multipliant par l’inverse de ce taux de réponse.















