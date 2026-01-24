# 4. Scoping, Data préparation et Annotations

ENFIN !!! Après cette loooooooooooongue introduction et loooooooooooongue présentation des concepts de base du 
développement logiciel, nous pouvons commencer à parler MLOps !

Pour rappel, voici une vision simplifiée des différentes étapes à observer :

![MLOps_Timeline.png](00_materials/MLOps_Timeline.png)

## Présentation du projet

Nous allons créer un Webservice de classification permettant de déterminer si un passager du Titanic a survécu.
Cette classification doit se faire en temps réel ! Le client n'aime pas les traitements batch qu'il juge d'un autre temps.
Et puis, si l'on est capable de traiter un flux de passagers en temps réel, nous serons parfaitement capables de le faire
en batch ;-)

## Présentation du dataset

Pour entrainer notre modèle, nous utiliserons des données tabulaires (csv). Elles comportent des informations sur les passagers
(titre, âge, sexe, classe, tarif payé, nombre de frères et sœurs à bord, nombre de parents à bord, port d'embarquement ...).

Exceptionnellement, le dataset est fourni et est stocké dans le repository git de notre projet. 
Ce n'est pas une bonne pratique, mais pour simplifier la prise en main de ce cours, nous avons fait ce choix.

Il serait préférable d'utiliser une solution de stockage et de versioning de la donnée brute [dédiée](#versioning-de-la-donnée-et-stockage). 

Analysez avec le professeur ce dataset.

## Préparation de la donnée

La donnée brute à notre disposition est déjà propre et prête à l'emploi. Nous ferons les étapes de splitting et analyse
des données directement dans notre pipeline de training. Nous ferons simple pour ce projet.

## Petites précisions sur l'annotation des données non structurées

Quand nous farons du Machine Learning supervisé, nous avons besoin d'annotations. Ces annotations sont la "source de vérité" que
notre modèle va apprendre à reproduire. Il est donc primordial que ces annotations soient de qualité.

Dans le cadre de données non structurées (images, vidéos, sons, textes), l'annotation peut s'avérer être une tâche longue et fastidieuse.
Mais importante. Nous avons déjà vu que la qualité de la donnée était primordiale. La qualité des annotations l'est tout autant.

Ecotag est une solution d'annotation sécurisée et open source, proposée initialement par Axa France. 
Nous allons l'utiliser ici pour
classifier nos images et en extraire les annotations. C'est une application Web dont voici quelques clichés que nous allons
commenter : 

Page d'authentification :

![ecotag_secure.png](00_materials/04_scoping_data_prep_label/ecotag_secure.png)

Menu principal : 

![ecotag_menu.png](00_materials/04_scoping_data_prep_label/ecotag_menu.png)

Gestion des équipes d'annotation (gestion des droits d'accès aux projets et datasets) : 

![ecotag_equipe.png](00_materials/04_scoping_data_prep_label/ecotag_equipe.png)

Gestion des datasets :

![ecotag_datasest.png](00_materials/04_scoping_data_prep_label/ecotag_datasest.png)

Gestion des projets :

![ecotag_project.png](00_materials/04_scoping_data_prep_label/ecotag_project.png)

Vous trouverez une demo en ligne d'ecotag [ici](https://axaguildev-ecotag.azurewebsites.net)

L'installation de la solution dans notre environnement
codespace va s'avérer malheureusement assez technique. Simplifier cette mise en place serait une belle évolution à 
apporter à l'outil. N'hésitez pas à vous lancer si le cœur vous en dit :)

Vous trouverez le code source d'ecotag [ici](https://github.com/AxaFrance/ecotag)

## Annotations des images (démonstration, à ne pas faire en pratique)

Voici une petite démonstration d'annotation d'un dataset d'images ! Pour ce faire, rendez-vous sur la page d'accueil de votre ecotag et
allez dans le menu Projets. Vous devriez voir un projet cats_dogs_other dont le statut est En cours.

![projets_en_cours.png](00_materials/04_scoping_data_prep_label/projets_en_cours.png)

Comme vous pouvez le voir, vous avez le pourcentage de complétion de votre projet qui est indiqué. Il est ici à 0%.
Il s'agit d'un projet de type classification d'image. Faisons un détour par les menus Datasets et Equipes. Dataset d'abord :

![dataset_locked.png](00_materials/04_scoping_data_prep_label/dataset_locked.png)

Nous constatons ici que le Dataset est Vérouillé. Cela veut dire que nous ne pouvons plus le modifier. C'est normal,
étant donné qu'un projet d'annotation a été créé, nous partons du principe que ce Dataset est en cours d'annotation.
Il ne doit donc pas changer ... Enfin, parlons de l'équipe : 

![equipes_created.png](00_materials/04_scoping_data_prep_label/equipes_created.png)

Cette équipe n'a qu'un seul utilisateur, vous, Bob ! Vous pouvez ajouter d'autres utilisateurs identifiés dans l'équipe.
Seuls les membres de cette équipe peuvent annoter dans notre projet.

Revenons donc sur notre projet en cliquant sur la loupe se trouvant dans la colonne actions : 

![projets_en_cours.png](00_materials/04_scoping_data_prep_label/projets_en_cours.png)

Nous arrivons sur cette page : 

![begin_label.png](00_materials/04_scoping_data_prep_label/begin_label.png)

Nous retrouvons ici toutes les informations utiles à notre projet, dont un récapitulatif de nos labels (cat, dog, other),
les emails des annotateurs, l'avancement ect... Pour commencer à annoter, cliquez sur Commencer à annoter ;)

Les images peuvent paraître grosses. Vous disposez d'outil graphique pour revoir l'aspect de la page, dont la taille de l'image.
N'hésitez pas à l'ajuster à votre goût !

![size.png](00_materials/04_scoping_data_prep_label/size.png)

Pendant votre exercice d'annotation, vous devriez tomber sur quelques surprises ... Parlons-en rapidement !

![surprise.png](00_materials/04_scoping_data_prep_label/surprise.png)

Une fois l'exercice d'annotation terminé (c'est l'affaire de quelques minutes), vous pouvez extraire vos annotations.
Cette extraction est précieuse et va servir de source de vérité pour notre modèle. Annotez sérieusement ! Voici la page
que vous aurez en fin d'exercice : 

![finished.png](00_materials/04_scoping_data_prep_label/finished.png)

Cliquez sur la flèche en haut à gauche pour revenir à la page du projet, puis cliquez en haut à droite sur Exporter pour
récupérer votre export : 

![export.png](00_materials/04_scoping_data_prep_label/export.png)

## Versioning de la donnée et stockage

### Pourquoi ?

Il est généralement déconseillé de pousser des datasets dans un repository Git pour les raisons suivantes :
- **Taille des fichiers** : Les fichiers de données peuvent être très volumineux, ce qui peut entraîner des problèmes 
de performance lors de la mise à jour du repository. De plus, cela peut augmenter considérablement la taille du 
repository, ce qui peut rendre le clonage et la synchronisation plus lents.
- **Historique des versions** : Les fichiers de données sont souvent mis à jour fréquemment, ce qui peut entraîner 
un historique de versions encombré et difficile à gérer. Cela peut également rendre difficile la comparaison des 
versions précédentes des fichiers de données.
- **Collaboration** : Les fichiers de données sont souvent partagés entre plusieurs personnes, ce qui peut entraîner 
des conflits lors d'un merge des modifications entre branches. De plus, il peut être difficile de gérer les 
autorisations d’accès aux fichiers de données.

Il est donc conseillé de passer par un système de stockage versionné et dédié pour les datasets de type fichier. Le
versioning est important, car votre donnée est vivante. Il faut que vous puissiez facilement retrouver la version
des données brutes avec laquelle vous avez entraîné votre modèle.

### Comment ?

Il existe plusieurs alternatives pour stocker les fichiers de données en dehors d’un repository Git, notamment :
- **Stockage de fichiers** : Les fichiers de données peuvent être stockés sur un système de fichiers partagé ou dans 
un service de stockage de fichiers cloud tel que Dropbox ou Google Drive. Cela permet de partager facilement les 
fichiers de données avec d’autres personnes et de les synchroniser automatiquement.
- **Bases de données** : Les fichiers de données peuvent être stockés dans une base de données, ce qui permet une 
gestion plus efficace des versions et une collaboration plus facile. Les bases de données peuvent également offrir 
des fonctionnalités telles que la recherche et la requête de données.
- **Services de stockage de données** : Il existe des services de stockage de données tels que Amazon S3 et 
Microsoft Azure Blob Storage qui sont conçus pour stocker des fichiers de données volumineux. Ces services offrent 
des fonctionnalités telles que la réplication de données, la sauvegarde et la restauration, ainsi que des options de 
sécurité avancées.

Pour faciliter également la gestion de ces fichiers, vous pouvez utiliser [Data Version Control, ou DVC](https://dvc.org/).

DVC est un outil open source qui permet de gérer et de versionner des fichiers de données. Contrairement à Git, 
DVC est conçu spécifiquement pour les fichiers de données volumineux et peut être utilisé en conjonction 
avec Git pour gérer les versions de code et de données. Ses avantages sont :
- **Taille des fichiers** : DVC utilise des liens symboliques pour stocker les fichiers de données, ce qui permet de 
gérer efficacement les fichiers volumineux sans les stocker directement dans le repository. Cela permet de réduire 
considérablement la taille du repository et d’améliorer les performances.
- **Historique des versions** : DVC stocke les fichiers de données dans un système de fichiers séparé, ce qui permet 
de gérer efficacement l’historique des versions des fichiers de données sans encombrer l’historique de versions de Git.
- **Collaboration** : DVC permet de partager facilement les fichiers de données avec d’autres personnes et de gérer 
les autorisations d’accès aux fichiers de données.

Pour simplifier ce cours, nous n'allons pas utiliser DVC, mais plutôt un service de stockage de données compris dans notre
solution de plateforme ML dédiée : [kto-mlflow](#installation-de-kto-mlflow-et-présentation-de-minio-et-dailyclean). 
Ce ne sera pas le plus idéal et peut-être trop manuel, mais cela fonctionnera pour l'exemple ;-)

### Installation de kto-mlflow et présentation de minio

**kto-mlflow** est une plateforme ML sur le Cloud de dernière génération développée spécifiquement pour ce cours. Nous
reviendrons dessus plus tard dans la partie qui lui est [consacrée](./06_ml_platforms.md).

Pour l'heure, et parce qu'elle contient un service de stockage de fichiers, nous allons installer préalablement 
cette plateforme dans notre Red Hat Developer Sandbox.

En voici la procédure : 
- Connectez-vous à votre compte [Red Hat Sandbox](https://sandbox.redhat.com/)
- Il n'est pas impossible que vous ayez à vous reconnecter

![006.png](img/006.png)
![007.png](img/007.png)

- Dans la section Devspaces, cliquez sur Try it!

![010.png](img/010.png)

- Il est possible que vous soyez amené.e à reconfirmer votre connexion.

![011.png](img/011.png)

- Ouvrer votre workspace déjà existant dans Red Hat DevSpace, pour ce faire, cliquez sur les trois petits points à droite de votre
workspace et cliquez sur Open

![162.png](img/162.png)
![163.png](img/163.png)
![164.png](img/164.png)

Pour installer kto-mlflow, nous allons utiliser l'outil oc (OpenShift CLI). Il est déjà installé et configuré dans votre DevSpace.
Pour simplifier l'installation, nous allons utiliser un script bash qui va faire le travail pour nous.

- Identifiez le script d'installation dans le répertoire scripts de votre projet
- Ouvrez un terminal dans votre DevSpace (Terminal -> New Terminal) si vous n'en avez pas déjà un d'ouvert
- Modifiez les droits de l'ensemble des scripts de ce répertoire pour les rendre exécutables
```bash
chmod -R 777 ./scripts
```

![124.png](img/124.png)

- Lancez le script d'installation de kto-mlflow
```bash
./scripts/install_kto_mlflow.sh
```
- Patientez une dizaine de minutes le temps que tous les pods se déploient

![125.png](img/125.png)

Une fois l'installation terminée, vous devriez voir apparaître de nouveaux projets dans votre OpenShift.

Nous utiliserons minio comme service de stockage de fichier. Soyez un bon citoyen ou une bonne citoyenne du Cloud,
utilisez DailClean (compris dans cette solution), pour éteindre kto-mlflow quand vous ne vous en servez plus. Pas de
panique, votre travail sera sauvegardé malgré l'extinction.

### Procédure de sauvegarde

Pour sauvegarder nos données, nous devons déjà les télécharger depuis votre projet sur notre machine locale.
- Depuis votre Devspace, identifiez le répertoire data de votre projet, un csv s'y trouve
- Faites un clic droit sur le fichier csv et cliquez sur Download

![139.png](img/139.png)

Maintenant, sauvegardez ces fichiers sur le minio de kto-mlflow. Pour cela :
- Retournez sur votre [Red Hat Sandbox](https://sandbox.redhat.com/)
- Dans la section Openshift, cliquez sur Try it

![127.png](img/127.png)

- Vous arrivez sur votre OpenShift, fermez la fenêtre de bienvenue et également l'assistant IA

![128.png](img/128.png)
![129.png](img/129.png)

- Passez la langue de votre OpenShift en anglais (US) pour éviter des problèmes d'affichage, cliquez sur votre profil 
en haut à droite puis sur Préférences utilisateur

![130.png](img/130.png)

- Cliquez sur Langue, décochez l'option Utilisez le paramètre de langue par défaut du navigateur 
et sélectionnez English

![131.png](img/131.png)

- Vérifiez que kto-mlflow est bien déployé, pour cela, dans le menu à gauche, cliquez sur Workloads -> Pods

![132.png](img/132.png)

- Validez que tous les pods de kto-mlflow sont bien en Running. Dans le menu projets, vérifiez qu'il est bien sélectionné
celui portant votre login Red Hat suivi de '-dev' (ex: bob-dev)

![133.png](img/133.png)

- Cliquez sur Networking -> Routes sur le menu à gauche de votre OpenShift
- Dans la liste des routes, identifiez la ligne minio-console et cliquez sur le lien de la colonne Location

![134.png](img/134.png)

- Vous arrivez sur la fenêtre de connection. Saisissez le login `minio` et le mot de passe `minio123`

![minio.png](img/135.png)

- Il est possible qu'une pop-up s'ouvre, cliquez sur Aknoledge

![136.png](img/136.png)

- Vous êtes automatiquement sur le navigateur d'objets
- Dans le menu à gauche, cliquez sur Create Bucket +

![137.png](img/137.png)

- Dans Bucket Name, saisissez le nom : `kto-titanic`. Attention, respectez bien ce nom. Les underscores ne sont pas autorisés !!
- Cliquez sur Create Bucket

![138.png](img/138.png)

- Par défaut, vous êtes dans le bucket que vous venez de créer, si ce n'est pas le cas, cliquez dessus dans la liste
des buckets dans le menu à gauche (il porte le nom kto-titanic)

![140.png](img/140.png)

- Cliquez sur Upload puis Upload File

![141.png](img/141.png)

- Sélectionnez votre fichier csv téléchargé précédemment puis cliquez sur Envoyer

![142.png](img/142.png)

- Confirmez l'envoie dans l'infobulle qui apparaît puis patientez quelques secondes
- Votre dataset est maintenant sauvegardé !!

![143.png](img/143.png)

- **Communiquez par mail une capture d'écran de la présence de votre dataset dans le bucket kto-titanic de votre minio console (évaluation)**

- Gardez le fichier téléchargé. Il est certes interdit de garder de la donnée confidentielle sur votre
machine, mais ici, ce n'est pas le cas. Nous ferons une exception pour ce cours.

Etant donné que l'espace de stockage va disparaître au bout des 30 jours d'activation de la Sandbox, nous allons donc malgré
tout garder une trace de notre dataset. **Ce N'EST PAS une bonne pratique**, mais 
nous devrons faire une exception à cause de ces limitations (que nous ne remettrons pas en cause, elles sont parfaitement 
justifiées).

## Présentation de DailyClean et comment démarrer kto-mlflow

Terminons cette partie en parlant de DailyClean. Notre plateforme kto-mlflow embarque un outil de nettoyage automatique 
des ressources nommé DailyClean. Cet outil permet d'éteindre automatiquement kto-mlflow chaque jour à une heure donnée.
Cela permet de ne pas consommer inutilement des ressources Cloud. 

L'outil permet également de redémarrer facilement kto-mlflow quand on en a besoin. En effet, la Sandbox Red Hat a 
également son système d'extinction automatique au bout de 8 heures d'inactivité. 

Voici comment utiliser DailyClean pour rallumer kto-mlflow quand vous en avez besoin :
- Connectez-vous à votre [Red Hat Sandbox](https://sandbox.redhat.com/)
- Dans la section Openshift, cliquez sur Try it
- Vous arrivez sur votre OpenShift
- Démarrez manuellement DailyClean s'il n'est pas déjà en route, cliquez sur Workloads -> Deployment dans le menu à gauche
- Identifiez dailyclean-api dans la liste des déploiements et cliquez dessus

![144.png](img/144.png)

- Un cercle avec indiqué Scaled to 0 devrait apparaître. Si ce n'est pas le cas, ne faites rien. 
Sinon, cliquez sur l'icône de la flèche vers le haut, à droite de ce cercle et attendez que le scaling soit terminé.

![145.png](img/145.png)
![146.png](img/146.png)

- Dans le menu à gauche, cliquez sur Networking -> Routes
- Dans la liste des routes, identifiez la ligne dailyclean et cliquez sur le lien de la colonne Location

![147.png](img/147.png)

- Vous arrivez sur la page d'accueil de DailyClean
- Constatez que le statut de votre environnement est bien Arrêté (Stopped)

![148.png](img/148.png)

- Cliquez sur On dans le menu Turn Environment, puis cliquez sur Submit pour démarrer kto-mlflow

![149.png](img/149.png)

- Patientez quelques minutes le temps que tous les pods se déploient

![150.png](img/150.png)
![151.png](img/151.png)

Cette partie est terminée ! Bravo !