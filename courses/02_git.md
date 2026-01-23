# 2. Git, GitHub, OpenSource et licencing

## Introduction

Cette partie a été développée à partir de :
- L'ouvrage de référence [Pro Git](https://git-scm.com/book/fr/v2)
- [Wikipédia](https://fr.wikipedia.org/wiki/Git)
- S'inspire grandement des supports de formation produits par la Communauté de Pratique Git d'AXA France que je remercie 
et salue chaleureusement

## Git en quelques mots ? 

Git est un logiciel de gestion de versions distribué et décentralisé. C'est un logiciel libre et gratuit, créé en 2005 par 
Linus Torvalds, auteur du noyau Linux. Git est largement utilisé aujourd'hui pour le contrôle de version dans de nombreux projets 
logiciels, y compris des projets commerciaux et en open source.

En plus d'être décentralisé, Git a été conçu avec la performance, la sécurité et la flexibilité à l'esprit. 
Il est utilisé par des dizaines de millions de personnes, sur tous les environnements (Windows, Mac, Linux). Git est 
aussi le système à la base du site web GitHub, que nous allons utiliser dans ce cours.

Il repose également sur un système de branches, pour des développements non linéaires.

## Comment collaborer et partager des fichiers ?

Illustration issue du site : https://tortoisesvn.net/docs/release/TortoiseSVN_fr/tsvn-basics-versioning.html

![ch02dia2.png](./00_materials/02_git/ch02dia2.png)

Un gestionnaire de version permet de répondre à cette problématique en enregistrant les différentes versions d'un ensemble
de fichiers.

## Les différences entre chaque type de gestionnaire

Les illustrations ci-dessous proviennent de l'excellent ouvrage Pro Git:

![gestion_locale.png](./00_materials/02_git/gestion_locale.png)

![gestion_centralisee.png](./00_materials/02_git/gestion_centralisee.png)

![gestion_distribuee.png](./00_materials/02_git/gestion_distribuee.png)

Git est, comme indiqué précédemment, un gestionnaire distribué et décentralisé. Cela veut dire que le dépôt central
n'est pas obligatoire. Il peut en effet être replacé en cas de problème.

## Les avantages du développement distribué

Chaque développeur et développeuses dispose donc de sa propre copie de travail sur sa machine locale et partage l'historique
complet des modifications faites sur le projet avec les autres personnes intervenant sur le projet.

![centralized-vs-distributed.png](./00_materials/02_git/centralized-vs-distributed.png)

Git se présente comme suit : 

![working_directory.png](./00_materials/02_git/working_directory.png)

Rappel : Chaque copie de travail du code est également un dépôt qui peut contenir l'historique complet de tous les
changements.

Travailler avec Git revient à travailler la majeure partie du temps en local. Chaque machine est bien une base de données
indépendante qui est synchronisable avec le dépot de référence, appelé communément le repository, 
partagé avec toutes les autres machines de développement.

Chaque modification se présente sous forme de commit, qui permet de faire avancer l'historique du projet. Chaque commit
est un instantané du projet. Chaque commit dispose d'un identifiant unique sous forme d'une clé de hashage.

![delta.png](./00_materials/02_git/delta.png)
![snapshot.png](./00_materials/02_git/snapshot.png)

## Les branches

Créer une branche signifie diverger de la ligne principale de développement et continuer à travailler sans impacter
cette ligne.

![gitflow.png](./00_materials/02_git/gitflow.png)

Le working directory représente une seule et unique branche. Il est possible de naviguer entre les branches pour porter
des modifications entre les différentes lignes de vie.

## Quelques mots sur GitHub

Présentation rapide de :
- GitHub
- GitHub Codespaces
- GitHub Actions

## Comment installer Git ?

Normalement, l'application Git doit s'installer sur votre machine. Il est multi-plateforme et donc compatible avec la
plupart des systèmes d'exploitation (Windows, Mac, Linux).

Pour les besoins de ce cours, nous n'aurons pas besoin de l'installer, car vous disposez d'un GitHub Codepaces sur
lequel git est préalablement installé.

## Les commandes de base

Connecter son projet à un repository distant :
- `git config --global user.name "votre nom"` (configurer votre nom d'utilisateur)
- `git config --global user.email "votre email"` (configurer votre email)
- `git init` (initialiser un repository git dans le dossier courant)
- `git remote add origin <adresse_du_repo_distant>` (connecter le repository local au repository distant)

Récupérer les modifications distantes :
- `git clone <adresse_du_repo_distant>` (copier le repository distant en local)
- `git fetch` (mise à jour de l'image locale à partir du repository distant)
- `git pull` (mise à jour de l'image locale et du working directory, donc de la branche courante)

Enregistrer une modification :
- `git status` (affichage des modifications sur le working directory)
- `git add <fichier>` (ajouter une modification dans l'index)
- `git commit -m "un message de commit"` (enregistrer l'index sous forme d'un incrément ou communément appelé, commit)
- `git push` (synchroniser ou pousser l'ensemble des incréments sur le repository distant et donc partager le travail avec toute l'équipe)

Normes de nommage des commits : https://www.conventionalcommits.org/fr/v1.0.0/

Gestion des branches :
- `git branch` (lister toutes les branches locales)
- `git remote show origin` (lister toutes les branches distantes)
- `git switch -c <nouvelle branche>` (permet de créer une nouvelle branche en local et travailler dessus)
- `git switch <branche existante>` (permet de naviguer sur une autre branche)
- `git push -u origin <nouvelle branche>` (pousser sur le repository distant la branche nouvellement créée)

Afficher l'historique:
- `git log --oneline --decorate --graph`

## Les commandes de base en pratique (EVALUATION)

Avant de manipuler git, nous allons tout d'abord, créez votre repository distant sur GitHub en le nommant : kto-titanic.
Nous allons ensuite démarrer un workspace RedHat DevSpace qui nous servira d'environnement de travail pour le reste du cours.
Nous allons également créer notre projet kto-titanic dans ce workspace, à partir d'un template cookiecutter et le configurer correctement.
Nous utiliserons enfin les commandes de bases git, pour versionner notre projet dans notre repository GitHub.

Commençons donc par créer votre repository GitHub.

- Connectez-vous sur [GitHub](www.github.com)
- Cliquez sur le bouton New 

![001.png](img/001.png)

- Autre méthode, vous pouvez également lister vos repositories en cliquant sur la tabulation Repositories de votre page GitHub et cliquer sur le bouton New

![009.png](img/009.png)

- Nommez votre repository kto-titanic, laissez-le en Public, **décochez** l'option Add a README file, sélectionnez No gitignore et pas de licence. 
- Cliquez sur Create repository

![002.png](img/002.png)

- Votre repository est créé. Vous devriez arriver sur une page similaire à celle-ci

![003.png](img/003.png)

Maintenant, nous allons créer le projet qui servira de base pour notre cours. Commençons par créer votre workspace dans
RedHat DevSpace.
- Connectez-vous sur [Sandbox Red Hat](https://sandbox.redhat.com/)
- Il n'est pas impossible que vous deviez vous reconnecter

![006.png](img/006.png)
![007.png](img/007.png)

- Si c'est la première fois que vous vous connectez, vous devrez remplir un formulaire avec vos informations personnelles

![008.png](img/008.png)

- Vous devriez arriver sur votre console Red Hat Developer.

![010.png](img/010.png)

- Dans la section Devspaces, cliquez sur Try it!

- Il n'est pas impossible que vous deviez vous logger et autoriser certains accès

![011.png](img/011.png)
![012.png](img/012.png)

- Cliquez sur Create Workspace, puis sur Empty Workspace

![013.png](img/013.png)

- Le workspace devrait se créer puis s'ouvrir dans un nouvel onglet

![014.png](img/014.png)
![015.png](img/015.png)

- Attention, vos 30 jours de sandbox commencent à partir de ce moment-là ! Vous pouvez suivre leur décompte sur [Sandbox Red Hat](https://sandbox.redhat.com/).

![029.png](img/029.png)

Très bien, vous disposez maintenant d'un environnement de développement complet dans le Cloud. Nous allons maintenant créer notre projet
kto-titanic à partir d'un template cookiecutter.
- Ouvrez un nouveau terminal dans votre DevSpace (Terminal > New Terminal)

![016.png](img/016.png)

- Installez cookiecutter et uv avec la commande `pip install cookiecutter uv`

![017.png](img/017.png)
![018.png](img/018.png)

Avant de créer notre projet, nous avons besoin de quelques informations qui nous serviront à configurer le projet correctement.
En effet, nous avons besoin d'un compte Quay.io pour héberger nos images Docker. En créant votre compte RedHat Developer, vous avez automatiquement un compte Quay.io.
- Connectez-vous sur [Quay.io](https://quay.io/)
- Il est possible que vous deviez vous connecter à votre compte Red Hat Developer, cliquez sur Sign in

![020.png](img/020.png)
![021.png](img/021.png)

- A droite de cette page, dans la section Users and Organization, cliquez sur votre nom d'utilisateur (c'est l'information que nous recherchons)

![022.png](img/022.png)

- Copiez votre nom d'utilisateur, nous en aurons besoin pour configurer notre projet. Notez le bien dans un bloc-notes.

![023.png](img/023.png)

- Revenez dans votre terminal DevSpace, nous avons besoin également de votre nom d'utilisateur OpenShift. Pour cela, tapez la commande 

```bash
oc whoami
```

![025.png](img/025.png)

- Notez également ce nom d'utilisateur dans votre bloc-notes. Normalement, il est identique à votre nom d'utilisateur 
Red Hat Developer et Quay.io. **MAIS IL EST POSSIBLE QUE CE NE SOIT PAS LE CAS. VÉRIFIEZ BIEN LES DEUX !**
- Nous avons maintenant toutes les informations nécessaires pour créer notre projet. Tapez la commande suivante dans votre terminal DevSpace :

```bash
cookiecutter https://github.com/guillaume-thomas/kto-api-and-cloud`
```


- Répondez aux différentes questions posées en utilisant les informations notées précédemment dans votre bloc-notes. Voici un exemple de réponses :

```
Enter the project name (should be your repository name) (kto-titanic): kto-titanic
Enter your Openshift sandbox name (to push images to your Quay.io account and configure OpenShift access to deploy mlflow). It should be the name of your namespace in OpenShift, without '-dev'. (your-openshift-sandbox-name): kto-gthomas
Your username on Quay.io (to push container images). (your-quay-username): kto_gthomas
```

![031.png](img/031.png)

**ATTENTION, comme vous pouvez le voir dans l'exemple ci-dessus, mon nom d'utilisateur Quay.io est différent de mon
nom d'utilisateur OpenShift. Assurez-vous de bien utiliser vos deux noms d'utilisateur respectifs.**

Votre projet est maintenant créé. Nous allons maintenant le configurer pour qu'il puisse être versionné avec git.
- Déplacez-vous dans le dossier de votre projet avec la commande `cd kto-titanic`

![035.png](img/035.png)

- Profitons-en pour ouvrir ce dossier dans l'interface graphique de votre DevSpace. Cliquez Open Folder.
- Sélectionnez le dossier /projects/kto-titanic dans la pop-up qui s'ouvre en haut de la fenêtre
- Cliquez sur Add

![036.png](img/036.png)

- Il est possible qu'une pop-up vous demande si vous demande si vous faites confiance à ce dossier. Cliquez sur Yes, I trust the authors

![037.png](img/037.png)

- Votre projet s'ouvre dans l'interface graphique de votre DevSpace, il est possible que la page se recharge. Veuillez patienter.

![038.png](img/038.png)

Prenons maintenant quelques instants pour observer la structure du projet. Vous y trouverez notamment :
- Un dossier scripts/ qui contiendra nos scripts d'installation de kto-mlflow, de lancement de nos expérimentations à la main, etc.
- Un dossier src/ qui contiendra le code source de notre API FastAPI, code d'entraînement de notre modèle, etc.
- Un dossier notebook/ 
- Un dossier k8s/ qui contiendra les fichiers de déploiement Kubernetes
- Un dossier .github/ qui contiendra nos GitHub Actions
- Un dossier tests/ qui contiendra nos tests unitaires
- Un fichier README.md qui contiendra la documentation de notre projet
- Un fichier .toml qui contiendra les dépendances Python de notre projet

![039.png](img/039.png)
![040.png](img/040.png)
![042.png](img/042.png)
![043.png](img/043.png)
![044.png](img/044.png)
![045.png](img/045.png)
![046.png](img/046.png)
![047.png](img/047.png)

- Très bien, nous allons maintenant initialiser git dans ce dossier et le connecter à notre repository GitHub distant. 
Commençons par configurer git avec votre nom et email. Définissons le nom de la branche par défaut.
```bash
git config --global init.defaultBranch main
```
```bash
git config --global user.email "<your_email>"
```
```bash
git config --global user.name "<your_name>"
```
- Initialisez git dans le dossier courant
```bash
git init
```
- Connectez votre repository local au repository distant. Pour cela, nous avons besoin d'un token GitHub.
- Connectez-vous sur [GitHub](www.github.com)
- Cliquez sur votre photo de profil en haut à droite, puis sur Settings

![061.png](img/061.png)

- Dans le menu de gauche, cliquez sur Developer settings

![062.png](img/062.png)

- Dans le menu de gauche, cliquez sur Personal access tokens
- Cliquez sur Tokens (classic)

![063.png](img/063.png)

- Cliquez sur Generate new token puis sur Generate new token (classic)

![064.png](img/064.png)

- Donnez un nom à votre token, sélectionnez une expiration (90 days est suffisant pour ce cours)
- Cochez les scopes repo et workflow

![065.png](img/065.png)
![066.png](img/066.png)

- Cliquez sur Generate token

![067.png](img/067.png)

- Copiez le token généré dans un bloc-notes, nous en aurons besoin pour la suite

![068.png](img/068.png)

- Revenez dans votre terminal DevSpace et tapez la commande suivante en remplaçant `<your_token>`, 
`<your_GITHUB_username>` et `<repository_name>` par vos informations respectives :

```bash
git remote add origin https://<your_token>@github.com/<your_GITHUB_username>/<repository_name>.git
```

Astuce, vous pouvez copier l'URL de votre repository distant en cliquant sur le bouton Code dans votre repository GitHub.
N'oubliez pas de remplacer `https://` par `https://<your_token>@` dans l'URL copiée.

![052.png](img/052.png)
![082.png](img/082.png)

Maintenant que votre repository local est connecté au repository distant, nous allons pouvoir faire notre premier commit et le pousser
sur GitHub.

- Ajoutez tous les fichiers à l'index avec la commande :
```bash
git add .
```
- Faites votre premier commit avec la commande :
```bash
git commit -m "feat: initial commit"
```
```bash
git push -u origin main
```

![083.png](img/083.png)

- Vérifier sur GitHub que vos fichiers ont bien été poussés.

![071.png](img/071.png)

- Vérifier également que vos github actions sont bien présentes dans l'onglet Actions de votre repository GitHub.

![074.png](img/074.png)
![075.png](img/075.png)

- Ils ne sont pas censés fonctionner pour le moment, mais nous les configurerons plus tard dans ce cours.

Nous allons maintenant pratiquer un peu les commandes de base git vues précédemment.

- Créer un nouveau dossier exercices dans le dossier racine de votre projet

![077.png](img/077.png)

- Crééz un fichier helloworld.txt dans ce dossier exercices et insérer le contenu que vous souhaitez

![078.png](img/078.png)

- Observez les changements via la commande `git status`
```bash
git status
```
- Ajoutez ce fichier à l'index avec la commande `git add`
```bash
git add exercices/helloworld.txt
```
- Committez ce fichier avec la commande `git commit`
```bash
git commit -m "feat: add helloworld.txt"
```
- Pushez ce commit avec la commande `git push`
```bash
git push
```

![084.png](img/084.png)

- Constatez sur GitHub que votre fichier a bien été poussé

![087.png](img/087.png)

- Notez que désormais, si vous modifiez quelque chose dans votre fichier helloworld.txt, votre IDE l'indiquera avec un petit M à côté du fichier.

![088.png](img/088.png)

Jouons un peu avec les branches maintenant.

- Créez une branche feat/exercice-branche avec `git switch -c`
- Créez un ou plusieurs fichiers et observez les changements via `git status`
- Commitez ces fichiers avec `git add` et `git commit`
- Poussez vos commits avec `git push` (attention, la branche n'existe pas encore sur le repository distant, il faudra donc utiliser l'option -u pour la première fois)

```bash
git switch -c feat/exercice-branche
git add .
git commit -m "feat: exercice branche"
git push -u origin feat/exercice-branche
```

![089.png](img/089.png)
![090.png](img/090.png)

- Vérifiez sur GitHub que votre branche et vos commits ont bien été poussés

![091.png](img/091.png)
![092.png](img/092.png)
![093.png](img/093.png)

## Comment revenir en arrière ?

Plusieurs commandes : 
- `git restore <mon-fichier>` (annule les modifications faites sur le fichier pour restaurer son état au dernier commit en date (HEAD))
- `git commit --amend --no-edit` (permet de modifier HEAD en ajoutant l'index actuel)
- `git push -f` (permet de repousser le dernier commit s'il a déjà été partagé. ATTENTION, cela écrase vos précédentes modifications partagées)
- `git reset` (permet de supprimer tous les fichiers de l'index)
- `git reset --hard <hash du commit>` (permet de remettre votre working directory à un état précédent)


## Merger

Il est possible de combiner les modifications faites entre deux branches différentes.

Il y a plusieurs types de merge avec Git. Voici le merge par défaut sur GitHub : 

![merge.png](./00_materials/02_git/merge.png)

Pour merger une branche dans la branche courante, utilisez la commande `git merge <nom de la branche à merger>`

Attention aux conflits ! Il existe des moyens de les corriger, mais nous ne les aborderons pas dans ce cours.

## Modifier l'historique (commandes avancées)

Nous n'irons pas plus avant sur ces commandes avancées, mais en voici malgré tout un petit inventaire : 
- `git stash` et ses dérivés pour mettre de côté des modifications sans avoir à les mettre dans un commit (et les remettre dans votre working directory)
- `git rebase` pour redéfinir un point de départ d'une branche par rapport à une autre sans merge
- `git rebase -i` permet de redéfinir en profondeur l'historique d'une branche

Exemple de merge :

![merge2.png](./00_materials/02_git/merge2.png)

Différence avec rebase : 

![rebase.png](./00_materials/02_git/rebase.png)

## Bonnes pratiques

Voici un petit éventail de bonnes pratiques :
- Respectez les règles de nommage des commits (rappel https://www.conventionalcommits.org/fr/v1.0.0/)
- Respectez et partagez un GitFlow précis avec votre équipe (gitflow, githubflow, ect...)
- Mergez vos branches principales avec des Pull Requests

## Evaluation : création d'une Pull Request

Nous allons maintenant créer une Pull Request pour merger la branche feat/exercice-branche dans main.
- Allez sur GitHub dans votre repository kto-titanic
- Cliquez sur l'onglet Pull Requests
- Cliquez sur New Pull Request
- Sélectionnez la branche feat/exercice-branche comme base et main comme branche de comparaison
- Cliquez sur Create Pull Request
- Donnez un titre et une description à votre Pull Request

![095.png](img/095.png)

- Cliquez sur Create Pull Request

![096.png](img/096.png)

- Modifiez la méthode de merge en cliquant sur la petite flèche à côté du bouton Merge Pull Request et sélectionnez Rebase and merge

![097.png](img/097.png)

- Cliquez sur Rebase and merge puis sur Confirm rebase and merge

![099.png](img/099.png)
![100.png](img/100.png)
![101.png](img/101.png)

- Constatez que votre branche feat/exercice-branche a bien été mergée dans main

![102.png](img/102.png)

- Retournez dans votre DevSpace et synchronisez votre branche main avec la commande `git pull`
```bash
git switch main
git pull --rebase
```

![103.png](img/103.png)
![104.png](img/104.png)
![105.png](img/105.png)

- Affichez votre historique avec `git log`, en jouant avec les différentes options pour voir leurs effets

![106.png](img/106.png)

- **Partagez par mail un screenshot de votre `git log` (évaluation). Indiquez bien également le lien vers votre repo github.**

## L'Open Source

En résumé, un logiciel open source est un code conçu pour être accessible au public : n’importe qui peut voir, 
modifier et distribuer le code à sa convenance. Ce type de logiciel est développé de manière collaborative et 
décentralisée, par une communauté, et repose sur l’examen par les pairs. Un logiciel open source est souvent moins 
cher (ce qui ne veut pas dire gratuit), plus flexible et profite d’une longévité supérieure par rapport à ses équivalents propriétaires.

Sources : https://www.redhat.com/fr/topics/open-source/what-is-open-source

## Les points d'attention sur le Licensing

### Qu'est-ce qu'une license ?

Il existe plus de 80 variantes de licences open source, mais elles entrent généralement dans l’une des deux catégories 
principales : le copyleft et le permissif.
- Copyleft : C’est un type de licence dans lequel le code dérivé du code open source original hérite de ses conditions 
de licence. Les licences open source copyleft les plus populaires, sont, par ordre de restriction : 
AGPL, GPL, LGPL, EPL et Mozilla.
- Permissif : Ce type de licence offre une plus grande liberté de réutilisation, de modification et de distribution.

Voici quelques exemples de licences open source couramment utilisées :
- Licences publiques générales GPL ou GNU GPL : Ces licences sont conçues pour garantir la liberté de partager et de modifier les versions du logiciel et garantir qu’il reste libre pour tous ses utilisateurs.
- Licence MIT : C’est une licence permissive qui est compatible avec de nombreuses autres licences, sans être une licence copyleft. Elle permet une utilisation illimitée et donne la possibilité de réutiliser le code sous une licence différente.
- Licence BSD : C’est une autre licence permissive qui est très flexible, permettant la redistribution et la modification du code source.

**Il est important de noter que chaque licence open source a ses propres conditions et restrictions juridiques, selon 
le type de licence open source appliqué. Il est donc crucial de respecter les termes des licences des logiciels open source.**

Comme vu précédemment, un logiciel open source **n'est pas forcément gratuit**. Il est régi par une license qui peut induire un coût
de license en cas d'usage commercial.

Sources : https://snyk.io/fr/learn/open-source-licenses/

### Prenons l'exemple de MuPDF 

MuPDF est un logiciel qui est disponible sous deux types de licences : 
une licence open source AGPL et une licence commerciale.

La licence AGPL est une licence de copyleft qui garantit que le logiciel reste libre pour tous ses utilisateurs. 
Si vous utilisez MuPDF sous la licence AGPL, vous devez utiliser uniquement la version AGPL et respecter les 
exigences de partage du code source de l’AGPL.

D’autre part, si vous utilisez MuPDF sous une de leurs licences commerciales, vous devez utiliser uniquement leur 
version commerciale et respecter les termes de leur licence commerciale. De plus, si vous achetez un support 
pour leur version commerciale, vous ne devez pas demander de support pour la version AGPL.

Il est important de noter que MuPDF est entièrement contrôlé par son éditeur. Par conséquent, toute utilisation commerciale 
de MuPDF nécessite une licence commerciale de ce dernier. Il n’existe pas de version “domaine public” de MuPDF.

En résumé, MuPDF ne peut pas être vendu sans licence, car il est protégé par ces licences. 
Toute utilisation commerciale de MuPDF nécessite une licence commerciale.

Sources : https://mupdf.com/licensing/index.html

