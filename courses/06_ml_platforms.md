# 6. Les plateformes de ML sur le Cloud

Dans cette partie, nous allons étudier comment déporter l'entraînement de notre modèle sur le Cloud. Le but étant de 
bénéficier des avantages de ce dernier dans le domaine du ML. 

Avant de commencer, afin que tout le monde parte du même point, vérifiez que vous n'avez aucune modification en
cours sur votre working directory avec `git status`.
Si c'est le cas, vérifiez que vous avez bien sauvegardé votre travail lors de l'étape précédente pour ne pas perdre
votre travail.
> ⚠️ **Attention** : En cas de doute, sollicitez le professeur.
> Votre contrôle continu pourrait être impacté.


Surtout, n'oubliez pas de commiter et pusher votre travail avant de continuer ! Pour rappel, les commandes sont : 
```bash
git add .
git commit -m "your message"
git push origin main
```

Dans ce module, vous allez :
- Découvrir les plateformes ML du marché (Databricks, Azure ML, OpenShift Data Science)
- Déployer votre propre plateforme MLflow dans Kubernetes
- Exécuter vos entraînements dans le Cloud via OpenShift

**Prérequis** : Avoir terminé le module 5 (entraînement en notebook)


Si nous revenons à notre sujet, nous allons voir dans cette partie comment utiliser des plateformes de ML dans le Cloud.
Nous nous trouvons toujours dans l'étape d'entraînement de notre modèle. Jusque-là, nous avons toujours travaillé en local,
c'est-à-dire que tout le traitement s'effectuait sur notre machine. Cela est très bien pour des petits jeux de données non critiques
et des modèles simples. Mais dès que l'on souhaite travailler sur des données plus volumineuses, ou entraîner des
modèles plus complexes, notre machine atteint rapidement ses limites. C'est là que le Cloud entre en jeu.

Dans la timeline MLOps, nous nous situons toujours dans la phase de **Modélisation**. Nous allons voir comment 
utiliser des plateformes de ML dans le Cloud pour entraîner nos modèles de manière plus efficace et plus rapide.

![MLOps_Timeline.png](00_materials/MLOps_Timeline.png)

## Présentation du cloud et ses avantages

Le cloud computing permet de fournir des services informatiques via Internet, tels que le stockage de données, 
le traitement de données, la gestion de bases de données, le déploiement de webservices et bien d'autres.

Au lieu d'installer et de maintenir des infrastructures de serveurs et de logiciels locaux dans leur propre centre 
de données, les entreprises peuvent accéder à des outils, des applications et des ressources informatiques via le cloud, 
généralement fournis par un fournisseur de services cloud tel qu'Amazon Web Services, Google Cloud Platform ou Microsoft Azure.

Les avantages du cloud computing comprennent la souplesse, la rapidité, la facilité d'accès et la réduction des coûts. 
Les entreprises peuvent facilement dimensionner les ressources en fonction de la demande, sans avoir à investir dans 
des infrastructures coûteuses. La mise en place de serveurs et de logiciels peut être accélérée grâce à l'utilisation 
de ressources cloud préconfigurées, ce qui permet aux entreprises de se concentrer sur le développement de leur activité
principale plutôt que sur les aspects informatiques.

L'avènement du cloud computing est une opportunité considérable pour le machine learning. Le cloud offre aux datascientists 
un accès simple et rapide à des ressources de calcul à grande échelle, ce qui est essentiel pour entraîner des modèles 
de machine learning complexes et gérer de très grandes quantités de données.

Voici quelques avantages clés pour le machine learning dans le cloud :

- **Évolutivité** : Le cloud permet aux scientifiques des données de dimensionner leurs ressources de calcul et de 
stockage en fonction de leurs besoins spécifiques pour un projet donné. Cela signifie qu'ils peuvent ajuster la capacité 
à la hausse ou à la baisse en fonction de la demande, sans avoir à investir dans des infrastructures ou des ressources coûteuses.
- **Accessibilité** : Le cloud permet aux scientifiques des données de travailler à distance et d'accéder à leurs 
projets de machine learning depuis n'importe où, ce qui peut améliorer la collaboration et la flexibilité du travail.
- **Coût** : Les ressources de machine learning peuvent être coûteuses car elles nécessitent un grand nombre de 
ressources de calcul et de stockage. Le cloud permet aux scientifiques des données de payer ce dont ils ont besoin, 
quand ils en ont besoin, sans encourir les coûts liés à l'achat et à la maintenance d'infrastructures de machine learning.
- **Sécurité** : Les fournisseurs de cloud ont des mesures de sécurité robustes pour protéger leurs ressources de 
machine learning. Les scientifiques des données peuvent donc utiliser des ressources de machine learning en toute 
sécurité dans le cloud sans avoir à se soucier de la sécurité de leur infrastructure.

En somme, l'avènement du cloud est une chance pour le machine learning car il offre aux scientifiques des données des 
ressources de calcul et de stockage puissantes et économiques, une grande souplesse, une meilleure accessibilité et une 
sécurité renforcée pour leurs projets de machine learning.

## Présentation des solutions du marché
### Databricks / MLflow

MLflow est une plateforme open source pour la gestion du cycle de vie des projets de Machine Learning. Elle a été créée 
par Databricks en 2018 et permet aux scientifiques des données de gérer leurs projets de machine learning de manière 
efficace, en suivant les données, les modèles, les résultats et les expériences de manière systématique.

Elle offre une fiabilité, une sécurité et une évolutivité de niveau entreprise. MLflow regroupe quatre composants : 
MLflow Tracking, MLflow Projects, MLflow Models et MLflow Model Registry : 

- **MLflow Tracking** permet de suivre les expériences de machine learning et de gérer les résultats associés. 
- **MLflow Projects** fournit un moyen simple de packager du code de machine learning dans un format reproductible, 
afin de le partager avec d’autres personnes ou de le déployer sur des systèmes de production. 
- **MLflow Models** permet de gérer les modèles de machine learning de manière centralisée, en prenant en charge les 
formats de modèles courants
- **MLflow Model Registry** permet de suivre les versions des modèles, de gérer les autorisations et de déployer les
modèles dans des environnements de production.

Databricks, de son côté, est une entreprise de logiciels fondée par les créateurs d'Apache Spark en 2013. Elle fournit 
une plateforme d'analyse de données unifiée dans le cloud (via un lakehouse, fusion entre un datalake et un data warehouse), 
qui permet aux entreprises de traiter des données à grande échelle, en utilisant des technologies distribuées. 
Cette plateforme est compatible avec les principales offres Cloud du marché et repose sur des technologies open sources 
éprouvées telles que Spark et MlFlow. 

### OpenShift DataScience

OpenShift Data Science est une solution développée par Red Hat pour aider les entreprises à gérer efficacement 
leur infrastructure de ML/DL (Machine Learning/Deep Learning) sur leurs propres serveurs ou dans le cloud. 
Cette solution fournit des outils pour créer, gérer et déployer des modèles de machine learning, en utilisant des 
frameworks tels que TensorFlow, scikit-learn, ou PyTorch, tous intégrés à la plateforme.

Les fonctionnalités d'OpenShift Data Science incluent également la gestion des environnements virtuels, la 
surveillance des modèles en cours d'exécution, le partage de modèles entre différents projets ou utilisateurs, 
ainsi que la possibilité d'utiliser des workflows de conteneurs pour déployer les modèles de manière automatisée.

### Azure ML

Azure Machine Learning Studio est une plateforme basée sur le cloud, développée par Microsoft, pour créer, déployer et
gérer des modèles de machine learning. Elle fournit une interface graphique pour la création de workflows de machine 
learning en assemblant des blocs de traitement de données et de modèles pré-conçus.

Azure Machine Learning Studio prend également en charge de nombreux langages de programmation, tels que Python, R et
C#, ce qui permet aux scientifiques des données de travailler avec les outils qu'ils connaissent déjà.

En outre, la plateforme d'Azure Machine Learning est hautement intégrée avec le reste des services Microsoft Azure, 
ce qui facilite la gestion et la maintenance des environnements de machine learning sur le cloud.

Comme d'autres plateformes vues ici, Azure ML permet également de déployer rapidement des modèles sous formes de 
Webservice directement dans ses infrastructures. Ce procédé est intéressant pour aller vite, mais pourrait se heurter
à quelques soucis d'intégration en entreprise, comme l'usage de bibliothèques spécifiques ou même la sécurisation
d'un tel service dans le Cloud, s'autant plus s'il est public. Cette approche ne sera pas étudiée dans ce cours, nous 
prendrons l'option moins packagée et ferons nous-même l'exercice à la main. Mais sachez que ça existe ! ;)


## Présentation de kto-mlflow

Ce cours est dédié à toutes et tous. Il doit également être suivi en pleine autonomie pour permettre aux absentes, absents,
de pouvoir rattraper le plus facilement possible. C'est pourquoi, nous n'utiliserons pas ici de plateforme du marché,
nécessitant des comptes payants provisionnés ou des comptes de formations issus de partenariat. 

Nous allons utiliser une plateforme maison, mise en place spécifiquement pour ce cours : kto-mlflow. Cette plateforme
nécessite uniquement d'avoir accès à un Kubernetes en ligne. C'est pourquoi, nous utiliserons la DevSandbox de RedHat,
car elle est gratuite. En d'autres termes, vous bénéficiez d'une plateforme OpenShift (une sorte de Kubernetes sous stéroïdes)
accessible en ligne pendant 30 jours, réactivables à volonté.

kto-mlflow se compose des composants suivants :
- mlflow : Vous devez déjà le connaitre, voir plus haut ;)
- minio : Pareil, vous connaissez déjà minio, c'est une sorte de S3 d'Amazone, donc un service de stockage de fichiers
- mysql : Célèbre base de données
- Dailyclean : Solution opensource permettant de libérer vos ressources Cloud quand vous n'en n'avez pas besoin. Permet
également de les rendre disponibles. C'est une sorte de bouton ON/OFF pour kto-mlflow

## Training avec MlFlow

Nous allons commencer par travailler avec MLflow en local, directement dans notre Codespace. Comme vous l'avez vu précédemment,
MLflow dispose d'un outil de tracking. Ce dernier permet donc de "logger" (ou journaliser) des informations utiles 
d'un run d'une expérience.

Dans MLflow, une **expérience** est la principale unité d'organisation. Toutes les exécutions (ou run) MLflow appartiennent
à une expérience. Chaque expérience vous permet de visualiser, de rechercher et de comparer des exécutions,
ainsi que de télécharger des artefacts ou des métadonnées à analyser dans d’autres outils. Les expériences sont
conservées dans le server de Tracking MLflow (local ici d'abord, puis hébergé dans kto-mlflow).

Dans MLflow, un **run** est une exécution d'un script ou d'un workflow de machine learning.
Il est associé à une expérience et contient des informations sur les paramètres, les métriques, les artefacts et les
tags de l'exécution. Vous pouvez utiliser les runs pour suivre les performances de votre modèle, comparer les
résultats de différentes exécutions et partager les résultats avec d'autres personnes.

En d'autres termes, chaque fois que vous lancerez un entraînement d'un modèle avec le support de MLflow, un nouveau run
dans votre expérience va se créer, et contiendra toutes les informations utiles que l'on aura souhaité y mettre. Nous
pouvons logger toutes sortes d'informations, que l'on verra par la suite. 
Plus d'informations [ici](https://mlflow.org/docs/latest/tracking.html) 

### Création d'un projet en local

Dans MLflow, un **projet** est un format pour empaqueter le code d'entraînement de manière réutilisable 
et reproductible. En outre, le composant **MLflow Projects** comprend une API et des outils en ligne de commande pour 
exécuter des projets. Chaque projet est simplement un répertoire de fichiers, ou un référentiel Git, contenant votre 
code. Vous pouvez décrire votre projet en détail en ajoutant un fichier MLproject, qui est un fichier texte formaté 
YAML. Nous pouvons spécifier pour un projet plusieurs propriétés, telles que son nom, ses points d'entrée et 
son environnement d'exécution.

Les **[environnements d’exécution](https://mlflow.org/docs/latest/ml/projects/#environment)** pour les projets MLflow sont utilisés pour spécifier les dépendances et les packages 
requis pour exécuter notre code. Les environnements d'exécution possibles pour les projets MLflow sont les suivants :
- **Environnement virtuel (Virtualenv)** : Les environnements virtuels prennent en charge les packages Python 
disponibles sur PyPI (nécessite pyenv pour fonctionner).
- **Environnement Conda** : Les environnements Conda prennent en charge les packages Python disponibles sur Anaconda.
(attention à la licence !!!)
- **Conteneur Docker** : Les conteneurs Docker sont des environnements d'exécution isolés qui contiennent tous les 
packages et dépendances nécessaires pour exécuter votre code.
- **Environnement système** : Les environnements système sont les environnements d'exécution par défaut qui 
utilisent les packages Python installés sur votre système.

>Pour commencer en local, cette section peut rester vide, l'environnement par défaut sera celui de votre machine.

Vous pouvez exécuter n'importe quel projet à partir d'un URI Git ou d'un répertoire local à l'aide de l'outil en 
ligne de commande `mlflow run` ou de l'API Python `mlflow.projects.run()`. Ces API permettent également de soumettre 
le projet pour une exécution à distance sur Databricks et Kubernetes. L'approche de l'exécution à distance est intéressante,
car elle nous permet d'entraîner nos modèles dans le Cloud et donc de bénéficier de ses avantages.

En guise d'environnement, étant donné que notre cible est de pouvoir entraîner nos modèles dans un Kubernetes en ligne,
nous allons préférer l'approche avec Docker. Nous reviendrons sur Docker et Kubernetes plus en détail plus loin dans ce cours.
Retenez pour l'instant que cette approche permet de créer une sorte d'environnement virtuel, un ordinateur dans votre 
ordinateur, complètement isolé, dans lequel nous allons ajouter notre code et nos dépendances.

Allez ! Commençons ! :-)

Identifiez le fichier `MLproject` dans le répertoire `./src/titanic/training/steps`, dont voici le contenu :
```yaml
name: kto-titanic

entry_points:

  main:
    parameters:
      path: str
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 10}
      random_state: {type: int, default: 42}
    command: "uv -n run --no-sync -m titanic.training.main --input_data_path {path} --n_estimators {n_estimators} --max_depth {max_depth} --random_state {random_state}"
```

Maintenant, testons ce MLProject en local. Pour ce faire, vous pouvez utiliser le script `./scripts/test_local_mlflow.sh`. Il devrait ressembler à ceci : 
```bash
export KUBE_MLFLOW_TRACKING_URI=https://mlflow-$$$$-dev.apps.$$$$$.openshiftapps.com # <--- mettez ici l'url de votre service mlflow
export MLFLOW_TRACKING_URI=https://mlflow-$$$$-dev.apps.$$$$.openshiftapps.com # <--- mettez ici l'url de votre service mlflow
export MLFLOW_S3_ENDPOINT_URL=https://minio-api-$$$$-dev.apps.$$$$.openshiftapps.com # <--- mettez ici l'url de votre service minio
export AWS_ACCESS_KEY_ID=$$$$ # <--- mettez ici le user minio
export AWS_SECRET_ACCESS_KEY=$$$$ # <--- mettez ici le password minio

uv run mlflow run ./src/titanic/training -e main --env-manager=local -P path=all_titanic.csv --experiment-name kto-titanic

```

> ⚠️ **Attention** : Remplacez les `$` par les informations de votre propre mlflow sur OpenShift, mais aussi avec les informations de connexion de votre service minio.

Pour retrouver l'url de votre mlflow, vous pouvez utiliser la commande suivante dans votre terminal : 
```bash
echo https://$(kubectl get route mlflow -o jsonpath='{.spec.host}')
```

![222.png](./img/222.png)

Vous pouvez également la retrouver dans l'interface web d'OpenShift, dans la section `Networking` > `Routes`, puis 
en identifiant la ligne `mlflow`.
L'url se trouve dans la colonne `Location`.

![221.png](./img/221.png)

Vous pouvez désormais exécuter ce script pour lancer votre expérience avec MLflow en local :
```bash
source ./scripts/test_local_mlflow.sh
```

![220.png](./img/220.png)

Comme vous pouvez le voir dans votre Terminal, MLflow a bien exécuté votre projet et a bien enregistré son exécution 
dans votre server de tracking mlflow distant.

Maintenant, connectez-vous à votre serveur MLflow dans le Cloud (kto-mlflow) en utilisant l'URL que vous aviez retrouvée précédemment.
Ouvrez un nouvel onglet dans votre navigateur et collez l'URL. Vous devriez voir l'interface web de MLflow.

Comme vous pouvez le constater, une nouvelle expérience `kto-titanic` a été créée, ainsi qu'un run d'entraînement. 
Cliquez sur l'expérience `kto-titanic` pour voir les détails.

![223.png](./img/223.png)

MlFlow a bien enregistré notre run d'entraînement. Il a également automatiquement détecté que votre projet était un projet de ML.
Cliquez sur le bouton Confirm. Puis, cliquez sur le run pour voir les détails.

![224.png](./img/224.png)

Vous devriez voir les détails de votre run, mais il n'y a pas beaucoup d'informations pour l'instant.

![225.png](./img/225.png)

> ⚠️ **Évaluation** : **Veuillez prendre une capture d'écran comme celle ci-dessus et veuillez la transmettre par mail
> à votre professeur.**

###  Ajout de logs mlflow dans notre code

Ajoutons maintenant quelques éléments dans nos run mlflow.

Pour ce faire, nous allons utiliser les capacités de MLflow Tracking, dont voici la [documentation](https://mlflow.org/docs/latest/tracking/tracking-api.html)
Prenons un peu de temps pour la parcourir !

Nous allons utiliser l'autolog pour commencer. Il s'agit d'une fonctionnalité puissante qui vous permet de logger
automatiquement les métriques, les paramètres et les modèles sans avoir besoin de déclarer explicitement des
instructions de journalisation. Tout ce que vous avez à faire est d'appeler `mlflow.autolog()` avant votre code
d'entraînement. Cette fonctionnalité est disponible pour plusieurs bibliothèques de machine learning telles que
scikit-learn, TensorFlow, PyTorch, XGBoost, LightGBM ...

Lorsque vous utilisez l'autolog, MLflow enregistre automatiquement diverses informations sur votre exécution, notamment :
- Les **métriques** : MLflow présélectionne un ensemble de métriques à enregistrer, en fonction du modèle et de la
  bibliothèque que vous utilisez.
- Les **paramètres** : les hyperparamètres spécifiés pour l'entraînement, ainsi que les valeurs par défaut fournies
  par la bibliothèque si elles ne sont pas explicitement définies.
- La **signature du modèle** : une instance de signature de modèle qui décrit le schéma d'entrée et de sortie du modèle.
- Les **artefacts** : par exemple, les points de contrôle du modèle.
- Le **jeu de données** : l'objet de jeu de données utilisé pour l'entraînement, le cas échéant
  (ne fonctionnera pas dans notre cas, cette fonctionnalité est encore en développement).


Pour utiliser l'autolog dans notre projet, ajoutez ces quelques lignes dans `./src/titanic/training/main.py`, pensez également à importer le
module `mlflow` :
```python
import mlflow
mlflow.autolog()
with mlflow.start_run():
    print('toto')
```

>Ajoutez donc autolog dans votre script `main.py`. N'oubliez pas d'importer le module `mlflow` en début de fichier.

Cela donne donc ce script :
```python
import logging

import fire
import mlflow

from titanic.training.steps.load_data import load_data
from titanic.training.steps.validate import validate
from titanic.training.steps.split_train_test import split_train_test
from titanic.training.steps.train import train

def workflow(input_data_path: str, n_estimators: int, max_depth: int, random_state: int) -> None:
    logging.warning(f"workflow input path : {input_data_path}")
    with mlflow.start_run():
        local_path = load_data(input_data_path)
        xtrain_path, xtest_path, ytrain_path, ytest_path = split_train_test(local_path)
        model_path = train(xtrain_path, ytrain_path, n_estimators, max_depth, random_state)
        validate(model_path, xtest_path, ytest_path)


if __name__ == "__main__":
    fire.Fire(workflow)


```

Dans notre cas, l'utilisation d'`mlflow.autolog()` ne donnera pas grand-chose de plus, car l'utilisation d'autolog 
se fait souvent de manière transparente par défaut. Il faut donc fréquemment ajouter des logs supplémentaires pour avoir
des informations plus pertinentes.

C'est ce que nous allons maintenant faire !

Afin de garantir une explicabilité et une transparence maximales de nos modèles, nous allons ajouter des logs pour toutes les données
utilisées dans notre pipeline de training. Cela inclut :
- Les données brutes utilisées pour l'entraînement
- Les splits de données (train, test, validation)
- Les modèles entraînés
- Les graphiques générés


En effet, en enregistrant toutes les données utilisées dans notre pipeline de training, nous pouvons mieux comprendre comment notre modèle
a été entraîné et quelles données ont été utilisées pour le faire. Cela nous permet également de reproduire les résultats 
si nécessaire, ce qui est essentiel pour garantir la fiabilité et la robustesse de nos modèles. C'est dans l'esprit de MLOps,
qui vise à rendre les processus de machine learning plus transparents et plus faciles à gérer. 

En effet, pour un modele de ML, il est crucial de pouvoir retracer l'origine des données utilisées pour son entraînement. 
Cela permet de garantir la qualité et la fiabilité du modèle, ainsi que de comprendre les biais potentiels dans les données. 
Nous voulons garantir qu'un modele qui part en Production a bien toutes les informations nécessaires stockées 
dans un run mlflow, pour être compris et maintenu dans le temps.

Nous allons donc enregistrer tous les fichiers en entrées et en sorties de chaque étape, le rapport y data profiling 
et également le model. Tout cela dans notre run mlflow.

Nous allons également en profiter pour utiliser les fichiers et dossiers temporaires, afin d'éviter d'écrire nos fichiers dans le répertoire `dist`
de notre projet, ce qui n'est pas une bonne pratique. Cela ralentit vos traitements et pollue votre espace de travail.
Dans ce cas précis, cela peut poser un problème de dataleakage, car les fichiers restent sur votre disque dur.
De plus, en cas de grosse volumétrie, vous risquez de saturer votre espace disque rapidement si vous ne programmez
pas le nettoyage de ces fichiers.
Enfin, ce code sera exécuté dans un environnement Docker dans le Cloud et il est préférable de ne pas écrire sur le disque dur, en
dehors des dossiers temporaires. Voire parfois, le disque dur n'est même pas accessible en écriture !
Pour utiliser les fichiers et dossiers temporaires, nous allons utiliser le module `tempfile` de Python :
```python
import tempfile
with tempfile.TemporaryDirectory() as temp_dir:
    # utilisez temp_dir pour stocker vos fichiers temporaires
    pass  # votre code ici
with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
    # utilisez temp_file.name pour accéder au nom du fichier temporaire
    pass  # votre code ici
```

Allez, c'est parti !

Commençons par load_data. Modifiez le fichier `./src/titanic/training/steps/load_data.py`. Ajoutez-y les imports manquants,
remplacez également la ligne suivante par l'utilisation d'un dossier temporaire :

```python
Path("./dist/").mkdir(parents=True, exist_ok=True)
```

> ⚠️ **Attention** : Faites bien attention aux nouvelles indentations !!

Enfin, enregistrez le fichier csv téléchargé et le rapport de y profiling généré avec la fonction `log_artifact`.
Afin de permettre à la prochaine étape de pouvoir récupérer le fichier, retournez le chemin du fichier stocké dans mlflow.
Votre code devrait ressembler à ceci :
```python
import logging
import os
from pathlib import Path
import tempfile # Nouvel import pour gérer les fichiers temporaires

import boto3
import mlflow # Nouvel import pour mlflow
import pandas as pd
from ydata_profiling import ProfileReport


ARTIFACT_PATH = "path_output"
PROFILING_PATH = "profiling_reports"


def load_data(path: str) -> str:
  logging.warning(f"load_data on path : {path}")

  with tempfile.TemporaryDirectory() as tmp_dir: # Utilisation d'un dossier temporaire
    local_path = Path(tmp_dir, "data.csv") # Fichier temporaire pour stocker les données
    logging.warning(f"to path : {local_path}")

    s3_client = boto3.client(
      "s3",
      endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
      aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
      aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    s3_client.download_file("kto-titanic", path, local_path)
    df = pd.read_csv(local_path)

    profile = ProfileReport(df, title=f"Profiling Report - {local_path.stem}")
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file: # Fichier temporaire pour le rapport de profiling
      profile.to_file(tmp_file.name)
      mlflow.log_artifact(tmp_file.name, PROFILING_PATH) # Log du rapport de profiling dans mlflow

    mlflow.log_artifact(str(local_path), ARTIFACT_PATH) # Log du fichier de données dans mlflow

  return f"{ARTIFACT_PATH}/{local_path.name}" # Retourne le chemin dans mlflow

```

Procédons de même pour le split. Modifiez le fichier `./src/titanic/training/steps/split_train_test.py`. Ajoutez-y les imports manquants,
créez un client mlflow afin de télécharger les données brutes de l'étape précédente depuis mlflow dans un dossier temporaire 
et enregistrez les fichiers de splits dans mlflow.
Enfin, n'oubliez pas de retourner les chemins des fichiers dans mlflow. Votre code devrait ressembler à ceci :
```python
import logging
from pathlib import Path
import tempfile # Nouvel import pour gérer les fichiers temporaires

import mlflow # Nouvel import pour mlflow
import pandas as pd
import sklearn.model_selection

client = mlflow.MlflowClient() # Client mlflow pour interagir avec le server de tracking

FEATURES = ["Pclass", "Sex", "SibSp", "Parch"]

TARGET = "Survived"


def split_train_test(data_path: str) -> tuple[str, str, str, str]:
    logging.warning(f"split on {data_path}")
    # Téléchargement des données brutes depuis mlflow
    df = pd.read_csv(client.download_artifacts(run_id=mlflow.active_run().info.run_id, path=data_path), index_col=False) 

    y = df[TARGET]
    x = df[FEATURES]
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3, random_state=42)

    datasets = [
      (x_train, "xtrain", "xtrain.csv"),
      (x_test, "xtest", "xtest.csv"),
      (y_train, "ytrain", "ytrain.csv"),
      (y_test, "ytest", "ytest.csv"),
    ]

    artifact_paths = []
    with tempfile.TemporaryDirectory() as tmp_dir: # Utilisation d'un dossier temporaire
        for data, artifact_path, filename in datasets:
            file_path = Path(tmp_dir, filename)
            data.to_csv(file_path, index=False)
            mlflow.log_artifact(str(file_path), artifact_path) # Log du fichier de split dans mlflow
            artifact_paths.append(f"{artifact_path}/{filename}") # Stockage du chemin dans mlflow

    return tuple(artifact_paths)

```

Passons à l'entraînement. Modifiez le fichier `./src/titanic/training/steps/train.py`. Ajoutez-y les imports manquants,
créez un client mlflow afin de télécharger les fichiers de splits depuis mlflow dans un dossier temporaire et 
enregistrez le modèle entraîné dans mlflow sous format de joblib comme un simple artefact.
Enfin, n'oubliez pas de retourner le chemin du modèle dans mlflow. Votre code devrait ressembler à ceci :
```python
import logging
from pathlib import Path
import tempfile # Nouvel import pour gérer les fichiers temporaires

import joblib
import mlflow # Nouvel import pour mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

client = mlflow.MlflowClient() # Client mlflow pour interagir avec le server de tracking

ARTIFACT_PATH = "model_trained"


def train(x_train_path: str, y_train_path: str, n_estimators: int, max_depth: int, random_state: int) -> str:
    logging.warning(f"train {x_train_path} {y_train_path}")
    x_train = pd.read_csv(
        client.download_artifacts(run_id=mlflow.active_run().info.run_id, path=x_train_path), index_col=False # Téléchargement des données depuis mlflow
    )
    y_train = pd.read_csv(
        client.download_artifacts(run_id=mlflow.active_run().info.run_id, path=y_train_path), index_col=False # Téléchargement des données depuis mlflow
    )

    x_train = pd.get_dummies(x_train)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(x_train, y_train)

    model_filename = "model.joblib"
    with tempfile.TemporaryDirectory() as tmp_dir: # Utilisation d'un dossier temporaire
        model_path = Path(tmp_dir, model_filename)
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), ARTIFACT_PATH) # Log du modèle dans mlflow

    return f"{ARTIFACT_PATH}/{model_filename}" # Retourne le chemin du modèle dans mlflow

```

Enfin, terminons par la validation. Modifiez le fichier `./src/titanic/training/steps/validate.py`. Ajoutez-y les imports manquants,
créez un client mlflow afin de télécharger les fichiers de splits et le modèle depuis mlflow dans un dossier temporaire.
Ensuite, calculez vos métriques, votre feature importance et enregistrez-les dans mlflow. Pour les métriques, utilisez la fonction `log_metric`.
Pour la feature importance, vous pouvez utiliser la fonction `log_dict`.

Enfin, sauvegardez votre modèle validé dans mlflow avec `log_model` et enregistrez le sous une nouvelle version dans le modèle registry 
de mlflow avec `register_model`. Attention, cette méthode peut renvoyer des exceptions. Veillez à bien les gérer et logger les erreurs avec le module `logging`.

Votre code devrait ressembler à ceci :
```python
import logging

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import mlflow # Nouvel import pour mlflow
from mlflow.models import infer_signature # Nouvel import pour inférer la signature du modèle

client = mlflow.MlflowClient()


def validate(model_path: str, x_test_path: str, y_test_path: str) -> None:
    logging.warning(f"validate {model_path}")
    model = joblib.load(client.download_artifacts(run_id=mlflow.active_run().info.run_id, path=model_path)) # Téléchargement du modèle depuis mlflow

    x_test = pd.read_csv(
        client.download_artifacts(run_id=mlflow.active_run().info.run_id, path=x_test_path), index_col=False # Téléchargement des données depuis mlflow
    )
    y_test = pd.read_csv(
        client.download_artifacts(run_id=mlflow.active_run().info.run_id, path=y_test_path), index_col=False # Téléchargement des données depuis mlflow
    )

    x_test = pd.get_dummies(x_test)

    if y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)

    feature_names = x_test.columns.tolist()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_importance = {
        name: float(importance) for name, importance in zip(feature_names, importances, strict=False)
        }
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        if hasattr(coefs, "shape") and len(coefs.shape) > 1:
            coefs = coefs[0]
        feature_importance = {name: float(coef) for name, coef in zip(feature_names, coefs, strict=False)}
    else:
        feature_importance = {name: 0.0 for name in feature_names}
        logging.warning("Model does not have feature importance attributes")

    mlflow.log_metric("mse", mse) # Log des métriques dans mlflow
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("medae", medae)
    mlflow.log_dict(feature_importance, "feature_importance.json") # Log de la feature importance dans mlflow

    model_info = mlflow.sklearn.log_model(
        model, name="model_final", signature=infer_signature(x_test, y_pred), input_example=x_test.head(10)
    ) # Log du modèle validé dans mlflow
    logging.warning(f"artifact path {model_info.artifact_path}") # Log des informations du modèle
    logging.warning(f"model uri {model_info.model_uri}")
    logging.warning(f"model uuid {model_info.model_uuid}")
    logging.warning(f"model metadata {model_info.metadata}")

    try:
        mlflow.register_model(model_info.model_uri, "model_registered") # Enregistrement du modèle dans le modèle registry
    except Exception as e:
        logging.error(f"Erreur registry: {e}") # Log de l'erreur si l'enregistrement échoue

```

Vous avez terminé ! Bravo !

Exécutez de nouveau votre projet mlflow en local à l'aide de votre script `./scripts/test_local_mlflow.sh`, avec la même commande que précédemment : 
```bash
source ./scripts/test_local_mlflow.sh
```

![226.png](img/226.png)

Notez que cette fois-ci, l'exécution est un peu plus longue, car nous avons ajouté des logs mlflow et utilisé des fichiers temporaires.
Les fichiers temporaires sont créés et supprimés automatiquement, ce qui évite de polluer votre espace de travail. Constatez les téléchargements
depuis mlflow dans votre terminal.

Notez enfin que votre run s'est bien terminé et qu'il a été enregistré dans votre serveur mlflow dans le Cloud avec un nouvel identifiant.

Retournez dans votre serveur mlflow dans le Cloud et constatez les différences ! Vous devriez avoir plus d'informations dans votre nouveau run.
Notez déjà la présence d'un modèle enregistré dans la colonne modèles. Quand vous ouvrez votre run, vous devriez avoir plus d'informations
dans les métriques, mais aussi dans les artefacts. Vous devriez voir le rapport de ydata profiling, les fichiers de splits, le modèle entraîné, la feature importance
et le modèle validé.

![227.png](./img/227.png)
![228.png](./img/228.png)
![229.png](./img/229.png)
![230.png](./img/230.png)

> ⚠️ **Évaluation** : **Veuillez prendre 4 captures d'écran comme celles ci-dessus et veuillez les transmettre par mail
> à votre professeur. N'oubliez pas de commiter et pusher vos modifications dans votre dépôt GitHub. 
> Ce sera également pris en compte dans votre évaluation.**

![236.png](./img/236.png)

Notez que nous avons désormais un modèle dans le modèle registry de mlflow. Vous pouvez cliquer dessus pour voir les différentes versions
de votre modèle. Pour l'instant, il n'y en a qu'une seule. Pour le voir, retournez sur la page principale de mlflow, 
puis cliquez sur l'onglet `Models` dans le menu de gauche.

![231.png](./img/231.png)
![232.png](./img/232.png)
![233.png](./img/233.png)
![234.png](./img/234.png)
![235.png](./img/235.png)

Enfin, dernière chose, vous avez modifié votre code et ce dernier est testé unitairement ! Prenez l'habitude d'exécuter vos
tests unitaires régulièrement pour vérifier que rien n'a été cassé. Comme vous pouvez le constater, mlflow ne modifie
en rien la bonne exécution de nos tests, ce qui est une excellente nouvelle !

Lancez donc les tests unitaires de la partie training avec la commande suivante : 
```bash
uv run pytest ./tests/training
```
Vous devriez voir que tous les tests passent correctement, y compris ceux que j'ai préalablement codés pour valider votre avancée.

![237.png](./img/237.png)

### Exécution dans un environnement Docker

Créez également un fichier `Dockerfile` dans le même répertoire : 
```Dockerfile
FROM python:3.11-slim

ARG MLFLOW_S3_ENDPOINT_URL
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

ENV MLFLOW_S3_ENDPOINT_URL=${MLFLOW_S3_ENDPOINT_URL}
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

COPY --chown=${USER} cats_dogs_other/train ./cats_dogs_other/train
COPY --chown=${USER} cats_dogs_other/requirements.txt ./cats_dogs_other/requirements.txt

RUN chmod -R 777 ./cats_dogs_other
RUN pip install -r /cats_dogs_other/requirements.txt
```

Vous devriez avoir ceci :

![mlproject.png](00_materials/06_ml_platforms/mlproject.png)

Créez l'image docker avec la commande (n'oubliez pas de remettre les variables d'environnement) : 
```bash
docker build -f ./cats_dogs_other/train/Dockerfile -t local/cats-dogs-other-train --build-arg MLFLOW_S3_ENDPOINT_URL=$MLFLOW_S3_ENDPOINT_URL --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY .
```

Notez que l'on nomme ici l'image `local/cats-dogs-other-train`. Ceci est raccord avec ce que l'on a mis dans le fichier
`MLproject`. 

Enfin, jouez la commande suivante : 
```bash
mlflow run ./cats_dogs_other/train --experiment-name cats-dogs-other
```

L'expérience est désormais enregistrée dans mlflow et exécutez dans un environnement Docker. 
Vous pouvez lancer un server mlflow local avec la commande suivante :
```bash
mlflow server --host 127.0.0.1 --port 8080
```

Rendez-vous dans l'onglet PORTS pour visiter la page web écoutant le port 8080.

![open_local_mlflow_server.png](00_materials/06_ml_platforms/open_local_mlflow_server.png)

Constatez que vous avez bien votre expérience crée, ainsi que votre run local. 

![run_in_local_mlflow.png](00_materials/06_ml_platforms/run_in_local_mlflow.png)

Notez également qu'il n'y a pas beaucoup d'informations dans votre run à part les paramètres d'entrée utilisés.

![run_local_is_empty.png](00_materials/06_ml_platforms/run_local_is_empty.png)

Éteignez votre serveur en faisant le raccourci CTRL+C dans votre Terminal.

![killing_local_mlflow.png](00_materials/06_ml_platforms/killing_local_mlflow.png)



Reconstruisez l'image docker avec la commande :
```bash
docker build -f ./cats_dogs_other/train/Dockerfile -t local/cats-dogs-other-train --build-arg MLFLOW_S3_ENDPOINT_URL=$MLFLOW_S3_ENDPOINT_URL --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY .
```

Relancez un nouveau run mlflow : 
```bash
mlflow run ./cats_dogs_other/train --experiment-name cats-dogs-other
```

Relancez le server mlflow et constatez les différences : 
```bash
mlflow server --host 127.0.0.1 --port 8080
```

*Attention ! Vous venez de modifier votre code et ce dernier est testé unitairement ! Prenez l'habitude d'exécuter vos
tests unitaires régulièrement pour vérifier que rien n'a été cassé. Comme vous pouvez le constater, mlflow ne modifie
en rien la bonne exécution de nos tests, ce qui est une excellente nouvelle !*

Notez que l'accès à certains Artifacts ne fonctionnent malheureusement pas. C'est un soucis de droits d'accès sur le système
de fichier de codespace, car l'exécution en local de mlflow créé les éléments directement sur votre disque. Pour corriger ce 
problème, vous pouvez utiliser la commande suivante : 
```bash
sudo chmod -R 777 mlruns/
```

Notez que désormais, nous avons plus d'information dans nos runs de train ! Merci autolog ! Mais il va nous manquer des choses !
Par exemple, pas trace de nos plots ! Ni du split qui a été effectué, pas d'info sur les datasets ! 

Nous allons donc ajouter des logs mlflow supplémentaires !

Prenons les choses dans l'ordre et occupons-nous d'abord de l'extraction de nos annotations.
Dans notre script `train/steps/extraction.py`, nous allons enregistrer le dictionnaire `extract` créé, car il récapitule
l'intégralité de nos annotations. Pour ce faire, nous pouvons utiliser la méthode [log_dict](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_dict)
de MLflow : 
```python
import mlflow.keras
mlflow.log_dict(extract, "annotations/extract.json")
```

Votre script devrait ressembler à ceci : 
```python
import json
from pathlib import Path
from typing import Any

import mlflow.keras

from .s3_wrapper import IS3ClientWrapper


def extraction_from_annotation_file(bucket_name: str, s3_path: str, filename: str, s3_client: IS3ClientWrapper) -> tuple[dict[Any, Any], set[Any]]:
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket_name, s3_path, filename)

    extract = {}
    classes = set()
    with open(filename) as file:
        annotations = json.load(file)["annotations"]
        for annotation in annotations:
            label = annotation["annotation"]["label"]
            extract[annotation["fileName"]] = label
            classes.add(label)
    mlflow.log_dict(extract, "annotations/extract.json")
    return extract, classes

```

Maintenant, sauvegardons le résultat du split, avec la même méthode `log_dict`. Ce qui va nous intéresser, c'est de 
garder une trace des dictionnaires créées dans notre fonction. Notre fichier `train/steps/split.py`
devrait ressembler à ceci : 

```python
import random
from pathlib import Path

import mlflow.keras

from .s3_wrapper import IS3ClientWrapper


def random_split_train_evaluate_test_from_extraction(extract: dict,
                                                     classes: set,
                                                     split_ratio_train: float,
                                                     split_ratio_evaluate: float,
                                                     split_ratio_test: float,
                                                     train_dir: str,
                                                     evaluate_dir: str,
                                                     test_dir: str,
                                                     bucket_name: str,
                                                     s3_path: str,
                                                     s3_client: IS3ClientWrapper):
    if split_ratio_train + split_ratio_evaluate + split_ratio_test != 1:
        raise Exception("sum of ratio must be equal to 1")

    keys_list = list(extract.keys())  # shuffle() wants a list
    random.shuffle(keys_list)  # randomize the order of the keys

    nkeys_train = int(split_ratio_train * len(keys_list))  # how many keys does split ratio train% equal
    keys_train = keys_list[:nkeys_train]
    keys_evaluate_and_test = keys_list[nkeys_train:]

    split_ratio_evaluate_and_test = split_ratio_evaluate + split_ratio_test
    nkeys_evaluate = int((split_ratio_evaluate / split_ratio_evaluate_and_test) * len(keys_evaluate_and_test))
    keys_evaluate = keys_evaluate_and_test[:nkeys_evaluate]
    keys_test = keys_evaluate_and_test[nkeys_evaluate:]

    extract_train = {k: extract[k] for k in keys_train}
    extract_evaluate = {k: extract[k] for k in keys_evaluate}
    extract_test = {k: extract[k] for k in keys_test}

    # create directories
    for existing_class in classes:
        Path(train_dir + "/" + existing_class).mkdir(parents=True, exist_ok=True)
        Path(evaluate_dir + "/" + existing_class).mkdir(parents=True, exist_ok=True)
        Path(test_dir + "/" + existing_class).mkdir(parents=True, exist_ok=True)

    # add files in directories
    download_files(extract_train, train_dir, bucket_name, s3_path, s3_client)
    download_files(extract_evaluate, evaluate_dir, bucket_name, s3_path, s3_client)
    download_files(extract_test, test_dir, bucket_name, s3_path, s3_client)

    mlflow.log_dict(extract_train, "annotations/split_train.json")
    mlflow.log_dict(extract_evaluate, "annotations/split_evaluate.json")
    mlflow.log_dict(extract_test, "annotations/split_test.json")


def download_files(extract: dict, directory: str, bucket_name: str, s3_path: str, s3_client: IS3ClientWrapper):
    for key, value in extract.items():
        s3_client.download_file(bucket_name, s3_path + key, directory + "/" + value + "/" + key)

```

Maintenant, sauvegardons notre modèle dans le format attendu, ainsi que nos jolis graphiques. Pour ce faire, nous allons
utiliser les méthodes [log_model](https://www.mlflow.org/docs/1.20.2/python_api/mlflow.keras.html#mlflow.keras.log_model) 
et [log_artifact](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_artifact) : 
```python
import mlflow.keras
#...
model.save(model_path)
mlflow.keras.log_model(model, "model")
#...
# save plot to file
pyplot.savefig(plot_filepath)
mlflow.log_artifact(plot_filepath)
```

Votre script `train/steps/train_and_evaluate.py` devrait donc ressembler à ceci : 
```python
from pathlib import Path

from keras import Model
from keras.src.applications.vgg16 import VGG16
from keras.src.callbacks import History
from keras.src.layers import Dropout, Flatten, Dense
from keras.src.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import mlflow.keras


def train_and_evaluate_model(train_dir: str,
                             evaluate_dir: str,
                             test_dir: str,
                             model_dir: str,
                             model_path: str,
                             plot_filepath: str,
                             batch_size: int,
                             epochs: int):
    model = define_model()

    # create data generator
    datagen = ImageDataGenerator(featurewise_center=True)
    # specify imagenet mean values for centering
    datagen.mean = [123.68, 116.779, 103.939]
    # prepare iterator
    train_it = datagen.flow_from_directory(
        train_dir,
        class_mode="binary",
        batch_size=batch_size,
        target_size=(224, 224)
    )
    validation_it = datagen.flow_from_directory(
        evaluate_dir,
        class_mode="binary",
        batch_size=batch_size,
        target_size=(224, 224)
    )
    # fit model
    history = model.fit(
        train_it,
        steps_per_epoch=len(train_it),
        validation_data=validation_it,
        validation_steps=len(validation_it),
        epochs=epochs,
        verbose=1,
    )
    # test model
    evaluate_it = datagen.flow_from_directory(
        test_dir,
        class_mode="binary",
        batch_size=batch_size,
        target_size=(224, 224)
    )
    _, acc = model.evaluate(evaluate_it, steps=len(evaluate_it), verbose=1)
    evaluate_accuracy_percentage = acc * 100.0
    print("> %.3f" % evaluate_accuracy_percentage)

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    create_history_plots(history, plot_filepath)

    model.save(model_path)
    mlflow.keras.log_model(model, "model")


def define_model() -> Model:
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    output = model.layers[-1].output
    drop1 = Dropout(0.2)(output)
    flat1 = Flatten()(drop1)
    class1 = Dense(64, activation="relu", kernel_initializer="he_uniform")(flat1)
    output = Dense(3, activation="sigmoid")(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def create_history_plots(history: History, plot_filepath: str):
    # plot loss
    pyplot.subplot(211)
    pyplot.title("Cross Entropy Loss")
    pyplot.plot(history.history["loss"], color="blue", label="train")
    pyplot.plot(history.history["val_loss"], color="orange", label="test")
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title("Classification Accuracy")
    pyplot.plot(history.history["accuracy"], color="blue", label="train")
    pyplot.plot(history.history["val_accuracy"], color="orange", label="test")
    # save plot to file
    pyplot.savefig(plot_filepath)
    mlflow.log_artifact(plot_filepath)
    
    pyplot.close()

```

Notez que dans ces conditions, autolog continue de générer le modèle dans son format par défaut. Pour ne pas gâcher notre
espace disque, nous pouvons demander à autolog d'ignorer la sauvegarde du modèle : 
```python
mlflow.autolog(log_models=False)
```

Faites-le dans votre projet et relancez un run en rebuildant votre image et en utilisant la commande mlflow.
Constatez les changements ! Bravo ! Vous avez terminé cette partie ! **Créez un nouveau commit et poussez vos
modifications sur votre branche. (évaluations)**

Maintenant, lançons nos runs dans le **Cloud** !!!

### Un peu de clean avant

Faites attention, construire des images Docker et exécuter des runs en lcal, occupent de l'espace disque sur votre Codespaces.
Il est peut-être temps de faire un brin de ménage.
Débarrassez-vous des répertoires `dist` et `mlruns` qui se trouvent dans votre Explorateur de fichier.

![cleaning_folders.png](00_materials/06_ml_platforms/cleaning_folders.png)

Si certains fichiers ne veulent pas se supprimer, utilisez cette méthode radicale (à utiliser prudemment) :
```bash
sudo rm -rf mlruns/
```

Il faut également néttoyer vos images docker en trop. Utilisez la commande suivante :
```bash
docker system prune
```

### Exécution dans kto-mlflow

Maintenant que nous avons fait fonctionner notre projet MLflow en local, faisons en sorte de le faire tourner dans
kto-mlflow, soit dans Kubernetes. Retenez pour l'instant que Kubernetes est, ce que l'on appelle, un orchestrateur
pour docker. Il permet donc de créer ce que l'on appelle des `Jobs`, qui exécutent docker et le code que l'on a mis dedans.
MLflow fonctionne ainsi. Quand il est en mode Kubernetes, il nous demande de configurer en amont les accès permettant
de se connecter à notre Kubernetes (ici notre OpenShift), pour créer un Job à partir d'un template et de l'image 
Docker que nous avons créé précédemment ;)

Etant donné que notre plateforme kto-mlflow est en ligne, il faut, avant tout autre chose, rendre notre image docker
de train disponible sur le net. Nous reviendrons plus en profondeur sur le fonctionnement de docker et sur ses concepts,
plus loin dans ce cours.

Pour rendre notre image disponible, nous allons la pousser sur Quay.io. Quay.io est disponible avec votre compte Red
Hat Developer et est gratuit sous certaines conditions. Pareil, nous en reparlerons plus en détail plus tard. Pour le 
moment, concentrons-nous sur le procédé.

Tout d'abord, connectez-vous sur [Quay.io](https://quay.io/) et cliquez en haut à droite sur Sign In :

![signin_quay.png](00_materials/06_ml_platforms/signin_quay.png)
![signin_quay2.png](00_materials/06_ml_platforms/signin_quay2.png) 

Normalement, vous devriez être redirigé vers le menu Repositories. Si ce n'est pas le cas, cliquez sur ce menu en haut :

![quay_repositories.png](00_materials/06_ml_platforms/quay_repositories.png)

Il faut maintenant que nous créions un nouveau repository. Cliquez sur ce bouton : 

![create_repo.png](00_materials/06_ml_platforms/create_repo.png)

Renseignez le nom de votre repo : `kto/train/cats-dogs-other-2023-2024`. Puis, surtout, veuillez faire en sorte que ce 
repository soit Public (cela me permettra d'y accéder et cela fait également partie des limitations de quay.io gratuit).
Enfin, cliquez sur Create Public Repository : 

![create_repo2.png](00_materials/06_ml_platforms/create_repo2.png)

Vous devriez désormais arriver sur cette page ! Bravo ! Vous avez créé votre espace de stockage en ligne !

![repo_created.png](00_materials/06_ml_platforms/repo_created.png)

Maintenant, il faut que l'on pousse notre image sur Quay. Pour ce faire, il faudra d'abord connecter votre client docker
de Codespace sur votre compte quay. Pour cela, nous allons créer un compte de type "Robot". Ces types de compte seront 
très utiles pour la suite de ce cours. Nous reviendrons dessus plus tard. Voyons désormais comment créer un compte !

Cliquez sur le menu en haut à droite et sélectionnez Account Settings : 

![account_settings_quay.png](00_materials/06_ml_platforms/account_settings_quay.png)

Cliquez ensuite sur le logo en forme de robot à gauche et cliquez sur Create Robot Account : 

![create_robot_account.png](00_materials/06_ml_platforms/create_robot_account.png)

Saisissez le nom du robot, par exemple `mlflow_train_2023_2024` et cliquez sur Create Robot Account : 

![create_robot_account2.png](00_materials/06_ml_platforms/create_robot_account2.png)

Maintenant, nous devons autoriser ce compte pour notre repository. Retournez dans le repository 
`kto/train/cats-dogs-other-2023-2024`, puis sélectionnez la roue crantée et ajoutez votre compte robot comme indiqué 
ci-dessous :

![add_robot_to_repo.png](00_materials/06_ml_platforms/add_robot_to_repo.png)

Mettez bien les droits admins à votre robot et ajoutez la permission :

![set_robot_admin_and_add.png](00_materials/06_ml_platforms/set_robot_admin_and_add.png)

Voici ce que vous devriez obtenir : 

![set_robot_admin_and_add.png](00_materials/06_ml_platforms/set_robot_admin_and_add.png)

Maintenant, retournez dans le menu des comptes robot et sélectionnez votre compte créé précédemment :

![return_to_robot_account.png](00_materials/06_ml_platforms/return_to_robot_account.png)

Dans la popup qui vient de s'ouvrir, sélectionnez Docker puis copiez la commande :

![copy_docker_login.png](00_materials/06_ml_platforms/copy_docker_login.png)

Jouez cette commande dans votre Terminal de Codespace et vous serez prêt à envoyer votre image !

![command_paste.png](00_materials/06_ml_platforms/command_paste.png)

Maintenant, il vous faut ajouter deux nouveaux fichiers dans votre projet, dans le répertoire `train` :
- `kubernetes_config.json` dans lequel vous mettrez ce contenu. Veillez-bien à mettre le nom de votre propre image dans quay
à la ligne `repository-uri` :
```json
{
  "kube-context": "openshift-context",
  "kube-job-template-path": "kubernetes_job_template.yaml",
  "repository-uri": "quay.io/gthomas59800/kto/train/cats-dogs-other-2023-2024"
}
```
- `kubernetes_job_template.yaml` dans lequel vous mettrez ce contenu. Veillez de nouveau à renseigner le nom de votre "namespace"
dans OpenShift à la 5eme ligne, au niveau de `namespace` :
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: "{replaced with MLflow Project name}"
  namespace: METTEZ_ICI_VOTRE_NAMESPACE
spec:
  ttlSecondsAfterFinished: 100
  backoffLimit: 0
  template:
    spec:
      containers:
        - name: "{replaced with MLflow Project name}"
          image: "{replaced with URI of Docker image created during Project execution}"
          command: ["{replaced with MLflow Project entry point command}"]
          resources:
            limits:
              cpu: 1300m
              memory: 4000Mi
            requests:
              cpu: 1300m
              memory: 4000Mi
      restartPolicy: Never
```
Vous trouverez cette information en vous connectant sur votre OpenShift : 

![open_sandbox.png](00_materials/06_ml_platforms/open_sandbox.png)
![open_sandbox2.png](00_materials/06_ml_platforms/open_sandbox2.png)
![open_sandbox3.png](00_materials/06_ml_platforms/open_sandbox3.png)
![get_namespace.png](00_materials/06_ml_platforms/get_namespace.png)

Vous devriez avoir ceci : 

![new_files.png](00_materials/06_ml_platforms/new_files.png)

Maintenant, dans votre fichier train/MLproject, changez ligne 4 le nom de l'image local/cats-dogs-other-train, par
le nom de votre image sur quay. Voici un exmeple : 
```yaml
name: cats-dogs-other

docker_env:
  image: quay.io/gthomas59800/kto/train/cats-dogs-other-2023-2024

entry_points:
  main:
    parameters:
        split_ratio_train: {type: float, default: 0.8}
        split_ratio_evaluate: {type: float, default: 0.1}
        split_ratio_test: {type: float, default: 0.1}
        batch_size: {type: int, default: 64}
        epochs: {type: int, default: 4}
        working_dir: {type: str, default: ./cats_dogs_other/train/dist}
        model_filename: {type: str, default: final_model.keras}
        model_plot_filename: {type: str, default: model_plot.png}
    command: "python ./cats_dogs_other/train/train.py --split_ratio_train={split_ratio_train} --split_ratio_evaluate={split_ratio_evaluate} --split_ratio_test={split_ratio_test} --batch_size={batch_size} --epochs={epochs} --working_dir={working_dir}"
```

En fait, ici, nous avons indiqué à MLflow le chemin où est stockée notre image. Elle ne sera plus sur votre machine
de développement, mais directement sur Quay.

Maintenant, construisez votre image avec cette commande, n'oubliez pas vos variables d'environnement si elles manquent,
notez également que le "tag" de votre image a changé pour laisser place à son nom dans quay.io : 
```bash
docker build -f ./cats_dogs_other/train/Dockerfile -t quay.io/gthomas59800/kto/train/cats-dogs-other-2023-2024 --build-arg MLFLOW_S3_ENDPOINT_URL=$MLFLOW_S3_ENDPOINT_URL --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY .
```

Une fois l'image buildée, poussez la dans quay avec la commande suivante :
```bash
docker push quay.io/gthomas59800/kto/train/cats-dogs-other-2023-2024
```

Constatez que votre image est bien dans votre repository quay.io : 

![select_repo.png](00_materials/06_ml_platforms/select_repo.png)
![image_pushed.png](00_materials/06_ml_platforms/image_pushed.png)

Maintenant, il est temps de réveiller kto-mlflow dans OpenShift. Allumez Dailyclean s'il est éteint dans le menu Workloads => Deployments,
Sélectionnez `dailyclean-api` et s'il est inscrit `scaled to zero` dans le cercle, cliquez une fois sur la flèche à droite.

![open_dailyclean_deployment.png](00_materials/06_ml_platforms/open_dailyclean_deployment.png)
![wake_dailyclean.png](00_materials/06_ml_platforms/wake_dailyclean.png)

Rendez-vous sur la page de votre dailyclean (vous trouverez le lien dans Network => Routes) et allumez votre environnement :

![open_dailyclean.png](00_materials/06_ml_platforms/open_dailyclean.png)
![turn_on.png](00_materials/06_ml_platforms/turn_on.png)

Attendez quelques instants que tout démarre. En attendant, il va falloir ajouter de nouveau quelques variables d'environnement
pour que tout fonctionne bien ! Grrrr. Bon, certaines sont normalement déjà configurée, mais dans le doute, voici toutes
les variables nécessaies : 
```bash
export KUBE_MLFLOW_TRACKING_URI=http://mlflow-balba-dev.apps.sandbox-m3.666.p1.openshiftapps.com
export MLFLOW_TRACKING_URI=http://mlflow-balba-dev.apps.sandbox-m3.666.p1.openshiftapps.com
export MLFLOW_S3_ENDPOINT_URL=http://minio-api-balba-dev.apps.sandbox-m3.666.p1.openshiftapps.com
export AWS_ACCESS_KEY_ID=minio 
export AWS_SECRET_ACCESS_KEY=minio123
```
Les deux premières variables sont nouvelles, en réalité, ce sont les mêmes, vous trouvez la bonne url ici dans OpenShift :

![get_mlflow_url.png](00_materials/06_ml_platforms/get_mlflow_url.png)

Ce n'est malheureusement pas terminé ... Vous aurez besoin de configurer kubectl, un client kubernetes, qui va permettre
la création de l'exécution de votre train DANS OpenShift. Voici les commandes à exécuter : 
```bash
kubectl config set-cluster openshift-cluster --server=METTEZ_ICI_LADRESSE_DE_VOTRE_OPENSHIFT
kubectl config set-credentials openshift-credentials --token=METTEZ_ICI_SON_JETON_DAUTHENTIFICATION
kubectl config set-context openshift-context --cluster=openshift-cluster --user=openshift-credentials --namespace=METTEZ_ICI_LE_NOM_DE_VOTRE_NAMESPACE
kubectl config use openshift-context
```

Pour trouver les informations à compléter, le nom de votre namespace et le même que celui précédemment renseigné dans le
fichier kubernetes_job_template.json. En ce qui concerne l'adresse du cluster et le jeton, vous trouverez les informations
dans openshift ici, en cliquant sur votre login en haut à droite et sur Copy login command :

![openshift_login.png](00_materials/06_ml_platforms/openshift_login.png)

Il n'est pas impossible que l'on vous demande de vous connecter sur votre SandBox :

![login_sandbox.png](00_materials/06_ml_platforms/login_sandbox.png)

Cliquez sur Display Token

![display_token.png](00_materials/06_ml_platforms/display_token.png)

Récupérez vos informations : 

![copy_token.png](00_materials/06_ml_platforms/copy_token.png)

Vous devriez avoir ce genre de réponses dans votre terminal : 

![kubectl_commands.png](00_materials/06_ml_platforms/kubectl_commands.png)

Ce n'est pas encore complètement terminé, pour que le mode "kubernetes" de mlflow fonctionne, vous devez installer la
dépendance python suivante:
```bash
pip install mlflow[extras]
```

Et voilà ! Tout est prêt ! Il ne vous reste qu'à jouer la commande mlflow suivante :
```bash
cd ./cats_dogs_other/train
mlflow run . --experiment-name cats-dogs-other --backend kubernetes --backend-config kubernetes_config.json
cd ../..
```

Suivez le déroulé de votre commande :-) Après quelques instants, votre job est lancé dans votre OpenShift : 

![mlflow_kto_command.png](00_materials/06_ml_platforms/mlflow_kto_command.png)
![job_strated.png](00_materials/06_ml_platforms/job_strated.png)

Si vous cliquez sur le job, vous verrez les logs d'exécution de votre train:

![trin_logs_in_openshift.png](00_materials/06_ml_platforms/trin_logs_in_openshift.png)

Une fois l'exécution terminée, votre job s'éteindra tout seul. Normalement, la commande mlflow depuis votre codespace
devrait ressembler à ceci :

![kto_mlflow_succeded.png](00_materials/06_ml_platforms/kto_mlflow_succeded.png)

Vous pouvez maintenant aller consulter vos runs et vos expériences dans votre mlflow en ligne ! Il vous suffit de 
chercher son lien dans Networking => Routes => mlflow :

![mlflow_url2.png](00_materials/06_ml_platforms/mlflow_url2.png)
![run_in_remote_mlflow.png](00_materials/06_ml_platforms/run_in_remote_mlflow.png)
![run_in_remote_mlflow2.png](00_materials/06_ml_platforms/run_in_remote_mlflow2.png)

**Maintenant, n'oubliez pas de créer un nouveau commit avec vos modifications et de le pousser sur votre branche (évaluations).
Veuillez également me partager par mail l'adresse de votre dailyclean (évaluations).**

N'oubliez pas d'éteindre votre environnement avec Dailyclean !!!

Bravo ! Cette partie est difficile, vous avez assuré ! Mais, comme vous avez pu le constater, faire tout ceci à la main
est très fastidieux. Le refaire à chaque fois va vous donner beaucoup de travail. C'est pourquoi, automatiser tout ce 
process est INDISPENSABLE ! Voyons maintenant comment faire dans le prochain chapitre !