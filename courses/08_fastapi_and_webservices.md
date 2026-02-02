# 8. Création d'API avec FastAPI

Dans ce chapitre, nous allons voir comment rendre votre IA disponible au monde entier et de manière sécurisée.

Avant de commencer, afin que tout le monde parte du même point, vérifiez que vous n'avez aucune modification en
cours sur votre working directory avec `git status`.
Si c'est le cas, vérifiez que vous avez bien sauvegardé votre travail lors de l'étape précédente pour ne pas perdre
votre travail.
Sollicitez le professeur, car il est possible que votre contrôle continue en soit affecté.

Sinon, annulez toutes vos modifications avec `git reset --hard HEAD`. Supprimez potentiellement les fichiers
non indexés.
Changez maintenant de branche avec `git switch step05`.
Créez désormais une branche avec votre nom : `git switch -c votrenom/step05`

Dans cette partie, nous allons désormais nous concentrer sur le code que vous trouverez dans `cats_dogs_other/api` :

![project_api.png](00_materials/08_fastapi_and_webservices/project_api.png)

## FastAPI
## OpenAPI / Swagger et la documentation
## Sécurité avec oAuth2

# API REST

## Introduction

### Résumé

Qu'est-ce que le MLOps ?

Où en sommes-nous dans la chronologie du MLOps ?

![ML project lifecylcle](00_materials/MLOps_Timeline.png)

Quel est l'objectif de l'étape de déploiement ? Pourquoi transformons-nous notre modèle en service web ?

Le contenu complet de ce cours [se trouve ici.](00_materials/08_fastapi_and_webservices/api_and_deployment.md)

### Les outils

Pour développer, nous utiliserons GitHub Codespaces. 

**N'OUBLIEZ JAMAIS D'ARRÊTER VOTRE ENVIRONNEMENT CODESPACES APRÈS UTILISATION**

Pour tester notre API, nous utiliserons Bruno

### Prérequis

Avoir suivi les étapes précédentes de ce cours.


## 1 - Développer un service web REST avec FastAPI

### a - Qu'est-ce qu'une API REST ? Une requête HTTP ? Que sont les verbes GET/PUT/POST/DELETE ?

Une API REST est une spécification de service web qui utilise les contraintes REST (Representational State Transfer). Nous
ne serons pas exhaustifs sur ce point dans ce cours (plus d'informations ici : https://en.wikipedia.org/wiki/Representational_state_transfer).

Dans ce cours, nous utiliserons ces termes pour spécifier notre service web. Cela signifie que nous allons créer une application qui
nous permet de partager notre modèle de ML avec d'autres applications web (dans ce cas, notre front) en temps réel.

Ces applications web interagiront entre elles avec des requêtes HTTP. HTTP pour HyperText Transfer Protocol, est un
protocole client-serveur utilisé pour les transmissions web.

![Webservices](00_materials/08_fastapi_and_webservices/1%20-%20rest/a%20-%20what%20is%20http/Webservice_1.png)

*Démonstration avec un navigateur web*

Les requêtes HTTP reposent sur des verbes, des corps et des URI afin d'échanger des informations entre les applications web, notamment un client
et un serveur.

URI : L'identifiant d'une ressource
Verbe : L'action que nous voulons faire sur une ressource spécifique
Corps : Le corps de la requête HTTP

Un exemple avec des produits :

![example 1](00_materials/08_fastapi_and_webservices/1%20-%20rest/a%20-%20what%20is%20http/Webservice_2.png)

Un deuxième exemple :

![example 2](00_materials/08_fastapi_and_webservices/1%20-%20rest/a%20-%20what%20is%20http/Webservice_3.png)

Notez que le format des réponses ici n'est pas spécifié mais souvent, les corps sont écrits en JSON.
Les types et formats peuvent être forcés avec des en-têtes HTTP.

Les verbes HTTP les plus utiles sont :

- **GET** : Pour obtenir des ressources
- **POST** : Pour créer des ressources
- **PUT** : Pour mettre à jour des ressources
- **DELETE** : Pour supprimer des ressources

### b - Installer Bruno et le tester

Téléchargez Bruno

Suivez le processus d'installation puis

**Maintenant, une démonstration rapide**

### c - Créer notre première route (/health) avec FastAPI et Uvicorn

Dans cette partie, nous allons créer notre première route HTTP : /health. Cette route retourne l'état actuel de notre service. S'il est correctement démarré, elle retourne "OK".

FastAPI est un framework utilisé pour construire des API en Python. Il nous apporte des outils pour nous permettre de développer notre premier service web HTTP.

Uvicorn est un serveur web pour Python. Il démarre une instance de serveur web pleinement utilisable afin d'exposer notre WS au MONDE !

Créez ou redémarrez votre environnement Codespaces.

Maintenant, nous installons FastAPI et uvicorn.
Dans `cats_dogs_other/api/requirements.txt`, ajoutez à la fin du fichier :
```
fastapi==0.115.6
fastapi-utils==0.2.1
uvicorn==0.27.0
httpx
```

Nous savons également, que nous allons avoir besoin du package wheel d'inférence. Tout comme pour l'entraînement,
ajoutons la copie du wheel dans le répertoire `cats_dogs_other/api/packages` dans le script `init_packages.sh`.
Cela donnerait ceci : 
```bash
pip install --upgrade pip
pip install build

pip install -e packages/inference

cd packages/inference/
python -m build --sdist --wheel
cd dist
cp *.whl ../../../cats_dogs_other/train/packages
cp *.whl ../../../cats_dogs_other/api/packages
cd ../../../

```
Testez que cela fonctionne avec la commande :
```bash
./init_packages.sh
```
Vous devriez avoir ceci :

![init_packages_update.png](00_materials/08_fastapi_and_webservices/init_packages_update.png)

Ajoutons maintenant la dépendance dans le fichier `cats_dogs_other/api/requirements.txt` :
```
fastapi==0.115.6
fastapi-utils==0.2.1
uvicorn==0.27.0
httpx
./cats_dogs_other/api/packages/kto_keras_inference-0.0.1-py3-none-any.whl
```

Jouez maintenant cette commande pour installer ces dépendances :
```bash
pip install -r cats_dogs_other/api/requirements.txt
```

Passons maintenant au code.

D'abord, nous ajoutons un nouveau script Python où nous allons créer nos routes : infer.py. Pour information, ce script existe déjà dans le repository. 
Il a été créé avec des trous pour vous par le cookiecutter. Vous le trouverez dans `./src/titanic/api/infer.py`.


Maintenant, nous créons notre route en complétant le script !
```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "OK"}

```

Notez que cette nouvelle route permet à nos clients d'obtenir l'état du service. Nous utilisons donc le verbe HTTP **GET**.

Maintenant, nous voulons tester cette nouvelle route !! Pour ce faire, nous devons lancer un serveur uvicorn. 
Pour le lancer, nous utiliserons le script `main.py` se trouvant dans le même répertoire que `infer.py`. Il est également à trous.
Ce script est responsable du démarrage de notre application FastAPI dans un serveur web uvicorn.



Dans main.py, nous lançons l'application FastAPI dans un serveur web local, sur le port classique : 8080
```python
import uvicorn

from cats_dogs_other.api.src import index

if __name__ == "__main__":
    uvicorn.run(index.app, host="0.0.0.0", port=8080)

```

Maintenant, nous lançons ce script depuis le Terminal :

![run_uvicorn.png](00_materials/08_fastapi_and_webservices/run_uvicorn.png)
![boot is running](00_materials/08_fastapi_and_webservices/1%20-%20rest/c%20-%20route%20health/boot_is_running.png)

Pour le tester, d'abord rendez votre processus Codespaces Public

![port private](00_materials/08_fastapi_and_webservices/1%20-%20rest/c%20-%20route%20health/port_8080_is_private.png)

Maintenant qu'il est Public, vous pouvez y accéder depuis votre navigateur ou votre Bruno local.
Copiez l'url de votre processus et collez-la dans votre outil favori. Ensuite, ajoutez la route /health à la fin de l'url.

Cela devrait ressembler à ceci :
https://blablabla-8080.preview.app.github.dev/health

Et le résultat est :

![health](00_materials/08_fastapi_and_webservices/1%20-%20rest/c%20-%20route%20health/itsaliiiiive.png)

Dans Bruno, créez une nouvelle Collection :

![create_bruno_collection.png](00_materials/08_fastapi_and_webservices/create_bruno_collection.png)
![create_bruno_collection2.png](00_materials/08_fastapi_and_webservices/create_bruno_collection2.png)

Maintenant, créez une requête : 

![create_request.png](00_materials/08_fastapi_and_webservices/create_request.png)
![create_get_health.png](00_materials/08_fastapi_and_webservices/create_get_health.png)

Saisissez l'url comme suit, sauvegardez et exécutez : 

![save_and_launch_health.png](00_materials/08_fastapi_and_webservices/save_and_launch_health.png)
![is_ok.png](00_materials/08_fastapi_and_webservices/is_ok.png)

Fermez maintenant votre serveur avec un CTRL+C dans le terminal : 

![kill.png](00_materials/08_fastapi_and_webservices/kill.png)

### d - Exposer notre modèle

Maintenant, nous voulons exposer notre modèle de classification de chats et de chiens. D'abord, nous nous assurons que nous avons notre module d'inférence
dans les requirements :

![requirements are ok](00_materials/08_fastapi_and_webservices/1%20-%20rest/d%20-%20model%20exposition/requirements_are_ok.png)

Nous allons utiliser le code du package pour faire la classification à partir d'un fichier.

Maintenant, nous devons créer une nouvelle route afin de permettre à nos utilisateurs d'envoyer une image. D'abord, parlons du multipart form data.

C'est un type de contenu spécifique pour le corps de notre requête. Il est très utile pour envoyer des fichiers via http.

Pour permettre le multipart dans notre projet de production, nous devons ajouter la dépendance suivante :

```
python-multipart==0.0.6
```

Votre fichier devrait ressembler à ceci : 
```
fastapi==0.115.6
fastapi-utils==0.2.1
uvicorn==0.27.0
httpx
./cats_dogs_other/api/packages/kto_keras_inference-0.0.1-py3-none-any.whl
python-multipart==0.0.6
```

Et ensuite, rafraîchissons notre environnement :

```bash
pip install -r ./cats_dogs_other/api/requirements.txt
```

Chargeons maintenant notre modèle. Dans un premier temps, nous allons ajouter à la main notre .keras directement dans notre
répertoire. Remarquez la présence d'un dossier `./production/api/resources`. C'est ici que nous mettrons notre .keras.
Notez qu'un fichier .gitignore est déjà présent. Il vous empêchera de pousser le modèle copié à la main dans votre 
repository git. Nous verrons comment automatiser le téléchargement de votre artifact directement depuis kto-mlflow,
plus loin dans ce cours.

Commencez par télécharger le modèle de votre dernier run réussi dans votre mlflow. Il faudra peut-être allumer
kto-mlflow avec Dailyclean voire, également démarrer Dailyclean à la main. Normalement, vous savez déjà le faire
par vous même ;-) Sinon, vous trouverez votre bonheur dans ce [chapitre](06_ml_platforms.md) :

![open_mlflow.png](00_materials/08_fastapi_and_webservices/open_mlflow.png)
![open_model.png](00_materials/08_fastapi_and_webservices/open_model.png)
![download_model.png](00_materials/08_fastapi_and_webservices/download_model.png)
![upload_model.png](00_materials/08_fastapi_and_webservices/upload_model.png)
![upload_model2.png](00_materials/08_fastapi_and_webservices/upload_model2.png)

Donc, pour créer une instance du modèle, nous utilisons le package d'inférence et utilisons ces lignes de code :
```python
from kto.inference import Inference

model = Inference("./cats_dogs_other/api/resources/final_model.keras")
```

Nous devons les ajouter dans notre script index.py. Notez que pour éviter des problèmes de performance, nous ne créerons pas une nouvelle instance de ce modèle pour chaque appel http.
Donc nous créons une instance de modèle unique (singleton) directement au démarrage de notre application FastApi. Maintenant, le script index.py devrait ressembler à ceci :
```python
from fastapi import FastAPI
from kto.inference import Inference

app = FastAPI()
model = Inference("./cats_dogs_other/api/resources/final_model.keras")


@app.get("/health")
def health():
    return {"status": "OK"}

```

Maintenant, nous créons notre nouvelle route /upload, qui permet à nos clients d'envoyer leurs images et de répondre le résultat de cette 
classification.

Notez que cette fois, les clients doivent envoyer des informations. Nous utiliserons donc le verbe **POST**.

```python
import io
from fastapi import FastAPI, UploadFile
from kto.inference import Inference

app = FastAPI()
model = Inference("./cats_dogs_other/api/resources/final_model.keras")


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/upload")
async def upload(file: UploadFile):
    file_readed = await file.read()
    file_bytes = io.BytesIO(file_readed)
    return model.execute(file_bytes)

```

Pour tester votre nouvelle route, redémarrez votre application python (lancez boot.py) et utilisez Bruno pour envoyer une image.

N'oubliez pas de définir votre Port Codespaces en Public.

Dans Bruno, créez une nouvelle Requête. C'est une requête POST, avec un corps dont le type de contenu est form/data. Votre partie
est nommée "file" (parce que vous l'avez définie en paramètre de la méthode upload dans index.py => async def upload(file: UploadFile))

![create_post_upload_request.png](00_materials/08_fastapi_and_webservices/create_post_upload_request.png)
![create_post_upload_request2.png](00_materials/08_fastapi_and_webservices/create_post_upload_request2.png)
![create_bruno_collection3.png](00_materials/08_fastapi_and_webservices/create_bruno_collection3.png)

Si finalement, Bruno ne propose toujours pas d'upload de fichiers, testez avec 
[Insomnium](https://github.com/ArchGPT/insomnium/releases). Préférez la version portable.

![create_post_upload_request_insomnium.png](00_materials/08_fastapi_and_webservices/create_post_upload_request_insomnium.png)
![create_post_upload_request_insomnium2.png](00_materials/08_fastapi_and_webservices/create_post_upload_request_insomnium2.png)
![create_post_upload_request_insomnium3.png](00_materials/08_fastapi_and_webservices/create_post_upload_request_insomnium3.png)
![create_post_upload_request_insomnium4.png](00_materials/08_fastapi_and_webservices/create_post_upload_request_insomnium4.png)
![create_post_upload_request_insomnium5.png](00_materials/08_fastapi_and_webservices/create_post_upload_request_insomnium5.png)
![is_ok_cat.png](00_materials/08_fastapi_and_webservices/is_ok_cat.png)

Finalement, votre test dans Postman devrait ressembler à ceci :

![test in postman](00_materials/08_fastapi_and_webservices/1%20-%20rest/d%20-%20model%20exposition/test%20in%20postman.png)

### e - Tests unitaires

Dans ce chapitre, nous allons créer des tests unitaires pour couvrir nos nouvelles routes http ! Pour ce faire, nous devons créer un 
nouveau script : `test_index.py`.

Créez-le dans le package python `cats_dogs_other.api.src.tests` :

![unit test localization](00_materials/08_fastapi_and_webservices/1%20-%20rest/e%20-%20unit%20testing/unit%20test%20script.png)

D'abord, nous devons créer un client de test depuis fastApi. Ce client nous permet de simuler un appel client à notre WS :
```python
from fastapi.testclient import TestClient
from cats_dogs_other.api.src import index

client = TestClient(index.app)
```

Notez que la classe TestClient prend en attribut une instance d'application FastApi. Nous ajouterons celle que nous voulons tester.

Maintenant, pour commencer, nous allons seulement tester notre route /health :
```python
import unittest

from fastapi.testclient import TestClient
from cats_dogs_other.api.src import index

client = TestClient(index.app)


class TestIndex(unittest.TestCase):
    def test_health(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "OK"})

```

Comme vous pouvez le voir, nous créons une nouvelle classe qui étend TestCase. Dans cette classe, nous créons une nouvelle méthode, test_health().
Dans cette méthode, nous simulons un appel à notre route /health en utilisant l'instance TestClient créée précédemment.

```python
response = client.get("/health")
```

Cet appel retourne une réponse sur laquelle nous pouvons faire des assertions. Nous affirmons que le code de réponse de la réponse http 
est 200 (ok) et nous affirmons le contenu json retourné.

Maintenant, nous testons notre route /upload en ajoutant une nouvelle méthode, test_upload() :
```python
import unittest

from fastapi.testclient import TestClient
from cats_dogs_other.api.src import index

client = TestClient(index.app)


class TestIndex(unittest.TestCase):
    def test_health(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "OK"})

    def test_upload(self):
        with open("./cats_dogs_other/api/src/tests/resources/cat.png", "rb") as file:
            response = client.post("/upload", files={"file": ("filename", file, "image/png")})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['prediction'], 'Cat')

```

Dans cette méthode, nous ouvrons l'image du chat dans resources (production/api/src/tests/resources). Et nous l'envoyons à la route /upload avec l'instance TestClient.

Cet appel retourne une nouvelle réponse. Nous affirmons le contenu de cette réponse.

Pour exécuter ce test, veuillez utiliser cette commande depuis un Terminal lancé depuis la racine du projet MLOpsPython :
```bash
python -m unittest discover -s cats_dogs_other.api.src.tests -t .
```

Vous devriez avoir ces résultats :

![results](00_materials/08_fastapi_and_webservices/1%20-%20rest/e%20-%20unit%20testing/results.png)

### f - Notre API avec Swagger / OpenAPI

Swagger / OpenAPI est une spécification pour écrire des API. Elle nous permet de définir un contrat depuis le serveur afin de dire aux clients de notre Webservice comment ils peuvent l'utiliser.

Vous avez deux workflows différents :
- contract first : vous écrivez votre API d'abord et vous générez votre code à partir de celle-ci (swagger et openapi fournissent des générateurs de code dans beaucoup de langages => Java, Python, .Net)
- code first : C'est la manière que nous avons choisie pour ce cours. Nous construisons notre Webservice et la documentation Swagger / OpenApi est générée à partir de celui-ci

FastApi nous donne la possibilité de voir le swagger de notre WebService automatiquement.

Pour le voir, utilisez la route /docs :

![swagger](00_materials/08_fastapi_and_webservices/1%20-%20rest/f%20-%20swagger/swagger.png)

FastAPI génère également de la documentation avec Redoc ! Pour le tester, essayez la route /redoc :

![redoc](00_materials/08_fastapi_and_webservices/1%20-%20rest/f%20-%20swagger/redoc.png)

C'est la fin de cette première partie. N'oubliez pas d'arrêter votre serveur uvicorn et votre environnement Codespaces !!

**Bravo ! Vous avez fait votre premier WebService ! Commitez et poussez vos modifications. Testez votre API de détection
de chat et chien avec Bruno, Insomnium ou plus simplement avec Swagger et faites-moi parvenir par mail une capture d'écran de votre test (évaluations).**

Une fois votre modification poussée, vous devriez remarquer que votre github action ne fonctionne plus. Notamment,
les tests unitaires ne fonctionnent plus. Prenons un peu de temps pour corriger ceci. Revenez sur le fichier 
`.github/workflows/cats-dogs-other.yaml`, identifiez la partie qui traite des tests unitaires et modifiez la par ceci :
```yaml
- name: Upgrade pip, install packages and run unittests
        run: |
          pip install --upgrade pip
          ./init_packages.sh
          pip install -r ./cats_dogs_other/requirements.txt
          pip install -r ./cats_dogs_other/label/requirements.txt
          pip install -r ./cats_dogs_other/api/requirements.txt
          # For tests purposes, we copy an existing .keras file in the folder api/resources
          # We will delete it just after the tests
          cp ./cats_dogs_other/train/steps/tests/test/input/model/final_model.keras ./cats_dogs_other/api/resources/final_model.keras 
          python -m unittest
          rm ./cats_dogs_other/api/resources/final_model.keras
```

Cela doit vous donner ce fichier final :
```yaml
name: Cats and dogs CI/CD
on: 
  push:
    branches:
      - step**

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python 3.11
        uses: actions/setup-python@v3
        with:
          python-version: 3.11
      - name: Upgrade pip, install packages and run unittests
        run: |
          pip install --upgrade pip
          ./init_packages.sh
          pip install -r ./cats_dogs_other/requirements.txt
          pip install -r ./cats_dogs_other/label/requirements.txt
          pip install -r ./cats_dogs_other/api/requirements.txt
          # For tests purposes, we copy an existing .keras file in the folder api/resources
          # We will delete it just after the tests
          cp ./cats_dogs_other/train/steps/tests/test/input/model/final_model.keras ./cats_dogs_other/api/resources/final_model.keras 
          python -m unittest
          rm ./cats_dogs_other/api/resources/final_model.keras
      - name: Install mlflow
        run: |
          pip install mlflow[extras]
      - name: Configure Docker (Quay) & Kubectl (Openshift Sandbox)
        run: |
          docker login -u="${{vars.QUAY_ROBOT_USERNAME}}" -p="${{secrets.QUAY_ROBOT_TOKEN}}" quay.io
          kubectl config set-cluster openshift-cluster --server=${{vars.OPENSHIFT_SERVER}}
          kubectl config set-credentials openshift-credentials --token=${{secrets.OPENSHIFT_TOKEN}}
          kubectl config set-context openshift-context --cluster=openshift-cluster --user=openshift-credentials --namespace=${{vars.OPENSHIFT_USERNAME}}-dev
          kubectl config use openshift-context
      - name: Wake up dailyclean and kto-mlflow
        run: |
          kubectl scale --replicas=1 deployment/dailyclean-api
          sleep 30
          curl -X POST ${{vars.DAILYCLEAN_ROUTE}}/pods/start
      - name: Build training image
        run: |
          docker build -f cats_dogs_other/train/Dockerfile -t quay.io/gthomas59800/kto/train/cats-dogs-other-2023-2024:latest --build-arg MLFLOW_S3_ENDPOINT_URL=${{vars.MLFLOW_S3_ENDPOINT_URL}} --build-arg AWS_ACCESS_KEY_ID=${{vars.AWS_ACCESS_KEY_ID}} --build-arg AWS_SECRET_ACCESS_KEY=${{secrets.AWS_SECRET_ACCESS_KEY}} .
      - name: Launch mlflow training in Openshift
        run: |
          export KUBE_MLFLOW_TRACKING_URI="${{vars.MLFLOW_TRACKING_URI}}"
          export MLFLOW_TRACKING_URI="${{vars.MLFLOW_TRACKING_URI}}"
          export MLFLOW_S3_ENDPOINT_URL="${{vars.MLFLOW_S3_ENDPOINT_URL}}"
          export AWS_ACCESS_KEY_ID="${{vars.AWS_ACCESS_KEY_ID}}" 
          export AWS_SECRET_ACCESS_KEY="${{secrets.AWS_SECRET_ACCESS_KEY}}"
          
          cd cats_dogs_other/train
          mlflow run . --experiment-name cats-dogs-other --backend kubernetes --backend-config kubernetes_config.json
      - name: Asleep kto-mlflow with dailyclean
        run: |
          curl -X POST ${{vars.DAILYCLEAN_ROUTE}}/pods/stop

```

## 2 - Sécurité de notre Webservice

### a - Pourquoi devrions-nous ajouter de la sécurité à notre Webservice ?

Quelques arguments :
- Assurer aux clients que leurs informations sont en sécurité et confidentielles
- Bloquer la rétro-ingénierie
- Permet un haut niveau d'Accord de Niveau de Service / Qualité de Service

Comment un pirate pourrait-il interférer dans notre relation client/serveur ?

![men in the middle](00_materials/08_fastapi_and_webservices/2%20-%20security/men%20in%20the%20middle.png)

Une première chose obligatoire à faire pour assurer la sécurité est de chiffrer les requêtes et les résultats entre les clients et les serveurs. Une solution est d'utiliser le protocole
Transport Layer Security. Il nous donne la possibilité d'étendre HTTP vers HTTP**S**.

*Démonstration avec un navigateur web*

Nous n'irons pas plus loin sur ce protocole car nos fournisseurs de Cloud nous assurent cette capacité. Mais il y a une chose simple que nous pouvons ajouter à notre application
pour maximiser sa sécurité.

### b - oAuth2

Nous utiliserons oAuth2. C'est un standard de sécurité. Son objectif est de déléguer la sécurité à un Serveur d'Autorisation. Il permettra aux clients de
s'autoriser et d'assurer au serveur (notre application est un Serveur de Ressources dans ce scénario) que ces clients sont autorisés à effectuer certaines actions.

![simple oAuth2 principle](00_materials/08_fastapi_and_webservices/2%20-%20security/oauth2%20simple.png)

Le client s'authentifiera et demandera au Serveur d'Autorisation un scope spécifique pour une application spécifique (notre Serveur de Ressources). Si les identifiants
du client sont corrects, le Serveur d'Autorisation lui donnera un token. Ce token est souvent limité dans le temps. Le client doit donner ce token
à notre application. Notre application vérifie que ce token est correct et non expiré.

### c - Un exemple avec Auth0

Nous allons créer un tenant gratuit sur Auth0 de Okta pour illustrer ces principes.

Créez un compte gratuit sur https://auth0.com/fr

![insciption1](00_materials/08_fastapi_and_webservices/2%20-%20security/oAuth0/inscription.png)


![insciption1](00_materials/08_fastapi_and_webservices/2%20-%20security/oAuth0/inscription2.png)


![insciption1](00_materials/08_fastapi_and_webservices/2%20-%20security/oAuth0/inscription3.png)

Dans cette dernière étape, veuillez vérifier l'option avancée afin de créer votre tenant dans l'UE

Maintenant, notre tenant est prêt. Nous devons créer une nouvelle API :

![create api](00_materials/08_fastapi_and_webservices/2%20-%20security/oAuth0/create%20api.png)

![new api](00_materials/08_fastapi_and_webservices/2%20-%20security/oAuth0/new%20api.png)

Cette API représente notre Webservice ML. Maintenant, nous créons un scope pour ajouter des permissions aux clients de cette api.

![create scope](00_materials/08_fastapi_and_webservices/2%20-%20security/oAuth0/create%20scope.png)

Maintenant nous devons créer l'application qui sera le client de notre API.

![create applicatino](00_materials/08_fastapi_and_webservices/2%20-%20security/oAuth0/create%20application.png)

Pour simplifier ce cours, nous utiliserons l'application par défaut créée par Auth0 à des fins de test. Notez que ce n'est pas une bonne pratique.

Maintenant nous ajoutons les permissions à cette application :

![add permissions](00_materials/08_fastapi_and_webservices/2%20-%20security/oAuth0/add%20permission%20to%20application.png)

Maintenant nous le testons avec postman !

![test](00_materials/08_fastapi_and_webservices/2%20-%20security/oAuth0/test.png)

*Maintenant, voyons ce que nous pouvons trouver dans un token avec jwt.io (démo)*

### d - Ajouter de la sécurité à notre Webservice

Pour nous assurer que nos clients utilisent un token oAuth2 correct de notre tenant Auth0, nous utiliserons des fonctionnalités avancées de FastAPI et une bibliothèque open source.

D'abord, nous ajoutons une nouvelle dépendance dans le requirements.txt de notre api :

```
oidc-jwt-validation==0.3.1
```

Ce module permet de valider facilement le token. En fait, FastAPI nous donne juste le token et fait des validations mineures dessus, comme la date d'expiration.
Nous devons aller plus loin. Par exemple, nous devons valider la signature du token, l'audience et l'émetteur.

Maintenant, nous rafraîchissons notre environnement :

```bash
pip install -r cats_dogs_other/api/requirements.txt
```

Maintenant, nous devons apporter quelques modifications au script index.py afin de protéger notre route /upload :

```python
import io
import logging
import os # cette ligne est nouvelle

from fastapi import FastAPI, UploadFile, Depends # ici, nous ajoutons Depends
from fastapi.security import OAuth2PasswordBearer # cette ligne est nouvelle
from kto.inference import Inference
from oidc_jwt_validation.authentication import Authentication # cette ligne est nouvelle
from oidc_jwt_validation.http_service import ServiceGet # cette ligne est nouvelle

app = FastAPI()
model = Inference("./cats_dogs_other/api/resources/final_model.keras")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # ce schéma vient de FastAPI. Il permet de vérifier certaines informations du token et de donner le token à la fonction

issuer = os.getenv("OAUTH2_ISSUER") # C'est une valeur d'une variable d'environnement. Elle nous permet de ne pas coder en dur certaines informations
audience = os.getenv("OAUTH2_AUDIENCE") # cette ligne est nouvelle
jwks_uri = os.getenv("OAUTH2_JWKS_URI") # cette ligne est nouvelle
logger = logging.getLogger(__name__) # cette ligne est nouvelle
authentication = Authentication(logger, issuer, ServiceGet(logger), jwks_uri) # Cet objet vérifiera plus en profondeur la validité du token
skip_oidc = False # ce booléen sera utilisé à des fins de tests. Par défaut et en production, il sera toujours False


@app.get("/health") # Notez que cette ligne ne change pas. Elle ne sera pas protégée
def health():
    return {"status": "OK"}


@app.post("/upload")
async def upload(file: UploadFile, token: str = Depends(oauth2_scheme)): # Regardez ce nouvel argument token
    if not skip_oidc:
        await authentication.validate_async(token, audience, "get::prediction") # Cette fonction validera le token
    file_readed = await file.read()
    file_bytes = io.BytesIO(file_readed)
    return model.execute(file_bytes)

```

Pour tester ce nouveau code, vous devez d'abord ajouter ces trois variables d'environnement :
- **OAUTH2_ISSUER**: L'émetteur est le Serveur d'Autorisation
- **OAUTH2_AUDIENCE**: L'audience est l'identifiant de l'API créée dans Auth0 plus tôt. Permet de spécifier 
l'entité (par exemple, une application) pour laquelle le jeton d'accès a été émis
- **OAUTH2_JWKS_URI**: Cet URI permet d'obtenir les clés publiques utilisées pour chiffrer la signature du token (avec Auth0, cette valeur est .well-known/jwks.json. Jetez-y un œil)

Pour ce faire, vous pouvez utiliser la commande linux `export`

Maintenant vous pouvez démarrer votre application, générer un nouveau token depuis Auth0 avec Postman et ensuite, essayer de prédire un chat depuis Postman.

Ce n'est pas fini. Maintenant vous devez corriger vos tests unitaires. En fait, parce que nous protégeons notre route /upload, nos tests ne fonctionnent plus.

Donc, pour les corriger, ajoutez ces lignes à votre script test_index.py :

```python
def skip_oauth():
    return {}


index.skip_oidc = True
index.app.dependency_overrides[index.oauth2_scheme] = skip_oauth
```

Ce code va mocker la fonctionnalité oAuth2 de votre script index.py.
Vous pouvez l'ajouter juste avant ceci :

```python
client = TestClient(index.app)
```

Maintenant ce chapitre est terminé. N'oubliez pas de terminer votre environnement Codespaces.

**De la même manière que la partie d'avant, commitez et poussez vos modifications (avec un nouveau commit) 
et prévenez-moi par mail que je puisse regarder votre code (évaluations)** 
Si possible, j'essaierai de tester en séance que tout fonctionne bien, avec un jeton oAuth fourni par vos soins.