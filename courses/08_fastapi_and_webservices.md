# 8. Création d'API avec FastAPI

Dans ce chapitre, nous allons voir comment rendre votre IA disponible au monde entier et de manière sécurisée.

Avant de commencer, afin que tout le monde parte du même point, vérifiez que vous n'avez aucune modification en
cours sur votre working directory avec `git status`.
Si c'est le cas, vérifiez que vous avez bien sauvegardé votre travail lors de l'étape précédente pour ne pas perdre
votre travail.
Sollicitez le professeur, car il est possible que votre contrôle continue en soit affecté.

> ⚠️ **Attention** : En cas de doute, sollicitez le professeur, car il est possible que votre contrôle continue en soit affecté.

Pour rappel, les commandes utiles sont :
```bash
git add .
git commit -m "your message"
git push origin main
```

Dans cette partie, nous allons désormais nous concentrer sur le code que vous trouverez dans `src/titanic/api` :

![288.png](img/288.png)

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

### b - Créer notre première route (/health) avec FastAPI et Uvicorn

Dans cette partie, nous allons créer notre première route HTTP : /health. Cette route retourne l'état actuel de notre service. S'il est correctement démarré, elle retourne "OK".

FastAPI est un framework utilisé pour construire des API en Python. Il nous apporte des outils pour nous permettre de développer notre premier service web HTTP.

Uvicorn est un serveur web pour Python. Il démarre une instance de serveur web pleinement utilisable afin d'exposer notre WS au MONDE !

Créez ou redémarrez votre environnement Codespaces.

Maintenant, nous installons FastAPI et uvicorn.
Dans `pyproject.toml`, ajoutez dans le groupe api les dépendances suivantes :
```toml
[dependency-groups]
api = [
    "fastapi>=0.119.0",
    "opentelemetry-api>=1.39.0",
    "opentelemetry-sdk>=1.39.0",
    "opentelemetry-exporter-otlp>=1.39.0",
    "opentelemetry-instrumentation-fastapi>=0.60b0",
    "uvicorn>=0.37.0",
    "pyjwt[crypto]>=2.8.0",
]
```

![289.png](img/289.png)
![290.png](img/290.png)

Mettez maintenant à jour votre environnement avec la commande :
```bash
uv sync --all-groups
```

![291.png](img/291.png)

Passons maintenant au code.

D'abord, nous ajoutons un nouveau script Python où nous allons créer nos routes : infer.py. Pour information, ce script existe déjà dans le repository. 
Il a été créé avec des trous pour vous par le cookiecutter. Vous le trouverez dans `./src/titanic/api/infer.py`.


Maintenant, nous créons notre route en complétant le script ! La fonction existe déjà, il vous suffit de compléter en 
ajoutant : 

- la bonne dépendance 
```python
from fastapi import FastAPI
```
![292.png](img/292.png)
- en créant votre application REST FastAPI
```python
app = FastAPI()
```
![293.png](img/293.png)
![294.png](img/294.png)
- en ajoutant le bon décorateur
```python
@app.get("/health")
```
![295.png](img/295.png)
![296.png](img/296.png)


Voici le code final de ce script :
```python
"""
Ce script permet d'inférer le model de machine learning et de le mettre à disposition
dans un Webservice. Il pourra donc être utilisé par notre chatbot par exemple,
ou directement par un front. Remplir ce script une fois l'entrainement du model fonctionne
"""

import os

# TODO : Importer les dépendances utiles au bon développement en Python (dataclass, enum, pandas)
# TODO : Importer les dépendances pour sérialiser / désérialiser le model

from fastapi import FastAPI
# TODO : Importer les dépendances OTEL pour le monitoring

from titanic.api.auth import verify_token


JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT", "http://jaeger.kto-gthomas-dev.svc.cluster.local:4318/v1/traces")

# TODO : Intégrer les configurations d'OTEL et instancier le tracer. Peut être fait plus tard si le cours
# sur l'observabilité n'est pas encore donné

app = FastAPI()

# TODO : Ouvrir et charger en mémoire le pickle qui sérialise le model

# TODO : Créer les class et dataclass représentant la donnée qui sera transmise au Webservice pour l'inférence

# TODO : Créer Pclass (enum)
# TODO : Créer Sex (enum)
# TODO : Créer Passenger (attention, l'objet doit pouvoir être transmis en dictionnaire au model. Il faudra créer une méthode d'instance

@app.get("/health")
def health() -> dict:
    return {"status": "OK"}

# TODO : Faire en sorte que cette fonction soit exposée via une route POST /infer
# TODO : Ajouter les paramètres de la fonction (peut se faire en deux fois avec la sécurisation via oAuth2)
def infer() -> list:
    # TODO : implémenter le corps de la fonction
    return [0]


```

Notez que cette nouvelle route permet à nos clients d'obtenir l'état du service. Nous utilisons donc le verbe HTTP **GET**.

Maintenant, nous voulons tester cette nouvelle route !! Pour ce faire, nous devons lancer un serveur uvicorn. 
Pour le lancer, nous utiliserons le script `main.py` se trouvant dans le même répertoire que `infer.py`. Il est également à trous.
Ce script est responsable du démarrage de notre application FastAPI dans un serveur web uvicorn.

![297.png](img/297.png)

Dans main.py, nous lançons l'application FastAPI dans un serveur web local, sur le port classique : 8080
```python
import uvicorn

from titanic.api import infer


def main() -> None:
    uvicorn.run(infer.app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()

```

![298.png](img/298.png)

Maintenant, nous lançons ce script depuis le Terminal. Pour ce faire, nous allons utiliser une des capacité d'uv, les project scripts.
Comme vous pouvez le voir dans le pyproject.toml, nous avons déjà défini un script pour lancer notre application FastAPI :
```toml
[tool.uv.scripts]
api = "titanic.api.main:main"
```

En utilisant l'alias api, nous disons à uv de lancer la fonction main() du script main.py. Donc, pour lancer notre application, utilisez cette commande :
```bash
uv run api
```

![299.png](img/299.png)

Comme vous pouvez le constater, votre application est maintenant démarrée et exposée sur le port 8080 de votre environnement. 
Pour le tester, Devspaces vous donne la possibilité d'exposer ce port vers l'extérieur. Vous pourrez donc accéder à votre application depuis votre navigateur.

Pour cela, en bas à droite, vous avez une pop-up qui vous permet d'exposer votre application sur une URL publique. Cliquez sur yes et exposez votre port 8080.

![300.png](img/300.png)

Maintenant que votre application est exposée, Devspaces vous donne une URL publique que vous pouvez utiliser pour accéder à cette dernière depuis n'importe où dans le monde.
Cliquez sur Open In New Tab et confirmez l'ouverture. Vous pourrez donc tester votre nouvelle route /health depuis votre navigateur.

![301.png](img/301.png)
![302.png](img/302.png)
![303.png](img/303.png)
![304.png](img/304.png)


Fermez maintenant votre serveur avec un CTRL+C dans le terminal : 

![305.png](img/305.png)
![306.png](img/306.png)

### c - Exposer notre modèle

Maintenant, nous voulons exposer notre modèle de classification de survivants du titanic. 

Chargeons maintenant notre modèle. Dans un premier temps, nous allons ajouter à la main notre model.pkl directement dans le bon
répertoire. Remarquez la présence d'un dossier `./src/titanic/api/resources`. C'est ici que nous mettrons notre fichier.
Notez qu'un fichier .gitignore est déjà présent. Il vous empêchera de pousser le modèle copié à la main dans votre 
repository git. Nous verrons comment automatiser le téléchargement de votre artifact directement depuis kto-mlflow
plus loin dans ce cours.

Commencez par télécharger le modèle de votre dernier run réussi dans votre mlflow. Il faudra peut-être allumer
kto-mlflow avec Dailyclean voire, également démarrer Dailyclean à la main. Normalement, vous savez déjà le faire
par vous même ;-) Sinon, vous trouverez votre bonheur [ici](./04_scoping_data_prep_label.md#présentation-de-dailyclean-et-comment-démarrer-kto-mlflow). :

![307.png](img/307.png)
![308.png](img/308.png)
![309.png](img/309.png)
![310.png](img/310.png)
![311.png](img/311.png)
![312.png](img/312.png)
![313.png](img/313.png)
![314.png](img/314.png)

Donc, pour créer une instance du modèle, nous utilisons le package d'inférence et utilisons ces lignes de code :
```python
import pickle

with open("./src/titanic/api/resources/model.pkl", "rb") as f:
    model = pickle.load(f)
```

Nous devons les ajouter dans notre script infer.py. Notez que pour éviter des problèmes de performance, nous ne créerons pas une nouvelle instance de ce modèle pour chaque appel http.
Donc nous créons une instance de modèle unique (singleton) directement au démarrage de notre application FastApi. Maintenant, le script infer.py devrait ressembler à ceci :
```python
"""
Ce script permet d'inférer le model de machine learning et de le mettre à disposition
dans un Webservice. Il pourra donc être utilisé par notre chatbot par exemple,
ou directement par un front. Remplir ce script une fois l'entrainement du model fonctionne
"""

import os
import pickle

# TODO : Importer les dépendances utiles au bon développement en Python (dataclass, enum, pandas)
# TODO : Importer les dépendances pour sérialiser / désérialiser le model

from fastapi import FastAPI
# TODO : Importer les dépendances OTEL pour le monitoring

from titanic.api.auth import verify_token


JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT", "http://jaeger.kto-gthomas-dev.svc.cluster.local:4318/v1/traces")

# TODO : Intégrer les configurations d'OTEL et instancier le tracer. Peut être fait plus tard si le cours
# sur l'observabilité n'est pas encore donné

app = FastAPI()

with open("./src/titanic/api/resources/model.pkl", "rb") as f:
    model = pickle.load(f)

# TODO : Créer les class et dataclass représentant la donnée qui sera transmise au Webservice pour l'inférence

# TODO : Créer Pclass (enum)
# TODO : Créer Sex (enum)
# TODO : Créer Passenger (attention, l'objet doit pouvoir être transmis en dictionnaire au model. Il faudra créer une méthode d'instance

@app.get("/health")
def health() -> dict:
    return {"status": "OK"}

# TODO : Faire en sorte que cette fonction soit exposée via une route POST /infer
# TODO : Ajouter les paramètres de la fonction (peut se faire en deux fois avec la sécurisation via oAuth2)
def infer() -> list:
    # TODO : implémenter le corps de la fonction
    return [0]


```

![315.png](img/315.png)
![316.png](img/316.png)

Maintenant, nous créons notre nouvelle route /infer, qui permet à nos clients d'envoyer leurs informations sur un passager et qui donne le résultat de la 
classification en réponse.

Notez que cette fois, les clients doivent envoyer des informations. Nous utiliserons donc le verbe **POST**.

N'oubliez pas de créer les dataclass qui permettront de sérialiser et désérialiser les données envoyées par les clients. 
En effet, FastAPI utilise Pydantic pour faire du parsing de données. 
Commencez par ajouter les importations nécessaires pour créer des dataclass et des enum. 
```python
from dataclasses import dataclass
from enum import Enum
import pandas as pd
```

![317.png](img/317.png)
![318.png](img/318.png)

Puis créez les enum et les dataclass nécessaires :
```python
class Pclass(Enum):
    UPPER = 1
    MIDDLE = 2
    LOW = 3


class Sex(Enum):
    MALE = "male"
    FEMALE = "female"


@dataclass
class Passenger:
    pclass: Pclass
    sex: Sex
    sibSp: int
    parch: int

    def to_dict(self) -> dict:
        return {"Pclass": self.pclass.value, "Sex": self.sex.value, "SibSp": self.sibSp, "Parch": self.parch}
    
```  

![319.png](img/319.png)

Modifiez également la fonction infer() déjà pour faire l'inférence avec le modèle chargé en mémoire et faites en sorte 
que cette fonction soit exposée via une route POST /infer.

```python
@app.post("/infer")
def infer(passenger: Passenger) -> list:

    df_passenger = pd.DataFrame([passenger.to_dict()])
    df_passenger["Sex"] = pd.Categorical(df_passenger["Sex"], categories=[Sex.FEMALE.value, Sex.MALE.value])
    df_to_predict = pd.get_dummies(df_passenger)

    res = model.predict(df_to_predict)

    return res.tolist()


```

![320.png](img/320.png)

Vous devez maintenant avoir ce code dans votre script infer.py :

```python
"""
Ce script permet d'inférer le model de machine learning et de le mettre à disposition
dans un Webservice. Il pourra donc être utilisé par notre chatbot par exemple,
ou directement par un front. Remplir ce script une fois l'entrainement du model fonctionne
"""

import os
import pickle
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from fastapi import FastAPI
# TODO : Importer les dépendances OTEL pour le monitoring

from titanic.api.auth import verify_token


JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT", "http://jaeger.kto-gthomas-dev.svc.cluster.local:4318/v1/traces")

# TODO : Intégrer les configurations d'OTEL et instancier le tracer. Peut être fait plus tard si le cours
# sur l'observabilité n'est pas encore donné

app = FastAPI()

with open("./src/titanic/api/resources/model.pkl", "rb") as f:
    model = pickle.load(f)

class Pclass(Enum):
    UPPER = 1
    MIDDLE = 2
    LOW = 3


class Sex(Enum):
    MALE = "male"
    FEMALE = "female"


@dataclass
class Passenger:
    pclass: Pclass
    sex: Sex
    sibSp: int
    parch: int

    def to_dict(self) -> dict:
        return {"Pclass": self.pclass.value, "Sex": self.sex.value, "SibSp": self.sibSp, "Parch": self.parch}

@app.get("/health")
def health() -> dict:
    return {"status": "OK"}


# TODO : Ajouter les paramètres de la fonction (peut se faire en deux fois avec la sécurisation via oAuth2)
@app.post("/infer")
def infer(passenger: Passenger) -> list:

    df_passenger = pd.DataFrame([passenger.to_dict()])
    df_passenger["Sex"] = pd.Categorical(df_passenger["Sex"], categories=[Sex.FEMALE.value, Sex.MALE.value])
    df_to_predict = pd.get_dummies(df_passenger)

    res = model.predict(df_to_predict)

    return res.tolist()

```

![321.png](img/321.png)

Pour tester votre nouvelle route, redémarrez votre application python avec uv comme précédemment.

![322.png](img/322.png)
![323.png](img/323.png)
![324.png](img/324.png)

Cette fois, nous allons utiliser OpenAPI / Swagger pour tester notre route. Nous verrons plus en détail ce qu'est 
OpenAPI / Swagger dans la section suivante, mais pour faire simple, c'est une documentation interactive de votre API 
qui vous permet de tester vos routes très facilement.

Dans le nouvel onglet, à la fin de l'URL, ajoutez /docs pour accéder à la documentation Swagger de votre API.

![325.png](img/325.png)

Dans cette documentation, vous pouvez voir toutes les routes de votre API, les verbes associés à ces routes, les paramètres à envoyer pour chaque route, etc.
Pour tester votre route /infer, cliquez dessus puis cliquez sur Try it out. Vous pouvez maintenant envoyer une requête à votre API en remplissant les champs nécessaires.

![326.png](img/326.png)
![327.png](img/327.png)

Cliquez sur Execute. Vous devriez obtenir une réponse de votre API avec le résultat de l'inférence.

![328.png](img/328.png)

Comme vous pouvez le voir, notre passager est classé comme un non survivant (0). Vous pouvez tester avec d'autres paramètres pour voir si vous obtenez des résultats différents.

![329.png](img/329.png)
![330.png](img/330.png)

> ⚠️ **Evaluation** : Envoyez une capture d'écran de votre test de la route /infer avec Swagger à votre professeur par mail. 

Retournez maintenant dans votre terminal et arrêtez votre serveur uvicorn avec un CTRL+C.

![331.png](img/331.png)


### d - Notre API avec Swagger / OpenAPI

Swagger / OpenAPI est une spécification pour écrire des API. Elle nous permet de définir un contrat depuis le serveur afin de dire aux clients de notre Webservice comment ils peuvent l'utiliser.

Vous avez deux workflows différents :
- contract first : vous écrivez votre API d'abord et vous générez votre code à partir de celle-ci (swagger et openapi fournissent des générateurs de code dans beaucoup de langages => Java, Python, .Net)
- code first : C'est la manière que nous avons choisie pour ce cours. Nous construisons notre Webservice et la documentation Swagger / OpenApi est générée à partir de celui-ci

FastApi nous donne la possibilité de voir le swagger de notre WebService automatiquement.

Pour le voir, utilisez la route /docs :

![swagger](00_materials/08_fastapi_and_webservices/1%20-%20rest/f%20-%20swagger/swagger.png)

FastAPI génère également de la documentation avec Redoc ! Pour le tester, essayez la route /redoc :

![redoc](00_materials/08_fastapi_and_webservices/1%20-%20rest/f%20-%20swagger/redoc.png)

C'est la fin de cette première partie. N'oubliez pas d'arrêter votre serveur uvicorn et votre environnement Desvspaces !!

**Bravo ! Vous avez fait votre premier WebService ! Maintenant, nous allons sécuriser notre API.**


## 2 - Sécurité de notre Webservice

### a - Pourquoi devrions-nous ajouter de la sécurité à notre Webservice ?

Quelques arguments :
- Assurer aux clients que leurs informations sont en sécurité et confidentielles
- Bloquer la rétro-ingénierie
- Permet un haut niveau d'Accord de Niveau de Service (SLA) / Qualité de Service (QOS)

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

![332.png](img/332.png)
![333.png](img/333.png)
![334.png](img/334.png)
![335.png](img/335.png)

Dans cette dernière étape, veuillez vérifier l'option avancée afin de créer votre tenant dans l'UE

![336.png](img/336.png)
![337.png](img/337.png)

Maintenant, notre tenant est prêt. Nous devons créer une nouvelle application et une nouvelle API :

![338.png](img/338.png)

Cliquez sur Create a new API.

![339.png](img/339.png)

Sélectionnez l'API et créez votre application.

![340.png](img/340.png)
![341.png](img/341.png)

Cette API représente notre Webservice ML. Maintenant, nous créons un scope pour ajouter des permissions aux clients de cette api.

![343.png](img/343.png)
![344.png](img/344.png)
![345.png](img/345.png)


Maintenant, nous ajoutons les permissions à cette application :

![346.png](img/346.png)
![347.png](img/347.png)

Maintenant nous le testons avec l'onglet Test !

![348.png](img/348.png)

*Maintenant, voyons ce que nous pouvons trouver dans un token avec jwt.io (démo)*

![349.png](img/349.png)

### d - Ajouter de la sécurité à notre Webservice

Pour nous assurer que nos clients utilisent un token oAuth2 correct de notre tenant Auth0, nous utiliserons des fonctionnalités avancées de FastAPI, les middlewares.

Nous avons déjà dans notre projet un script de validation de jetons oAuth2, `auth.py`, se trouvant dans `./src/titanic/api/`.

![350.png](img/350.png)

Ce script de valider facilement le token. En fait, FastAPI nous donne juste le token et fait des validations mineures dessus, comme la date d'expiration.
Nous devons aller plus loin. Par exemple, nous devons valider la signature du token, l'audience et l'émetteur.

Maintenant, nous devons apporter quelques modifications au script infer.py afin de protéger notre route /infer. Nous devons ajouter
l'import Depends de FastAPI et la fonction de validation verify_token de notre script. 

```python
from fastapi import FastAPI, Depends
# TODO : Importer les dépendances OTEL pour le monitoring

from titanic.api.auth import verify_token
```
Voici à quoi devrait ressembler le début de votre script infer.py après ces modifications.
![351.png](img/351.png)

Enfin, nous devons ajouter la validation du token dans notre route /infer. 

```python
@app.post("/infer")
def infer(passenger: Passenger, token: str = Depends(verify_token("api:read"))) -> list
```
![352.png](img/352.png)


Voici à quoi devrait ressembler votre script après cette modification.

```python
"""
Ce script permet d'inférer le model de machine learning et de le mettre à disposition
dans un Webservice. Il pourra donc être utilisé par notre chatbot par exemple,
ou directement par un front. Remplir ce script une fois l'entrainement du model fonctionne
"""

import os
import pickle
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from fastapi import FastAPI, Depends
# TODO : Importer les dépendances OTEL pour le monitoring

from titanic.api.auth import verify_token


JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT", "http://jaeger.kto-gthomas-dev.svc.cluster.local:4318/v1/traces")

# TODO : Intégrer les configurations d'OTEL et instancier le tracer. Peut être fait plus tard si le cours
# sur l'observabilité n'est pas encore donné

app = FastAPI()

with open("./src/titanic/api/resources/model.pkl", "rb") as f:
    model = pickle.load(f)

class Pclass(Enum):
    UPPER = 1
    MIDDLE = 2
    LOW = 3


class Sex(Enum):
    MALE = "male"
    FEMALE = "female"


@dataclass
class Passenger:
    pclass: Pclass
    sex: Sex
    sibSp: int
    parch: int

    def to_dict(self) -> dict:
        return {"Pclass": self.pclass.value, "Sex": self.sex.value, "SibSp": self.sibSp, "Parch": self.parch}

@app.get("/health")
def health() -> dict:
    return {"status": "OK"}


# TODO : Ajouter les paramètres de la fonction (peut se faire en deux fois avec la sécurisation via oAuth2)
@app.post("/infer")
def infer(passenger: Passenger, token: str = Depends(verify_token("api:read"))) -> list:

    df_passenger = pd.DataFrame([passenger.to_dict()])
    df_passenger["Sex"] = pd.Categorical(df_passenger["Sex"], categories=[Sex.FEMALE.value, Sex.MALE.value])
    df_to_predict = pd.get_dummies(df_passenger)

    res = model.predict(df_to_predict)

    return res.tolist()


```

Pour tester ce nouveau code, vous devez d'abord ajouter cette variable d'environnement :
- **OAUTH2_DOMAIN**: Le nom de domain de l'émetteur qui est le Serveur d'Autorisation. Celui-ci est disponible dans la 
requête de test de votre API dans Auth0. Il est aussi disponible dans les paramètres de votre API dans Auth0. Il doit ressembler à ça : `https://dev-1234567.eu.auth0.com/`

Pour ce faire, vous pouvez utiliser la commande linux `export`

Maintenant, vous pouvez démarrer votre application, générer un nouveau token depuis Auth0 et ensuite, essayer de prédire un passager depuis OpenAPI avec et sans jeton.

![353.png](img/353.png)

Testez d'abord sans jeton. Vous devriez obtenir une erreur 401 Unauthorized.

![358.png](img/358.png)

Testez maintenant avec un jeton valide. Vous devriez obtenir une réponse 200 OK avec le résultat de l'inférence.

![354.png](img/354.png)
![355.png](img/355.png)
![356.png](img/356.png)
![357.png](img/357.png)


> ⚠️ **Evaluation** : Faites deux captures d'écran de vos tests de la route /infer avec Swagger sans jeton ET en 
utilisant un token oAuth2 valide et envoyez-la à votre professeur par mail (exemples ci-dessus).

N'oubliez pas d'arrêter votre serveur uvicorn !!

### d - Tests unitaires

Dans ce chapitre, nous allons exécutez les tests unitaires sur notre API. Les tests unitaires sont des tests qui permettent de tester une unité de code de manière isolée.
Ils ont déjà été vus dans le chapitre sur les tests unitaires, mais nous allons les appliquer à notre API cette fois.
Ils sont déjà écrits pour vous dans le script `test_infer.py` se trouvant dans `./tests/api/`.

Commentons rapidement le test sur la route /health pour comprendre comment il fonctionne.

D'abord, nous devons créer un client de test depuis fastApi. Ce client nous permet de simuler un appel client à notre WS :
```python
from unittest.mock import Mock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient
import builtins

with (
    patch("builtins.open", side_effect=selective_mock_open),
    patch("pickle.load", return_value=mock_model),
    patch("titanic.api.infer.verify_token", mock_verify_factory),
):
    from titanic.api.infer import app

@pytest.fixture
def client():
    """Client de test."""
    return TestClient(app)
```

Notez que la classe TestClient prend en attribut une instance d'application FastApi. Nous ajouterons celle que nous voulons tester.

Maintenant, nous allons seulement commenter notre test sur notre route /health :
```python
def test_health_endpoint(client):
    """Test que le endpoint /health fonctionne."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

```

Comme vous pouvez le voir, nous créons une nouvelle fonction, test_health_endpoint(client).
Dans cette méthode, nous simulons un appel à notre route /health en utilisant l'instance TestClient créée précédemment.

```python
response = client.get("/health")
```

Cet appel retourne une réponse sur laquelle nous pouvons faire des assertions. Nous affirmons que le code de réponse de la réponse http
est 200 (ok) et nous affirmons le contenu json retourné.

Maintenant, nous testons notre route /infer en ajoutant une nouvelle fonction :
```python

mock_model = Mock()
mock_model.predict.return_value = np.array([1])


def mock_verify_factory(scope):
    async def _verify(credentials=None):
        return "mock-token"

    return _verify


original_open = builtins.open


def selective_mock_open(file, *args, **kwargs):
    if "model.pkl" in str(file):
        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        return mock_file
    return original_open(file, *args, **kwargs)


with (
    patch("builtins.open", side_effect=selective_mock_open),
    patch("pickle.load", return_value=mock_model),
    patch("titanic.api.infer.verify_token", mock_verify_factory),
):
    from titanic.api.infer import app

@pytest.fixture(autouse=True)
def reset_oauth_env():
    import os
    """Force OAUTH2_DOMAIN à vide pour tous les tests."""
    with patch.dict(os.environ, {"OAUTH2_DOMAIN": ""}, clear=False):
        yield

@pytest.fixture
def mock_infer_model():
    """Mock du modèle ML pour les tests."""
    model = Mock()
    model.predict.return_value = np.array([1])

    with patch("titanic.api.infer.model", model):
        yield model

def test_infer_first_class_female(client, mock_infer_model):
    """Test prédiction pour une femme de 1ère classe."""
    mock_infer_model.predict.return_value = np.array([1])
    payload = {"pclass": 1, "sex": "female", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    result = response.json()
    assert result == [1]
    mock_infer_model.predict.assert_called_once()

```

Dans cette fonction, nous envoyons un payload json à la route /infer avec l'instance TestClient.

Cet appel retourne une nouvelle réponse. Nous affirmons le contenu de cette réponse.
Comme vous pouvez le voir, nous mockons également le modèle d'inférence pour nous assurer que nous testons uniquement la logique de notre route et pas la logique de notre modèle.
Nous mockons aussi la validation du token oAuth2 pour nous assurer que nos tests fonctionnent même sans un token valide.

Pour information, une coquille s'est glissée dans les tests unitaires. Veuillez au préalable les corriger en ajoutant le code suivant : 
```python
@pytest.fixture(autouse=True)
def reset_oauth_env():
    import os
    """Force OAUTH2_DOMAIN à vide pour tous les tests."""
    with patch.dict(os.environ, {"OAUTH2_DOMAIN": ""}, clear=False):
        yield
        
```

Votre script devrait ressembler à ceci : 
```python
from unittest.mock import Mock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient
import builtins


mock_model = Mock()
mock_model.predict.return_value = np.array([1])


def mock_verify_factory(scope):
    async def _verify(credentials=None):
        return "mock-token"

    return _verify


original_open = builtins.open


def selective_mock_open(file, *args, **kwargs):
    if "model.pkl" in str(file):
        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        return mock_file
    return original_open(file, *args, **kwargs)


with (
    patch("builtins.open", side_effect=selective_mock_open),
    patch("pickle.load", return_value=mock_model),
    patch("titanic.api.infer.verify_token", mock_verify_factory),
):
    from titanic.api.infer import app

@pytest.fixture(autouse=True)
def reset_oauth_env():
    import os
    """Force OAUTH2_DOMAIN à vide pour tous les tests."""
    with patch.dict(os.environ, {"OAUTH2_DOMAIN": ""}, clear=False):
        yield

@pytest.fixture
def mock_infer_model():
    """Mock du modèle ML pour les tests."""
    model = Mock()
    model.predict.return_value = np.array([1])

    with patch("titanic.api.infer.model", model):
        yield model


@pytest.fixture
def client():
    """Client de test."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test que le endpoint /health fonctionne."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_infer_first_class_female(client, mock_infer_model):
    """Test prédiction pour une femme de 1ère classe."""
    mock_infer_model.predict.return_value = np.array([1])
    payload = {"pclass": 1, "sex": "female", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    result = response.json()
    assert result == [1]
    mock_infer_model.predict.assert_called_once()


def test_infer_third_class_male(client, mock_infer_model):
    """Test prédiction pour un homme de 3ème classe."""
    mock_infer_model.reset_mock()
    mock_infer_model.predict.return_value = np.array([0])
    payload = {"pclass": 3, "sex": "male", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    result = response.json()
    assert result == [0]
    mock_infer_model.predict.assert_called_once()


def test_infer_with_family(client, mock_infer_model):
    """Test prédiction avec des membres de la famille."""
    mock_infer_model.reset_mock()
    mock_infer_model.predict.return_value = np.array([1])
    payload = {"pclass": 2, "sex": "female", "sibSp": 1, "parch": 2}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    result = response.json()
    assert result == [1]
    mock_infer_model.predict.assert_called_once()


def test_infer_invalid_pclass(client):
    """Test validation avec une classe invalide."""
    payload = {"pclass": 5, "sex": "female", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 422


def test_infer_invalid_sex(client):
    """Test validation avec un sexe invalide."""
    payload = {"pclass": 1, "sex": "unknown", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 422


def test_infer_missing_field(client):
    """Test avec un champ manquant."""
    payload = {"pclass": 1, "sex": "male", "sibSp": 0}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 422


def test_infer_without_token(client):
    """Test que l'API refuse les requêtes sans token."""
    payload = {"pclass": 1, "sex": "female", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload)
    assert response.status_code == 401

```

Vous devriez avoir ces résultats :

![359.png](img/359.png)

Ajoutez vos tests unitaires de votre api dans votre github action pour qu'ils soient exécutés à chaque push. 

Dans votre fichier ct-ci-cd.yaml dans .github/workflows, modifiez la ligne 46 pour ajouter vos tests unitaires d'API :
```yaml
      - name: Launch unit tests
        run: |
          uv run pytest tests/ci tests/training tests/api
```

Votre fichier devrait ressembler à ceci : 
```yaml
name: Train KTO Titanic model and Deploy API

on:
  push:
    branches:
      - main
    paths:
      - 'src/titanic/api/**'
      - 'src/titanic/training/**'
      - 'src/titanic/ci/**'
      - '/tests/api/**'
      - '/tests/training/**'
      - '/tests/ci/**'
      - 'k8s/experiment/**'
      - 'k8s/api/**'
      - '.github/workflows/ct-ci-cd.yaml'
  pull_request:
    branches:
      - main

env:
  EXPERIMENT_NAME: kto-titanic
  EXPERIMENT_IMAGE_NAME: quay.io/kto_gthomas/titanic/experiment
  API_IMAGE_NAME: quay.io/kto_gthomas/titanic/api
  API_ROUTE_NAME: titanic-api
  DAILYCLEAN_ROUTE_NAME: dailyclean
  MINIO_API_ROUTE_NAME: minio-api
  MLFLOW_TRACKING_ROUTE_NAME: mlflow

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python 3.13
        uses: actions/setup-python@v3
        with:
          python-version: 3.13
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install uv
          uv sync --group training --group dev
      - name: Launch unit tests
        run: |
          uv run pytest tests/ci tests/training tests/api
      - name: Resync only training group
        run: |
          uv sync --group training
      - name: Configure docker and kubectl
        run: |
          docker login -u="${{vars.QUAY_ROBOT_USERNAME}}" -p="${{secrets.QUAY_ROBOT_TOKEN}}" quay.io
          kubectl config set-cluster openshift-cluster --server=${{vars.OPENSHIFT_SERVER}}
          kubectl config set-credentials openshift-credentials --token=${{secrets.OPENSHIFT_TOKEN}}
          kubectl config set-context openshift-context --cluster=openshift-cluster --user=openshift-credentials --namespace=${{vars.OPENSHIFT_USERNAME}}-dev
          kubectl config use openshift-context
      - name: Get Routes from Kubernetes and add them to env
        run: |
          DAILYCLEAN_ROUTE_URL=$(kubectl get route ${{env.DAILYCLEAN_ROUTE_NAME}} -o jsonpath='{.spec.host}')
          MINIO_API_ROUTE_URL=$(kubectl get route ${{env.MINIO_API_ROUTE_NAME}} -o jsonpath='{.spec.host}')
          MLFLOW_TRACKING_ROUTE_URL=$(kubectl get route ${{env.MLFLOW_TRACKING_ROUTE_NAME}} -o jsonpath='{.spec.host}')
          
          echo "DAILYCLEAN_ROUTE_URL=https://$DAILYCLEAN_ROUTE_URL" >> $GITHUB_ENV
          echo "MINIO_API_ROUTE_URL=https://$MINIO_API_ROUTE_URL" >> $GITHUB_ENV
          echo "MLFLOW_TRACKING_ROUTE_URL=https://$MLFLOW_TRACKING_ROUTE_URL" >> $GITHUB_ENV
      - name: Wake up dailyclean and mlflow
        run: |
          kubectl scale --replicas=1 deployment/dailyclean-api
          sleep 30
          curl -X POST $DAILYCLEAN_ROUTE_URL/pods/start
      - name: Build training image
        run: |
          docker build -f k8s/experiment/Dockerfile -t ${{ env.EXPERIMENT_IMAGE_NAME }}:latest --build-arg MLFLOW_S3_ENDPOINT_URL=$MINIO_API_ROUTE_URL --build-arg AWS_ACCESS_KEY_ID=${{vars.AWS_ACCESS_KEY_ID}} --build-arg AWS_SECRET_ACCESS_KEY=${{secrets.AWS_SECRET_ACCESS_KEY}} .
      - name: Launch mlflow training in Openshift
        run: |
          export KUBE_MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_ROUTE_URL
          export MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_ROUTE_URL
          export MLFLOW_S3_ENDPOINT_URL=$MINIO_API_ROUTE_URL
          export AWS_ACCESS_KEY_ID="${{vars.AWS_ACCESS_KEY_ID}}" 
          export AWS_SECRET_ACCESS_KEY="${{secrets.AWS_SECRET_ACCESS_KEY}}"

          uv run mlflow run ./src/titanic/training -P path=all_titanic.csv --experiment-name ${{ env.EXPERIMENT_NAME }} --backend kubernetes --backend-config ./k8s/experiment/kubernetes_config.json
      - name: Asleep kto-mlflow with dailyclean
        run: |
          curl -X POST $DAILYCLEAN_ROUTE_URL/pods/stop
          
          # TODO: Saisir la suite de cette pipeline. Devrait apparaître : 
          # Install depencies, Launch unit tests, Resync only training group,
          # Configure docker and kubectl, Get Routes from Kubernetes and add them to env
          # Wake up dailyclean and mlflow, Build training image, Launch mlflow training in Openshift.
          # Une fois l'API développée, et sécurisée intégrer : 
          # Download model artifact, Build and push api image, Configure API manifest with OAuth2 domain
          # Deploy api to Openshift with OAuth2 protection, Get OAuth2 token for integration test
          # Test api with OAuth2 authentication, Asleep kto-mlflow with dailyclean
          
```

Maintenant ce chapitre est terminé. N'oubliez pas de fermer votre environnement Devspaces.

> ⚠️ **Evaluation** : De la même manière que la partie d'avant, commitez et poussez vos modifications (avec un nouveau commit) 
et prévenez-moi par mail que je puisse regarder votre code. Votre github action doit fonctionner (attention, l'entrainement peut échouer si votre
mot de passe OpenShift n'est pas mis à jour) et les tests unitaires doivent être passants.

Si possible, j'essaierai de tester en séance que tout fonctionne bien, avec un jeton oAuth fourni par vos soins.