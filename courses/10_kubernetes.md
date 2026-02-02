# 10. Kubernetes

Dans ce chapitre, nous allons voir comment déployer votre API sous forme de conteneur Docker, directement dans openshift.

Avant de commencer, afin que tout le monde parte du même point, vérifiez que vous n'avez aucune modification en
cours sur votre working directory avec `git status`.
Si c'est le cas, vérifiez que vous avez bien sauvegardé votre travail lors de l'étape précédente pour ne pas perdre
votre travail.
Sollicitez le professeur, car il est possible que votre contrôle continue en soit affecté.

Sinon, annulez toutes vos modifications avec `git reset --hard HEAD`. Supprimez potentiellement les fichiers
non indexés.
Changez maintenant de branche avec `git switch step07`.
Créez désormais une branche avec votre nom : `git switch -c votrenom/step07`

## Qu'est-ce que c'est ?
## À quoi ça sert ?
## Comment ça fonctionne ?
## Manipulation sur OpenShift
## Cloud act, cloud souverain et réversibilité


## 4 - Orchestrer nos conteneurs avec Kubernetes

Pourquoi devrions-nous orchestrer nos conteneurs ?

Vous pouvez consulter la documentation ici : https://www.redhat.com/en/topics/containers/what-is-container-orchestration

Pour orchestrer nos conteneurs, nous pouvons utiliser Kubernetes. Cet outil est majeur sur le marché. Kubernetes a été initialement créé
par Google, développé en Go. Kubernetes est open source. Dans ce cours, nous utiliserons Openshift, qui est un Kubernetes avec
plus de fonctionnalités. Openshift est également open source et développé par RedHat.

Documentation sur Kubernetes ici : https://www.redhat.com/en/topics/containers/what-is-kubernetes

Votre environnement d'exécution de conteneurs est nommé Pod dans Kubernetes.
Kubernetes est un cluster. La réplication des Pods sur plusieurs nœuds est possible.

Pour déployer vos Pods, vous pouvez utiliser une ressource Kubernetes nommée Deployment. Dans cet objet, vous pouvez spécifier les images des conteneurs
qui composent votre Pod, les ressources pour chacun d'entre eux, les volumes, les replicas, les variables d'environnement...

Les replicas vous permettent d'augmenter le nombre de vos pods afin de traiter plus de requêtes en même temps.

Pour rendre votre service disponible sur le réseau, vous devez créer une autre ressource Kubernetes qui est nommée Service. Avec
elle, vous pouvez indiquer que votre service sera sollicité en http, vous pouvez lier le port de vos webservices, etc...

Enfin, vous pouvez créer une Route (spécifique à Openshift) afin de créer une URL appropriée pour votre webservice.

Pour créer ces ressources, nous pouvons les décrire dans des manifestes. Pour ce faire, nous allons créer un fichier yaml, `mlops-api.yaml`, dans un nouveau dossier,
`deploy`. Deploy doit être à la racine du projet.

![add_manifest.png](00_materials/10_kubernetes/add_manifest.png)

Voici une proposition de manifeste. Discutons-en :

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlops-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlops-api
  template:
    metadata:
      labels:
        app: mlops-api
    spec:
      containers:
        - name: mlops-api
          image: quay.io/gthomas59800/kto/mlops_python_2023_2024
          env:
            - name: OAUTH2_ISSUER
              value: https://dev-57n2oiz6kv1zyfh6.eu.auth0.com/
            - name: OAUTH2_AUDIENCE
              value: https://gthomas-cats-dogs.com
            - name: OAUTH2_JWKS_URI
              value: .well-known/jwks.json
          ports:
            - containerPort: 8080
          resources:
            limits:
              memory: "1000Mi"
              cpu: "200m"
            requests:
              memory: "500Mi"
              cpu: "200m"
---
apiVersion: v1
kind: Service
metadata:
  name: mlops-api-service
spec:
  selector:
    app: mlops-api
  ports:
    - port: 8080
      name: http-port
      targetPort: 8080
---
kind: Route
apiVersion: route.openshift.io/v1
metadata:
  name: mlops-api
spec:
  to:
    kind: Service
    name: mlops-api-service
    weight: 100
  port:
    targetPort: http-port
  tls:
    termination: edge
    insecureEdgeTerminationPolicy: None
  wildcardPolicy: None
```

Maintenant, nous allons utiliser notre manifeste afin de déployer notre application dans le Cloud. Pour ce faire, nous avons besoin d'un Kubernetes disponible dans le Cloud.

RedHat offre à tous les développeurs un bac à sable OpenShift de développement gratuit, disponible dans le Cloud. Ce bac à sable est disponible 30 jours et est supprimé automatiquement.

Vous pouvez recréer un nouveau bac à sable gratuitement après cette suppression. Alors maintenant, créons notre bac à sable !

Utilisez votre compte développeur RedHat pour créer un Red Hat Developer Sandbox. Vous devez vous connecter sur : https://developers.redhat.com/

Maintenant, cliquez sur le menu Developer Sandbox et sur le bouton Explore the free Developer Sandbox.

![create openshift sandbox](00_materials/10_kubernetes/create%20an%20openshift%20sandbox.png)

Ensuite, cliquez sur le bouton Start your sandbox for free

![start your sandbox](00_materials/10_kubernetes/start%20your%20sandbox.png)

Vous pouvez vérifier votre compte avec la validation par téléphone si vous avez un accès restreint à vos e-mails

![phone validation](00_materials/10_kubernetes/using%20phone%20validation%20est%20possible.png)

Installer le client openshift dans notre Codespace (éléments trouvés ici https://docs.okd.io/4.10/cli_reference/openshift_cli/getting-started-cli.html):

```bash
mkdir oc
cd oc
curl https://mirror.openshift.com/pub/openshift-v4/clients/oc/latest/linux/oc.tar.gz --output oc.tar.gz
tar xvf oc.tar.gz
pwd
PATH=$PATH:/workspaces/MLOpsPython/oc
cd ../production/kubernetes
oc apply -f mlops-api.yaml
```

Maintenant ! Testons notre application dans le Cloud depuis Postman !

**Bravo, votre application fonctionne ! Veuillez me communiquer par mail la route vers votre service, ainsi qu'un jeton oAuth2, 
que je puisse le tester (évaluations).**

N'oubliez pas de supprimer votre projet d'openshift

```bash
oc delete -f mlops-api.yaml
```