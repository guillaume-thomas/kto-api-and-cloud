ATTENTION de bien préciser quand on créé le remote add qu'il faut mettre le token dans l'URL pour que ça fonctionne sans demander le mot de passe à chaque fois.

ATTENTION donc, si on doit refaire un git clone, de bien penser à mettre le token dans l'URL du clone également.
Exemple : git clone https://<TOKEN>@github.com/username/repo.git

Ajouter dans le cours l'exclusion à postériori du dossier scripts dans le .gitignore si on l'a déjà commit.
Exemple :
```bash
echo "
scripts/
" >> .gitignore
git add .gitignore
git rm -r --cached scripts/
git commit -m "feat: Supprime le répertoire scripts du suivi Git."
git push
```

Pour ne pas le perdre : 

- Pour poursuivre, nous allons ajouter en variable d'environnement, les informations de connexion à minio. Pour cela, créez
  un nouveau script dans le répertoire ./scripts avec le contenu suivant :
```bash
export MINIO_ENDPOINT_URL=$(oc get route minio-api -o jsonpath='{.spec.host}')
export MINIO_ACCESS_KEY=minio
export MINIO_SECRET_KEY=minio123
```

![169.png](img/169.png)
![170.png](img/170.png)
![171.png](img/171.png)

- Exécutez ce script dans votre terminal avec les commandes suivantes :
```bash
chmod 777 ./scripts/export-environment-variables.sh
source ./scripts/export-environment-variables.sh
```

![172.png](img/172.png)