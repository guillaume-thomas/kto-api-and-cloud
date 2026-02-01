# Section d'aide en cas de problèmes techniques

## Ma Sandbox a été recréée ou supprimée

Si votre Sandbox a été supprimée, vous pouvez la recréer gratuitement en suivant les étapes définies dans 
[l'introduction du cours](./courses/01_intro.md#création-de-notre-compte-red-hat-et-provisionnement-de-notre-red-hat-developer-sandbox). 
Notez que la Sandbox est valide pour 30 jours à partir de la date de création. Après cette période, vous devrez également la recréer.

Une fois que la Sandbox est recréée, veuillez suivre les étapes suivantes pour vous reconnecter à votre projet :

- Ouvrez votre Sandbox Red Hat Developer.

![197.png](./courses/img/197.png)

- Accédez à la section Devspaces.
- Créez un nouveau Devspace en utilisant le repository GitHub de votre projet.
- Cliquez sur Create & Open.

![198.png](./courses/img/198.png)

- Dites que vous faites confiance au repository en cliquant sur Continue.

![199.png](./courses/img/199.png)

- Acceptez la connection vers GitHub en cliquant sur Authorize.

![200.png](./courses/img/200.png)

- Votre Codespace s'ouvre alors avec votre projet.

![201.png](./courses/img/201.png)

- Il vous faut reconfigurer votre git et git remote pour pointer vers votre repository GitHub personnel avec un jeton. 
Pour cela, utilisez les commande suivante en remplaçant `<votre_nom_utilisateur>`, `<votre_email>` et `<nom_du_repository>` par vos informations :

```bash
git remote set-url origin https://<votre_nom_utilisateur>:<votre_token>@github.com/<votre_nom_utilisateur>/<nom_du_repository>.git
git config user.name "<votre_nom_utilisateur>"
git config user.email "<votre_email>"
git config core.editor "vim"
```

![202.png](./courses/img/202.png)
![203.png](./courses/img/203.png)

- Réinstallez uv avec la commande :

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

![204.png](./courses/img/204.png)

- Synchronisez votre projet avec la commande :

```bash
uv sync --all-groups
```

![205.png](./courses/img/205.png)

- Réinstallez mlflow avec les commandes : 

```bash
chmod -R 777 scripts
./scripts/install-mlflow.sh
```

![206.png](./courses/img/206.png)

- Vous pouvez maintenant continuer à travailler sur votre projet.

![207.png](./courses/img/207.png)

- Votre dataset a également été supprimé. Reportez vous à la section 
[Chargement du dataset dans votre Sandbox](./courses/04_scoping_data_prep_label.md#procédure-de-sauvegarde) 
pour le recharger.

- Rejouez vos dernières github actions pour régénérer les artefacts de build et de déploiement.

![208.png](./courses/img/208.png)
![209.png](./courses/img/209.png)
![210.png](./courses/img/210.png)
![211.png](./courses/img/211.png)

## Dailyclean a visiblement éteint mon environnement tout seul

Il est possible qu'à partir de 18H, votre Sandbox soit mise en veille par le processus Dailyclean.
Vous pouvez vous en rendre compte si vous voyez un écran similaire à celui-ci depuis votre espace Openshift :

![219.png](./courses/img/219.png)

Pour continuer à travailler, il vous suffit de redémarrer votre kto-mlflow avec Dailyclean. Suivez les étapes décrites [ici](./courses/04_scoping_data_prep_label.md#présentation-de-dailyclean-et-comment-démarrer-kto-mlflow).