# Troubleshooting

Cette page vous permettra de répondre à certains problèmes que vous pourriez rencontrer pendant la réalisation de ce projet.
Voici les problèmes connus et les moyens de les résoudre.

## Erreur 401 sur openshift dans mes github actions, que faire ?

Il n'est pas impossible que certaines fois, quand on lance la github action, l'erreur suivante apparaisse :

HTTP response body:
```json 
{
  "kind": "Status",
  "apiVersion": "v1",
  "metadata": {},
  "status": "Failure",
  "message": "Unauthorized",
  "reason": "Unauthorized",
  "code": 401
}
```

Il s'agit de votre jeton OnpenShift qui n'est plus valide. Il faut donc le [renouveler](07_ci.md). Reprenez l'étape
des variables d'environnement et les secrets dans GitHub, et renseignez un nouveau jeton dans le secret OPENSHIFT_TOKEN.
Vous trouverez le lien vers ce jeton ici, puis Display token :

![create_secrets_and_variables.png](00_materials/07_ci/create_secrets_and_variables.png)


## Ma Sandbox a été recréée ou supprimée

Si votre Sandbox a été supprimée, vous pouvez la recréer gratuitement en suivant les étapes définies dans
[l'introduction du cours](./01_intro.md#création-de-notre-compte-red-hat-et-provisionnement-de-notre-red-hat-developer-sandbox).
Notez que la Sandbox est valide pour 30 jours à partir de la date de création. Après cette période, vous devrez également la recréer.

Une fois que la Sandbox est recréée, veuillez suivre les étapes suivantes pour vous reconnecter à votre projet :

- Ouvrez votre Sandbox Red Hat Developer.

![197.png](./img/197.png)

- Accédez à la section Devspaces.
- Créez un nouveau Devspace en utilisant le repository GitHub de votre projet.
- Cliquez sur Create & Open.

![198.png](./img/198.png)

- Dites que vous faites confiance au repository en cliquant sur Continue.

![199.png](./img/199.png)

- Acceptez la connection vers GitHub en cliquant sur Authorize.

![200.png](./img/200.png)

- Votre Codespace s'ouvre alors avec votre projet.

![201.png](./img/201.png)

- Il vous faut reconfigurer votre git et git remote pour pointer vers votre repository GitHub personnel avec un jeton.
  Pour cela, utilisez les commande suivante en remplaçant `<votre_nom_utilisateur>`, `<votre_email>` et `<nom_du_repository>` par vos informations :

```bash
git remote set-url origin https://<votre_nom_utilisateur>:<votre_token>@github.com/<votre_nom_utilisateur>/<nom_du_repository>.git
git config user.name "<votre_nom_utilisateur>"
git config user.email "<votre_email>"
git config core.editor "vim"
```

![202.png](./img/202.png)
![203.png](./img/203.png)

- Réinstallez uv avec la commande :

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

![204.png](./img/204.png)

- Synchronisez votre projet avec la commande :

```bash
uv sync --all-groups
```

![205.png](./img/205.png)

- Réinstallez mlflow avec les commandes :

```bash
chmod -R 777 scripts
./scripts/install-mlflow.sh
```

![206.png](./img/206.png)

- Vous pouvez maintenant continuer à travailler sur votre projet.

![207.png](./img/207.png)

- Votre dataset a également été supprimé. Reportez vous à la section
  [Chargement du dataset dans votre Sandbox](./04_scoping_data_prep_label.md#procédure-de-sauvegarde)
  pour le recharger.

- Rejouez vos dernières github actions pour régénérer les artefacts de build et de déploiement.

![208.png](./img/208.png)
![209.png](./img/209.png)
![210.png](./img/210.png)
![211.png](./img/211.png)

## Dailyclean a visiblement éteint mon environnement tout seul

Il est possible qu'à partir de 18H, votre Sandbox soit mise en veille par le processus Dailyclean.
Vous pouvez vous en rendre compte si vous voyez un écran similaire à celui-ci depuis votre espace Openshift :

![219.png](./img/219.png)

Pour continuer à travailler, il vous suffit de redémarrer votre kto-mlflow avec Dailyclean. Suivez les étapes décrites [ici](./04_scoping_data_prep_label.md#présentation-de-dailyclean-et-comment-démarrer-kto-mlflow).

## Red Hat Developer Sandbox est lent, ça ne fonctionne pas ... Que faire ?

Malheureusement, cela peut arriver et ce n'est pas illogique. Il s'agit d'un cluster partagé pour expérimenter la solution.
Il ne s'agit en aucun cas d'une plateforme de production avec garantie de disponibilité. N'oubliez pas, cette Sandbox est 
gratuite.