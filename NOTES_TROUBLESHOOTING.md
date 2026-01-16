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