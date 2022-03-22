# R&T Reconstruction de surface 3D, texturée et sémantisée

## Connexion au dépôt Gitlab CNES depuis l'extérieur

Voir la [Documentation Gitlab CNES](https://confluence.cnes.fr/pages/viewpage.action?pageId=26159013) sur 
Confluence pour plus d'information.

Toute connexion doit commencer par:

    ssh-agent /bin/bash
    ssh-add ~/.ssh/gitlab-ssh-cnes-fr

Pour clôner un répertoire, il faut utiliser la commande suivante:

    git clone gu=<user.name>@git@gitlab-ssh.cnes.fr:<groug-name>/<project-slug>.git