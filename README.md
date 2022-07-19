# R&T Reconstruction de surface 3D, texturée et sémantisée

## Données

### Simu3D
Les données utilisées sont tirées du modèle 3D texturé de Nanterre (`/work/OT/siqi/simu3D/data/mesh/Nanterre15cm_ECEF/`). 
Ces dernières sont injectées dans la [chaîne complète de Simu3D](https://gitlab.cnes.fr/ctsiqi/Simulation_3D_complete).

La configuration utilisée est la suivante:

```json
{
  "steps": {
    "enable_ibg": true,
    "enable_csi": true,
    "enable_mnsref": true,
    "enable_rpc": true,
    "enable_cars": false,
    "enable_demcompare": false
  },
  "config_ibg": {
    "mesh": "/work/OT/siqi/simu3D/data/mesh/Nanterre15cm_ECEF/Nanterre.obj",
    "models": {
      "PDV1": {
        "bloc": "/work/OT/siqi/simu3D/data/modeles_geo/NANTERRE/PHR/BLOC_STEREO/",
        "id_scene": "P1AP--2013072139324950CP",
        "step": 0.25
      },
      "PDV2": {
        "bloc": "/work/OT/siqi/simu3D/data/modeles_geo/NANTERRE/PHR/BLOC_STEREO/",
        "id_scene": "P1AP--2013072139303958CP",
        "step": 0.25
      }
    },
    "parameters": {"clean_tmp": false, "tile_size": 200, "generate_ibg": true},
    "pbs": {
      "nb_procs": "8",
      "memory": "80gb",
      "nb_nodes": "8",
      "walltime": "04:59:00"
    }
  },
  "config_mnsref": {
    "crs": "EPSG:32631",
    "parameters": {"tile_size": 100},
    "pbs": {
      "nb_procs": "12",
      "memory": "120gb",
      "nb_nodes": "12",
      "walltime": "04:59:00"
    }
  },

  "config_csi": {
    "PF": {
      "PF_REECH": {
        "path": "/work/scratch/thenozc/rtsurface3d/20220309_formation_simu3d/configs/conf_CSI_template_reech4_quantif.cmdfile"
      }
    },
    "band_order": {"B0": 1, "B1": 2, "B2": 3},
    "lance_simili_path": "/softs/projets/outils_qi/simu3Dcomplete/lance_simili.sh"
  },
  "simulation_name": "Nanterre15cm",
  "resolution_pdv": 0.5,
  "out_dir": "/work/scratch/thenozc/rtsurface3d/20220309_formation_simu3d/test/nanterre_2"
}
```

Simu3D permet d'obtenir les images en géométrie capteur ainsi que les RPC qui peuvent ensuite être fournis à CARS pour la stéréorestitution.
Toutefois, la texture du modèle 3D de Nanterre ne contient que les bandes RGB. Les images produites
par Simu3D sont alors à 50cm en RGB. Il manque la bande NIR pour être au plus près de données Pléiades.

Une étape de passage en nuances de gris est nécessaire pour fournir en entrée de CARS une paire d'images de type "panchromatique".
La combinaison linéaire suivante est utilisée (c'est celle utilisée par `scikit-image`):

    Grayscale = 0.2125 * R + 0.7154 * G + 0.0721 * B

CARS (version `0.4.0`) est lancé sur ces données (`prepare` et `compute_dsm` en mode `mp`). Les scripts
du répertoire [cars-pctolas](https://gitlab.cnes.fr/cars/tools/cars-pctolas) sont ensuite utilisés pour obtenir un seul nuage de points au format `las`.

La dernière transformation consiste à passer les points du nuage du repère cartésien au repère projeté UTM adéquate.


### PHR
Des données PHR sont utilisées en guise de comparaison.
Ces dernières sont situées ici: `/work/OT/siaa/3D/Development/rt_mesh_3d/data/Nanterre/PHR`

## Connexion au dépôt Gitlab CNES depuis l'extérieur

Voir la [Documentation Gitlab CNES](https://confluence.cnes.fr/pages/viewpage.action?pageId=26159013) sur 
Confluence pour plus d'information.

Toute connexion doit commencer par:

    ssh-agent /bin/bash
    ssh-add ~/.ssh/gitlab-ssh-cnes-fr

Pour clôner un répertoire, il faut utiliser la commande suivante:

    git clone gu=<user.name>@git@gitlab-ssh.cnes.fr:<groug-name>/<project-slug>.git
