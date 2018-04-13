#!/usr/bin/env python

from __future__ import print_function
import re
from PRO_Rebouchage import *
from PRO_Medicis import *
from PRO_DecMoy import *
from PRO_Stats import *
from BibOrionXmodule import *


def runProMedicis(images, outputs, window, initial_disp, disp_range, pas={'y':1,'x':1}, roi=None, mask_value=None):
    # init
    sp_medicis = PME_Init()

    PME_SetParam(sp_medicis, "imaref_in", str(images['ref']))
    PME_SetParam(sp_medicis, "imasec_in", str(images['sec']))
    PME_SetParam(sp_medicis, "flog_out", str(outputs['log']))
    PME_SetParam(sp_medicis, "fconfi_out", str(outputs['log_params']))
    PME_SetParam(sp_medicis, "gri_out", str(outputs['res']))

    PME_SetParam_v(sp_medicis, "NOMBRE_DE_PROCESSUS", 1)

    PME_SetParam_v(sp_medicis, "PAS_GRILLE_LIGNE", pas['y'])
    PME_SetParam_v(sp_medicis, "PAS_GRILLE_COLONNE", pas['x'])
    if roi:
        PME_SetParam(sp_medicis, "GRILLE_REGULIERE", 'EXTRAIT')
        PME_SetParam_v(sp_medicis, "PREMIER_POINT_GRILLE_LIGNE", roi['y'])
        PME_SetParam_v(sp_medicis, "PREMIER_POINT_GRILLE_COLONNE", roi['x'])
        PME_SetParam_v(sp_medicis, "NB_LIGNE_GRILLE", roi["h"])
        PME_SetParam_v(sp_medicis, "NB_COLONNE_GRILLE", roi["w"])

    PME_SetParam_v(sp_medicis, "ZONE_CORREL_NB_LI", window['y'])
    PME_SetParam_v(sp_medicis, "ZONE_CORREL_NB_CO", window['x'])
    PME_SetParam_v(sp_medicis, "DEC_MOY_NB_LI", initial_disp['y'])
    PME_SetParam_v(sp_medicis, "DEC_MOY_NB_CO", initial_disp['x'])
    PME_SetParam_v(sp_medicis, "NB_LI_EXPLORATION", disp_range['y'])
    PME_SetParam_v(sp_medicis, "NB_CO_EXPLORATION", disp_range['x'])

    PME_SetParam(sp_medicis, "TYPE_METHODE", 'CORREL_LINEAIRE')
    PME_SetParam(sp_medicis, "CALCUL_CORREL", 'CORRELATION_SPATIALE')
    PME_SetParam(sp_medicis, "METHODE_CORREL", 'RECHERCHE_ITERATIVE_PAR_REECHANT')
    PME_SetParam(sp_medicis, "IMAGE_A_REECHANT", 'IMAGE_RECHERCHE')
    PME_SetParam(sp_medicis, "TYPE_RECHERCHE", 'DICHOTOMIE')
    PME_SetParam_v(sp_medicis, "PRECISION_LOC", 0.125)

    if mask_value:
        PME_SetParam(sp_medicis, "MASQUE_A_NODATA", "LISTE_VALEURS")
        PME_SetParam_v(sp_medicis, "NOMBRE_D_ELEMENTS", 1)
        PME_SetParam(sp_medicis, "VALEURS_A_NODATA", str(mask_value))

    # run
    PME_Run(sp_medicis)

    # end
    PME_Free(sp_medicis)


def runProStats(grid, outputs):
    # init
    sp_stats = PST_Init()

    PST_SetParam(sp_stats, "gri_in", str(grid))
    PST_SetParam(sp_stats, "flog_out", str(outputs['cr']))
    PST_SetParam(sp_stats, "fstats_out", str(outputs['res']))

    # run
    PST_Run(sp_stats)

    # end
    PST_Free(sp_stats)

    # get back information needed (mean disp, std, and percentage of valid points)
    with open(outputs['res'], 'r') as file:
        content = file.readlines()
    mean={}
    std={}
    percent=0.
    for line in content:
        if line.startswith('<MOY_DEC_LIG>'):
            mean['y'] = float(re.findall(r"[-+]?\d*\.\d+|\d+",line)[0])
        if line.startswith('<MOY_DEC_COL>'):
            mean['x'] = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
        if line.startswith('<ECART_TYPE_DEC_LIG>'):
            std['y'] = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
        if line.startswith('<ECART_TYPE_DEC_COL>'):
            std['x'] = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
        if line.startswith('<POURCENTAGE_POINTS_VALIDES>'):
            percent = int(re.findall(r"[-+]?\d+", line)[0])

    return mean, std, percent


def runProDecMoy(inputs, outputs, initial_disp={'y':0,'x':0}, roi=None):
    #init
    sp_decmoy = PDM_Init()

    PDM_SetParam(sp_decmoy, "imaref_in",  str(inputs['ref']))
    PDM_SetParam(sp_decmoy, "imasec_in", str(inputs['sec']))

    PDM_SetParam(sp_decmoy, "flog_out", str(outputs['log']))
    PDM_SetParam(sp_decmoy, "fconfi_out", str(outputs['log_params']))

    PDM_SetParam_v(sp_decmoy, "REDUC_LI", 1)
    PDM_SetParam_v(sp_decmoy, "REDUC_CO", 1)
    PDM_SetParam_v(sp_decmoy, "DEC_MOY_NB_LI", initial_disp['y'])
    PDM_SetParam_v(sp_decmoy, "DEC_MOY_NB_CO", initial_disp['x'])

    if roi:
        PDM_SetParam(sp_decmoy, "MODE_OPERATOIRE", 'DEUX_EXTRAITS')
        PDM_SetParam_v(sp_decmoy, "EXTRAIT_PREMIERE_LIGNE_IMAREF", roi['y'])
        PDM_SetParam_v(sp_decmoy, "EXTRAIT_PREMIERE_COLONNE_IMAREF", roi['x'])
        PDM_SetParam_v(sp_decmoy, "EXTRAIT_NOMBRE_LIGNES", roi["h"])
        PDM_SetParam_v(sp_decmoy, "EXTRAIT_NOMBRE_COLONNES", roi["w"])
        PDM_SetParam_v(sp_decmoy, "EXTRAIT_PREMIERE_LIGNE_IMASEC", roi['y'])
        PDM_SetParam_v(sp_decmoy, "EXTRAIT_PREMIERE_COLONNE_IMASEC", roi['x'])


    #run
    PDM_Run(sp_decmoy)

    #end
    PDM_Free(sp_decmoy)


def runProRebouchage(inputs, outputs):
    #init
    sp_rebouchage = PRE_Init()

    PRE_SetParam(sp_rebouchage, "gri_in",  inputs['image'])
    PRE_SetParam(sp_rebouchage, "gri_out", outputs['image'])

    PRE_SetParam(sp_rebouchage, "flog_out", outputs['log'])
    PRE_SetParam(sp_rebouchage, "fconfi_out", outputs['log_params'])

    PRE_SetParam(sp_rebouchage, "NATURE_GRILLE", 'IMAGE')
    PRE_SetParam_v(sp_rebouchage, "NOMBRE_MASQUES", 1)
    PRE_SetParam(sp_rebouchage, "LISTE_NOMBRE_VALEURS_PAR_MASQUE", "1")
    PRE_SetParam(sp_rebouchage, "TABLEAU_VALEURS_MASQUEES", inputs['noDataValue'])
    PRE_SetParam_v(sp_rebouchage, "DISTANCE_MAX_RECHERCHE", 10)
    PRE_SetParam_v(sp_rebouchage, "NOMBRE_FILTRAGES_HOMOGENEISATION", 0)

    #run
    PRE_Run(sp_rebouchage)

    #end
    PRE_Free(sp_rebouchage)


def runOrion_translate(images, translate, type_out="REEL_D"):
    #init
    orion = BibOrionX()

    #set param
    orion.set_param("FORMAT_SOURCE", "AUTO")
    orion.set_param("MODE_GESTION_SOURCE", "BANDEAU")
    orion.set_param("FORMAT_CIBLE", "TIFF")
    orion.set_param("TYPE_DATA_CIBLE", type_out)
    orion.set_param("TYPE_RADIO", "BCO")
    orion.set_param("MODE_GESTION_BORD", "MIROIR")
    orion.set_param_v("EXPORT_GEOREF", 1)

    #run
    orion.gene(str(images['input']),
               str(images['output']),
               translate['row'], translate['col'],
               1, 0, 0, 1,
               True)