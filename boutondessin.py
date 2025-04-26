# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys # Needed for sys.platform if opening folder

# ---------------------------------------------------------------------------
# --- DÉBUT DE VOTRE CODE ORIGINAL (INTACT) ---
# ---------------------------------------------------------------------------
from random import *
import math
# import argparse # Pas nécessaire pour l'interface graphique simple
from PIL import Image, ImageDraw, ImageOps

# --- Importation des modules locaux ---
# Assurez-vous que ces fichiers .py sont dans le même dossier ou dans le PYTHONPATH
try:
    from filters import *
    from strokesort import *
    import perlin
    from util import * # Assurez-vous que distsum est ici si utilisé dans connectdots
except ImportError as e:
    # Utiliser print car messagebox n'est peut-être pas encore prêt si l'erreur est très tôt
    print(f"ERREUR CRITIQUE: Impossible d'importer un module local nécessaire : {e}\nAssurez-vous que filters.py, strokesort.py, perlin.py, et util.py sont présents et corrects.")
    # Essayer d'afficher une messagebox si Tkinter est chargé
    try:
        root = tk.Tk()
        root.withdraw() # Cacher la fenêtre principale inutile
        messagebox.showerror("Erreur d'Importation", f"Impossible d'importer un module local nécessaire : {e}\nVérifiez la console et assurez-vous que les fichiers .py sont présents.")
    except Exception:
        pass # Si Tkinter échoue aussi, l'erreur console suffira
    exit() # Arrêter l'exécution

no_cv = False
export_path = "output/out.svg" # Sera écrasé par main()
draw_contours = True
draw_hatch = True
show_bitmap = False
resolution = 1024
hatch_size = 1
contour_simplify = 1

# --- Définition des fonctions (votre code original) ---

def simplify_vectors(lines, tolerance=5.0):
    """
    Réduit le nombre de vecteurs en fusionnant ceux qui sont proches.
    (Votre code original ici)
    """
    if not lines:
        return lines
    print(f"Simplifying vectors from {len(lines)} polylines...")
    all_segments = []
    for polyline in lines:
        # Gérer le cas où une "polyline" est juste une paire de points
        if isinstance(polyline, (list, tuple)) and len(polyline) >= 2:
             # S'assurer que les éléments sont des paires (x,y)
             if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in polyline):
                 for i in range(len(polyline)-1):
                     all_segments.append((polyline[i], polyline[i+1]))
             else:
                 print(f"  Avertissement simplify_vectors: polyline invalide ignorée: {polyline}")
        else:
             print(f"  Avertissement simplify_vectors: polyline trop courte ou invalide ignorée: {polyline}")

    simplified = []
    while all_segments:
        current = all_segments.pop(0)
        # S'assurer que current est un tuple de deux points valides
        if not (isinstance(current, tuple) and len(current) == 2 and
                isinstance(current[0], (tuple, list)) and len(current[0]) == 2 and
                isinstance(current[1], (tuple, list)) and len(current[1]) == 2):
            print(f"  Avertissement simplify_vectors: segment initial invalide ignoré: {current}")
            continue

        merged = True
        while merged:
            merged = False
            i = 0
            while i < len(all_segments):
                segment = all_segments[i]
                # S'assurer que segment est aussi valide
                if not (isinstance(segment, tuple) and len(segment) == 2 and
                        isinstance(segment[0], (tuple, list)) and len(segment[0]) == 2 and
                        isinstance(segment[1], (tuple, list)) and len(segment[1]) == 2):
                    print(f"  Avertissement simplify_vectors: segment de comparaison invalide ignoré: {segment}")
                    i += 1
                    continue

                try:
                    dist_start_current = math.dist(current[0], segment[0])
                    # Attention: current peut devenir plus long que 2 points après fusion
                    # Utiliser current[-1] pour le dernier point
                    dist_end_current = math.dist(current[-1], segment[1])
                    dist_start_end = math.dist(current[0], segment[1])
                    dist_end_start = math.dist(current[-1], segment[0])
                except (TypeError, IndexError):
                    print(f"  Avertissement simplify_vectors: erreur de distance avec current={current}, segment={segment}")
                    i += 1
                    continue # Passer au segment suivant si calcul de distance échoue

                min_dist = min(dist_start_current, dist_end_current, dist_start_end, dist_end_start)

                if min_dist <= tolerance:
                    merged = True
                    segment_to_merge = all_segments.pop(i) # Retirer le segment fusionné

                    # Logique de fusion (adapter pour 'current' qui peut être > 2 points)
                    if min_dist == dist_start_current:
                         # segment inversé + current
                         current = tuple(reversed(segment_to_merge)) + current
                    elif min_dist == dist_end_current:
                         # current + segment
                         current = current + segment_to_merge
                    elif min_dist == dist_start_end:
                         # segment + current (inverser segment ?)
                         # current = segment + current[1:] # Potentiellement incorrect si current a > 2 pts
                         current = segment_to_merge + current # Fusion simple bout à bout
                    else:  # dist_end_start
                         # current + segment inversé
                         # current = current[:-1] + segment # Potentiellement incorrect
                         current = current + tuple(reversed(segment_to_merge)) # Fusion simple bout à bout

                    # Après une fusion, on recommence la recherche depuis le début de all_segments
                    # pour ce 'current' modifié, car il pourrait maintenant fusionner avec d'autres.
                    # Donc, on sort de la boucle interne 'while i < len(all_segments)' et
                    # la boucle 'while merged' refait une passe.
                    break # Sortir de la boucle interne (while i < ...)
                else:
                    i += 1 # Passer au segment suivant si pas de fusion
            # Si on a break (fusionné), la boucle 'while merged' continue
            # Si on a terminé la boucle 'while i < len(all_segments)' sans fusion (merged=False),
            # la boucle 'while merged' se termine.

        simplified.append(current)

    # Convertir les segments fusionnés (qui sont maintenant des tuples de points) en listes de points
    result = [list(seg) for seg in simplified if isinstance(seg, (list, tuple)) and len(seg) >= 2]

    print(f"Simplified to {len(result)} polylines")
    return result


try:
    import numpy as np
    import cv2
except ImportError: # Utiliser ImportError qui est plus spécifique
    print("Cannot import numpy/openCV. Switching to NO_CV mode.")
    no_cv = True

def find_edges(IM):
    print("finding edges...")
    global no_cv # Accéder à la variable globale
    # S'assurer que IM est bien une image PIL en mode 'L'
    if not isinstance(IM, Image.Image):
        print("Erreur find_edges: IM n'est pas une image PIL.")
        return Image.new("L", (100, 100), 0) # Retourner image noire
    if IM.mode != 'L':
        IM = IM.convert('L')

    if no_cv:
        print("  using NO_CV mode (Sobel from util/filters)")
        try:
            # Assurez-vous que appmask, F_SobelX, F_SobelY sont définis
            # Appliquer sur des copies pour éviter de modifier l'original si appmask le fait
            im_x = appmask(IM.copy(), [F_SobelX])
            im_y = appmask(IM.copy(), [F_SobelY])
            # Combiner les magnitudes (exemple)
            im_out = Image.new("L", IM.size)
            px_out = im_out.load()
            px_x = im_x.load()
            px_y = im_y.load()
            w, h = IM.size
            for y in range(h):
                for x in range(w):
                     magnitude = int(math.sqrt(px_x[x, y]**2 + px_y[x, y]**2))
                     px_out[x, y] = min(255, magnitude) # Clamper à 255
            IM = im_out
        except NameError as e:
             print(f"ERREUR: Fonction {e} non trouvée pour le mode NO_CV.")
             # Retourner image noire en cas d'erreur
             return Image.new("L", IM.size, 0)
        except Exception as e:
             print(f"ERREUR inattendu en mode NO_CV: {e}")
             return Image.new("L", IM.size, 0)
    else:
        print("  using CV mode (Canny)")
        try:
            im = np.array(IM)
            if im is None or im.size == 0:
                 print("Erreur find_edges: np.array(IM) a échoué ou est vide.")
                 return Image.new("L", IM.size, 0)
            im = cv2.GaussianBlur(im,(7,7),1.5)
            im = cv2.Canny(im,75,150) # Seuils pour Canny
            # Dilate/erode optionnel
            kernel = np.ones((2,2), np.uint8)
            im = cv2.dilate(im, kernel, iterations=1)
            im = cv2.erode(im, kernel, iterations=1)
            IM = Image.fromarray(im)
        except cv2.error as e:
             print(f"Erreur OpenCV dans find_edges: {e}")
             print("  Passage en mode NO_CV pour la suite.")
             no_cv = True # Forcer no_cv s'il y a une erreur ici
             # Tenter de relancer find_edges en mode no_cv
             return find_edges(IM) # Attention: récursion potentielle si no_cv échoue aussi
        except Exception as e:
             print(f"Erreur inattendue avec OpenCV: {e}")
             return Image.new("L", IM.size, 0) # Retourner image noire

    # Binariser le résultat
    return IM.point(lambda p: 255 if p > 128 else 0)

def getdots(IM):
    """ Extrait les segments horizontaux de pixels blancs. """
    print("getting contour points...")
    if not isinstance(IM, Image.Image) or IM.mode != 'L':
         print("Erreur getdots: IM invalide.")
         return []
    PX = IM.load()
    dots = []
    w,h = IM.size
    for y in range(h): # Itérer jusqu'à h-1 ou h? Original était h-1. Gardons h-1.
        row = []
        current_start = -1
        current_len = 0
        for x in range(w): # Itérer sur toute la largeur
            is_white = (PX[x,y] == 255)

            if is_white:
                if current_start == -1: # Début d'un segment
                    current_start = x
                    current_len = 1
                else: # Continuation d'un segment
                    current_len += 1
            else: # Pixel noir
                if current_start != -1: # Fin d'un segment précédent
                    # Format original: (start_x, 0) ou (start_x, length-1) ?
                    # Le code original ajoutait (x, 0) puis modifiait le 2e elem.
                    # Ceci suggère que le 2e elem était une longueur ou un index.
                    # Tentons de reproduire : (start_x, length-1)
                    row.append((current_start, current_len - 1))
                    current_start = -1 # Réinitialiser
                    current_len = 0

        # Gérer segment qui finit au bord droit
        if current_start != -1:
            row.append((current_start, current_len - 1))

        dots.append(row)
    return dots


def connectdots(dots):
    """ Connecte les points/segments. Code original. """
    print("connecting contour points...")
    contours = []
    if not dots: return contours # Gérer liste vide

    for y in range(len(dots)):
        # S'assurer que dots[y] est une liste avant d'itérer
        if not isinstance(dots[y], list):
             print(f"Avertissement connectdots: dots[{y}] n'est pas une liste, ignoré.")
             continue

        processed_indices_current_row = set() # Pour éviter de réutiliser un point/segment

        for idx, (x, v) in enumerate(dots[y]): # idx est l'index dans la ligne y
             # 'v' semble être la longueur-1 du segment démarrant en x.
             # Le point représentatif pourrait être le début x, ou le milieu x + v/2 ?
             # Le code original utilise juste x dans la comparaison abs(x0-x). Utilisons x.
             current_point = (x, y) # Le point de référence pour la connexion

             if y == 0: # Première ligne, chaque segment démarre un contour
                 contours.append([current_point])
                 processed_indices_current_row.add(idx)
             else:
                 # Chercher le point/segment le plus proche dans la ligne précédente (y-1)
                 closest_x0 = -1
                 cdist = 100 # Seuil de distance max? Ou juste une grande valeur initiale? Original = 100
                 best_prev_idx = -1 # Index du meilleur point/segment dans la ligne y-1

                 # S'assurer que dots[y-1] est valide
                 if y > 0 and isinstance(dots[y-1], list):
                     for prev_idx, (x0, v0) in enumerate(dots[y-1]):
                          dist = abs(x0 - x)
                          if dist < cdist:
                              # Est-ce que le point précédent était déjà connecté ?
                              # Difficile à savoir ici sans marquer les points utilisés dans les lignes précédentes.
                              # Le code original ne semble pas vérifier cela explicitement.
                              cdist = dist
                              closest_x0 = x0
                              best_prev_idx = prev_idx # On garde l'index du point x0 trouvé
                 # else: # Pas de ligne précédente valide

                 # Décider de la connexion
                 if cdist > 3: # Si trop loin (seuil original=3), démarrer un nouveau contour
                     contours.append([current_point])
                     processed_indices_current_row.add(idx)
                 else:
                     # Essayer de trouver un contour existant qui se termine au point (closest_x0, y-1)
                     found_contour = False
                     # Itérer sur les contours existants pour trouver celui qui finit au bon endroit
                     # C'est potentiellement lent si beaucoup de contours
                     for i in range(len(contours)):
                         # Vérifier si le contour n'est pas vide et si le dernier point correspond
                         if contours[i] and contours[i][-1] == (closest_x0, y - 1):
                             # S'assurer que ce point (closest_x0, y-1) n'a pas déjà servi
                             # à étendre un *autre* contour vers cette ligne y.
                             # Le code original ne vérifie pas cela explicitement, ce qui pourrait
                             # mener à des "branches" si un point (y-1) est proche de plusieurs points (y).
                             # On va suivre l'original pour l'instant.
                             contours[i].append(current_point)
                             found_contour = True
                             processed_indices_current_row.add(idx)
                             # Marquer le point (closest_x0, y-1) comme utilisé ? Non fait dans l'original.
                             break # Arrêter dès qu'on a trouvé un contour à étendre

                     if not found_contour: # Si aucun contour ne finissait là, en créer un nouveau
                         contours.append([current_point])
                         processed_indices_current_row.add(idx)

        # Nettoyage des contours courts à la fin de chaque ligne (original)
        # Attention: modifier une liste pendant qu'on itère dessus est risqué.
        # Il vaut mieux créer une nouvelle liste ou itérer à l'envers.
        # Faisons une copie pour l'itération.
        temp_contours = contours[:] # Copie superficielle
        contours = [] # Liste résultat
        for c in temp_contours:
            # Si le dernier point est bien avant y-1 ET le contour est très court (<4 points)
            if c and c[-1][1] < y - 1 and len(c) < 4:
                 pass # Ne pas ajouter ce contour à la liste résultat (équivaut à remove)
            else:
                 contours.append(c) # Garder le contour

    return contours


def getcontours(IM, sc=2):
    """ Génère les contours. Code original. """
    print("generating contours...")
    # S'assurer que IM est une image PIL
    if not isinstance(IM, Image.Image):
         print("Erreur getcontours: IM n'est pas une image PIL.")
         return []

    IM_edges = find_edges(IM.copy()) # Travailler sur une copie des bords
    if not IM_edges: return [] # find_edges a échoué

    # Approche originale avec double rotation
    IM1 = IM_edges
    try:
        IM2 = IM_edges.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    except Exception as e:
        print(f"Erreur lors de la rotation/transposition: {e}")
        IM2 = Image.new("L", (IM_edges.height, IM_edges.width), 0) # Image noire de taille inversée

    dots1 = getdots(IM1)
    contours1 = connectdots(dots1)
    dots2 = getdots(IM2)
    contours2_raw = connectdots(dots2)

    # Transformer les coordonnées de contours2
    contours2 = []
    for c_raw in contours2_raw:
         transformed_c = []
         valid = True
         for p in c_raw:
             if isinstance(p, (list, tuple)) and len(p) == 2:
                 transformed_c.append((p[1], p[0])) # Inverser x et y
             else:
                  print(f"Avertissement getcontours: point invalide dans contours2_raw ignoré: {p}")
                  valid = False
                  break
         if valid:
             contours2.append(transformed_c)

    contours = contours1 + contours2
    print(f"  Contours initiaux (avant fusion): {len(contours)}")

    # Fusion des contours si l'extrémité de l'un est proche du début de l'autre
    # Utilise distsum de util.py - assurez-vous qu'elle existe et fonctionne
    merged_indices = set()
    final_contours = []
    try:
        for i in range(len(contours)):
            if i in merged_indices: continue # Déjà fusionné dans un autre
            current_contour = contours[i][:] # Copie
            if not current_contour: continue # Ignorer contour vide

            merged_in_this_pass = True
            while merged_in_this_pass:
                 merged_in_this_pass = False
                 for j in range(len(contours)):
                     # Ne pas fusionner avec soi-même ou un contour déjà utilisé ou vide
                     if i == j or j in merged_indices or not contours[j]: continue

                     # Vérifier si la fin de 'current_contour' est proche du début de 'contours[j]'
                     if distsum(contours[j][0], current_contour[-1]) < 8: # Seuil original = 8
                          # Fusionner: ajouter contours[j] (sans son premier point?) à current_contour
                          # L'original fait current = current + contours[j], ce qui duplique le point de jonction.
                          # Gardons l'original pour l'instant.
                          current_contour.extend(contours[j]) # Ajouter tous les points de l'autre
                          merged_indices.add(j) # Marquer j comme utilisé
                          merged_in_this_pass = True # On a fusionné, refaire une passe
                          # print(f"    Fusionné contour {j} dans {i}")
                          break # Sortir de la boucle interne (for j) et refaire une passe

                 # Pourrait aussi vérifier si le début de current est proche de la fin de j, etc.
                 # L'original ne semble vérifier que fin(i) -> début(j).

            final_contours.append(current_contour) # Ajouter le contour potentiellement allongé

    except NameError:
         print("ERREUR: La fonction 'distsum' (de util.py?) n'est pas définie. Fusion de contours impossible.")
         # Retourner les contours non fusionnés comme fallback
         final_contours = [c for c in contours if c] # Filtrer les vides potentiels
    except Exception as e:
        print(f"Erreur pendant la fusion des contours: {e}")
        final_contours = [c for c in contours if c]

    contours = final_contours
    print(f"  Contours après fusion: {len(contours)}")


    # Sous-échantillonnage (original: prendre 1 point sur 8)
    sampled_contours = []
    for c in contours:
        if len(c) > 8 : # Garder au moins 1 point si moins de 8 ? Non, l'original filtre après.
             sampled_contours.append([c[j] for j in range(0, len(c), 8)])
        elif len(c) > 1: # Garder les contours courts mais > 1 point
             sampled_contours.append(c)
    contours = sampled_contours
    print(f"  Contours après sous-échantillonnage: {len(contours)}")

    # Filtrer les contours avec moins de 2 points (après échantillonnage)
    contours = [c for c in contours if isinstance(c, list) and len(c) > 1]
    print(f"  Contours après filtrage (<2 pts): {len(contours)}")


    # Mise à l'échelle (facteur 'sc', original=contour_simplify)
    scaled_contours = []
    for c in contours:
        scaled_c = []
        valid = True
        for v in c:
             if isinstance(v, (list, tuple)) and len(v) == 2:
                  try:
                      scaled_c.append((v[0] * sc, v[1] * sc))
                  except TypeError:
                       print(f"Avertissement getcontours: point non numérique lors de la mise à l'échelle: {v}")
                       valid = False
                       break
             else:
                  print(f"Avertissement getcontours: point invalide lors de la mise à l'échelle: {v}")
                  valid = False
                  break
        if valid:
            scaled_contours.append(scaled_c)
    contours = scaled_contours
    print(f"  Contours après mise à l'échelle: {len(contours)}")


    # Ajout de bruit Perlin (original)
    noisy_contours = []
    for i in range(len(contours)): # Utiliser l'index i pour Perlin
         c = contours[i]
         noisy_c = []
         valid = True
         for j in range(len(c)): # Utiliser l'index j pour Perlin
             p = c[j]
             if isinstance(p, (list, tuple)) and len(p) == 2:
                  try:
                      noise_x = 10 * perlin.noise(i * 0.5, j * 0.1, 1) # Amplitude originale = 10
                      noise_y = 10 * perlin.noise(i * 0.5, j * 0.1, 2)
                      noisy_c.append(
                          (int(p[0] + noise_x), int(p[1] + noise_y))
                      )
                  except Exception as e:
                       print(f"Erreur Perlin pour contour {i}, point {j}: {e}")
                       valid = False
                       break # Arrêter ce contour si Perlin échoue
             else:
                  print(f"Avertissement getcontours: point invalide avant Perlin: {p}")
                  valid = False
                  break
         if valid and len(noisy_c) > 1: # S'assurer qu'il reste au moins 2 points
             noisy_contours.append(noisy_c)

    contours = noisy_contours
    print(f"  Contours après bruit Perlin: {len(contours)}")

    return contours

def hatch(IM, sc=16):
    """ Génère les hachures. Code original. """
    print("hatching...")
    if not isinstance(IM, Image.Image) or IM.mode != 'L':
         print("Erreur hatch: IM invalide.")
         return []
    PX = IM.load()
    w,h = IM.size
    # Listes séparées pour différents types/angles de hachures (original)
    lg1 = [] # Hachures horizontales
    lg2 = [] # Hachures diagonales (type \)

    for x0 in range(w):
        for y0 in range(h):
            x = x0*sc
            y = y0*sc
            try:
                pixel_val = PX[x0,y0]
            except IndexError:
                print(f"Erreur hatch: Index hors limites ({x0},{y0}) pour taille ({w},{h})")
                continue # Ignorer ce pixel

            # Logique originale basée sur la valeur du pixel
            if pixel_val > 144: # Clair -> rien
                pass
            elif pixel_val > 64: # Moyen -> une ligne horizontale
                lg1.append([(x,y+sc/4),(x+sc,y+sc/4)])
            elif pixel_val > 16: # Sombre -> horiz + diag
                lg1.append([(x,y+sc/4),(x+sc,y+sc/4)])
                lg2.append([(x+sc,y),(x,y+sc)]) # Diag \
            else: # Très sombre -> deux horiz + diag
                lg1.append([(x,y+sc/4),(x+sc,y+sc/4)])
                lg1.append([(x,y+sc/2+sc/4),(x+sc,y+sc/2+sc/4)])
                lg2.append([(x+sc,y),(x,y+sc)])

    # Fusion des lignes contiguës (original)
    # Semble fusionner uniquement si la fin exacte de l'un == début exact de l'autre
    lines_to_process = [lg1, lg2] # Traiter les deux types de lignes
    merged_lines = []

    for k in range(len(lines_to_process)): # Pour chaque type (lg1, lg2)
         current_lines = lines_to_process[k][:] # Copie de la liste de lignes
         lines_in_group = [] # Pour stocker les lignes fusionnées de ce groupe

         # Marquer les lignes utilisées dans une fusion pour ne pas les réutiliser seules
         used_indices = set()

         for i in range(len(current_lines)):
             if i in used_indices or not current_lines[i]: continue # Ignorer si vide ou déjà fusionnée

             base_line = current_lines[i][:] # Commencer avec cette ligne (copie)

             merged_in_pass = True
             while merged_in_pass:
                 merged_in_pass = False
                 for j in range(len(current_lines)):
                     # Ne pas fusionner avec soi-même ou une ligne vide ou déjà utilisée
                     if i == j or j in used_indices or not current_lines[j]: continue

                     # Condition de fusion originale: fin(base_line) == début(ligne j)
                     if base_line and current_lines[j] and base_line[-1] == current_lines[j][0]:
                         # Fusionner en ajoutant la suite de la ligne j (sans le premier point)
                         base_line.extend(current_lines[j][1:])
                         used_indices.add(j) # Marquer j comme utilisée
                         merged_in_pass = True
                         # print(f"  Hatch fusion: ligne {j} dans {i} (groupe {k})")
                         # On ne sort pas de la boucle j, on continue de chercher d'autres fusions pour base_line
                 # Fin de la boucle for j

             # Après avoir essayé toutes les fusions possibles pour base_line
             lines_in_group.append(base_line)
             used_indices.add(i) # Marquer i comme utilisée (même si non fusionnée)

         merged_lines.extend(lines_in_group) # Ajouter les lignes (fusionnées ou non) de ce groupe

    lines = merged_lines
    print(f"  Hachures après fusion: {len(lines)}")

    # Filtrer les lignes vides qui auraient pu être créées (original 'lines[k] = [l ...]')
    lines = [l for l in lines if isinstance(l, list) and len(l) > 0]
    print(f"  Hachures après filtrage vide: {len(lines)}")

    # Ajout de bruit Perlin (original)
    noisy_lines = []
    for i in range(len(lines)):
        l = lines[i]
        noisy_l = []
        valid = True
        for j in range(len(l)):
            p = l[j]
            if isinstance(p, (list, tuple)) and len(p) == 2:
                try:
                    noise_x = sc * perlin.noise(i * 0.5, j * 0.1, 1) # Amplitude liée à sc? Original: int(...)
                    noise_y = sc * perlin.noise(i * 0.5, j * 0.1, 2)
                    # Le "-j" original sur Y
                    noisy_l.append(
                        (int(p[0] + noise_x), int(p[1] + noise_y - j))
                    )
                except Exception as e:
                    print(f"Erreur Perlin pour hachure {i}, point {j}: {e}")
                    valid = False
                    break
            else:
                 print(f"Avertissement hatch: point invalide avant Perlin: {p}")
                 valid = False
                 break
        if valid and len(noisy_l) > 1: # Garder seulement si valide et a au moins 2 points
            noisy_lines.append(noisy_l)

    lines = noisy_lines
    print(f"  Hachures après bruit Perlin: {len(lines)}")

    return lines


def sketch(path):
    """ Fonction principale de traitement. Code original (adapté pour chemins). """
    # Accès aux globales pour configuration
    global draw_contours, draw_hatch, resolution, contour_simplify, hatch_size, show_bitmap, export_path, no_cv

    print(f"--- Lancement de sketch pour: {path} ---")
    IM = None
    # L'original cherchait dans plusieurs endroits, simplifions: utiliser le chemin fourni
    try:
        IM = Image.open(path)
        print(f"  Image chargée: {path}")
    except FileNotFoundError:
        # Ne pas faire exit(0), renvoyer une exception pour que l'appelant gère
        print(f"ERREUR: Le fichier d'entrée '{path}' n'a pas été trouvé.")
        raise FileNotFoundError(f"Le fichier d'entrée '{path}' n'existe pas.")
    except Exception as e:
        print(f"ERREUR: Impossible d'ouvrir l'image '{path}': {e}")
        raise IOError(f"Impossible d'ouvrir ou lire le fichier image: {e}")

    w_orig, h_orig = IM.size
    if w_orig == 0 or h_orig == 0:
        raise ValueError("L'image d'entrée a une dimension nulle.")
    aspect_ratio = h_orig / w_orig

    # --- Prétraitement ---
    try:
        IM = IM.convert("L")
        IM = ImageOps.autocontrast(IM, 10) # Original = 10
    except Exception as e:
         raise ValueError(f"Erreur lors de la conversion/contraste: {e}")

    lines = [] # Pour collecter toutes les lignes générées

    # --- Génération des Contours (si activé) ---
    if draw_contours:
        print("\n  Génération des contours...")
        contour_res_w = resolution // contour_simplify
        contour_res_h = int(contour_res_w * aspect_ratio)
        if contour_res_w > 0 and contour_res_h > 0:
             try:
                 im_resized = IM.resize((contour_res_w, contour_res_h), Image.Resampling.LANCZOS) # Utiliser Lanczos pour meilleure qualité
                 contour_lines = getcontours(im_resized, contour_simplify) # sc = contour_simplify
                 lines.extend(contour_lines)
                 print(f"    Contours ajoutés: {len(contour_lines)}")
             except Exception as e:
                  print(f"    ERREUR pendant getcontours: {e}")
                  # Continuer sans les contours si ça échoue? Ou arrêter? Pour l'instant, on continue.
        else:
             print(f"    Avertissement: Résolution pour contours ({contour_res_w}x{contour_res_h}) invalide. Contours ignorés.")

    # --- Génération des Hachures (si activé) ---
    if draw_hatch:
        print("\n  Génération des hachures...")
        hatch_res_w = resolution // hatch_size
        hatch_res_h = int(hatch_res_w * aspect_ratio)
        if hatch_res_w > 0 and hatch_res_h > 0:
             try:
                 # NEAREST peut être préférable pour préserver les paliers de gris pour hatch
                 im_resized = IM.resize((hatch_res_w, hatch_res_h), Image.Resampling.NEAREST)
                 hatch_lines = hatch(im_resized, hatch_size) # sc = hatch_size
                 lines.extend(hatch_lines)
                 print(f"    Hachures ajoutées: {len(hatch_lines)}")
             except Exception as e:
                  print(f"    ERREUR pendant hatch: {e}")
                  # Continuer sans les hachures si ça échoue?
        else:
             print(f"    Avertissement: Résolution pour hachures ({hatch_res_w}x{hatch_res_h}) invalide. Hachures ignorées.")

    # --- Tri des lignes ---
    if lines:
        print("\n  Tri des lignes...")
        try:
            lines = sortlines(lines) # Utilise la fonction de strokesort.py
            print(f"    Tri terminé pour {len(lines)} lignes.")
        except NameError:
            print("    Avertissement: Fonction 'sortlines' non trouvée. Les lignes ne sont pas triées.")
        except Exception as e:
            print(f"    ERREUR pendant sortlines: {e}")
    else:
        print("\n  Aucune ligne générée (ni contour, ni hachure).")
        # Le SVG sera vide mais on continue pour créer le fichier.

    # --- Affichage Bitmap (si activé) ---
    if show_bitmap:
        print("\n  Affichage de l'aperçu bitmap...")
        try:
            # Utiliser la résolution cible pour l'affichage
            disp_w = resolution
            disp_h = int(resolution * aspect_ratio)
            if disp_h > 0:
                 disp = Image.new("RGB",(disp_w, disp_h),(255,255,255))
                 draw = ImageDraw.Draw(disp)
                 for l in lines:
                     # Vérifier que l est une liste/tuple de points valides
                     if isinstance(l, (list, tuple)) and len(l) >= 2:
                          valid_points = []
                          valid_line = True
                          for p in l:
                              if isinstance(p, (list, tuple)) and len(p) == 2:
                                   try:
                                       # Clamper les points pour être dans l'image
                                       x = max(0, min(disp_w - 1, float(p[0])))
                                       y = max(0, min(disp_h - 1, float(p[1])))
                                       valid_points.append((x, y))
                                   except (ValueError, TypeError):
                                        print(f"    Avertissement show_bitmap: point invalide ignoré {p}")
                                        valid_line = False; break
                              else: valid_line = False; break
                          if valid_line and len(valid_points) >= 2:
                               draw.line(valid_points,(0,0,0),1) # Trait fin pour aperçu
                 disp.show()
            else:
                 print("    Erreur show_bitmap: hauteur d'affichage nulle.")
        except Exception as e:
            print(f"    ERREUR pendant show_bitmap: {e}")

    # --- Écriture du fichier SVG ---
    print(f"\n  Écriture du fichier SVG vers: {export_path}")
    # S'assurer que le dossier de sortie existe
    output_dir = os.path.dirname(export_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"    Dossier de sortie créé: {output_dir}")
        except OSError as e:
            print(f"    ERREUR: Impossible de créer le dossier '{output_dir}': {e}")
            # Lever une exception pour arrêter proprement
            raise IOError(f"Impossible de créer le dossier de sortie '{output_dir}'")

    # Générer le contenu SVG
    svg_content = makesvg(lines) # Appel de la fonction originale

    # Écrire dans le fichier
    try:
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        print(f"  {len(lines)} lignes écrites dans le SVG.")
        print("--- Sketch terminé avec succès. ---")
    except Exception as e:
        print(f"  ERREUR lors de l'écriture du fichier SVG '{export_path}': {e}")
        raise IOError(f"Impossible d'écrire le fichier SVG '{export_path}': {e}")

    return lines # Retourner les lignes (même si pas utilisé par l'interface simple)


def makesvg(lines):
    """ Génère la chaîne SVG. Code original. """
    print("generating svg file...")
    out = '<svg xmlns="http://www.w3.org/2000/svg" version="1.1">\n' # Ajouter newline
    # Style par défaut (peut être mis dans l'entête SVG)
    default_style = 'stroke="black" stroke-width="1" fill="none"' # Largeur de 1 (original était 2 * 0.5)

    lines_written = 0
    for l in lines:
         # Vérifier que l est une liste/tuple non vide de points
         if isinstance(l, (list, tuple)) and len(l) >= 2:
             points_str_list = []
             valid_line = True
             for p in l:
                  # Vérifier que p est une paire de coordonnées
                  if isinstance(p, (list, tuple)) and len(p) == 2:
                      try:
                          # Original faisait * 0.5 - appliquons-le ici
                          x = round(float(p[0]) * 0.5, 2)
                          y = round(float(p[1]) * 0.5, 2)
                          if math.isfinite(x) and math.isfinite(y):
                             points_str_list.append(f"{x},{y}")
                          else: valid_line = False; break # Ignorer ligne avec NaN/Inf
                      except (ValueError, TypeError):
                           valid_line = False; break # Ignorer ligne avec point non numérique
                  else: valid_line = False; break # Ignorer ligne avec point mal formé

             if valid_line and len(points_str_list) >= 2:
                  points_str = " ".join(points_str_list) # Utiliser espace comme séparateur standard
                  out += f' <polyline points="{points_str}" {default_style} />\n'
                  lines_written += 1
         # else: Ignorer les lignes invalides ou trop courtes

    out += '</svg>'
    if lines_written < len(lines):
        print(f"  Avertissement makesvg: {len(lines) - lines_written} lignes invalides ou trop courtes ignorées.")
    return out


def main(input_path='IMG_0450.jpg', output_path='output/out.svg', show_bmp=False,
         no_contour=False, no_hatch=False, no_cv_mode=False,
         # Attention: les tailles n'étaient pas dans l'original 'main', elles étaient globales
         # Ajoutons-les pour pouvoir les configurer si besoin, mais avec les valeurs globales par défaut
         main_hatch_size=hatch_size, main_contour_simplify=contour_simplify, main_resolution=resolution):
    """ Fonction principale d'orchestration. Code original (adapté). """
    # Mise à jour des variables globales basée sur les arguments de main()
    global export_path, draw_hatch, draw_contours, hatch_size, contour_simplify, resolution, show_bitmap, no_cv

    # Valider et assigner les chemins
    if not isinstance(input_path, str) or not input_path:
        raise ValueError("Chemin d'entrée invalide fourni à main()")
    if not isinstance(output_path, str) or not output_path:
        raise ValueError("Chemin de sortie invalide fourni à main()")
    export_path = output_path # Définit où sketch va écrire

    # Assigner les booléens et tailles
    draw_hatch = not no_hatch
    draw_contours = not no_contour
    show_bitmap = show_bmp
    no_cv = no_cv_mode # Mettre à jour le flag no_cv global

    # Valider et assigner les tailles/résolution
    try:
         hatch_size = int(main_hatch_size)
         contour_simplify = int(main_contour_simplify)
         resolution = int(main_resolution)
         if hatch_size <= 0 or contour_simplify <= 0 or resolution <= 0:
             raise ValueError("Tailles/Résolution doivent être > 0")
    except (ValueError, TypeError) as e:
         raise ValueError(f"Paramètre numérique invalide dans main(): {e}")

    # Appel à la fonction sketch (qui utilise les globales mises à jour)
    print(f"Lancement de sketch via main() - Input: {input_path}, Output: {export_path}")
    print(f"Options - Contours: {draw_contours}, Hachures: {draw_hatch}, Res: {resolution}, Simp: {contour_simplify}, HatchSize: {hatch_size}, NoCV: {no_cv}")

    # L'appel à sketch peut lever des exceptions (FileNotFound, IOError, ValueError...)
    # Laisser l'appelant (l'interface Tkinter) gérer ces exceptions
    sketch(input_path)

    print("Processus 'main' terminé.") # sketch() affiche déjà son propre message de fin

# --- FIN DE VOTRE CODE ORIGINAL (INTACT) ---
# ---------------------------------------------------------------------------


# -------------------------------------------------------------------------
# --- DÉBUT DE L'INTERFACE TKINTER MINIMALE ---
# -------------------------------------------------------------------------

def run_conversion():
    """ Fonction appelée par le bouton Tkinter """
    input_path = filedialog.askopenfilename(
        title="Sélectionnez une image d'entrée",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Tous les fichiers", "*.*")]
    )
    if not input_path:
        lbl_status.config(text="Annulé: Pas de fichier d'entrée sélectionné.")
        return

    # Suggérer un nom de sortie basé sur l'entrée
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    suggested_svg_name = base_name + ".svg"
    initial_dir = "output" # Dossier par défaut

    output_path = filedialog.asksaveasfilename(
        title="Enregistrer le fichier SVG de sortie sous...",
        initialdir=initial_dir,
        initialfile=suggested_svg_name,
        defaultextension=".svg",
        filetypes=[("Fichier SVG", "*.svg"), ("Tous les fichiers", "*.*")]
    )
    if not output_path:
        lbl_status.config(text="Annulé: Pas de fichier de sortie sélectionné.")
        return

    # Mettre à jour le statut et désactiver le bouton
    lbl_status.config(text=f"Traitement en cours...\nEntrée: {os.path.basename(input_path)}\nSortie: {os.path.basename(output_path)}")
    btn_run.config(state=tk.DISABLED)
    root.update_idletasks() # Forcer l'affichage du nouveau statut

    # --- Appel de votre fonction 'main' originale ---
    try:
        # Appeler main avec les chemins et les options par défaut souhaitées
        # Par exemple, pour correspondre à votre dernier appel original `no_hatch=True`:
        main(
            input_path=input_path,
            output_path=output_path,
            no_hatch=True, # Garder cette option comme dans votre exemple
            # Les autres options utiliseront les valeurs globales par défaut
            # ou celles définies dans la fonction main elle-même si vous les laissez
            no_contour=False, # Valeur par défaut
            show_bmp=False,  # Valeur par défaut
            no_cv_mode=no_cv # Utilise la valeur détectée au démarrage
            # Les tailles utiliseront les globales initiales: hatch_size=16, contour_simplify=10, resolution=1024
        )
        # Si main() réussit sans exception:
        messagebox.showinfo("Succès", f"Conversion terminée !\nSVG enregistré sous:\n{output_path}")
        lbl_status.config(text="Terminé avec succès.")
        # Optionnel: ouvrir le dossier de sortie
        try:
            output_dir_final = os.path.dirname(output_path)
            if os.path.exists(output_dir_final):
                 if os.name == 'nt': # Windows
                     os.startfile(output_dir_final)
                 elif sys.platform == 'darwin': # macOS
                     os.system(f'open "{output_dir_final}"')
                 else: # Linux
                     os.system(f'xdg-open "{output_dir_final}"')
        except Exception as e:
            print(f"Impossible d'ouvrir le dossier de sortie: {e}")

    except (FileNotFoundError, IOError, ValueError) as e:
        # Gérer les erreurs attendues (fichier non trouvé, problème lecture/écriture, param invalide)
        print(f"ERREUR gérée pendant la conversion: {e}")
        messagebox.showerror("Erreur", f"Une erreur est survenue:\n{e}")
        lbl_status.config(text=f"Erreur: {e}")
    except Exception as e:
        # Gérer les autres erreurs inattendues (ex: dans Perlin, OpenCV, PIL...)
        print(f"ERREUR INATTENDUE pendant la conversion: {e}")
        import traceback
        traceback.print_exc() # Imprimer la trace complète dans la console
        messagebox.showerror("Erreur Inattendue", f"Une erreur imprévue est survenue:\n{e}\nConsultez la console pour les détails.")
        lbl_status.config(text="Erreur inattendue (voir console).")

    finally:
        # Réactiver le bouton dans tous les cas (succès ou échec)
        btn_run.config(state=tk.NORMAL)


# --- Configuration de la fenêtre principale Tkinter ---
root = tk.Tk()
root.title("Convertisseur Image vers SVG (Original)")
root.geometry("400x150") # Taille simple

# Frame principal pour l'organisation
frame = tk.Frame(root, padx=10, pady=10)
frame.pack(expand=True, fill=tk.BOTH)

# Bouton pour lancer la conversion
btn_run = tk.Button(frame, text="Charger Image et Convertir en SVG", command=run_conversion, width=30, height=2)
btn_run.pack(pady=10)

# Label pour afficher le statut
lbl_status = tk.Label(frame, text="Prêt.", justify=tk.LEFT, wraplength=380)
lbl_status.pack(pady=10, fill=tk.X)

# --- Lancement de la boucle principale Tkinter ---
# Assurez-vous que l'appel original à main() à la fin de votre script est supprimé ou commenté
# Ex: # main(input_path='IMG_0450.jpg', output_path='output/out.svg',no_hatch=True)



# -------------------------------------------------------------------------
# --- FIN DE L'INTERFACE TKINTER MINIMALE ---
# -------------------------------------------------------------------------