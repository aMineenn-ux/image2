import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from svgpathtools import svg2paths
import os
import traceback
import time
import main 
import math
import cinema1
from scipy.interpolate import interp1d
import matplotlib.animation as animation
from bisect import bisect_left
import shutil
import subprocess
import os

# DÉPENDANCES POUR L'APERÇU SVG ET IMAGES (Installer avec: pip install Pillow svglib reportlab)
try:
    from PIL import Image, ImageTk
except ImportError:
    messagebox.showerror("Erreur Dépendance", "La bibliothèque 'Pillow' est requise.\nInstallez-la avec : pip install Pillow")
    exit()

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
except ImportError:
    svg2rlg = None
    renderPM = None
    print("INFO: 'svglib' et 'reportlab' non trouvées. La fonction 'Voir SVG' sera indisponible.")
    print("Installez-les avec : pip install svglib reportlab")


# =============================================================================
# FONCTIONS DE TRAITEMENT SVG (Fournies par l'utilisateur)
# =============================================================================

def compute_curvature(points):
    """ Calcule la courbure pour chaque point basé sur l'angle entre les segments adjacents. """
    curvatures = [0] # Courbure nulle au premier point
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1] # Vecteur du point précédent au point actuel
        v2 = points[i + 1] - points[i] # Vecteur du point actuel au point suivant
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # Éviter division par zéro ou calculs sur des points identiques
        if norm_v1 == 0 or norm_v2 == 0:
            curvatures.append(0) # Pas de courbure si pas de mouvement
            continue

        # Calcul du cosinus de l'angle entre v1 et v2
        # S'assurer que le produit scalaire est dans [-1, 1] pour éviter les erreurs acos
        cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        angle = np.arccos(cos_theta) # Angle en radians
        curvatures.append(angle) # Angle comme mesure de courbure

    curvatures.append(0) # Courbure nulle au dernier point
    return np.array(curvatures)

def remove_consecutive_duplicates_xyz(df_in):
    """Supprime les lignes consécutives qui sont des doublons exacts sur X, Y et Z."""
    if df_in.empty:
        return df_in
    # Crée un masque booléen. True signifie que la ligne est différente de la suivante OU c'est la dernière ligne.
    mask = (df_in[['X', 'Y', 'Z']].shift(-1) != df_in[['X', 'Y', 'Z']]).any(axis=1)
    # S'assure de toujours garder la dernière ligne du DataFrame original
    mask.iloc[-1] = True
    # Applique le masque et réinitialise l'index
    return df_in[mask].reset_index(drop=True)
def process_svg_to_coordinates(filename):
    
    # =============================================
    # PARTIE ORIGINALE (identique au code initial)
    # =============================================
    
    # Paramètres pour les dimensions du mur (en cm)
    wall_width = 4000
    wall_height = 3000
    margin = 600
    min_points_per_segment = 3
    max_points_per_segment = 200
    curvature_threshold = 0.01

    # Charger les chemins SVG
    paths, attributes = svg2paths(filename)

    # Initialisation des listes
    all_coordinates = []
    points = []
    z_values = []
    previous_end = None

    # Calcul des limites du dessin
    for path in paths:
        for segment in path:
            segment_points = []
            for t in np.linspace(0, 1, num=100):
                point = segment.point(t)
                segment_points.append((point.real, point.imag))
            all_coordinates.append(np.array(segment_points))

    all_coordinates = np.concatenate(all_coordinates)
    x_min, x_max = np.min(all_coordinates[:, 0]), np.max(all_coordinates[:, 0])
    y_min, y_max = np.min(all_coordinates[:, 1]), np.max(all_coordinates[:, 1])

    # Fonction de normalisation
    def normalize_coordinates(coords, x_min, x_max, y_min, y_max):
        normalized_x = (coords[:, 0] - x_min) / (x_max - x_min) * (wall_width - 2 * margin) + margin
        normalized_y = (coords[:, 1] - y_min) / (y_max - y_min) * (wall_height - 2 * margin) + margin
        return np.column_stack((normalized_x, normalized_y))

    # Traitement des segments et détection des discontinuités
    for path in paths:
        for segment in path:
            segment_points = []
            for t in np.linspace(0, 1, num=100):
                point = segment.point(t)
                segment_points.append((point.real, point.imag))

            curvature = compute_curvature(np.array(segment_points))
            avg_curvature = np.mean(curvature)
            
            if avg_curvature < curvature_threshold:
                num_points = min_points_per_segment
            else:
                num_points = int(min_points_per_segment + (avg_curvature * (max_points_per_segment - min_points_per_segment)))
            
            num_points = min(num_points, max_points_per_segment)

            segment_points = []
            for t in np.linspace(0, 1, num=num_points):
                point = segment.point(t)
                segment_points.append((point.real, point.imag))

            segment_points = np.array(segment_points)
            normalized_segment_points = normalize_coordinates(segment_points, x_min, x_max, y_min, y_max)
            
            if previous_end is not None:
                discontinuity_distance = np.linalg.norm(normalized_segment_points[0] - previous_end)
                num_red_points = min(int(discontinuity_distance), 1)
                if num_red_points > 0:
                    midpoint = (previous_end + normalized_segment_points[0]) / 2
                    points.append(midpoint)
                    z_values.append(0)

            
            for point in normalized_segment_points:
                points.append(point)
                z_values.append(1)

            
            previous_end = normalized_segment_points[-1]

    points = np.array(points)
    z_values = np.array(z_values)

    df = pd.DataFrame(points, columns=['X', 'Y']).round(2)
    df['Z'] = z_values

    def remove_consecutive_duplicates(df):
        mask = (df['Z'] == 1) & (df[['X', 'Y']].shift(1) == df[['X', 'Y']]).all(axis=1)
        return df[~mask]

    df = remove_consecutive_duplicates(df)
    df.to_csv('coordinates_X_Y_Z_dynamic_with_discontinuity_no_duplicates.csv', index=False)


    # =============================================
    # PARTIE AJOUTÉE (optimisation des points)
    # =============================================
    
    def optimize_points(data):
        # Estimation de la longueur totale
        points = data[['X', 'Y']].to_numpy()
        distances = np.sqrt(np.diff(points[:, 0])**2 + np.diff(points[:, 1])**2)
        total_length = np.sum(distances)
        
        # Paramètre d'échelle (peut être ajusté)
        scaling_factor = 1
        target_points = int(total_length * scaling_factor)
        
        # Calcul de courbure avec tolérance
        def compute_curvature_optimized(points, tolerance=0.0001):
            curvatures = [0]
            for i in range(1, len(points) - 1):
                v1 = points[i] - points[i - 1]
                v2 = points[i + 1] - points[i]
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)

                if norm_v1 == 0 or norm_v2 == 0:
                    curvatures.append(0)
                    continue

                cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
                angle = np.arccos(cos_theta)

                if angle < tolerance:
                    curvatures.append(0)
                else:
                    curvatures.append(angle)

            curvatures.append(0)
            return np.array(curvatures)
        
        curvature = compute_curvature_optimized(points)
        z_values = data['Z'].to_numpy()
        
        # Protection des points Z=0 et leurs voisins
        z0_indices = np.where(z_values == 0)[0]
        protected_indices = set(z0_indices)
        
        for idx in z0_indices:
            if idx - 1 >= 0:
                protected_indices.add(idx - 1)
            if idx + 1 < len(z_values):
                protected_indices.add(idx + 1)
        
        protected_indices.add(0)
        protected_indices.add(len(data) - 1)
        
        # Calcul des poids pour les points Z=1
        z1_indices = np.where(z_values == 1)[0]
        weights = curvature[z1_indices] + 0.0001
        weights /= np.sum(weights)
        points_to_keep = np.round(weights * target_points).astype(int)
        
        # Sélection des points optimisés
        optimized_indices = []
        for i, num_points in enumerate(points_to_keep):
            segment_indices = z1_indices[i:i + 2]
            if len(segment_indices) > 1:
                segment_points = points[segment_indices[0]:segment_indices[1]]
                sampled_indices = np.linspace(0, len(segment_points) - 1, num=num_points, dtype=int)
                for idx in sampled_indices.tolist():
                    if (segment_indices[0] + idx) not in protected_indices:
                        optimized_indices.append(segment_indices[0] + idx)
        
        optimized_indices.extend(sorted(protected_indices))
        optimized_indices = sorted(optimized_indices)
        
        # Création du DataFrame optimisé
        optimized_data = data.iloc[optimized_indices]
        optimized_data = remove_consecutive_duplicates(optimized_data)
        
        # Vérification des points Z=0
        z0_count_original = np.sum(z_values == 0)
        z0_count_optimized = np.sum(optimized_data['Z'] == 0)
        
        if z0_count_original != z0_count_optimized:
            print(f"Avertissement : Nombre de points Z=0 changé : {z0_count_original} -> {z0_count_optimized}")
        
        return optimized_data
    
    # Application de l'optimisation
    optimized_df = optimize_points(df)
    def distance(p1, p2):
        """Calcule la distance euclidienne entre deux points."""
        return np.sqrt((p1['X'] - p2['X'])**2 + (p1['Y'] - p2['Y'])**2)
    
    filtered_points = [optimized_df.iloc[0]]  # Ajouter le premier point
    for i in range(1, len(optimized_df)):
        if distance(optimized_df.iloc[i], filtered_points[-1]) >= 50:
            filtered_points.append(optimized_df.iloc[i])
    
    # Création du DataFrame filtré
    optimized_df = pd.DataFrame(filtered_points).reset_index(drop=True)
    return optimized_df


def calculate_curvature(x1, y1, x2, y2, x3, y3):
    """
    Calcule la courbure d'un arc défini par trois points.
    """
    denom = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    if abs(denom) < 1e-8:
        return 0  # Ligne droite
    curvature = 2 * abs(denom) / (
        np.linalg.norm([x2 - x1, y2 - y1]) * np.linalg.norm([x3 - x2, y3 - y2]) * np.linalg.norm([x3 - x1, y3 - y1])
    )
    return curvature

def detect_critical_curvatures(drawn_points_x, drawn_points_y, curvature_threshold=0.01):
    """
    Identifie les segments avec des courbures graves et retourne une liste d'indices de segments.
    Chaque élément de la liste contient les indices des segments formant une courbure critique.
    """
    critical_curvatures = []
    critical_curvatures_values=[]
    current_group = []

    for i in range(1, len(drawn_points_x) - 1):
        curvature = calculate_curvature(
            drawn_points_x[i - 1], drawn_points_y[i - 1],
            drawn_points_x[i], drawn_points_y[i],
            drawn_points_x[i + 1], drawn_points_y[i + 1]
        )

        if curvature > curvature_threshold:
            current_group.append(i-1)
        else:
            if current_group:
                current_group.append(i-1)
                critical_curvatures.append(current_group)
                critical_curvatures_values.append(curvature)
                current_group = []

    if current_group:
        critical_curvatures.append(current_group)
        critical_curvatures_values.append(curvature)

    return (critical_curvatures,critical_curvatures_values)


def process_svg(filename):
    
    # =============================================
    # PARTIE ORIGINALE (identique au code initial)
    # =============================================
    
    # Paramètres pour les dimensions du mur (en cm)
    wall_width = 4000
    wall_height = 3000
    margin = 600
    min_points_per_segment = 3
    max_points_per_segment = 200
    curvature_threshold = 0.1

    # Charger les chemins SVG
    paths, attributes = svg2paths(filename)

    # Initialisation des listes
    all_coordinates = []
    points = []
    z_values = []
    previous_end = None

    # Calcul des limites du dessin
    for path in paths:
        for segment in path:
            segment_points = []
            for t in np.linspace(0, 1, num=100):
                point = segment.point(t)
                segment_points.append((point.real, point.imag))
            all_coordinates.append(np.array(segment_points))

    all_coordinates = np.concatenate(all_coordinates)
    x_min, x_max = np.min(all_coordinates[:, 0]), np.max(all_coordinates[:, 0])
    y_min, y_max = np.min(all_coordinates[:, 1]), np.max(all_coordinates[:, 1])

    # Fonction de normalisation
    def normalize_coordinates(coords, x_min, x_max, y_min, y_max):
        normalized_x = (coords[:, 0] - x_min) / (x_max - x_min) * (wall_width - 2 * margin) + margin
        normalized_y = (coords[:, 1] - y_min) / (y_max - y_min) * (wall_height - 2 * margin) + margin
        return np.column_stack((normalized_x, normalized_y))

    # Traitement des segments et détection des discontinuités
    for path in paths:
        for segment in path:
            segment_points = []
            for t in np.linspace(0, 1, num=100):
                point = segment.point(t)
                segment_points.append((point.real, point.imag))

            curvature = compute_curvature(np.array(segment_points))
            avg_curvature = np.mean(curvature)
            
            if avg_curvature < curvature_threshold:
                num_points = min_points_per_segment
            else:
                num_points = int(min_points_per_segment + (avg_curvature * (max_points_per_segment - min_points_per_segment)))
            
            num_points = min(num_points, max_points_per_segment)

            segment_points = []
            for t in np.linspace(0, 1, num=num_points):
                point = segment.point(t)
                segment_points.append((point.real, point.imag))

            segment_points = np.array(segment_points)
            normalized_segment_points = normalize_coordinates(segment_points, x_min, x_max, y_min, y_max)
            
            if previous_end is not None:
                discontinuity_distance = np.linalg.norm(normalized_segment_points[0] - previous_end)
                num_red_points = min(int(discontinuity_distance), 1)
                if num_red_points > 0:
                    midpoint = (previous_end + normalized_segment_points[0]) / 2
                    points.append(midpoint)
                    z_values.append(0)

            
            for point in normalized_segment_points:
                points.append(point)
                z_values.append(1)

            
            previous_end = normalized_segment_points[-1]

    points = np.array(points)
    z_values = np.array(z_values)

    df = pd.DataFrame(points, columns=['X', 'Y']).round(2)
    df['Z'] = z_values

    def remove_consecutive_duplicates(df):
        mask = (df['Z'] == 1) & (df[['X', 'Y']].shift(1) == df[['X', 'Y']]).all(axis=1)
        return df[~mask]

    df = remove_consecutive_duplicates(df)
    df.to_csv('coordinates_X_Y_Z_dynamic_with_discontinuity_no_duplicates.csv', index=False)


    # =============================================
    # PARTIE AJOUTÉE (optimisation des points)
    # =============================================
    
    def optimize_points(data):
        # Estimation de la longueur totale
        points = data[['X', 'Y']].to_numpy()
        distances = np.sqrt(np.diff(points[:, 0])**2 + np.diff(points[:, 1])**2)
        total_length = np.sum(distances)
        
        # Paramètre d'échelle (peut être ajusté)
        scaling_factor = 1
        target_points = int(total_length * scaling_factor)
        
        # Calcul de courbure avec tolérance
        def compute_curvature_optimized(points, tolerance=0.0001):
            curvatures = [0]
            for i in range(1, len(points) - 1):
                v1 = points[i] - points[i - 1]
                v2 = points[i + 1] - points[i]
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)

                if norm_v1 == 0 or norm_v2 == 0:
                    curvatures.append(0)
                    continue

                cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
                angle = np.arccos(cos_theta)

                if angle < tolerance:
                    curvatures.append(0)
                else:
                    curvatures.append(angle)

            curvatures.append(0)
            return np.array(curvatures)
        
        curvature = compute_curvature_optimized(points)
        z_values = data['Z'].to_numpy()
        
        # Protection des points Z=0 et leurs voisins
        z0_indices = np.where(z_values == 0)[0]
        protected_indices = set(z0_indices)
        
        for idx in z0_indices:
            if idx - 1 >= 0:
                protected_indices.add(idx - 1)
            if idx + 1 < len(z_values):
                protected_indices.add(idx + 1)
        
        protected_indices.add(0)
        protected_indices.add(len(data) - 1)
        
        # Calcul des poids pour les points Z=1
        z1_indices = np.where(z_values == 1)[0]
        weights = curvature[z1_indices] + 0.0001
        weights /= np.sum(weights)
        points_to_keep = np.round(weights * target_points).astype(int)
        
        # Sélection des points optimisés
        optimized_indices = []
        for i, num_points in enumerate(points_to_keep):
            segment_indices = z1_indices[i:i + 2]
            if len(segment_indices) > 1:
                segment_points = points[segment_indices[0]:segment_indices[1]]
                sampled_indices = np.linspace(0, len(segment_points) - 1, num=num_points, dtype=int)
                for idx in sampled_indices.tolist():
                    if (segment_indices[0] + idx) not in protected_indices:
                        optimized_indices.append(segment_indices[0] + idx)
        
        optimized_indices.extend(sorted(protected_indices))
        optimized_indices = sorted(optimized_indices)
        
        # Création du DataFrame optimisé
        optimized_data = data.iloc[optimized_indices]
        optimized_data = remove_consecutive_duplicates(optimized_data)
        
        # Vérification des points Z=0
        z0_count_original = np.sum(z_values == 0)
        z0_count_optimized = np.sum(optimized_data['Z'] == 0)
        
        if z0_count_original != z0_count_optimized:
            print(f"Avertissement : Nombre de points Z=0 changé : {z0_count_original} -> {z0_count_optimized}")
        
        return optimized_data
    
    # Application de l'optimisation
    optimized_df = optimize_points(df)
    
    optimized_df.to_csv('optimized_points.csv', index=False)
    
    if not optimized_df.iloc[-1].equals(df.iloc[-1]):
        print("Attention : Le dernier point de l'original n'est pas dans le fichier optimisé.")
    
    
    # Premier filtrage : Éliminer les points trop proches
    def distance(p1, p2):
        """Calcule la distance euclidienne entre deux points."""
        return np.sqrt((p1['X'] - p2['X'])**2 + (p1['Y'] - p2['Y'])**2)
    
    filtered_points = [optimized_df.iloc[0]]  # Ajouter le premier point
    for i in range(1, len(optimized_df)):
        if distance(optimized_df.iloc[i], filtered_points[-1]) >= 1:
            filtered_points.append(optimized_df.iloc[i])
    
    # Création du DataFrame filtré
    optimized_df = pd.DataFrame(filtered_points).reset_index(drop=True)

    

        
    all_curves_x = []
    all_curves_y = []
    current_curve_x = []
    current_curve_y = []
    for i in range(len(optimized_df)):
        point = (optimized_df['X'].iloc[i], optimized_df['Y'].iloc[i], optimized_df['Z'].iloc[i])
        if optimized_df['Z'].iloc[i] == 0:
            if current_curve_x:
                all_curves_x.append(current_curve_x)
                all_curves_y.append(current_curve_y)
                current_curve_x = []
                current_curve_y = []
        else:
            current_curve_x.append(point[0])
            current_curve_y.append(point[1])
    if current_curve_x:
        all_curves_x.append(current_curve_x)
        all_curves_y.append(current_curve_y)
    print(len(all_curves_x))
    smooth_x = []
    smooth_y = []
    time_stamps = []
    colors = []
    pause = 5
    initial_time = 0

    def dilate_segment(times, start_i, end_i, factor):
        """Dilate les écarts de temps entre start_i et end_i, renvoie (new_times, time_shift)."""
        adjusted = times.copy()
        # Assurer un i-1 valide
        s = max(1, start_i)
        for i in range(s, min(end_i + 1, len(times))):
            adjusted[i] = adjusted[i-1] + factor * (times[i] - times[i-1])
        time_shift = adjusted[min(end_i, len(times)-1)] - times[min(end_i, len(times)-1)]
        return adjusted, time_shift

    def shift_following(times, from_i, shift):
        """Décale tous les temps à partir de from_i par shift."""
        for i in range(from_i, len(times)):
            times[i] += shift

    def adjust_times_for_curvatures(times, concatenated_segments, critical_intervals, dilation_factor=2):
        """
        Dilate les temps dans les segments critiques puis décale les suivants.

        times: list[float] des instants
        concatenated_segments: list[float] instants repères
        critical_intervals: list[tuple[int]] indices dans concatenated_segments (inclusifs)
        dilation_factor: float
        """
        for interval in critical_intervals:
            start_idx, end_idx = interval[0], interval[-1]
            start_time = concatenated_segments[start_idx-1]
            end_time = concatenated_segments[min(end_idx,len(concatenated_segments)-1)]
            # Trouver indices dans times
            start_i = bisect_left(times, start_time)
            end_i   = bisect_left(times, end_time)

            # Dilater et récupérer le shift
            times, shift = dilate_segment(times, start_i, end_i, dilation_factor)
            # Appliquer le shift aux instants suivants
            shift_following(times, end_i + 1, shift)

        return times
    z=[]
    for i in range(len(all_curves_x)):
        curve_x, curve_y = all_curves_x[i], all_curves_y[i]
        curve_length = 0
        for j in range(1, len(curve_x)):
            dx = curve_x[j] - curve_x[j - 1]
            dy = curve_y[j] - curve_y[j - 1]
            curve_length += math.sqrt(dx**2 + dy**2)
        
        if curve_length > 3:
            times, x_t, y_t, concatenated_segments = main.courbe_profil(curve_x, curve_y)
            z.extend([1] * len(x_t))  # pour les courbes
            
            critical, _ = detect_critical_curvatures(curve_x, curve_y)

            adjusted_times = adjust_times_for_curvatures(times, concatenated_segments, critical, dilation_factor=2)
            adjusted_times = [t + initial_time for t in adjusted_times]
            time_stamps.extend(adjusted_times)
            smooth_x.extend(x_t)
            smooth_y.extend(y_t)
            colors.extend(['blue'] * len(x_t))
            if i < len(all_curves_x) - 1:
                next_curve_x, next_curve_y = all_curves_x[i + 1][0], all_curves_y[i + 1][0]
                transition_x = np.linspace(curve_x[-1], next_curve_x, 10)
                transition_y = np.linspace(curve_y[-1], next_curve_y, 10)
                t_points_transition, x_transition, y_transition, _ = main.courbe_profil(transition_x, transition_y)
                z.extend([0] * len(x_transition))  # pour les transitions
                t_points_transition = [t + adjusted_times[-1] + pause for t in t_points_transition]
                smooth_x.extend(x_transition)
                smooth_y.extend(y_transition)
                time_stamps.extend(t_points_transition)
                colors.extend(['red'] * len(x_transition))
                initial_time = time_stamps[-1] + pause
        else:
            
            continue
    
    def regularize_time_stamps(time_stamps, smooth_x, smooth_y,colors, z,dt):
        z_array = np.array(z)
        regular_z = []
        start_time = time_stamps[0]
        end_time = time_stamps[-1]
        regular_time_stamps = np.arange(start_time, end_time, dt)
        smooth_x_interpolated = interp1d(time_stamps, smooth_x, kind='linear', fill_value="extrapolate")(regular_time_stamps)
        smooth_y_interpolated = interp1d(time_stamps, smooth_y, kind='linear', fill_value="extrapolate")(regular_time_stamps)
        # Adapter les couleurs par plus proche voisin
        color_array = np.array(colors)
        original_times = np.array(time_stamps)
        regular_colors = []

        for t in regular_time_stamps:
            idx = np.argmin(np.abs(original_times - t))  # Trouver l’indice le plus proche
            regular_colors.append(color_array[idx])
            regular_z.append(z_array[idx])

        return regular_time_stamps, smooth_x_interpolated, smooth_y_interpolated, regular_colors,regular_z

    dt = 0.2
    regular_time_stamps, smooth_x_interpolated, smooth_y_interpolated, regular_colors,regular_z = regularize_time_stamps(
        time_stamps, smooth_x, smooth_y, colors,z,dt
    )
    theta1 = [cinema1.theta1(x, y) for x, y in zip(smooth_x_interpolated, smooth_y_interpolated)]
    theta2 = [cinema1.theta2(x, y) for x, y in zip(smooth_x_interpolated, smooth_y_interpolated)]
    theta3 = [cinema1.theta3(x, y) for x, y in zip(smooth_x_interpolated, smooth_y_interpolated)]
    theta4 = [cinema1.theta4(x, y) for x, y in zip(smooth_x_interpolated, smooth_y_interpolated)]
    
    premiere_valeur = [theta1[0], theta2[0], theta3[0], theta4[0]]
    valeurs_traitees = [
        (theta1[i] - premiere_valeur[0], 
         theta2[i] - premiere_valeur[1], 
         theta3[i] - premiere_valeur[2], 
         theta4[i] - premiere_valeur[3]) 
        for i in range(len(theta1))
    ]
    nom_fichier = "valeurs.txt"
    
    with open(nom_fichier, "w") as fichier:
        for i, valeur in enumerate(valeurs_traitees):
            fichier.write(f"{valeur[0]},{valeur[1]},{valeur[2]},{valeur[3]},{regular_z[i]}\n")
    

    return regular_time_stamps, smooth_x_interpolated, smooth_y_interpolated, all_curves_x, all_curves_y, regular_colors
# =============================================================================
# CLASSE DE L'APPLICATION TKINTER (MODIFIÉE AVEC STYLE ROBOT)
# =============================================================================
class SVGApp:
    def __init__(self, master):
        self.master = master
        master.title("Contrôleur de Trajectoire Robot - Style Clam")
        master.geometry("1000x750")

        # --- Forcer le thème CLAM et récupérer les couleurs ---
        self.style = ttk.Style(master)
        self.clam_bg_color = "#d9d9d9" # Couleur typique clam (à ajuster)
        self.clam_fg_color = "#000000" # Noir
        self.clam_accent_color = "#0078D7" # Bleu windows/office comme accent
        self.clam_button_color = "#e1e1e1"
        self.clam_disabled_fg = "#a3a3a3"
        self.clam_plot_bg = "#ffffff" # Fond blanc pour le plot

        try:
            self.style.theme_use('clam')
            print("Thème 'clam' appliqué.")
            # Essayer de récupérer les vraies couleurs du thème clam
            # Note: Les clés peuvent varier (TFrame, TButton etc.)
            try: self.clam_bg_color = self.style.lookup('TFrame', 'background')
            except: pass # Garder la valeur par défaut
            try: self.clam_fg_color = self.style.lookup('TLabel', 'foreground')
            except: pass
            try: self.clam_button_color = self.style.lookup('TButton', 'background')
            except: pass
            # L'accent et plot bg sont souvent mieux définis manuellement
        except tk.TclError:
            print("AVERTISSEMENT: Thème 'clam' non trouvé, utilisation des styles par défaut Tkinter.")
            # Dans ce cas, les couleurs personnalisées seront quand même utilisées

        # Appliquer couleur de fond principale à la fenêtre racine
        master.config(bg=self.clam_bg_color)

        # --- Redéfinir les couleurs de classe avec les couleurs CLAM ---
        self.BG_COLOR = self.clam_bg_color
        self.FG_COLOR = self.clam_fg_color
        self.ACCENT_COLOR = self.clam_accent_color
        self.BUTTON_COLOR = self.clam_button_color
        self.BUTTON_FG = self.clam_fg_color # Texte noir sur bouton clair
        self.DISABLED_FG = self.clam_disabled_fg
        self.PLOT_BG = self.clam_plot_bg
        self.RED_COLOR = "#E53935" # Rouge un peu plus vif

        # --- Variables d'état (inchangées) ---
        self.selected_svg_file = None
        self.dataframe = None
        self.output_csv_file = "optimized_points.csv"
        # ... (autres variables d'état : simulation_index, simulation_running, etc.) ...
        self.simulation_index = 0
        self.simulation_running = False
        self.default_min_x, self.default_max_x = 0, 4000
        self.default_min_y, self.default_max_y = 0, 3000
        self.ani = None
        self.animation_running = False
        self.start_time = 0
        self.current_frame = 0
        self.logo_photo = None
        self.placeholder_img = None
        self.svg_preview_photo = None
        self.paused = False
        self.paused_time = 0
        self.accumulated_pause = 0
        self.pause_start_time = 0

        # Données pour l'animation (initialisées à None)
        self.regular_time_stamps = None
        self.smooth_x_interpolated = None
        self.smooth_y_interpolated = None
        self.colors = None # Note: c'était regular_colors dans l'original

        # --- Configuration des styles ttk (basés sur les couleurs Clam) ---
        self._configure_styles()

        # --- Layout Frames (utilise le style App.TFrame maintenant défini) ---
        top_frame = ttk.Frame(master, style="App.TFrame", padding=(10, 5))
        top_frame.pack(side=tk.TOP, fill=tk.X)

        action_frame = ttk.Frame(master, style="App.TFrame", padding=(10, 5))
        action_frame.pack(side=tk.TOP, fill=tk.X)

        plot_frame = ttk.Frame(master, style="App.TFrame", padding=(5, 0))
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        status_frame = ttk.Frame(master, style="Status.TFrame", padding=(0, 2))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)


        # --- Widgets Haut (Logo + Fichier) ---
        # Le logo utilisera un Label tk standard, donc besoin de configurer son bg
        try:
            logo_img_path = "/Users/amineennadzim/Desktop/linedraw-master/output/amine png vangogh.png" # Garder ton nom de fichier
            if os.path.exists(logo_img_path):
                logo_img = Image.open(logo_img_path)
                logo_img = logo_img.resize((48, 48), Image.Resampling.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo_img)
                logo_label = tk.Label(top_frame, image=self.logo_photo, bg=self.BG_COLOR) # bg tk Label
                logo_label.pack(side=tk.LEFT, padx=(10, 20), pady=5)
            else: print(f"Info: Logo '{logo_img_path}' non trouvé.")
        except Exception as e: print(f"Erreur chargement logo: {e}")

        # Boutons ttk utilisent les styles configurés
        self.btn_load = ttk.Button(top_frame, text="Charger SVG", command=self.load_svg, style="Accent.TButton", width=15)
        self.btn_load.pack(side=tk.LEFT, padx=5, pady=10)

        # Label ttk utilise style configuré
        self.lbl_filename = ttk.Label(top_frame, text="Aucun fichier SVG chargé", style="Filename.TLabel", width=50, anchor="w")
        self.lbl_filename.pack(side=tk.LEFT, padx=15, pady=10, fill=tk.X, expand=True)

        # --- Widgets Actions ---
        self.btn_process = ttk.Button(action_frame, text="Traiter SVG (points)", command=self.process_svg, state=tk.DISABLED)
        self.btn_process.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_arduino = ttk.Button(action_frame, text="Traiter pour Arduino", command=self.traitement_arduino, state=tk.DISABLED) # Désactivé au début
        self.btn_arduino.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_realtime = ttk.Button(action_frame, text="Simuler Trajectoire", command=self.start_real_time_simulation, style="Accent.TButton", state=tk.DISABLED) # Désactivé au début
        self.btn_realtime.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_pause = ttk.Button(action_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=5, pady=5)


        # --- Zone de Plot Matplotlib (fond blanc) ---
        self.fig = Figure(figsize=(7, 6), dpi=100, facecolor=self.PLOT_BG) # Fond blanc pour le plot
        self.ax = self.fig.add_subplot(111)
        self._setup_plot_style() # Configurer couleurs axes, etc.

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.config(bg=self.PLOT_BG) # Fond tk widget du canvas = fond du plot
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        # Charger l'image placeholder (vérifier le nom)
        self.placeholder_img_path = "coe.png" # Garder ton nom
        self.load_placeholder_image(self.placeholder_img_path)
        self.show_placeholder_image()

        # Barre d'outils Matplotlib (adapter au style Clam)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self._style_matplotlib_toolbar(toolbar) # Appliquer style adapté
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0,5))

        # --- Barre de Statut ---
        self.status_label = ttk.Label(status_frame, text="Prêt. Chargez un fichier SVG.", style="Status.TLabel", anchor="w")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=3)

        self.update_button_states()

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("App.TFrame", background=self.BG_COLOR)
        style.configure("Status.TFrame", background=self.BUTTON_COLOR)

        # --- Boutons Généraux ---
        style.configure("TButton",
                        background=self.BUTTON_COLOR, foreground=self.BUTTON_FG,
                        padding=(10, 6), relief=tk.FLAT, borderwidth=0,
                        font=('Segoe UI', 10))
        style.map("TButton",
                  background=[('active', self.ACCENT_COLOR), ('disabled', '#5A5A5A')],
                  foreground=[('disabled', self.DISABLED_FG)])

        # --- Bouton Accent (Charger, Etape Suivante) ---
        style.configure("Accent.TButton",
                        background=self.ACCENT_COLOR, foreground=self.BUTTON_FG,
                        font=('Segoe UI', 10, 'bold'))
        style.map("Accent.TButton",
                   background=[('active', '#007AA5'), ('disabled', '#5A5A5A')], # Plus foncé actif
                   foreground=[('disabled', self.DISABLED_FG)])

        # --- Labels ---
        style.configure("TLabel", background=self.BG_COLOR, foreground=self.FG_COLOR, padding=2, font=('Segoe UI', 10))
        style.configure("Filename.TLabel", font=('Segoe UI', 9, 'italic'))
        style.configure("Status.TLabel", background=self.BUTTON_COLOR, foreground=self.FG_COLOR, font=('Segoe UI', 9))

    def _setup_plot_style(self):
        """ Configure les couleurs des éléments Matplotlib. """
        self.ax.set_facecolor(self.PLOT_BG)
        self.ax.tick_params(axis='x', colors=self.FG_COLOR)
        self.ax.tick_params(axis='y', colors=self.FG_COLOR)
        self.ax.xaxis.label.set_color(self.FG_COLOR)
        self.ax.yaxis.label.set_color(self.FG_COLOR)
        self.ax.title.set_color(self.FG_COLOR)
        # Couleur des bordures du graphe
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.FG_COLOR)
            spine.set_linewidth(0.5) # Plus fin

    def _style_matplotlib_toolbar(self, toolbar):
        """Applique le style sombre à la barre d'outils Matplotlib."""
        toolbar.config(background=self.BG_COLOR)
        toolbar._message_label.config(background=self.BG_COLOR, foreground=self.FG_COLOR)
        try:
            # Essayer de trouver les boutons spécifiques et les styliser
            for item in toolbar.winfo_children():
                if isinstance(item, (tk.Button, tk.Checkbutton, ttk.Button, ttk.Checkbutton)):
                    item.config(background=self.BUTTON_COLOR, foreground=self.BUTTON_FG, relief=tk.FLAT, borderwidth=0)
                    if hasattr(item, 'config'): # Vérifier si config existe
                         # Tenter de changer couleur quand survolé (peut ne pas marcher pour tous)
                         item.bind("<Enter>", lambda e, w=item: w.config(background=self.ACCENT_COLOR))
                         item.bind("<Leave>", lambda e, w=item: w.config(background=self.BUTTON_COLOR))
        except Exception as e:
            print(f"Avertissement: N'a pas pu styliser complètement la barre d'outils MPL: {e}")


    def load_placeholder_image(self, image_path):
        try:
            if os.path.exists(image_path):
                img = Image.open(image_path)
                self.placeholder_img = img # Garder l'objet Image PIL
            else:
                print(f"Info: Image placeholder '{image_path}' non trouvée.")
                self.placeholder_img = None
        except Exception as e:
            self.placeholder_img = None
            print(f"Erreur chargement image placeholder: {e}")

    def show_placeholder_image(self):
        self.ax.cla()
        self._setup_plot_style() # Réappliquer le style des axes
        if self.placeholder_img:
            try:
                 # Essayer d'afficher l'image sans toucher aux limites auto de Matplotlib
                self.ax.imshow(self.placeholder_img, aspect='auto')
                self.ax.set_title("Chargez un SVG pour commencer", color=self.FG_COLOR, fontsize=11)
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                # Cacher les spines (bordures) pour une vue épurée de l'image
                for spine in self.ax.spines.values(): spine.set_visible(False)
            except Exception as e:
                print(f"Erreur affichage placeholder: {e}")
                self.prepare_plot_axes() # Fallback sur grille vide
                self.ax.set_title("Prêt (erreur placeholder)", color=self.FG_COLOR)
        else:
            self.prepare_plot_axes() # Grille vide si pas de placeholder
            self.ax.set_title("Prêt à visualiser", color=self.FG_COLOR)
        self.canvas.draw()

    def set_status(self, message):
        self.status_label.config(text=message)
        self.master.update_idletasks()

    def update_button_states(self):
        svg_loaded = self.selected_svg_file is not None
        data_processed = hasattr(self, 'regular_time_stamps')  # Vérifie si le traitement Arduino est fait
        
        self.btn_process.config(state=tk.NORMAL if svg_loaded else tk.DISABLED)
        
        # Bouton Arduino - toujours actif si SVG chargé
        self.btn_arduino.config(state=tk.NORMAL if svg_loaded else tk.DISABLED)
        
        # Bouton Realtime - seulement actif si traitement Arduino fait
        self.btn_realtime.config(state=tk.NORMAL if data_processed else tk.DISABLED)
        
        # Gestion de la pause
        animation_active = self.animation_running
        self.btn_pause.config(state=tk.NORMAL if animation_active else tk.DISABLED)


    def prepare_plot_axes(self, title="Visualisation"):
        self.ax.cla()
        self._setup_plot_style() # Appliquer couleurs et style axes
        margin = 50 # Marge en cm

        # Utiliser limites des données traitées si disponibles, sinon défaut
        if self.dataframe is not None and not self.dataframe.empty:
             plot_min_x = self.dataframe['X'].min() - margin
             plot_max_x = self.dataframe['X'].max() + margin
             plot_min_y = self.dataframe['Y'].min() - margin
             plot_max_y = self.dataframe['Y'].max() + margin
             # Assurer que min < max
             if plot_min_x == plot_max_x: plot_max_x += 100
             if plot_min_y == plot_max_y: plot_max_y += 100
        else: # Limites par défaut (mur)
             plot_min_x, plot_max_x = self.default_min_x - margin, self.default_max_x + margin
             plot_min_y, plot_max_y = self.default_min_y - margin, self.default_max_y + margin

        self.ax.set_xlim(plot_min_x, plot_max_x)
        self.ax.set_ylim(plot_min_y, plot_max_y)
        self.ax.set_xlabel('Coordonnée X (cm)')
        self.ax.set_ylabel('Coordonnée Y (cm)')
        self.ax.set_title(title, color=self.FG_COLOR, fontsize=11)
        self.ax.invert_yaxis() # Garder Y inversé comme demandé initialement
        self.ax.set_aspect('equal', adjustable='datalim') # Aspect ratio basé sur données
        self.ax.grid(True, color='#555555', linestyle=':', linewidth=0.5) # Grille plus discrète
        # Afficher les spines (bordures) pour la vue de données
        for spine in self.ax.spines.values(): spine.set_visible(True)
        self.canvas.draw_idle()


    def load_svg(self):
        filepath = filedialog.askopenfilename(
            title="Ouvrir un fichier SVG",
            filetypes=(("Fichiers SVG", "*.svg"), ("Tous les fichiers", "*.*"))
        )
        if filepath:
            self.selected_svg_file = filepath
            display_name = os.path.basename(filepath)
            if len(display_name) > 45: display_name = display_name[:20] + "..." + display_name[-20:]
            self.lbl_filename.config(text=display_name)
            self.set_status(f"Fichier: {os.path.basename(filepath)}. Prêt à traiter ou visualiser.")

            self.dataframe = None # Réinitialiser les données précédentes
            self.simulation_running = False
            # Ne pas réafficher le placeholder ici, afficher une grille vide prête
            self.prepare_plot_axes("SVG chargé. Traitez ou affichez l'aperçu.")
            self.update_button_states()

        else:
            self.set_status("Chargement annulé.")

    
    def process_svg(self):
        if not self.selected_svg_file: return

        self.set_status("Traitement du SVG en cours... Merci de patienter.")
        self.master.config(cursor="watch")
        # Désactiver tous les boutons d'action pendant le traitement
        for btn in [self.btn_load,  self.btn_process]:
             btn.config(state=tk.DISABLED)
        self.master.update()

        processed_df = None
        try:
            # APPEL DE VOTRE FONCTION DE TRAITEMENT INTÉGRÉE
            processed_df = process_svg_to_coordinates(self.selected_svg_file)

            if processed_df is not None and not processed_df.empty:
                self.dataframe = processed_df
                num_points = len(self.dataframe)
                self.set_status(f"Traitement terminé. {num_points} points générés dans '{self.output_csv_file}'.")
                # Afficher directement le résultat après traitement ? ou juste préparer axes?
                self.display_plot() # Afficher le résultat statique
                messagebox.showinfo("Traitement Réussi", f"Le fichier SVG a été traité avec succès.\n{num_points} points ont été générés.")

            elif processed_df is not None and processed_df.empty:
                 self.dataframe = None
                 self.set_status("Traitement terminé, mais aucun point généré (vérifiez SVG?).")
                 messagebox.showwarning("Aucun Point", "Le traitement s'est terminé mais n'a produit aucun point de données.")
                 self.show_placeholder_image() # Retour au placeholder
            else: # Erreur déjà gérée dans process_svg via messagebox
                self.dataframe = None
                self.set_status("Échec du traitement du SVG.")
                self.show_placeholder_image()

        except Exception as e: # Double sécurité
             self.dataframe = None
             self.set_status(f"Erreur critique pendant le traitement global: {e}")
             messagebox.showerror("Erreur Critique", f"Une erreur majeure est survenue:\n{e}\nVoir console.")
             traceback.print_exc()
             self.show_placeholder_image()
        finally:
            self.master.config(cursor="") # Curseur normal
            self.update_button_states() # Réactiver boutons selon état


    def display_plot(self):
        if self.dataframe is None or self.dataframe.empty: return

        self.set_status("Affichage du résultat statique...")
        self.simulation_running = False # Stopper simu si affichage statique
        self.update_button_states()

        try:
            self.prepare_plot_axes(title='Résultat Traité (Statique)')
            x, y, z = self.dataframe['X'], self.dataframe['Y'], self.dataframe['Z']

            # Points Z=1 (Tracé) en bleu
            self.ax.plot(x[z==1], y[z==1], marker='o', linestyle='', markersize=4, color=self.ACCENT_COLOR, label='Tracé (Z=1)')
            # Points Z=0 (Sauts) en rouge
            self.ax.plot(x[z==0], y[z==0], marker='x', linestyle='', markersize=6, color=self.RED_COLOR, label='Saut (Z=0)')
            # Ligne d'ordre des points (très discrète)
            self.ax.plot(x, y, color='#666666', linestyle='-', linewidth=0.4, alpha=0.6)

            # Légende stylisée
            legend = self.ax.legend(facecolor=self.PLOT_BG, edgecolor='#777777', labelcolor=self.FG_COLOR, fontsize=9)
            self.canvas.draw_idle()
            self.set_status("Plot statique affiché.")

        except KeyError as e:
             self.set_status(f"Erreur données: Colonne manquante '{e}'.")
             messagebox.showerror("Erreur Données", f"Colonne '{e}' manquante dans les données traitées.")
             self.show_placeholder_image()
        except Exception as e:
            self.set_status(f"Erreur affichage plot: {e}")
            messagebox.showerror("Erreur Affichage", f"Impossible d'afficher le plot:\n{e}")
            traceback.print_exc()
            self.show_placeholder_image()


    
    def traitement_arduino(self):
        if not self.selected_svg_file:
            messagebox.showwarning("Avertissement", "Aucun fichier SVG sélectionné.")
            return

        try:
            # Désactiver le bouton pendant le traitement
            self.btn_arduino.config(state=tk.DISABLED, text="Traitement en cours...")
            self.set_status("Traitement Arduino en cours...")
            self.master.update()  # Force la mise à jour de l'interface
            
            # Simuler un traitement long (à remplacer par votre vrai traitement)
            self.master.after(100, self.finish_arduino_processing)
            
        except Exception as e:
            self.btn_arduino.config(state=tk.NORMAL, text="traitement arduino")
            messagebox.showerror("Erreur", f"Échec du traitement du SVG : {str(e)}")
            traceback.print_exc()

    def finish_arduino_processing(self):
        """Cette méthode est appelée lorsque le traitement est terminé"""
        try:
            # Ici vous mettez votre vrai traitement
            (self.regular_time_stamps,
            self.smooth_x_interpolated,
            self.smooth_y_interpolated,
            self.all_curves_x,
            self.all_curves_y,
            self.colors
            ) = process_svg(self.selected_svg_file)
            
            # Mettre à jour l'interface
            self.btn_arduino.config(state=tk.NORMAL, text="traitement arduino")
            self.btn_realtime.config(state=tk.NORMAL)  # Activer le bouton simulation
            self.set_status("Prêt pour simulation - Traitement Arduino terminé!")
            messagebox.showinfo("Succès", "Traitement Arduino terminé. Prêt pour simulation.")
            
        except Exception as e:
            self.btn_arduino.config(state=tk.NORMAL, text="traitement arduino")
            self.set_status("Erreur pendant le traitement Arduino")
            messagebox.showerror("Erreur", f"Échec du traitement : {str(e)}")
            traceback.print_exc()
    
    def start_real_time_simulation(self):
        if getattr(self, 'ani', None):
            self.stop_animation()

        # Vérifie que les données ont été traitées
        if not hasattr(self, 'regular_time_stamps'):
            messagebox.showwarning("Avertissement", "Veuillez d'abord charger et traiter un fichier SVG.")
            return

        try:
            self.create_animation()
            self.set_status("Simulation temps réel démarrée !")

        except Exception as e:
            messagebox.showerror("Erreur", f"Échec de la simulation : {str(e)}")
            traceback.print_exc()


    def create_animation(self):
        # Nettoyage & style
        self.ax.cla()
        self._setup_plot_style()
        self.ax.set_xlim(0, 4000)
        self.ax.set_ylim(0, 3000)
        self.ax.invert_yaxis()
        self.ax.set_title('Simulation Temps Réel', color=self.FG_COLOR)
        self.ax.grid(True, color='#555555', linestyle=':')

        # États de pause
        self.paused = False
        self.btn_pause.config(text="Pause", state='normal')
        self.start_time = time.time()

        # Initialisation des lignes
        self.line_blue, = self.ax.plot([], [], color=self.ACCENT_COLOR, linewidth=2)
        self.line_red, = self.ax.plot([], [], color=self.RED_COLOR, linewidth=1, linestyle=':')

        # Création de l'animation sur N frames
        n_pts = len(self.regular_time_stamps)
        self.animation_running = True
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update_animation,
            frames=range(n_pts),
            interval=200,       # 200 ms = 0.2 s par point
            blit=False,         # évite parfois des frames manquantes
            repeat=False
        )
        self.canvas.draw()


    import numpy as np

    def update_animation(self, frame_idx):
        if not self.animation_running or self.paused:
            return self.line_blue, self.line_red

        # on crée des listes de la longueur courante,
        # mais on met NaN là où la couleur ne correspond pas
        x_blue = [
            self.smooth_x_interpolated[i] if self.colors[i] == 'blue' else np.nan
            for i in range(frame_idx + 1)
        ]
        y_blue = [
            self.smooth_y_interpolated[i] if self.colors[i] == 'blue' else np.nan
            for i in range(frame_idx + 1)
        ]

        x_red = [
            self.smooth_x_interpolated[i] if self.colors[i] == 'red' else np.nan
            for i in range(frame_idx + 1)
        ]
        y_red = [
            self.smooth_y_interpolated[i] if self.colors[i] == 'red' else np.nan
            for i in range(frame_idx + 1)
        ]

        # on met à jour les artistes : chaque ligne est “cassée”
        # là où il y a un NaN, donc pas de raccord indésirable
        self.line_blue.set_data(x_blue, y_blue)
        self.line_red .set_data(x_red,  y_red)

        # arrêt en fin d’animation
        if frame_idx == len(self.regular_time_stamps) - 1:
            self.animation_running = False
            self.btn_pause.config(state='disabled')

        return self.line_blue, self.line_red



    def toggle_pause(self):
        if self.paused:
            self.resume_animation()
        else:
            self.pause_animation()


    def pause_animation(self):
        if self.animation_running and not self.paused:
            self.paused = True
            self.pause_start_time = time.time()
            self.btn_pause.config(text="Reprendre")
            if self.ani:
                self.ani.event_source.stop()
            self.set_status("Animation en pause")


    def resume_animation(self):
        if self.animation_running and self.paused:
            self.paused = False
            pause_duration = time.time() - self.pause_start_time
            # Ajustement du start_time pour que l’animation reprenne là où elle en était
            self.start_time += pause_duration
            self.btn_pause.config(text="Pause")
            if self.ani:
                self.ani.event_source.start()
            self.set_status("Animation reprise")


    def stop_animation(self):
        """À appeler pour forcer l'arrêt propre."""
        if getattr(self, 'ani', None):
            self.ani.event_source.stop()
        self.animation_running = False
        self.paused = False
        self.btn_pause.config(text="Pause", state='disabled')
        self.ax.cla()
        self.show_placeholder_image()

# =============================================================================
# LANCEMENT DE L'APPLICATION
# =============================================================================
if __name__ == "__main__":
    # Vérification basique que les fonctions existent (sécurité)
    required_funcs = ['process_svg_to_coordinates', 'compute_curvature']
    if not all(func in globals() for func in required_funcs):
         messagebox.showerror("Erreur Critique Code", "Les fonctions de traitement SVG essentielles sont manquantes dans le script !")
    else:
        # Rappel pour les images
        print("------------------------------------------------------------")
        print("RAPPEL : Assurez-vous d'avoir les fichiers image :")
        print("  - 'robot_logo.png' (pour le logo)")
        print("  - 'starry_night.jpg' (ou autre nom, pour le fond initial)")
        print("dans le même dossier que ce script.")
        print("------------------------------------------------------------")
        root = tk.Tk()
        app = SVGApp(root)
        root.mainloop() 