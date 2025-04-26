import numpy as np
import math
from scipy.optimize import fsolve
from scipy.integrate import quad

# Paramètres du système
L0, H0 = 4000, 3000    # Dimensions du cadre
a, b = 300, 300        # Dimensions de la nacelle
r = np.sqrt(a**2 + b**2) / 2
alpha = math.atan(a / b)
#l_tot = np.sqrt(3450**2 + 2450**2)  # Longueur totale du câble
l_tot=5000
t = 0.35                  # Épaisseur du câble
R0 = 15                # Rayon initial de la poulie

# Fonctions qui calculent la longueur du câble déployé (li) en fonction de (x,y)
def l1(x, y):
    return np.sqrt((x - r * np.cos(alpha))**2 + (y - r * np.sin(alpha))**2)

def l2(x, y):
    return np.sqrt((L0 - x - r * np.cos(alpha))**2 + (y - r * np.sin(alpha))**2)

def l3(x, y):
    return np.sqrt((L0 - x - r * np.cos(alpha))**2 + (H0 - y + r * np.sin(alpha))**2)

def l4(x, y):
    return np.sqrt((x - r * np.cos(alpha))**2 + (H0 - y - r * np.sin(alpha))**2)

# Fonction qui calcule le rayon de la poulie pour un angle theta (modèle de spirale d'Archimède)
def calculate_R(theta):
    return R0 + (t / (2 * np.pi)) * theta

# Fonction intégrale qui calcule la longueur de câble enroulé sur la spirale jusqu'à l'angle theta
def spiral_length(theta):
    integrand = lambda u: np.sqrt((t / (2*np.pi))**2 + (R0 + (t / (2*np.pi)) * u)**2)
    return quad(integrand, 0, theta)[0]

# Fonction qui calcule la longueur du segment tangent AB, en fonction de theta, du point B et du centre O.
def tangent_length(theta, B, O):
    R_current = calculate_R(theta)
    d = np.linalg.norm(B - O)
    #print(d)
    if d <= R_current:
        raise ValueError("Le point B se trouve à l'intérieur ou sur la poulie (d <= R). Vérifiez la géométrie.")
    return np.sqrt(d**2 - R_current**2)

# Fonction pour trouver theta tel que :
# spiral_length(theta) + tangent_length(theta, B, O) = l_tot - l_i
def find_theta(l_i, B, O):
    def f(theta):
        return spiral_length(theta) + tangent_length(theta, B, O) - (l_tot - l_i)
    return fsolve(f, 10)[0]  # Initialisation à 10 rad

# Fonctions pour obtenir theta en fonction de (x, y) pour chaque câble/poulie.
# Les points B et O sont définis de la manière suivante :
# Pour la poulie 1 : B1 = (x-150, y-150), O1 = (x-100, y-100)
def theta1(x, y):
    B1 = np.array([x - 150, y - 150])
    O1 = np.array([x - 105, y - 105])
    return find_theta(l1(x, y), B1, O1)

# Pour la poulie 2 : B2 = (x+150, y-150), O2 = (x+100, y-100)
def theta2(x, y):
    B2 = np.array([x + 150, y - 150])
    O2 = np.array([x + 105, y - 105])
    return find_theta(l2(x, y), B2, O2)

# Pour la poulie 3 : B3 = (x+150, y+150), O3 = (x+100, y+100)
def theta3(x, y):
    B3 = np.array([x + 150, y + 150])
    O3 = np.array([x + 105, y + 105])
    return find_theta(l3(x, y), B3, O3)

# Pour la poulie 4 : B4 = (x-150, y+150), O4 = (x-100, y+100)
def theta4(x, y):
    B4 = np.array([x - 150, y + 150])
    O4 = np.array([x - 105, y + 105])
    return find_theta(l4(x, y), B4, O4)

def calculate_R1(l_i): #cas discret pour comparaison
    # Longueur du câble déjà enroulée
    l_remaining = l_tot - l_i  # Longueur restante du câble à enrouler
    
    R_current = R0  # Initialiser avec le rayon initial
    while l_remaining > 0 :
        # Longueur de câble à enrouler sur cette couche
        delta_l = 2 * np.pi * R_current  # Périmètre de la couche actuelle
        if l_remaining < delta_l:
            # Si la longueur restante est inférieure au périmètre, ajouter seulement la portion restante
            delta_l = l_remaining

        # Mettre à jour le rayon pour la couche suivante
        R_current += t
        # Réduire la longueur restante
        l_remaining -= delta_l
    return R_current
# Exemple d'utilisation
if __name__ == '__main__':
    # Coordonnées de la nacelle (exemple)
    x, y = 1500, 500

    theta1_val = theta1(x, y)
    R1_val = calculate_R(theta1_val)
    AB1_val = tangent_length(theta1_val, np.array([x - 150, y - 150]), np.array([x - 105, y - 105]))

    print("Pour la poulie 1:")
    print(f"  θ = {theta1_val:.3f} rad")
    print(f"  Rayon R = {R1_val:.3f} mm")
    print(f"  Longueur du segment tangent AB = {AB1_val:.3f} mm")
    print(calculate_R1(l1(x,y)))
