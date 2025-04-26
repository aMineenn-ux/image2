import numpy as np
from scipy.optimize import fsolve
import math
from itertools import accumulate
A=14567
B=185
C=-2.73

def courbe_profil(drawn_points_x,drawn_points_y):
    distances = [np.linalg.norm([drawn_points_x[i+1] - drawn_points_x[i], drawn_points_y[i+1] - drawn_points_y[i]])
             for i in range(len(drawn_points_x) - 1)]
    d_total = sum(distances)
    #print(d_total)
    J =A/(d_total+B)+C if A/(d_total+B)+C>=3 else 3
    V_max = 50 / (1 + np.exp(-0.03*(d_total-90)))
    t2=np.sqrt(V_max/J)
    segment_index = 0  # Indice du segment où t2 se situe
    remaining_time = t2  # Temps restant à analyser
    segment_instant=[0]
    # Boucle pour trouver le segment où t2 se situe
    for i, segment_length in enumerate(distances):
        # Calcul de la distance parcourue dans ce segment pour le temps restant
        distance_t = (1 / 6) * J * remaining_time**3
        A_i = np.array([drawn_points_x[i], drawn_points_y[i]])
        A_i1 = np.array([drawn_points_x[i+1], drawn_points_y[i+1]])
        u_i = A_i1 - A_i
        u_i_normalized = u_i / (segment_length+ 1e-8)

        if distance_t  <= segment_length:
            # Si la distance calculée reste dans le segment courant
            segment_index = i
            x_1= drawn_points_x[i]+(1 / 6) * J * remaining_time**3 * u_i_normalized[0]
            y_1= drawn_points_y[i]+(1 / 6) * J * remaining_time**3 * u_i_normalized[1]
            segment_instant.append(t2)
            break
        else:
            # Si on dépasse le segment courant, passer au suivant
            remaining_time -= (6 * segment_length / J)**(1/3)  # Réduction du temps restant
            c=segment_instant[-1]
            segment_instant.append((6 * segment_length / J)**(1/3)+c)
   


    t= np.linspace(0, t2, 20)
    segments = []
    for i in range(len(segment_instant)-1):
            sous_liste = [val for val in t if segment_instant[i] <= val <= segment_instant[i+1]]
            segments.append(sous_liste)
    t= [item for sublist in segments for item in sublist]
    # Exécution des calculs pour chaque sous-liste
    x_t=[]
    y_t=[]
    for i, segment_instants in enumerate(segments):
        A_i = np.array([drawn_points_x[i], drawn_points_y[i]])
        A_i1 = np.array([drawn_points_x[i + 1], drawn_points_y[i + 1]])
        u_i = A_i1 - A_i
        segment_length = distances[i]  # Longueur du segment actuel
        u_i_normalized = u_i / (segment_length+ 1e-8)
        for instant in segment_instants:
            # Calculs de la phase 1 pour chaque instant
            x_phase1 = A_i[0] + (1 / 6) * J * (instant-segment_instant[i])**3 * u_i_normalized[0]
            y_phase1 = A_i[1] + (1 / 6) * J * (instant-segment_instant[i])**3 * u_i_normalized[1]
            x_t.append(x_phase1)
            y_t.append(y_phase1)
   

    def find_remaining_time(segment_length, t2, J):
        def equation(t):
            return  (1 / 2) * J * t2**2 * t + (1 / 2) * J * t2 * t**2 - (1 / 6) * J * t**3  - segment_length

        # Recherche initiale pour résoudre numériquement
        t_initial = t2  # Temps initial probable
        t_solution = fsolve(equation, t_initial)
        return t_solution[0]

    t3 = 2*t2  # Instant t3
    # Calcul des distances restantes après t2
    distance_after_t2 = [
        np.linalg.norm([drawn_points_x[i + 1] - x_1, drawn_points_y[i + 1] - y_1])
        if i == segment_index
        else distances[i]
        for i in range(segment_index, len(distances))
    ]
    distance_demarrage = (1 / 6) * J *(t2**3-(t3-t2)**3)  + (1 / 2) * J * t2**2 * (t3-t2) + (1 / 2) * J * t2 * (t3-t2)**2
    remaining_time = t3 - t2
    segment_instant_t3 = [t2]  # Liste des instants pour t3

    # Localisation du segment pour t3
    for i, segment_length in enumerate(distance_after_t2):
        distance_t = (1 / 2) * J * t2**2 * remaining_time + (1 / 2) * J * t2 * remaining_time**2 - (1 / 6) * J * remaining_time**3
        if distance_t <= segment_length:
            # Si la distance calculée reste dans le segment courant
            segment_index_t3 = segment_index + i
            segment_instant_t3.append(t3)
           
            break
        else:
            # Trouver le temps restant via résolution numérique
            time_for_segment = find_remaining_time(segment_length, t2, J)
            remaining_time -= time_for_segment
            c=segment_instant_t3[-1]
            segment_instant_t3.append(c + time_for_segment)

    # Création de la liste des instants t1 entre t2 et t3
    t1 = np.linspace(t2, t3, 20)
    segments_t3 = []
    for i in range(len(segment_instant_t3) - 1):
        sous_liste = [val for val in t1 if segment_instant_t3[i] <= val <= segment_instant_t3[i + 1]]
        segments_t3.append(sous_liste)
    t1= [item for sublist in segments_t3 for item in sublist]
    # Calcul des positions pour chaque instant dans t1
    for i, segment_instants in enumerate(segments_t3):
        A_i = np.array([drawn_points_x[segment_index + i], drawn_points_y[segment_index + i]])
        A_i1 = np.array([drawn_points_x[segment_index + i + 1], drawn_points_y[segment_index + i + 1]])
        u_i = A_i1 - A_i
        segment_length = distances[segment_index + i]
        u_i_normalized = u_i / (segment_length+ 1e-8)
        if i == 0:  # Premier segment
            for instant in segment_instants:
                # Appliquer une formule différente pour les x et y du premier segment
                x_phase = x_1+((1 / 2) * J * t2**2 * (instant-t2) + (1 / 2) * J * t2 * (instant-t2)**2 - (1 / 6) * J * (instant-t2)**3 )*u_i[0]/np.linalg.norm(u_i)
                y_phase = y_1+((1 / 2) * J * t2**2 * (instant-t2) + (1 / 2) * J * t2 * (instant-t2)**2 - (1 / 6) * J * (instant-t2)**3 )*u_i[1]/np.linalg.norm(u_i)
                x_t.append(x_phase)
                y_t.append(y_phase)
               
        else:  # Pour les segments suivants
            total_distance = sum(distance_after_t2[:i])
            for instant in segment_instants:
                # Garder la formule standard pour les autres segments
                x_phase = A_i[0] + ((1 / 2) * J * t2**2 * (instant-t2) + (1 / 2) * J * t2 * (instant-t2)**2 - (1 / 6) * J * (instant-t2)**3 -total_distance)*u_i_normalized[0]
                y_phase = A_i[1] + ((1 / 2) * J * t2**2 * (instant-t2) + (1 / 2) * J * t2 * (instant-t2)**2 - (1 / 6) * J * (instant-t2)**3 -total_distance)*u_i_normalized[1]
                x_t.append(x_phase)
                y_t.append(y_phase)


    v_const = J * t2**2  # Vitesse à l'instant t3
    # Somme cumulée des distances après t3
    distance_cumulee = 0
    distance_after_t3 = []
    segment_index_t4 = segment_index_t3  # Initialiser à l'indice du segment après t3

    for i, distance in enumerate(distances[segment_index_t3:], start=segment_index_t3):
        if i==segment_index_t3:
            distance=np.linalg.norm([drawn_points_x[i + 1] - x_t[-1], drawn_points_y[i + 1] - y_t[-1]])

        distance_cumulee += distance
        if distance_cumulee >= d_total-2*distance_demarrage: #2 fois distance demaragge pour englober aussi la phase d'arret symetrique au demarage en terme de profil de vitesse
            # On a trouvé le segment contenant t4
            segment_index_t4 = i
           
            # Ajouter seulement la partie nécessaire de ce segment
            surplus = distance_cumulee - (d_total-2*distance_demarrage)
            distance_after_t3.append(distance -surplus)
            break
        else:
            # Ajouter la distance complète de ce segment
            distance_after_t3.append(distance)

    t4=t3+sum(distance_after_t3[:])/v_const
   
    instant_after_t3 = np.linspace(t3, t4, math.floor(t4*30/1.5))
    time_segments = [distance / v_const for distance in distance_after_t3]  # Temps pour chaque segment
    cumulative_times = list(accumulate(time_segments))  # Somme cumulative des temps
    segment_instant_after_t3 = [t3] + [t3 + t for t in cumulative_times]

    segments_after_t3 = []
    for i in range(len(segment_instant_after_t3) - 1):
        sous_liste = [val for val in instant_after_t3 if segment_instant_after_t3[i] <= val <= segment_instant_after_t3[i + 1]]
        segments_after_t3.append(sous_liste)
    instant_after_t3=[item for sublist in segments_after_t3 for item in sublist]
    for i, segment in enumerate(segments_after_t3):
        if i==0:
            A_i = np.array([x_t[-1],y_t[-1]])
            A_i2 = np.array([drawn_points_x[segment_index_t3 ],drawn_points_y[segment_index_t3 ]])
            A_i1 = np.array([drawn_points_x[segment_index_t3 + 1],drawn_points_y[segment_index_t3 + 1]])
            u_i = A_i1-A_i2
            u_i_normalized = u_i / (np.linalg.norm(u_i)+ 1e-8)  # Direction normalisée
        else:
            A_i = np.array([drawn_points_x[segment_index_t3 + i], drawn_points_y[segment_index_t3 + i]])
            A_i1 = np.array([drawn_points_x[segment_index_t3 + i + 1], drawn_points_y[segment_index_t3 + i + 1]])
            u_i = A_i1 - A_i
            u_i_normalized = u_i / (np.linalg.norm(u_i)+ 1e-8)  # Direction normalisée
            # Calcul de la distance parcourue dans ce segment
        for instant in segment:
            distance_covered = v_const * (instant-segment_instant_after_t3[i])
            x_phase = A_i[0] + distance_covered * u_i_normalized[0]
            y_phase = A_i[1] + distance_covered * u_i_normalized[1]
            x_t.append(x_phase)
            y_t.append(y_phase)



    # phase apres t4
    distance_after_t4 = [
        np.linalg.norm([drawn_points_x[i + 1] - x_t[-1], drawn_points_y[i + 1] - y_t[-1]])
        if i == segment_index_t4
        else distances[i]
        for i in range(segment_index_t4, len(distances))
    ]

    t5=t4+t2
    remaining_time = t5-t4
    segment_instant_t4 = [t4]  
    def find_remaining_time2(segment_length, t2, J):
        def equation(t):
            return  J * t2**2 * t -  (1 / 6) * J * t**3  - segment_length

        # Recherche initiale pour résoudre numériquement
        t_initial = 0  # Temps initial probable
        t_solution = fsolve(equation, t_initial)
        return t_solution[0]
    # Localisation du segment pour t5
    for i, segment_length in enumerate(distance_after_t4):
        distance_t = J * t2**2 * remaining_time - (1 / 6) * J * remaining_time**3
        if distance_t <= segment_length:
            # Si la distance calculée reste dans le segment courant
            segment_index_t5 = segment_index_t4 + i
            segment_instant_t4.append(t4+t2)
           
            break
        else:
            time_for_segment = find_remaining_time2(segment_length, t2, J)
            remaining_time -= time_for_segment
            c=segment_instant_t4[-1]
            segment_instant_t4.append(c + time_for_segment)

    t0 = np.linspace(t4, t5, 20)
    segments_t4 = []
    for i in range(len(segment_instant_t4) - 1):
        sous_liste = [val for val in t0 if segment_instant_t4[i] <= val <= segment_instant_t4[i + 1]]
        segments_t4.append(sous_liste)
    t0= [item for sublist in segments_t4 for item in sublist]
    # Calcul des positions pour chaque instant dans t1
    for i, segment_instants in enumerate(segments_t4):
        A_i = np.array([drawn_points_x[segment_index_t4 + i], drawn_points_y[segment_index_t4 + i]])
        A_i1 = np.array([drawn_points_x[segment_index_t4 + i + 1], drawn_points_y[segment_index_t4 + i + 1]])
        u_i = A_i1 - A_i
        segment_length = distance_after_t4[i]
        u_i_normalized = u_i / (segment_length+ 1e-8)
       

        if i == 0:  # Premier segment
            x=x_t[-1]
            y=y_t[-1]
            for instant in segment_instants:
                # Appliquer une formule différente pour les x et y du premier segment
                x_phase = x+(J * t2**2 * (instant-t4)  - (1 / 6) * J * (instant-t4)**3 )*(u_i[0]/(np.linalg.norm(u_i)))
                y_phase = y+(J * t2**2 * (instant-t4)  - (1 / 6) * J * (instant-t4)**3 )*(u_i[1]/(np.linalg.norm(u_i)))                
                x_t.append(x_phase)
                y_t.append(y_phase)
               
               
               
        else:  # Pour les segments suivants
            total_distance = sum(distance_after_t4[:i])
            for instant in segment_instants:
                # Garder la formule standard pour les autres segments
                x_phase = A_i[0] + (J * t2**2 * (instant-t4)  - (1 / 6) * J * (instant-t4)**3 -total_distance)*u_i_normalized[0]
                y_phase = A_i[1] + (J * t2**2 * (instant-t4)  - (1 / 6) * J * (instant-t4)**3 -total_distance)*u_i_normalized[1]
                x_t.append(x_phase)
                y_t.append(y_phase)
               
             

    def find_remaining_time1(segment_length, t2, J):
        def equation(t):
            return  (1 / 2) * J * t2**2 * t -(1 / 2) * J * t2 * t**2 +  (1 / 6) * J * t**3  - segment_length

        # Recherche initiale pour résoudre numériquement
        t_initial = 0  # Temps initial probable
        t_solution = fsolve(equation, t_initial)
        return t_solution[0]
   
    distance_after_t5 = [
        np.linalg.norm([drawn_points_x[i + 1] - x_t[-1], drawn_points_y[i + 1] - y_t[-1]])
        if i == segment_index_t5
        else distances[i]
        for i in range(segment_index_t5, len(distances))
    ]
    t6=t5+t2
    remaining_time = t6-t5
    segment_instant_t5 = [t5]  


    for i, segment_length in enumerate(distance_after_t5):
        distance_t = (1 / 2) * J * t2**2 * remaining_time -(1 / 2) * J * t2 * remaining_time**2+  (1 / 6) * J * remaining_time**3
        if distance_t <= segment_length:
            # Si la distance calculée reste dans le segment courant
            segment_instant_t5.append(t6)
            break
        else:
            # Trouver le temps restant via résolution numérique
            time_for_segment = find_remaining_time1(segment_length, t2, J)
            remaining_time -= time_for_segment
            c=segment_instant_t5[-1]
            segment_instant_t5.append(c + time_for_segment)

    t_0 = np.linspace(t5, t6, 20)
    segments_t5 = []

    for i in range(len(segment_instant_t5) - 1):
        sous_liste = [val for val in t_0 if segment_instant_t5[i] <= val <= segment_instant_t5[i + 1]]
        segments_t5.append(sous_liste)
    t_0 = [item for sublist in segments_t5 for item in sublist]

    for i, segment_instants in enumerate(segments_t5):
        A_i = np.array([drawn_points_x[segment_index_t5 + i], drawn_points_y[segment_index_t5 + i]])
        A_i1 = np.array([drawn_points_x[segment_index_t5 + i + 1], drawn_points_y[segment_index_t5 + i + 1]])
        u_i = A_i1 - A_i
       
        segment_length = distance_after_t5[i]
       
        u_i_normalized = u_i / (segment_length+ 1e-8)
        if i == 0:  # Premier segment
            x=x_t[-1]
            y=y_t[-1]
            for instant in segment_instants:
               
                # Appliquer une formule différente pour les x et y du premier segment
                x_phase = x+((1 / 2) * J * t2**2 * (instant-t5) - (1 / 2) * J * t2 * (instant-t5)**2 + (1 / 6) * J * (instant-t5)**3 )*(u_i[0]/(np.linalg.norm(u_i)+ 1e-4))
                y_phase = y+((1 / 2) * J * t2**2 * (instant-t5) - (1 / 2) * J * t2 * (instant-t5)**2 + (1 / 6) * J * (instant-t5)**3 )*(u_i[1]/(np.linalg.norm(u_i)+ 1e-4))
                x_t.append(x_phase)
                y_t.append(y_phase)

        else:  # Pour les segments suivants
            total_distance = sum(distance_after_t5[:i])
            for instant in segment_instants:
                # Garder la formule standard pour les autres segments
                x_phase = A_i[0] + ((1 / 2) * J * t2**2 * (instant-t5) - (1 / 2) * J * t2 * (instant-t5)**2 + (1 / 6) * J * (instant-t5)**3 -total_distance)*u_i_normalized[0]
                y_phase = A_i[1] + ((1 / 2) * J * t2**2 * (instant-t5) - (1 / 2) * J * t2 * (instant-t5)**2 + (1 / 6) * J * (instant-t5)**3 -total_distance)*u_i_normalized[1]
                x_t.append(x_phase)
                y_t.append(y_phase)
    times=np.concatenate((t, t1, instant_after_t3,t0,t_0))    
    #return(times,x_t,y_t)
    concatenated_segments = (segment_instant[:-1] +segment_instant_t3[1:-1] +segment_instant_after_t3[1:-1] +segment_instant_t4[1:-1] +segment_instant_t5[1:] )
    return(times,x_t,y_t,concatenated_segments)
