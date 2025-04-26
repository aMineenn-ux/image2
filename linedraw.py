from random import *
import math
import argparse
from PIL import Image, ImageDraw, ImageOps
from filters import *
from strokesort import *
import perlin
from util import *

no_cv = False
export_path = "output/out.svg"
draw_contours = True
draw_hatch = True
show_bitmap = False
resolution = 1024
hatch_size = 16
contour_simplify = 10
def simplify_vectors(lines, tolerance=5.0):
    """
    Réduit le nombre de vecteurs en fusionnant ceux qui sont proches.
    
    Args:
        lines: Liste des polylignes [(x1,y1), (x2,y2), ...]
        tolerance: Distance maximale pour fusionner deux points (en pixels)
    
    Returns:
        Liste simplifiée des polylignes
    """
    if not lines:
        return lines

    print(f"Simplifying vectors from {len(lines)} polylines...")
    
    # Convertir toutes les lignes en une seule séquence de segments
    all_segments = []
    for polyline in lines:
        for i in range(len(polyline)-1):
            all_segments.append((polyline[i], polyline[i+1]))
    
    # Algorithme de simplification
    simplified = []
    while all_segments:
        current = all_segments.pop(0)
        merged = True
        
        while merged:
            merged = False
            for i, segment in enumerate(all_segments):
                # Vérifier la proximité entre les extrémités
                dist_start_current = math.dist(current[0], segment[0])
                dist_end_current = math.dist(current[1], segment[1])
                dist_start_end = math.dist(current[0], segment[1])
                dist_end_start = math.dist(current[1], segment[0])
                
                # Trouver la meilleure correspondance
                min_dist = min(dist_start_current, dist_end_current, dist_start_end, dist_end_start)
                
                if min_dist <= tolerance:
                    merged = True
                    if min_dist == dist_start_current:
                        # Concaténer en inversant le segment
                        current = (segment[1],) + current
                    elif min_dist == dist_end_current:
                        # Concaténer normalement
                        current = current + (segment[1],)
                    elif min_dist == dist_start_end:
                        # Concaténer les deux
                        current = segment + current[1:]
                    else:  # dist_end_start
                        current = current[:-1] + segment
                    
                    all_segments.pop(i)
                    break
        
        simplified.append(current)
    
    # Convertir les segments fusionnés en polylignes
    result = []
    for segment in simplified:
        if len(segment) > 2:  # Si on a plus qu'un simple segment
            result.append(list(segment))
        else:
            result.append([segment[0], segment[1]])
    
    print(f"Simplified to {len(result)} polylines")
    return result
try:
    import numpy as np
    import cv2
except:
    print("Cannot import numpy/openCV. Switching to NO_CV mode.")
    no_cv = True

def find_edges(IM):
    print("finding edges...")
    if no_cv:
        #appmask(IM,[F_Blur])
        appmask(IM,[F_SobelX,F_SobelY])
    else:
        im = np.array(IM) 
        im = cv2.GaussianBlur(im,(7,7),1.5)
        im = cv2.Canny(im,75,150)
        kernel = np.ones((2,2), np.uint8)  # Taille du noyau
        im = cv2.dilate(im, kernel, iterations=1)
        im = cv2.erode(im, kernel, iterations=1)
        IM = Image.fromarray(im)
    return IM.point(lambda p: p > 128 and 255)  


def getdots(IM):
    print("getting contour points...")
    PX = IM.load()
    dots = []
    w,h = IM.size
    for y in range(h-1):
        row = []
        for x in range(1,w):
            if PX[x,y] == 255:
                if len(row) > 0:
                    if x-row[-1][0] == row[-1][-1]+1:
                        row[-1] = (row[-1][0],row[-1][-1]+1)
                    else:
                        row.append((x,0))
                else:
                    row.append((x,0))
        dots.append(row)
    return dots
    
def connectdots(dots):
    print("connecting contour points...")
    contours = []
    for y in range(len(dots)):
        for x,v in dots[y]:
            if v > -1:
                if y == 0:
                    contours.append([(x,y)])
                else:
                    closest = -1
                    cdist = 100
                    for x0,v0 in dots[y-1]:
                        if abs(x0-x) < cdist:
                            cdist = abs(x0-x)
                            closest = x0

                    if cdist > 3:
                        contours.append([(x,y)])
                    else:
                        found = 0
                        for i in range(len(contours)):
                            if contours[i][-1] == (closest,y-1):
                                contours[i].append((x,y,))
                                found = 1
                                break
                        if found == 0:
                            contours.append([(x,y)])
        for c in contours:
            if c[-1][1] < y-1 and len(c)<4:
                contours.remove(c)
    return contours


def getcontours(IM,sc=2):
    print("generating contours...")
    IM = find_edges(IM)
    IM1 = IM.copy()
    IM2 = IM.rotate(-90,expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    dots1 = getdots(IM1)
    contours1 = connectdots(dots1)
    dots2 = getdots(IM2)
    contours2 = connectdots(dots2)

    for i in range(len(contours2)):
        contours2[i] = [(c[1],c[0]) for c in contours2[i]]    
    contours = contours1+contours2

    for i in range(len(contours)):
        for j in range(len(contours)):
            if len(contours[i]) > 0 and len(contours[j])>0:
                if distsum(contours[j][0],contours[i][-1]) < 8:
                    contours[i] = contours[i]+contours[j]
                    contours[j] = []

    for i in range(len(contours)):
        contours[i] = [contours[i][j] for j in range(0,len(contours[i]),8)]


    contours = [c for c in contours if len(c) > 1]

    for i in range(0,len(contours)):
        contours[i] = [(v[0]*sc,v[1]*sc) for v in contours[i]]

    for i in range(0,len(contours)):
        for j in range(0,len(contours[i])):
            contours[i][j] = int(contours[i][j][0]+10*perlin.noise(i*0.5,j*0.1,1)),int(contours[i][j][1]+10*perlin.noise(i*0.5,j*0.1,2))

    return contours


def hatch(IM,sc=16):
    print("hatching...")
    PX = IM.load()
    w,h = IM.size
    lg1 = []
    lg2 = []
    for x0 in range(w):
        for y0 in range(h):
            x = x0*sc
            y = y0*sc
            if PX[x0,y0] > 144:
                pass
                
            elif PX[x0,y0] > 64:
                lg1.append([(x,y+sc/4),(x+sc,y+sc/4)])
            elif PX[x0,y0] > 16:
                lg1.append([(x,y+sc/4),(x+sc,y+sc/4)])
                lg2.append([(x+sc,y),(x,y+sc)])

            else:
                lg1.append([(x,y+sc/4),(x+sc,y+sc/4)])
                lg1.append([(x,y+sc/2+sc/4),(x+sc,y+sc/2+sc/4)])
                lg2.append([(x+sc,y),(x,y+sc)])

    lines = [lg1,lg2]
    for k in range(0,len(lines)):
        for i in range(0,len(lines[k])):
            for j in range(0,len(lines[k])):
                if lines[k][i] != [] and lines[k][j] != []:
                    if lines[k][i][-1] == lines[k][j][0]:
                        lines[k][i] = lines[k][i]+lines[k][j][1:]
                        lines[k][j] = []
        lines[k] = [l for l in lines[k] if len(l) > 0]
    lines = lines[0]+lines[1]

    for i in range(0,len(lines)):
        for j in range(0,len(lines[i])):
            lines[i][j] = int(lines[i][j][0]+sc*perlin.noise(i*0.5,j*0.1,1)),int(lines[i][j][1]+sc*perlin.noise(i*0.5,j*0.1,2))-j
    return lines


def sketch(path):
    IM = None
    possible = [path,"images/"+path,"images/"+path+".jpg","images/"+path+".png","images/"+path+".tif"]
    for p in possible:
        try:
            IM = Image.open(p)
            break
        except FileNotFoundError:
            print("The Input File wasn't found. Check Path")
            exit(0)
            pass
    w,h = IM.size

    IM = IM.convert("L")
    IM=ImageOps.autocontrast(IM,10)

    lines = []
    if draw_contours:
        lines += getcontours(IM.resize((resolution//contour_simplify,resolution//contour_simplify*h//w)),contour_simplify)
    if draw_hatch:
        lines += hatch(IM.resize((resolution//hatch_size,resolution//hatch_size*h//w)),hatch_size)

    lines = sortlines(lines)
    if show_bitmap:
        disp = Image.new("RGB",(resolution,resolution*h//w),(255,255,255))
        draw = ImageDraw.Draw(disp)
        for l in lines:
            draw.line(l,(0,0,0),5)
        disp.show()

    f = open(export_path,'w')
    f.write(makesvg(lines))
    f.close()
    print(len(lines),"strokes.")
    print("done.")
    return lines


def makesvg(lines):
    print("generating svg file...")
    out = '<svg xmlns="http://www.w3.org/2000/svg" version="1.1">'
    for l in lines:
        l = ",".join([str(p[0]*0.5)+","+str(p[1]*0.5) for p in l])
        out += '<polyline points="'+l+'" stroke="black" stroke-width="2" fill="none" />\n'
    out += '</svg>'
    return out

def main(input_path='IMG_0450.jpg', output_path='output/out.svg', show_bmp=False, 
         no_contour=False, no_hatch=False, no_cv_mode=False, main_hatch_size=1, main_contour_simplify=1):
    # Utilisation de noms de variables internes pour éviter le conflit avec les globales
    global export_path, draw_hatch, draw_contours, hatch_size, contour_simplify, show_bitmap, no_cv
    export_path = output_path
    draw_hatch = not no_hatch
    draw_contours = not no_contour
    hatch_size = main_hatch_size  # Assignation de la variable interne à la globale
    contour_simplify = main_contour_simplify  # Assignation de la variable interne à la globale
    show_bitmap = show_bmp  # Renommé pour éviter le conflit
    no_cv = no_cv_mode  # Renommé pour éviter le conflit
    
    # Appel à la fonction sketch
    print("Starting sketch process...")
    sketch(input_path)
    print("Sketch process completed.")

# Exécution de la fonction principale avec les paramètres souhaités
main(input_path='IMG_0450.jpg', output_path='output/out.svg',no_hatch=True)