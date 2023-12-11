import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    '''Muestra la imagen pasada por parámetro'''
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

def detectar_dados(frame):
    '''Crea una nueva imagen con los dados y sus centroides detectados'''
    # Proceso de detección de dados
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detectar dados
    red_channel = frame[:,:,0]  # Seleccionamos el canal rojo
    frame_binary = (red_channel > 70).astype(np.uint8)  # imagen binaria

    # Aplicar componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_binary, connectivity=8)
    dice_centroids = []
    statss = []
    labelss= []

    labeled_image = np.zeros_like(frame_binary)  # Creamos una imagen para volcar los dados
    image_area = stats[0][4]  # Area de la imagen entera
    for n in range(1, num_labels):  # Filtramos el fondo
        obj = (labels == n).astype(np.uint8)
        height = stats[n][cv2.CC_STAT_HEIGHT]
        width = stats[n][cv2.CC_STAT_WIDTH]
        ratio = width / height 
        area = stats[n][cv2.CC_STAT_AREA]
        delta = 0.2 # Margen de error para el ratio
        ratio_condition = 1 - delta < ratio < 1 + delta # Buscamos aquellas componentes conectadas cuyo bounding box sea un cuadrado perfecto, lo que es el caso para cualquier posición de un dado 
        area_condition = image_area * 0.001 < area < image_area * 0.01 # Filtramos areas muy chicas o muy grandes
        if ratio_condition and area_condition:
            labeled_image[obj == 1] = 255
            labeled_image = cv2.circle(labeled_image, center=(int(centroids[n][0]),int(centroids[n][1])),radius=0,thickness=20,color=0)
            dice_centroids.append(centroids[n])
            statss.append(stats[n])
            labelss.append(labels[n])            
    return labeled_image, dice_centroids, statss, labelss


def distanciaEuclideana(p1,p2):
    '''Siendo X1 y X2 dos pares ordenados de coordenadas, calcula la distancia euclideana entre ellos dos'''
    x1, y1 = p1
    x2, y2 = p2
    return ((x2-x1)**2 + (y2-y1)**2)**(1/2)


def dadosQuietos(previousFrameCentroids, currentFrameCentroids, threshold):
    '''Detecta si los dados ya están quietos dados dos conjunto de centroides y un umbral.
    El umbral representa el movimiento máximo que pueden tener los centroides más proximos 
    al próximo frame para que se tome como que no hay movimiento'''
    # Primero, chequeamos que en cada frame hay 5 centroides
    if len(previousFrameCentroids) == 5 and len(currentFrameCentroids) == 5:
        # Comparar distancia de los centroides
        # Para esto, iteramos sobre los centroides actuales y buscamos la mínima distancia entre cada centroide anterior
        # Si se cumple la condición de que la distancia es menor que el umbral para alguno de los puntos, se toma como válida
        # la distancia. En el caso de que haya un centroide muy lejos que el resto se termina el programa.
        # Este no es el método más óptimo pero sí el más fácil de entender.
        for currentCentroid in currentFrameCentroids:
            # Hallamos el centroide anterior más cercano usando la función min usando como clave la distancia euclideana
            nearestCentroid = min(previousFrameCentroids, key= lambda c: distanciaEuclideana(currentCentroid, c))
            distance = distanciaEuclideana(nearestCentroid, currentCentroid)
            if distance > threshold:
                # En el caso de que haya un punto muy alejado, todavía los dados no están quietos
                #print(distance)
                return False
        # En el caso de que se cumpla la condición del umbral para todos los centroides, entonces se toma como que se está quieto
        return True
    else:
        # Si no hay 5 centroides en alguno de los dos frames, devolver falso
        return False


def buscovalordados(image):
  '''Recibe la imagen de un dado y determina su valor'''
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, thresh_img = cv2.threshold(gray_image, thresh=128, maxval=255, type=cv2.THRESH_BINARY)  # Umbralamos

  # Hacemos componentes conectadas para separar cada punto del dado
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, connectivity = 8, ltype=cv2.CV_32S)

  labeled_image = np.zeros_like(thresh_img)
  numero_dado = 0
  for i in range(1, num_labels):

      # --- Selecciono el objeto actual -----------------------------------------
      obj = (labels == i).astype(np.uint8)

       # --- Calculo Rho ---------------------------------------------------------
      ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      area = cv2.contourArea(ext_contours[0])
      perimeter = cv2.arcLength(ext_contours[0], True)

      if perimeter == 0 :
        continue
      fp = area / perimeter**2

      # Removemos el caso donde fp = 0, es decir, una componente conectada de un solo punto
      if fp == 0:
        continue

      # Identificamos círculos y nos quedamos con aquellos más grandes (filtramos según área)
      if 1/fp > 8 and 1/fp < 17 and stats[i][4] > 15:
        labeled_image[obj == 1] = 255
        # Contamos los puntos
        numero_dado += 1
  return(str(numero_dado))



# --- Leer y grabar un video ------------------------------------------------

# Video path... Para ejecutar desde la terminal
video_path = sys.argv[1] +'.mp4'

#Sino ingresando a mano
#video_path = 'tirada_1.mp4'

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


frames_con_dado = 0
n_frame = 0

# Verificar si el archivo ya existe
output_file = 'Video-Output.mp4'
counter = 1
while os.path.exists(output_file):
    # Si el archivo ya existe, cambiar el nombre añadiendo un sufijo numerado
    output_file = 'Video-Output_{}.mp4'.format(counter)
    counter += 1

# Ahora, crear el objeto VideoWriter con el nombre de archivo adecuado
out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

distancia_entre_centroides_actual = None
distancia_entre_centroides_anterior = None
previous_centroids=[]
while cap.isOpened():
    ret, frame = cap.read()

    if ret == True:
        if n_frame: 
            example = frame  # Capturamos un frame de ejemplo
            example, current_centroids, statss, labels = detectar_dados(example)
            #print(len(current_centroids))
            example = cv2.resize(example, dsize=(int(width/3), int(height/3)))

            if n_frame % 1 == 0:
                are_dices_steady = dadosQuietos(previous_centroids, current_centroids, 3)
                
                # Luego de comparar, declaramos los centroides del frame actual como centroides anteriores para el próximo frame
                previous_centroids = current_centroids

                if are_dices_steady:
                  frames_consecutivos +=1
                else:
                  frames_consecutivos = 0

                if frames_consecutivos >=2:
                    for st in statss:
                      # Recortar la región del componente conectado
                      x, y, w, h = st[:4]  # Coordenadas del cuadro delimitador
                      dado = frame[y:y + h, x:x + w]
                      valor = buscovalordados(dado)
                      # print('valor= ',valor)

                      cv2.rectangle(frame, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]),(255,0,0), 2)  # Agrego un rectángulo en cada dado
                      cv2.arrowedLine(frame, (round(st[0]+st[2]/2), round(st[1]-40)), (round(st[0]+st[2]/2), st[1]), (255,0,0), 2, tipLength=0.5)
                      cv2.putText(frame, valor, (st[0], round(st[1]-45)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2)

            # --- Muestro por pantalla ------------
            frame_show = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
            #imshow(frame_show)
            # ---------------------------------------------------------------
            out.write(frame)  # grabo frame --> IMPORTANTE: frame debe tener el mismo tamaño que se definio al crear out.


        n_frame += 1


        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()