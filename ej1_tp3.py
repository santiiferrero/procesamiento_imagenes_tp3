import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

#Funcion para mostrar la imagen
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
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



"""Recibe un video y devuelve un frame con los datos quietos """
def detectar_frame(cap):

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_con_dados = []
    n_frame = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if n_frame % 10 == 0:
                frame_example = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Detectar dados
                red_channel = frame_example[:,:,0]  # Seleccionamos el canal rojo
                green_channel = frame_example[:,:,1]  # Seleccionamos el canal verde
                frame_binary = ((red_channel > 80) & (green_channel < 70)).astype(np.uint8)  # imagen binaria
                # Aplicar componentes conectadas
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_binary, connectivity=8)
                dice_centroids = []
                image_area = stats[0][4]
                for n in range(1, num_labels):  # Filtramos el fondo
                    if image_area * 0.001 < stats[n][4] < image_area * 0.01:
                       dice_centroids.append(centroids[n])

                #Frame con color
                frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
                if len(dice_centroids) == 6:
                    frames_con_dados.append(frame)
                 
            n_frame += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break

    return frames_con_dados[-1] #Elegimos el último frame así nos aseguramos que los dados no estén en movimiento



'''Crea una nueva imagen con los dados y también devuelve una lista de diccionarios con características de los dados'''
def detectar_dados(frame):
   
    # Proceso de detección de dados
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar dados
    red_channel = frame[:,:,0]  # Seleccionamos el canal rojo
    frame_binary = (red_channel > 70).astype(np.uint8)  # imagen binaria

    lista_objetos = []
    # Aplicar componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_binary, connectivity=8)
    dice_centroids = []

    labeled_image = np.zeros_like(frame_binary)  # Creamos una imagen para volcar los dados
    image_area = stats[0][4]  # Área de la imagen entera
    for n in range(1, num_labels):  # Filtramos el fondo

        # Creamos diccionario para almacenar el objeto y sus datos
        objeto_dicc = {}

        obj = (labels == n).astype(np.uint8)
        if image_area * 0.002 < stats[n][4] < image_area * 0.01:

            # Guardamos el centroide
            objeto_dicc['labels'] = obj
            objeto_dicc['centroid'] = centroids[n]

            # Recortamos la region del dado y la guardamos
            x, y, w, h = stats[n][:4]  # Coordenadas del cuadro delimitador
            objeto_dicc['recorte'] = frame[y:y + h, x:x + w]

            labeled_image[obj == 1] = 255
            dice_centroids.append(centroids[n])
            lista_objetos.append(objeto_dicc)
        

    return labeled_image, lista_objetos




'''Al ingresarle un dado, te devuelve cual es la puntuacion del mismo'''

def valor(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, thresh_img = cv2.threshold(gray_image, thresh=128, maxval=255, type=cv2.THRESH_BINARY)  # Umbralamos

  # Hacemos componentes conectadas para separar cada punto del dado
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, connectivity = 8, ltype=cv2.CV_32S)
  #labels_color = np.uint8(255/(num_labels-1)*labels)                  # Llevo el rango de valores a [0 255] para diferenciar mejor los colores
                                      
  #im_color = cv2.applyColorMap(labels_color, cv2.COLORMAP_JET)
  #im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)                # El mapa de color que se aplica está en BGR --> convierto a RGB
  #imshow(im_color)
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
      #print(stats[i][4])
      if 1/fp > 8 and 1/fp < 17 and stats[i][4] > 4 and stats[i][4] < 15:
        labeled_image[obj == 1] = 255
        # Contamos los puntos
        numero_dado += 1
  return(str(numero_dado))




'''
Le asigna la puntuacion correspondiente a cada dado
'''
def numero_dados(image):

    #devuelve el ulitmo frame y el ultimo frame con gcan aplicado
    frame = detectar_frame(image)

    #devuelve una imagen con los dados, y una lista de diccionarios con caracteritisticas de los mismos
    dados,lista_dados = detectar_dados(frame)


    for dado in lista_dados:
      dado['valor'] = valor(dado['recorte'])

    return lista_dados,frame


'''
Ejecucion del codigo, donde pedimos al usuario que ingrese el nombre del video
'''
# Video path... Para ejecutar desde la terminal
video_path = sys.argv[1] +'.mp4'

#Sino ingresando a mano
#video_path = 'tirada_1.mp4'

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    dados_list, frame = numero_dados(cap)

    # Parámetros para el texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 4
    font_color = (0, 255, 0)

    for objeto in dados_list:
        obj = (objeto['labels']).astype(np.uint8)
        if objeto['valor'] != '0' :
            frame[obj == 1, 2] = 255  
            cv2.putText(frame, objeto['valor'], (int(objeto['centroid'][0]), int(objeto['centroid'][1])),
                        font, font_scale, font_color, font_thickness)

    # Muestra la imagen resultante
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convertimos de BGR A RGB
    plt.show()

    # Espera la tecla 'q' durante 1 milisegundo
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Libera el objeto VideoCapture
cap.release()
cv2.destroyAllWindows()







#PASOS: VIDEO 1 DE EJEMPLO

#video_path = 'tirada_1.mp4'

#cap = cv2.VideoCapture(video_path)

#imshow(detectar_frame(cap)) #aplicamos detectar_frame

#frame = detectar_frame(cap) # guardamos en frame 

#labeled_image, lista_objetos = detectar_dados(frame) #aplicamos detectar dados

#imshow(labeled_image)

#lista_objetos 

#Luego con las funciones valor y numero_dados iteraran sobre los dados de los frames y le asignaran un valor, ingresandolo como 'valor' en lista_objetos

#Por ultimo se grafica en la imagen original tomada como frame.

