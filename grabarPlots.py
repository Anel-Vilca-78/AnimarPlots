import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2 as cv
import numpy as np
import os
## Este programita tiene métodos que permiten animar una gráfica, grabarVideo de una gráfica, 
## Además se incorpora un método para grabar varios videos a partir de varias animaciones
## unirVideos permite unir todos los videos previamente guardados

##Nota: En mi caso no puedo reproducir el video creado directamente en vscode, por algún detalle de compatibilidad
##Entre paquetes o algo así, pero si se reproduce con algún reproductor externo.
def animarPlot(x,y):    
    fig,ax = plt.subplots();    
    def actualizarPlot(i):
        ax.clear()        
        ax.scatter(x[:i],y[:i])
        #Esta es una forma para ajustar los ejes, pero uds pueden usar la que gusten se adapten
        ##mejor a sus gráficas
        ax.set_xlim([1.1*np.min(x),1.1*np.max(x)])
        ax.set_ylim([1.1*np.min(y),1.1*np.max(y)])
    ##Usen interval como 0 en caso que quieran que su animación sea lo más rápida posible
    animar = FuncAnimation(fig, actualizarPlot,range(len(x)),interval=0, cache_frame_data=False, repeat = False)
    #plt.show() #Solo para visualizar, no se requiere para generar los videos
    return fig, animar

def grabarVideo(animacion,nombre_video):
        fig, animar = animacion
        animar.save(nombre_video,writer = 'ffmpeg',fps = 1, dpi = 100)
        plt.close(fig)

def unirVariosVideos(listaAnimaciones,listaVideos):
    videos = []
    for i in range(len(listaAnimaciones)):
        fig,animar = listaAnimaciones[i]    
        animar.save(listaVideos[i],writer = 'ffmpeg', fps = 1, dpi=100)
        plt.close(fig)
        videos.append(cv.VideoCapture(listaVideos[i]))
        #os.remove(listaVideos[i])
    ancho = int(videos[0].get(cv.CAP_PROP_FRAME_WIDTH))
    alto = int(videos[0].get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(videos[0].get(cv.CAP_PROP_FPS))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_combinado = cv.VideoWriter('video_final.mp4', fourcc, fps,(ancho,alto))    
    for video in videos:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            video_combinado.write(frame)
    for video in videos:
        video.release()
    video_combinado.release()
    
def reproducirVideo(nombre_video):
    video = cv.VideoCapture(nombre_video)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        cv.imshow('Video Final', frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    video.release()
    cv.destroyAllWindows()