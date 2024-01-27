import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import math
import os
import numpy as np
from flask import Flask, render_template, request
import copy
import re
import glob
import cv2
import time
import grabarPlots

listaVideos = []
listaAnimaciones = []
img_files = []
inicio = time.time()

app = Flask(__name__)

@app.route('/')
def pagina():
    return render_template('IA.html', bandera=False)

img_files = []
intervaloInferior = 0
intervaloSuperior = 0
rango = 0
resolucion = 0
nResolucion = 0
bits = 0
puntos = 0
poblacionMaxima = 0
poblacionActual = 0
poblacionInicial = 0
Pmi = 0
Pmg = 0
n = 0
k = 0
g = 1
x = []
fx = []
iDecimal = []
peorCaso = []
mejorCaso = []
promedio = []
hijos = []
iteraciones = 0
generacion = 0
copia = []
generaciones = []
poblacionCopia = []
xCopia = []
fxCopia = []
iDecimalCopia = []
problema = ""

poblacion = []

def calculos():
    global n, nResolucion, bits, rango, puntos, intervaloInferior, intervaloSuperior, resolucion, poblacionInicial, Pmi, Pmg, poblacionMaxima
    rango = intervaloSuperior - intervaloInferior

    puntos = (rango / resolucion) + 1

    while True:
        if 2 ** (n - 1) < puntos <= 2 ** n:
            bits = n
            break
        n += 1

    nResolucion = rango / ((2 ** n) - 1)

def crearIndividuos():
    global poblacionInicial, poblacion

    for i in range(poblacionInicial):
        individuo = ""
        for j in range(bits):
            bit = random.choice([0, 1])
            individuo = individuo + str(bit)
        poblacion.append(individuo)
    print("individuos al inicio")
    print(poblacion)  


def evaluacion():
    global poblacion, iDecimal, intervaloInferior, intervaloSuperior, rango, fx, peorCaso, mejorCaso, promedio, x, generaciones, generacion

    x = []
    fx = []
    iDecimal = []

    generaciones.append(generacion)
    generacion += 1

    for i in range(len(poblacion)):
        iDecimal.append(int(poblacion[i], 2))

        xTemp = intervaloInferior + iDecimal[i] * (rango / ((2 ** n) - 1))
        x.append(xTemp)
        #par1 = 5 * x[i]
        #par2 = 3*((x[i])**2)
        #par3 =  (x[i]**3)
        #par1 = math.sqrt(abs(x[i]**3))
        #par2 = math.sin(x[i]**2)

        fxTemp = (3 * x[i]) * (math.cos(math.radians(x[i]))) * (math.sin(math.radians(x[i]))) * (math.log(abs(x[i]) + 1))
        #fxTemp = math.sqrt(abs(x[i]**3)) * math.sin(x[i]**2)

        #par3-((x[i]) ** 3)*(math.cos(math.radians(par1)))

         #fxTemp = (((math.sin(math.radians(x[i])))* (x[i]) ** 3)/100) + ((x[i]) ** 2 * (math.cos(math.radians(x[i]))))
        #fxTemp = ((x[i]))*(math.cos(math.radians(x[i])))*(math.sin(math.radians(2*x[i]))) + 2*(x[i])
        #fxTemp = (x[i] ** 2) * (math.cos(math.radians(wi))) - 3 * x[i]
        fx.append(fxTemp)

    if problema == "minimizar":
        peorCaso.append(max(fx))
        mejorCaso.append(min(fx))
    elif problema == "maximizar":
        peorCaso.append(min(fx))
        mejorCaso.append(max(fx))
        
    promedio.append(sum(fx) / len(fx))

    print("mejor caso")
    print(mejorCaso)
    print("peor caso")
    print(peorCaso)
    print("caso promedio")
    print(promedio)
    print("x")
    print(x)
    print("fx")
    print(fx)
    print("la i decimal")
    print(iDecimal)

def reproduccion():
    global poblacion, hijos, x, fx, iDecimal

    #probabilidad de cruza
    for i in range(len(poblacion)):
        for j in range(len(poblacion)):
            print("el individuo numero" + str(i) + "va a cruzarse")
            #buscar pareja
            pareja = j
            print("se cruza con la pareja" + str(pareja))
            #cruza
            puntoMaximo = len(poblacion[0]) - 2
            
            puntoCorte = random.randint(0, puntoMaximo)

            print("corte")
            print(puntoCorte)
            
            padreTemp = poblacion[i]
            parejaTemp = poblacion[pareja]

            parte1 = padreTemp[0:puntoCorte+1]
            parte3 = parejaTemp[0:puntoCorte+1]

            parte2 = parejaTemp[puntoCorte+1:]
            parte4 = padreTemp[puntoCorte+1:]

            hijo1 = parte1 + parte2
            hijo2 = parte3 + parte4

            hijos.append(hijo1)
            hijos.append(hijo2)

def mutacion():
    global Pmg, Pmi, hijos, poblacion
    print("hijos antes de mutar")
    print(hijos)
    # mutacion del individuo
    for i in range(len(hijos)):
        prob = random.randint(0, 100)
        print("probabilidad de la posibilidad de mutar" + str(prob))
        if prob <= Pmi:
            # mutacion del gen
            print("el hijo numero" + hijos[i] + "muta")
            hijo = hijos[i]
            for j in range(len(hijo)+1):
                if j != 0:
                    prob2 = random.randint(0, 100)
                    print("probabilidad de mutar un bit" + str(prob2))
                    if prob2 <= Pmg:
                        print("el bit numero")
                        print(j)
                        print("esta mutando")
                        bitHijo = hijo[j-1]
                        if bitHijo == "0":
                            hijo = hijo[:j-1] + "1" + hijo[j:]#1
                        else:
                            hijo = hijo[:j-1] + "0" + hijo[j:]#0
                        hijos[i] = hijo
                        
    print("hijos despues de mutar")
    print(hijos)
    poblacion.extend(hijos)
    print("poblacion con hijos")
    print(poblacion)
    hijos = []

def eliminate_repetitions(population_complete, bits_complete, x_complete, fx_complete):
    global fx, x, iDecimal, poblacion

    seen = set()
    indices_to_keep = [i for i, x in enumerate(population_complete) if not (x in seen or seen.add(x))]

    iDecimal[:] = [population_complete[i] for i in indices_to_keep]
    poblacion[:] = [bits_complete[i] for i in indices_to_keep]
    x[:] = [x_complete[i] for i in indices_to_keep]
    fx[:] = [fx_complete[i] for i in indices_to_keep]

def poda():
    global fx, poblacion, x, iDecimal   

    print("imprimiendo la poblacion antes de la eliminacion de iguales y el fx")
    print(poblacion)
    print(fx)
    print(iDecimal)
    print("imprimiendo la poblacion despues de la eliminacion de iguales")

    eliminate_repetitions(iDecimal, poblacion, x, fx)
    print(poblacion)
    print(iDecimal)

    print("imprimiendo la longitud de la problacioNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
    print(len(poblacion))
    while len(poblacion) > poblacionMaxima:
        ultimaPosicion = len(poblacion) - 1
        individuoAleatoreo = random.randint(0, ultimaPosicion)
        print("tamaño de la poblacion")
        print(len(poblacion))


        if problema == "minimizar":
            mejorIndividuo = min(fx)
        elif problema == "maximizar":
            mejorIndividuo = max(fx)

        if individuoAleatoreo != fx.index(mejorIndividuo):
            print("PODAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            print("imprimiendo la poblacion antes de la poda y el fx")
            print(poblacion)
            print(fx)
            
            print("imprimiendo la posicion del individuo a remover")
            print(individuoAleatoreo)
            
            del poblacion[individuoAleatoreo]
            del fx[individuoAleatoreo]
            del x[individuoAleatoreo]
            del iDecimal[individuoAleatoreo]
            print("imprimiendo la poblacion despues de la poda y el fx y el x y iDecimal")
            print(poblacion)
            print(fx)
            print(x)
            print(iDecimal)

def guardarGeneracion():
    global poblacion, poblacionCopia, xCopia, fxCopia, iDecimalCopia, x, fx, iDecimal

    poblacionCopia.append(copy.deepcopy(poblacion))
    iDecimalCopia.append(copy.deepcopy(iDecimal))
    xCopia.append(copy.deepcopy(x))
    fxCopia.append(copy.deepcopy(fx))

def generarGrafica0():
    global poblacion, x, fx

    if problema == "minimizar":
        fxMejor = fx.index(min(fx))
        fxPeor = fx.index(max(fx))
    elif problema == "maximizar":
        fxMejor = fx.index(max(fx))
        fxPeor = fx.index(min(fx))

    plt.figure(figsize=(10, 6))

    color_primero = 'green'
    color_segundo = 'yellow'
    color_tercero = 'red'

    fxMejorTemp = fx[fxMejor]
    fxPeorTemp = fx[fxPeor]

    xMejorTemp = x[fxMejor]
    xPeorTemp = x[fxPeor]
    fxPromTemp = sum(fx) / len(fx)
    xPromTemp = sum(x) / len(x)

    posiciones_especificas = [fxMejor, fxPeor]

    color_resto = 'blue'
    for i in range(len(x)):
        if i not in posiciones_especificas:
            plt.scatter(x[i], fx[i], color=color_resto, marker='o', label=iDecimal[i])

    plt.scatter(xMejorTemp, fxMejorTemp, color=color_primero, marker='o', label=iDecimal[fxMejor])
    plt.scatter(xPeorTemp, fxPeorTemp, color=color_tercero, marker='o', label=iDecimal[fxPeor])
    plt.scatter(xPromTemp, fxPromTemp, color=color_segundo, marker='o', label="promedio")

    plt.title('generacion 0')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)



    plt.legend(loc='upper right', bbox_to_anchor=(1.13, 1.165))
    plt.savefig(os.path.join('static', 'generaciones', '0.png'))

def generarOtrasGraficas(g):
    if problema == "minimizar":
        fxMejor = fx.index(min(fx))
        fxPeor = fx.index(max(fx))
    elif problema == "maximizar":
        fxMejor = fx.index(max(fx))
        fxPeor = fx.index(min(fx))

    plt.figure(figsize=(10, 6))

    color_primero = 'green'
    color_segundo = 'yellow'
    color_tercero = 'red'

    fxMejorTemp = fx[fxMejor]
    fxPeorTemp = fx[fxPeor]

    xMejorTemp = x[fxMejor]
    xPeorTemp = x[fxPeor]
    fxPromTemp = sum(fx) / len(fx)
    xPromTemp = sum(x) / len(x)

    posiciones_especificas = [fxMejor, fxPeor]

    color_resto = 'blue'
    for i in range(len(x)):
        if i not in posiciones_especificas:
            plt.scatter(x[i], fx[i], color=color_resto, marker='o', label=iDecimal[i])

    plt.scatter(xMejorTemp, fxMejorTemp, color=color_primero, marker='o', label=iDecimal[fxMejor])
    plt.scatter(xPeorTemp, fxPeorTemp, color=color_tercero, marker='o', label=iDecimal[fxPeor])
    plt.scatter(xPromTemp, fxPromTemp, color=color_segundo, marker='o', label="promedio")

    plt.title('generacion ' + str(g))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.13, 1.165))

    plt.savefig(os.path.join('static', 'generaciones', f'{g}.png'))

def generarGraficaEvolucionFitness():
    global generaciones, peorCaso, mejorCaso

    plt.figure(figsize=(10, 6))

    plt.plot(generaciones, peorCaso, color='red', linestyle='-', label='Peor Caso')
    plt.plot(generaciones, promedio, color='yellow', linestyle='-', label='Promedio')
    plt.plot(generaciones, mejorCaso, color='green', linestyle='-', label='Mejor Caso')

    plt.scatter(generaciones, peorCaso, color='red', marker='o')
    plt.scatter(generaciones, promedio, color='yellow', marker='o')
    plt.scatter(generaciones, mejorCaso, color='green', marker='o')

    plt.title('Evaluación del fitness')
    plt.xlabel('Iteraciones/generaciones')
    plt.ylabel('f(x)')
    plt.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.13, 1.165))
    plt.savefig(os.path.join('static', f'grafica_evaluacion.png'))

def inicializacion():
    calculos()
    crearIndividuos()
    evaluacion()
    guardarGeneracion()
    generarGrafica0()

def optimizacion(g):
    reproduccion()
    mutacion()
    evaluacion()
    generarOtrasGraficas(g)
    guardarGeneracion()
    poda()

@app.route('/procesar_formulario', methods=['POST'])
def procesar_formulario():
    global intervaloSuperior, img_files, intervaloInferior, poblacionInicial, poblacionMaxima, iteraciones, resolucion, Pmi, Pmg, rango, nResolucion, bits, puntos, poblacion, iDecimal, x, fx, peorCaso, mejorCaso, promedio, generaciones, k, g, problema

    intervaloInferior = int(request.form.get('inferior'))
    intervaloSuperior = int(request.form.get('superior'))
    problema = request.form.get('problema')
    poblacionInicial = int(request.form.get('inicial'))
    poblacionMaxima = int(request.form.get('maxima'))
    iteraciones = int(request.form.get('iteraciones'))
    resolucion = float(request.form.get('resolucion'))
    Pmi = int(request.form.get('pmi'))
    Pmg = int(request.form.get('pmg'))

    print("imprimiendo problema")
    print(problema)

    # iteraciones = iteraciones - 1

    inicializacion()
    while k < iteraciones:
        optimizacion(g)

        k += 1
        g += 1

        print("imprimiendo mejor caso")
    print(mejorCaso)

    generarGraficaEvolucionFitness()

    print("POBLACION FINAL")
    print(poblacion)
    
    if problema == "minimizar":
        fxMejor = fx.index(min(fx))
        fxPeor = fx.index(max(fx))
    elif problema == "maximizar":
        fxMejor = fx.index(max(fx))
        fxPeor = fx.index(min(fx))

    plt.figure(figsize=(10, 6))

    color_primero = 'green'
    color_segundo = 'yellow'
    color_tercero = 'red'

    fxMejorTemp = fx[fxMejor]
    fxPeorTemp = fx[fxPeor]

    xMejorTemp = x[fxMejor]
    xPeorTemp = x[fxPeor]
    fxPromTemp = sum(fx) / len(fx)
    xPromTemp = sum(x) / len(x)
    plt.scatter(xMejorTemp, fxMejorTemp, color=color_primero, marker='o', label=iDecimal[fxMejor])
    plt.scatter(xPeorTemp, fxPeorTemp, color=color_tercero, marker='o', label=iDecimal[fxPeor])
    plt.scatter(xPromTemp, fxPromTemp, color=color_segundo, marker='o', label="promedio")

    posiciones_especificas = [fxMejor, fxPeor]

    # Establecer un color específico para los demás puntos
    color_resto = 'blue'
    for i in range(len(x)):
        if i not in posiciones_especificas:
            plt.scatter(x[i], fx[i], color=color_resto, marker='o', label=iDecimal[i])

    plt.title('generacion final')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.13, 1.165))
    plt.savefig(os.path.join('static', f'grafica_generacion_final.png'))

    for i in range(g):  
        xTemp = xCopia[i].copy()
        yTemp = fxCopia[i].copy()
        listaAnimaciones.append(grabarPlots.animarPlot(xTemp,yTemp))
        nombre_video = f"video{i+1}.mp4"
        listaVideos.append(nombre_video)
    grabarPlots.unirVariosVideos(listaAnimaciones,listaVideos)

    create_video()

    return render_template('IA2.html', intervaloSuperior=intervaloSuperior, intervaloInferior=intervaloInferior, poblacionInicial=poblacionInicial, poblacionMaxima=poblacionMaxima, resolucion=resolucion, Pmi=Pmi, Pmg=Pmg, rango=rango, nResolucion=nResolucion, bits=bits, puntos=puntos, poblacion=poblacion, iDecimal=iDecimal, x=x, fx=fx, peorCaso=peorCaso, mejorCaso=mejorCaso, promedio=promedio, k=k, iteraciones=iteraciones, bandera=True, poblacionCopia=poblacionCopia, iDecimalCopia=iDecimalCopia, xCopia=xCopia, fxCopia=fxCopia)

@app.route('/agregar', methods=['POST'])
def agregar():
    global intervaloInferior, intervaloSuperior, listaAnimaciones, img_files, poblacionInicial, poblacionMaxima, poblacionActual, nResolucion, iteraciones, generacion, resolucion, puntos, rango, bits, Pmi, Pmg, n, k, g, poblacionCopia, iDecimalCopia, generaciones, mejorCaso, peorCaso, promedio, fxCopia, xCopia, hijos, copia, fx, x, poblacion, iDecimal

    intervaloInferior = 0
    intervaloSuperior = 0
    poblacionInicial = 0
    poblacionMaxima = 0
    poblacionActual = 0
    nResolucion = 0
    iteraciones = 0
    generacion = 0
    resolucion = 0
    puntos = 0
    rango = 0
    bits = 0
    Pmi = 0
    Pmg = 0
    n = 0
    k = 0
    g = 1

    img_files = []
    poblacionCopia = []
    iDecimalCopia = []
    generaciones = []
    mejorCaso = []
    poblacion = []
    promedio = []
    iDecimal = []
    peorCaso = []
    fxCopia = []
    xCopia = []
    hijos = []
    copia = []
    fx = []
    x = []

    listaAnimaciones = []

    img_files = glob.glob("./static/generaciones/*.png")

    for i in range(k, len(img_files)):
        img_file = img_files[i]
        os.remove(img_file)

    return render_template('IA.html', bandera=False)

def create_video():
    img_files = glob.glob("./static/generaciones/*.png")
    img_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    img = cv2.imread(img_files[0])
    height, width, layers = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('./static/generaciones/generations.mp4', fourcc, 0.5, (width, height))
    for img_file in img_files:
        img = cv2.imread(img_file)
        video.write(img)
    video.release()

if __name__ == '__main__':
    app.run(debug=True)

