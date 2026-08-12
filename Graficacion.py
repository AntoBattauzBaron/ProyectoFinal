# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:39:28 2026

@author: antob
"""
import matplotlib.pyplot as plt
import numpy as np
"""
    Grafica los 3 canales de EEG (C3, C4, Cz) para las 3 condiciones
    en una grilla de 3 filas x 3 columnas, todo en una sola figura.
 
    Parametros
    ----------
    eeg_mov, eeg_OA, eeg_OC : pd.DataFrame. Cada uno con columnas "C3", "C4", "Cz" 
    fs : float
        Frecuencia de muestreo en Hz
    unidades : str
        Texto para rotular el eje Y 
    duracion_seg : float o None
        Si se especifica (ej: 30), grafica solo los primeros `duracion_seg`
        segundos de cada condicion, en vez del registro completo.
    """
def graficar_EEG(eeg_mov, eeg_OA, eeg_OC, fs, unidades="uV", duracion_seg=None):
    
    # Agrupamos los 3 DataFrames en un diccionario (par clave-valor) para poder recorrerlos
    # con un for en vez de repetir el mismo bloque de codigo 3 veces.
    # El ORDEN de este diccionario define el orden de las filas.
    condiciones = {
        "Movimiento": eeg_mov,
        "Ojos abiertos": eeg_OA,
        "Ojos cerrados": eeg_OC,
    }
 
    canales = ["C3", "C4", "Cz"]  # define el orden de las columnas
 
    # plt.subplots(3, 3, ...) crea UNA figura con una MATRIZ de 3x3 ejes.
    # 'axs' da la posición de cada gráfica: axs[fila, columna].
    # sharex='col' hace que los 3 subplots de una misma columna compartan
    # el mismo eje temporal 
    fig, axs = plt.subplots(3, 3, figsize=(13, 7), sharex='col')
 
    # enumerate(...) el par (condicion, dataframe) de cada condicion.
    # Durante el for, por cada ciclo se otorga a (nombre_cond, df) el par de enumerate()
    for i, (nombre_cond, df) in enumerate(condiciones.items()):
        
        if duracion_seg is not None:
           df = df.iloc[:int(duracion_seg * fs)]
           
        # np.arange(len(df)) genera [0, 1, 2, ..., N-1], un numero por muestra.
        t = np.arange(len(df)) / fs
 
        # Recorremos los 3 canales para llenar las 3 columnas de ESTA fila.
        for j, canal in enumerate(canales):
            ax = axs[i, j]  # el subplot puntual: fila i, columna j
 
            ax.plot(t, df[canal].values, linewidth=0.4, color="#2c5f8a")
            ax.grid(alpha=0.3)  # grilla de fondo tenue, ayuda a leer valores
 
            # Titulo de columna (nombre del canal) solo en la fila de arriba,
            # para no repetirlo 3 veces de forma redundante.
            if i == 0:
                ax.set_title(canal, fontsize=11)
 
            # Etiqueta del eje Y solo en la primera columna: ahi aprovechamos
            # para indicar tambien a que condicion corresponde esa fila.
            if j == 0:
                ax.set_ylabel(f"{nombre_cond}\n[{unidades}]", fontsize=9)
 
            # "Tiempo [s]" solo en la ultima fila (abajo de todo).
            if i == 2:
                ax.set_xlabel("Tiempo [s]")
 
    fig.suptitle("Registro de EEG por condicion", fontsize=13)
    fig.tight_layout()  # ajusta espaciados para que no se superpongan etiquetas
    return fig  
 
#%%
    """
    Grafica los 2 canales de EMG (EMGizq, EMGder) para las 3 condiciones
    en una grilla de 3 filas x 2 columnas, todo en una sola figura.
 
    Parametros
    ----------
    emg_mov, emg_OA, emg_OC : pd.DataFrame. Cada uno con EMGizq y EMGder
        Cada uno con columnas "EMGizq", "EMGder" (salida de obtener_datos_EMG)
    fs : float
        Frecuencia de muestreo en Hz
    unidades : str
        Texto para rotular el eje Y (ej: "uV")
    duracion_seg : float o None
            Mismo parametro de zoom que en graficar_EEG.
    """
def graficar_EMG(emg_mov, emg_OA, emg_OC, fs, unidades="uV", duracion_seg=None):

    condiciones = {
        "Movimiento": emg_mov,
        "Ojos abiertos": emg_OA,
        "Ojos cerrados": emg_OC,
    }
    canales = ["EMGizq", "EMGder"]
 
    fig, axs = plt.subplots(3, 2, figsize=(9, 7), sharex='col')
 
    for i, (nombre_cond, df) in enumerate(condiciones.items()):
 
        if duracion_seg is not None:
            df = df.iloc[:int(duracion_seg * fs)]
 
        t = np.arange(len(df)) / fs
 
        for j, canal in enumerate(canales):
            ax = axs[i, j]
            ax.plot(t, df[canal].values, linewidth=0.4, color="#8a3c2c")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(canal, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{nombre_cond}\n[{unidades}]", fontsize=9)
            if i == 2:
                ax.set_xlabel("Tiempo [s]")
 
    fig.suptitle("Registro de EMG por condicion", fontsize=13)
    fig.tight_layout()
    return fig
 