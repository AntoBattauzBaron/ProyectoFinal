# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:08:08 2026

@author: Antonella Battauz Baron

Este script permite acceder y abrir los registros de EEG y EMG

"""
import tkinter as tk
from tkinter import filedialog
import pandas as pd


# %%
"""
    Permite seleccionar la carpeta con los archivos
    
    Retorna:
        ruta de acceso a los archivos 
    """
def elegir_datos():
    try:
        root = tk.Tk()         # Inicializar Tkinter, crea un objeto (root) de la clase TK
        root.attributes('-topmost', True) #Se coloca encima de cualquier ventana
        root.withdraw()         #Oculta un cuadro vacío sin destruirlo
        
        # Busco las rutas a los archivos
        file_paths = tk.filedialog.askopenfilenames(    #File_paths es un vector
            title="Selecciona los archivos con datos",
            filetypes=[("Archivos de texto", "*.txt"), 
                      ("Todos los archivos", "*.*")]
        )
        root.destroy()          #Cierra la ventana una vez seleccionado los archivos
        return file_paths
        
        if not file_paths:
            print("No se seleccionó ningún archivo.")
            return None
        
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        if 'root' in locals():
            root.destroy()
        return None
# %%
    """
    Carga un archivo de registro OpenBCI, saltando las líneas de metadata.
    
    Retorna:
        pd.DataFrame con los datos crudos
    """

def cargar_datos():
    
    posicion_ruta_mov = 0
    posicion_ruta_OA = 1
    posicion_ruta_OC = 2
    
    rutas = elegir_datos()
    datos_movimiento = pd.read_csv(rutas[posicion_ruta_mov], skiprows=4, header=0, skipinitialspace = True)
    datos_ojos_abiertos = pd.read_csv(rutas[posicion_ruta_OA], skiprows=4, header=0, skipinitialspace = True)
    datos_ojos_cerrados = pd.read_csv(rutas[posicion_ruta_OC], skiprows=4, header=0, skipinitialspace = True)
    
    return datos_movimiento, datos_ojos_abiertos, datos_ojos_cerrados

# %%
    """
    Extrae y renombra los canales de EEG de un dataframe crudo de OpenBCI.
    
    Retorna: canales C3, C4 y Cz de EEG para cada condición de evaluación
    
    """

def obtener_datos_EEG(datos_movimiento, datos_ojos_abiertos, datos_ojos_cerrados):
    
    eeg_movimiento = datos_movimiento[["EXG Channel 2", "EXG Channel 3", "EXG Channel 4"]].rename(columns={"EXG Channel 2": "C3", "EXG Channel 3": "C4", "EXG Channel 4": "Cz"})  
    eeg_ojos_abiertos = datos_ojos_abiertos[["EXG Channel 2", "EXG Channel 3", "EXG Channel 4"]].rename(columns={"EXG Channel 2": "C3", "EXG Channel 3": "C4", "EXG Channel 4": "Cz"}) 
    eeg_ojos_cerrados = datos_ojos_cerrados[["EXG Channel 2", "EXG Channel 3", "EXG Channel 4"]].rename(columns={"EXG Channel 2": "C3", "EXG Channel 3": "C4", "EXG Channel 4": "Cz"}) 
    
    return eeg_movimiento, eeg_ojos_abiertos, eeg_ojos_cerrados

#%%
    """
    Extrae y renombra los canales de EMG de un dataframe crudo de OpenBCI.

    Retorna: canales EMG izquierdo y derecho para cada condición de evaluación

    """
def obtener_datos_EMG(datos_movimiento, datos_ojos_abiertos, datos_ojos_cerrados):
    
    emg_movimiento = datos_movimiento[["EXG Channel 0", "EXG Channel 1"]].rename(columns={"EXG Channel 0": "EMGizq", "EXG Channel 1": "EMGder"})  #EMG izq, EMG der
    emg_ojos_abiertos = datos_ojos_abiertos[["EXG Channel 0", "EXG Channel 1"]].rename(columns={"EXG Channel 0": "EMGizq", "EXG Channel 1": "EMGder"})
    emg_ojos_cerrados = datos_ojos_cerrados[["EXG Channel 0", "EXG Channel 1"]].rename(columns={"EXG Channel 0": "EMGizq", "EXG Channel 1": "EMGder"})
    
    return emg_movimiento, emg_ojos_abiertos, emg_ojos_cerrados

#%%

    
    
    
    
    
    
    
    
    