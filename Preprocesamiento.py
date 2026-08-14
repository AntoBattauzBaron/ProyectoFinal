# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:30:46 2026

@author: antob
"""

import numpy as np
import mne
import AccesoArchivos as Acceso
import Graficacion as gf
import pandas as pd


#%%
"""
    Arma una sola matriz con los datos crudos de todos los canales 
    para las distintas condiciones.
 
    Parametros
    emg_df, eeg_df: los dataframes de las dos señales biológicas.
    
    Retorna:
        Arreglo con los datos en un formato y organización permitido por las 
        funciones y que permite preprocesar todo a la vez
    """
def armar_raw(emg_df, eeg_df):
    
    # np.vstack apila los datos de los registros.
    
    datos_uv = np.vstack([
        emg_df['EMGizq'].values,
        emg_df['EMGder'].values,
        eeg_df['C3'].values,
        eeg_df['C4'].values,
        eeg_df['Cz'].values,
    ])
    datos_v = datos_uv * 1e-6  # pasa los datos a V que es lo que requiere mne
    #crea una instancia con la info de adquisición
    info = mne.create_info(ch_names=nombres_canales, sfreq=fs, ch_types=tipos_canales)
    return mne.io.RawArray(datos_v, info) #Crea un objeto de datos a partir de arreglos de registros crudos

# Con los datos en las condiciones necesarias se puede filtrar 
#%%
"""
    Aplica el acondicionamiento en frecuencia a un objeto Raw de MNE que ya
    contiene los 5 canales (EMGizq, EMGder, C3, C4, Cz) con sus ch_types
    asignados. Consideraciones
    
    1. Pasa-altos de EMG con Butterworth de orden 1 (method='iir',
       phase='zero'), que MNE aplica adelante y atrás sobre la señal
       (equivalente a filtfilt), logrando una atenuación resultante de
       orden 2 (40 dB/decada) con fase cero. La frecuencia de diseño
       (fc_emg_diseño) está corregida para que el corte -3dB EFECTIVO, 
       ya con la doble pasada aplicada, caiga en 20 Hz.
    
    Parametros
    ----------
    raw : mne.io.RawArray
        Objeto Raw con los 5 canales EMG/EEG ya armados (ver armar_raw),
        con los tipos de canal ('eeg' / 'emg') correctamente asignados,
        ya que los filtros de EEG y EMG se aplican por separado mediante
        picks.
    
    Retorna
    -------
    mne.io.RawArray
        El mismo objeto raw, ya filtrado. 
"""
def limpiar_emg(raw): 
    tmin = 500 / raw.info['sfreq'] #Recorto las primeras 500 muestras, saco transitorios
    raw.crop(tmin=tmin)
    
    pasadas = 2
    orden = 1
    factor = (2**(1/pasadas) - 1)**(1/(2*orden))
    fc_deseada_pa = 20 #Se debe corregir la frecuencia de corte por doble pasada
    fc_deseada_pb = 450 #Se debe corregir la frecuencia de corte por doble pasada

    raw.filter(l_freq = factor*fc_deseada_pa, h_freq=None, picks='emg',
               method='iir', iir_params = dict(order=1, ftype='butter', phase='zero')) #Recomendación técnica
    
    armonicos_50hz = np.arange(50, fs/2, 50)  # 50, 100, 150, ..., hasta Nyquist
    raw.notch_filter(freqs=armonicos_50hz,notch_widths=2,trans_bandwidth=3.0, picks='all') #Saca todos los armónicos de la frecuencia de línea
    
    return raw

def raw_a_dataframes(raw):
    datos_uv = raw.get_data() * 1e6  # de Volts (MNE) otra vez a uV
    df = pd.DataFrame(datos_uv.T, columns=raw.ch_names)
    return df[['C3', 'C4', 'Cz']], df[['EMGizq', 'EMGder']]

#%% Sección tipo main

datos_movimiento, datos_ojos_abiertos, datos_ojos_cerrados = Acceso.cargar_datos()

eeg_mov, eeg_OA, eeg_OC = Acceso.obtener_datos_EEG(datos_movimiento, datos_ojos_abiertos, datos_ojos_cerrados)
emg_mov, emg_OA, emg_OC = Acceso.obtener_datos_EMG(datos_movimiento, datos_ojos_abiertos, datos_ojos_cerrados)

fs = 2000
nombres_canales = ['EMGizq', 'EMGder', 'C3', 'C4', 'Cz']
tipos_canales   = ['emg', 'emg', 'eeg', 'eeg', 'eeg']

raw_mov = armar_raw(emg_mov, eeg_mov)
raw_OA  = armar_raw(emg_OA, eeg_OA)
raw_OC  = armar_raw(emg_OC, eeg_OC)

raw_mov = limpiar_emg(raw_mov)
raw_OA  = limpiar_emg(raw_OA)
raw_OC  = limpiar_emg(raw_OC)

eeg_mov_f, emg_mov_f = raw_a_dataframes(raw_mov)
eeg_OA_f,  emg_OA_f  = raw_a_dataframes(raw_OA)
eeg_OC_f,  emg_OC_f  = raw_a_dataframes(raw_OC)

gf.graficar_EEG(eeg_mov_f, eeg_OA_f, eeg_OC_f, fs, unidades="uV",duracion_seg=5)
gf.graficar_EMG(emg_mov_f, emg_OA_f, emg_OC_f, fs, unidades="uV",duracion_seg=10)

gf.graficar_espectro_EEG(eeg_mov, eeg_OA, eeg_OC, fs)
gf.graficar_espectro_EMG(emg_mov, emg_OA, emg_OC, fs)

gf.graficar_espectro_EEG(eeg_mov_f, eeg_OA_f, eeg_OC_f, fs)
gf.graficar_espectro_EMG(emg_mov_f, emg_OA_f, emg_OC_f, fs)

