# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 18:25:20 2026

@author: antob
"""

import AccesoArchivos as Acceso
import Graficacion as gf

datos_movimiento, datos_ojos_abiertos, datos_ojos_cerrados = Acceso.cargar_datos()

eeg_mov, eeg_OA, eeg_OC = Acceso.obtener_datos_EEG(datos_movimiento, datos_ojos_abiertos, datos_ojos_cerrados)
emg_mov, emg_OA, emg_OC = Acceso.obtener_datos_EMG(datos_movimiento, datos_ojos_abiertos, datos_ojos_cerrados)

Frec_muestreo = 2000
gf.graficar_EEG(eeg_mov, eeg_OA, eeg_OC, Frec_muestreo, unidades="uV",duracion_seg=1)
gf.graficar_EMG(emg_mov, emg_OA, emg_OC, Frec_muestreo, unidades="uV", duracion_seg=5)
