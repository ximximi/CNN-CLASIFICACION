# Clasificación Automatizada de Entornos Inmobiliarios mediante CNN (Pipeline End-to-End)

**Asignatura:** Técnicas Avanzadas de Machine Learning  
**Integrantes:** Felipe Rubio, Ximena Soliz, Yassmin Bazan  
**Fecha:** Noviembre 2026  

## 1. Problema de Negocio
Automatización de la clasificación visual de fotografías aéreas y de entorno tomadas por drones para tasación inmobiliaria en 6 categorías (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`), superando las limitaciones espaciales de redes densas clásicas (MLP).

## 2. Arquitectura del Pipeline (9 Etapas)
El proyecto está encapsulado en un flujo modular y reproducible (`SEED = 29`) con persistencia incremental:
1. **Ingesta (`cargar_datos`):** Descarga automatizada y partición `train (80%) / val (20%) / test`.
2. **EDA y Balance (`analizar_dataset`):** Auditoría de distribución de clases (*Ratio Max/Min = 1.15x*).
3. **Preprocesamiento (`preprocesar_datos`):** Normalización `[0, 1]` + `.cache().prefetch(AUTOTUNE)`.
4. **Arquitectura (`construir_modelo`):** Constructor paramétrico CNN (`3 bloques Conv2D + MaxPool`, `Flatten`, `Dense`, `Softmax`).
5. **Entrenamiento (`entrenar_modelo`):** Optimización con `Adam` y `categorical_crossentropy`.
6. **Regularización:** *Data Augmentation* (`RandomFlip`, `RandomRotation(0.03)`, `RandomZoom(0.1)`), *Dropout*, *BatchNorm*, *L1/L2* y *Early Stopping*.
7. **Optimización (`ejecutar_catalogo_regularizacion` + `optimizar_hiperparametros`):** Evaluación sistemática en 3 fases (14 experimentos) + búsqueda controlada de hiperparámetros con checkpoints en `.csv`/`.json`.
8. **Evaluación (`evaluar_modelo` + `evaluate_roc_auc`):** Métricas multiclase, Matriz de Confusión y curvas ROC-AUC (One-vs-Rest).
9. **Auditoría XAI y Ética (`auditar_modelo` + `auditar_sesgos_modelo`):** Mapas de calor **Grad-CAM** sobre `Conv_3`, análisis de pares de confusión y detección de errores de alta confianza (`>= 80%`).

## 3. Evolución de Resultados
| Hito | Arquitectura | Val Loss | Test Accuracy | F1-Score | Parámetros |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ev. 1** | MLP Base | 1.3200 | 54.37% | 0.5410 | 8.65M |
| **Ev. 2** | CNN Sin Regularizar | ~0.68 | ~78.40% | ~0.78 | 4.83M |
| **Ev. 3** | **CNN + DA + Drop(0.5) + ES (`Dense 256`)** | **0.3950** | **86.17%** | **0.8609** | **9.56M** |

## 4. Cómo Ejecutar
1. Abrir `notebooks/E3_Pipeline_CNN_Entornos_Inmobiliarios.ipynb` en Google Colab con entorno **GPU T4**.
2. Ejecutar `ctx = ejecutar_pipeline_completo(forzar_reinicio=False)` para correr las 9 etapas de extremo a extremo.
