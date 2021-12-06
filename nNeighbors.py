
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

def getCSV(featuresRoute, testRoute):
    '''
    Devuelve los archivos CSV en forma de Dateframe para poder ser utilizados
    
    Parametros
    -----------
    featuresRoute : str
        la ruta al archivo features.csv
        
    testRoute : str
        la ruta al archivo respuesta.csv
        
    Returns
    -----------
    featureData, testData : Dataframe
        Los dos archivos CSV convertido a dataframe
    '''
    dataHead = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    testHead = ['C1']
    featureData = pd.read_csv(featuresRoute, names=dataHead, na_values=['no info', '?'])
    testData = pd.read_csv(testRoute, names=testHead, na_values=['no info', '?'])
    featureData = featureData.fillna(0)
    testData = testData.fillna(0)
    
    return featureData,testData

def nearestNeightborModel(featureData,testData):
    '''
    Recibe 2 dataframe para crear y entrenar un modelo de vecino mas cercano
    
    Parametros
    -------------
    featureData, testData : Dataframe
        recibe los 2 dataframe
        
    Returns
    ---------------
    modelo : kNeighborsClassifier
        el clasificador ya entrenado
        
    x_trainData, y_trainData : list
        los datos de entrenamiento divididos
        
    x_testData, y_testData : list
        los datos de prueba divididos
    '''
    # Normalizando Datos 
    scaler = StandardScaler()

    scaler.fit(featureData)
    scaled_features = scaler.transform(featureData)
    scaled_data = pd.DataFrame(scaled_features, columns=featureData.columns)
    
      #Dividiendo Muestra para entrenamiento y prueba donde X es scaled_data y Y es testData
    x_trainData, x_testData, y_trainData, y_testData = train_test_split(
        scaled_data, testData['C1'], test_size=0.2)

    #Creando el modelo
    modelo = KNeighborsClassifier(n_neighbors=1)
    #Entrenando el modelo
    modelo.fit(x_trainData, y_trainData)
    
    return modelo,x_trainData, x_testData, y_trainData, y_testData

def evalua(y_testData,prediccion):
    '''
    Imprime el desempeño de la tecnica.
    
    Parametros
    ----------------
    y_testData : list
        Lista de los valores usados para las pruebas
    prediccion : ndarray
        prediccion del modelo para los datos provistos
        
    Returns
    ---------------
    Void
    '''
    print(classification_report(y_testData, prediccion))
    print(confusion_matrix(y_testData, prediccion))

def main():
    
    featureData,testData = getCSV('./data/features.csv','./data/respuesta.csv')
    
    modelo,x_trainData, x_testData, y_trainData, y_testData = nearestNeightborModel(featureData,testData)
    
    prediccion = modelo.predict(x_testData)

    # Generando matriz de confusion y rep de clasificacion
    evalua(y_testData,prediccion)

    # Obteniendo mejor K
    rango = range(1, 40)
    error = []

    for i in rango:
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(x_trainData, y_trainData)
        pred_i = knn.predict(x_testData)
        error.append(np.mean(pred_i != y_testData))

    # Generando grafica
    plt.figure(figsize=(12, 6))
    plt.plot(rango, error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Valor x Error')
    plt.xlabel('Valor de K')
    plt.ylabel('Error')
    plt.show()


if __name__ == "__main__":
    main()
