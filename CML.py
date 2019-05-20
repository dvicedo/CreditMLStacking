import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from warnings import simplefilter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from vecstack import stacking

#Ignore warnings in this first example
simplefilter(action='ignore', category=FutureWarning)
#Feed Dataframe with learning's dataset
datos= pd.read_csv('AprendizajeCredito.csv',delimiter=',',decimal='.')
#Encoding values of string type related with the column BuenPagador
le = LabelEncoder()
datos['BuenPagador']= le.fit_transform(datos['BuenPagador'].astype('str'))
#Execute Undersampling
Cant_MalosPagadores = len(datos[datos['BuenPagador'] == 0])
MalPagadorIndices = datos[datos.BuenPagador== 0].index
BuenPagadorIndices = datos[datos.BuenPagador== 1].index
BuenPagador_random_indices = np.random.choice(BuenPagadorIndices, Cant_MalosPagadores, replace=False)
undersample= np.concatenate([MalPagadorIndices,BuenPagador_random_indices])
#Store balanced DataFrame  and split x,y
dfbalanceado = datos.loc[undersample]
x=dfbalanceado.iloc[:,1:6]
y=dfbalanceado.iloc[:,6:7]
#Data split with 80% dedicated to training and 20% to test.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#Feed new DataFrame with data that we need predict
datosFinal= pd.read_csv('data/nuevos_individuos_credito.csv',delimiter=',',decimal='.')
#Data split of data, specially X_Test
X_test=datosFinal.iloc[:,1:6]
#Configure different level 1 classifier related with stacking methodology.
models = [BaggingClassifier(),SVC(),ExtraTreeClassifier(),KNeighborsClassifier(n_neighbors=5,n_jobs=-1),RandomForestClassifier(random_state=0,n_jobs=-1,n_estimators=100),XGBClassifier(random_state=0,n_jobs=-1,learning_rate=0.1, n_estimators=100, max_depth=3)]
S_train, S_test = stacking(models,X_train, y_train.values.ravel(), X_test,regression=False, mode='oof_pred_bag',needs_proba=False,save_dir=None,metric=accuracy_score,n_folds=4,stratified=True,shuffle=True,random_state=0,verbose=2)
#Configure 2nd  level classifier related with stacking methodology.
model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,n_estimators=100, max_depth=3)
#Feed model with retrieved data S_train and y_train
model = model.fit(S_train, y_train.values.ravel())
#Predict test data.
y_pred = model.predict(S_test)
df= pd.DataFrame(y_pred)
#Start index from 1 instead of 0.
df.index = np.arange(1, len(df)+1)
#Output results to a csv file
df.to_csv('data/Ejemplo_envio.csv', header=False)



