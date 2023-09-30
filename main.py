import pandas as pd

dataset = pd.read_csv('cancer.csv') # ler arquivo csv

x = dataset.drop(columns=["diagnosis(1=m, 0=b)"]) # x recebe todas colunas, exceto 'diagnosis'
y = dataset["diagnosis(1=m, 0=b)"] # y recebe a coluna 'diagnosis'

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2) # repartindo a base para treino e para avaliação, 0.2 é porcentagem de teste

import tensorflow as tf

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape, activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000)

model.evaluate(x_test, y_test)
