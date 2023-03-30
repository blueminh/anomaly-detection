from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
import numpy as np
import pandas as pd
import csv

"""
CODE WAS INSPIRED BY KAGGLE ARTICLE ON AUTOENCODERS GIVEN HERE: 
https://www.kaggle.com/code/robinteuwens/anomaly-detection-with-auto-encoders


"""




"""
Stuff we need to drop 
"""
to_drop = ['Timestamp','Label']

to_drop_test = ['Timestamp']

RANDOM_SEED = 42




"""
Importing data from given files and drop columns 
"""
X_train = pd.read_csv('train_data.csv').drop(to_drop,axis=1).values
validation_data = pd.read_csv('val_data.csv')
X_validation, y_validation = validation_data.drop(to_drop,axis=1).values, validation_data.Label.values
test_data = pd.read_csv('test_data.csv')
X_test = test_data.drop(to_drop_test,axis=1).values


"""
Setting up the data transforming pipeline
"""
pipeline = Pipeline([('normalizer', Normalizer()),('scaler', MinMaxScaler())])
pipeline.fit(X_train)
X_train_transformed = pipeline.transform(X_train)
X_validation_transformed = pipeline.transform(X_validation)

"""
Training the Neural Network 
"""
feature_dimensions = X_train_transformed.shape[1]
BATCH_SIZE = 500 
EPOCHS = 100

auto_encoder = tf.keras.models.Sequential([
    
    # deconstruct / encode
    tf.keras.layers.Dense(feature_dimensions, activation='elu', input_shape=(feature_dimensions, )), 
    tf.keras.layers.Dense(16, activation='elu'),
    tf.keras.layers.Dense(8, activation='elu'),
    tf.keras.layers.Dense(4, activation='elu'),
    tf.keras.layers.Dense(2, activation='elu'),
    
    # reconstruction / decode
    tf.keras.layers.Dense(4, activation='elu'),
    tf.keras.layers.Dense(8, activation='elu'),
    tf.keras.layers.Dense(16, activation='elu'),
    tf.keras.layers.Dense(feature_dimensions, activation='elu')
])

auto_encoder.compile(optimizer="adam", 
                    loss="mse",
                    metrics=["acc"])

auto_encoder.fit(X_train_transformed,X_train_transformed,shuffle=True,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_data=(X_validation_transformed, y_validation))

"""
Predict anomalous points in the test data
"""
# Transforming the test data using the pipeline fitted by the training data
X_test_transformed = pipeline.transform(X_test)
results = auto_encoder.predict(X_test_transformed)

error = np.mean(np.power(X_test_transformed - results, 2), axis=1)

ninetyPercentile = np.percentile(error,86)

indexesToCheck = []


for idx, error in enumerate(error):
    if error > ninetyPercentile:
        indexesToCheck.append(idx)

"""
Extracting the anomalous points predicted by the model
"""
anomaly_ids = X_test[indexesToCheck,0]

csv_data = []

for id in test_data.Id.values:
    if (id in anomaly_ids):
        csv_data.append([id,1])
    else:
        csv_data.append([id,0])

"""
Store the predictions
"""
with open('anomalies.csv','w',newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)