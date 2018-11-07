import numpy

import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt


df_train = pd.read_csv('train.csv')
df_train = pd.get_dummies(df_train, dtype=float)

#added personal_status_A95 
df_train['personal_status_A95'] = 0

#initial credit row for train data
def credit(row):
    if 4000 <= row <= 20000:
        return 1
    if 1500 <= row < 4000:
        return 2
    return 3


df_train['cluster'] = df_train.apply(lambda row: credit(row.credit_amount), axis=1)


X_train = df_train.drop(['cluster', 'credit_amount', 'serial number'], axis=1)
y_train = df_train['cluster']
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
y_train = np_utils.to_categorical(encoded_Y, 3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


#creating a model using this function

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=61, activation='relu'))
    model.add(Dropout(0.9))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.9))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = baseline_model()
history = model.fit(X_train, y_train, batch_size=800, verbose=1, validation_split=0.2, epochs=2000)



# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


test = pd.read_csv("test.csv")
test = pd.get_dummies(test, dtype=float)
test['personal_status_A95'] = 0

x_test = test.copy()
x_test = x_test.drop(['serial number'], axis=1)

x_test = scaler.transform(x_test)
pred = model.predict(x_test)
pred = [numpy.argmax(x) + 1 for x in pred]
test['cluster_number'] = pred
predict = test[['serial number', 'cluster_number']]
predict.to_csv('predicted.csv', index=False)
