from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from Evaluate_Error import evaluat_error_

def Model_ANN(Train_Data, Train_Target, Test_Data, Test_Target):
    model = Sequential()
    model.add(Dense(64, input_dim=Train_Data.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',   # for binary classification
                  metrics=['accuracy'])
    model.fit(Train_Data, Train_Target, epochs=20, batch_size=32, verbose=0)
    predictions = model.predict(Test_Data)
    Eval = evaluat_error_(predictions,Test_Target)
    return Eval