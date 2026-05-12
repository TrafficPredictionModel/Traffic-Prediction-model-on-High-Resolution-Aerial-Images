from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from Evaluate_Error import evaluat_error_


def Model_LSTM_GRU(Train_Data, Train_Target, Test_Data, Test_Target):
    if len(Train_Data.shape) == 2:
        Train_Data = Train_Data.reshape((Train_Data.shape[0], 1, Train_Data.shape[1]))
        Test_Data = Test_Data.reshape((Test_Data.shape[0], 1, Test_Data.shape[1]))
    # ----------- LSTM Model -----------
    lstm_model = Sequential()
    lstm_model.add(LSTM(64, input_shape=(Train_Data.shape[1], Train_Data.shape[2])))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.compile(optimizer=Adam(learning_rate=0.001),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    lstm_model.fit(Train_Data, Train_Target, epochs=20, batch_size=32, verbose=0)
    lstm_pred = lstm_model.predict(Test_Data)
    # ----------- GRU Model -----------
    gru_model = Sequential()
    gru_model.add(GRU(64, input_shape=(Train_Data.shape[1], Train_Data.shape[2])))
    gru_model.add(Dropout(0.3))
    gru_model.add(Dense(1, activation='sigmoid'))
    gru_model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    gru_model.fit(Train_Data, Train_Target, epochs=20, batch_size=32, verbose=0)
    gru_pred = gru_model.predict(Test_Data)
    avg_pred = (lstm_pred + gru_pred) / 2.0
    Eval = evaluat_error_(avg_pred, Test_Target)
    return Eval
