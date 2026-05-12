from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
from Evaluate_Error import evaluat_error_

def Model_YOLOv3_Mask_RCNN(Train_Data, Train_Target, Test_Data, Test_Target,
                           epochs=5, lr=0.001):
    input_shape = Train_Data.shape[1:]

    # --- YOLOv3- model ---
    inp1 = Input(shape=input_shape)
    x1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inp1)
    x1 = Conv2D(64, (3, 3), activation="relu", padding="same")(x1)
    x1 = Flatten()(x1)
    out1 = Dense(1, activation="sigmoid")(x1)
    yolov3 = Model(inp1, out1, name="YOLOv3")
    yolov3.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    yolov3.fit(Train_Data, Train_Target, epochs=epochs, verbose=0)
    pred_yolo = yolov3.predict(Test_Data)

    # --- Mask-RCNN- model ---
    inp2 = Input(shape=input_shape)
    x2 = Conv2D(32, (3, 3), activation="relu", padding="same")(inp2)
    x2 = Conv2D(64, (3, 3), activation="relu", padding="same")(x2)
    x2 = Flatten()(x2)
    out2 = Dense(1, activation="sigmoid")(x2)
    mask_rcnn = Model(inp2, out2, name="Mask_RCNN")
    mask_rcnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    mask_rcnn.fit(Train_Data, Train_Target, epochs=epochs, verbose=0)
    pred_mask = mask_rcnn.predict(Test_Data)
    avg_pred = (pred_yolo + pred_mask) / 2.0
    Eval = evaluat_error_(avg_pred, Test_Target)
    return Eval
