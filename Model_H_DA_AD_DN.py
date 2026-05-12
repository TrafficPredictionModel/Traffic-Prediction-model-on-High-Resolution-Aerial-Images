import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam

from Evaluate_Error import evaluat_error_

def _channel_attention(x, reduction=8):
    ch = x.shape[-1]
    avg = layers.GlobalAveragePooling2D()(x)
    maxp = layers.GlobalMaxPooling2D()(x)
    dense = layers.Dense(ch // reduction, activation='relu')
    dense2 = layers.Dense(ch, activation='sigmoid')
    avg_out = dense2(dense(avg))
    max_out = dense2(dense(maxp))
    att = layers.Add()([avg_out, max_out])
    att = layers.Reshape((1,1,ch))(att)
    return layers.Multiply()([x, att])

def _spatial_attention(x):
    avg = K.mean(x, axis=-1, keepdims=True)
    mx = K.max(x, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1)([avg, mx])
    conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    return layers.Multiply()([x, conv])

def _cbam_block(x, reduction=8):
    x = _channel_attention(x, reduction=reduction)
    x = _spatial_attention(x)
    return x

def _dense_dilated_block(x, growth_rate=16, layers_in_block=3, dilation_rates=(1,2,3)):
    concat_feats = [x]
    for i in range(layers_in_block):
        dr = dilation_rates[i % len(dilation_rates)]
        out = layers.BatchNormalization()(x)
        out = layers.ReLU()(out)
        out = layers.Conv2D(growth_rate, kernel_size=3, padding='same', dilation_rate=dr)(out)
        concat_feats.append(out)
        x = layers.Concatenate(axis=-1)(concat_feats)
    return x

def _aspp_fusion(x, filters=32, rates=(1,2,4)):
    aspp = []
    for r in rates:
        t = layers.Conv2D(filters, 3, padding='same', dilation_rate=r, activation='relu')(x)
        aspp.append(t)
    aspp.append(layers.GlobalAveragePooling2D()(x))
    aspp[-1] = layers.Reshape((1,1,filters))(layers.Dense(filters, activation='relu')(aspp[-1]))
    aspp = [layers.UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation='bilinear')(aspp[-1])] + aspp[:-1] if isinstance(aspp[-1], tf.Tensor) else aspp
    # safer: just concat the convs
    convs = []
    for r in rates:
        convs.append(layers.Conv2D(filters, 1, activation='relu', padding='same')(layers.Conv2D(filters, 3, padding='same', dilation_rate=r)(x)))
    return layers.Concatenate(axis=-1)(convs)

def _make_2d_path(input_shape, init_filters=32, growth_rate=16, dense_layers=3):
    inp = layers.Input(shape=input_shape, name='feat1_input')
    x = layers.Conv2D(init_filters, 3, padding='same', activation='relu')(inp)
    # two adaptive dilated dense blocks + ASPP fusion
    x = _dense_dilated_block(x, growth_rate=growth_rate, layers_in_block=dense_layers, dilation_rates=(1,2,3))
    x = _aspp_fusion(x, filters=init_filters, rates=(1,2,4))
    x = _cbam_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    model = models.Model(inputs=inp, outputs=x, name='2d_path')
    return model

def _make_1d_path(input_shape, filters=64, kernel_size=3):
    inp = layers.Input(shape=input_shape, name='feat2_input')
    # expand and use Conv1D blocks
    x = layers.Reshape((input_shape[0], 1))(inp) if len(input_shape)==1 else inp
    x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.Conv1D(filters//2, kernel_size, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    model = models.Model(inputs=inp, outputs=x, name='1d_path')
    return model

def Model_H_DA_AD_DN(Feat1, Feat2, Target, sol=None):
    if sol is None:
        sol=[5, 0.01, 100]
    epochs = 10
    verbose = 1
    Feat1 = np.asarray(Feat1)
    Feat2 = np.asarray(Feat2)
    Target = np.asarray(Target)
    N = Feat1.shape[0]
    assert Feat2.shape[0] == N and Target.shape[0] == N, "Feat1, Feat2 and Target must have same first dim (N samples)."

    hidden_neurons = int(sol[0])
    lr = float(sol[1])
    steps_per_epoch = int(sol[2])
    steps_per_epoch = max(1, steps_per_epoch)

    # build subpaths
    path2d = _make_2d_path(Feat1.shape[1:], init_filters=32, growth_rate=16, dense_layers=3)
    path1d = _make_1d_path(Feat2.shape[1:], filters=64)

    # fuse
    inp1 = path2d.input
    inp2 = path1d.input
    out1 = path2d.output  # global pooled vector
    out2 = path1d.output

    # project and combine
    p1 = layers.Dense(hidden_neurons, activation='relu')(out1)
    p2 = layers.Dense(hidden_neurons, activation='relu')(out2)
    fused = layers.Concatenate()([p1, p2])
    # Dual attention on fused vector: channel attention implemented via Dense gating
    gate = layers.Dense(hidden_neurons//2, activation='relu')(fused)
    gate = layers.Dense(fused.shape[-1], activation='sigmoid')(gate)
    fused = layers.Multiply()([fused, gate])
    x = layers.Dense(64, activation='relu')(fused)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', name='out')(x)
    model = models.Model(inputs=[inp1, inp2], outputs=out, name='H_DA_AD_DN_cls')
    loss = 'binary_crossentropy'
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss)
    # derive batch_size from steps_per_epoch to approximate the user's steps concept
    batch_size = max(1, math.ceil(N / steps_per_epoch))
    # train
    model.fit([Feat1, Feat2], Target, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # predictions and evaluation
    preds = model.predict([Feat1, Feat2], batch_size=batch_size)
    Eval = evaluat_error_(preds, Target)
    return Eval
