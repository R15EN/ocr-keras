import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, Dense, Dropout
from tensorflow.keras.layers import LSTM, Bidirectional, BatchNormalization, Activation

class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred

def conv_block(filters, x, conv_name, pool_name):
    x = Conv2D(
        filters, 
        kernel_size=(3, 3), 
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
        name=conv_name
        )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name=pool_name)(x)
    x = Dropout(0.3)(x)
    return x

def BiLSTM(units, x):
    x = Bidirectional(LSTM(units, return_sequences=True, dropout=0.25))(x)
    return x

def ocr_model(image_width, image_height, char_to_num):

    input_img = tf.keras.Input(shape=(image_width, image_height, 3), name="image")
    labels = tf.keras.layers.Input(name="label", shape=(None,))

    # Conv block 1
    x = conv_block(32, input_img, 'conv1', 'pool1')

    # Conv block 2
    x = conv_block(64, x, 'conv2', 'pool2')
    
    # Conv block 3
    x = conv_block(128, x, 'conv3', 'pool3')

    new_shape = ((image_width // 8), (image_height // 8) * 128)
    x = Reshape(target_shape=new_shape, name="reshape")(x)
    x = Dense(256, activation="relu", name="dense1")(x)
    x = Dropout(0.2)(x)
    
    # RNNs.
    x = BiLSTM(266, x)
    x = BiLSTM(128, x)

    x = Dense(len(char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = tf.keras.models.Model(inputs=[input_img, labels], outputs=output)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer)
    
    return model
