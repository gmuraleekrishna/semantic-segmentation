from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Activation, MaxPooling2D, UpSampling2D


class SegNetBasic(Sequential):
    def __init__(self, no_of_classes, height, width):
        super(Model, self).__init__()
        self.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(height, width, 3)))   # Input layer
        self.add(BatchNormalization())
        self.add(Activation(activation='relu'))
        max1, max2, max3, max4, max5 = self.encoder()
        self.decoder(max1, max2, max3, max4, max5)
        self.add(Conv2D(filters=no_of_classes, kernel_size=(1, 1), padding='same'))
        self.add(Activation('softmax')) # output size (None, 480, 640, 11)

    def decoder(self, max1, max2, max3, max4, max5):
        up1 = self.add(UpSampling2D())
        self.add(Merge(up1, max5))
        self.convolution_block(512)
        self.convolution_block(512)
        self.convolution_block(512)
        self.add(Dropout(0.2))

        up2 = self.add(UpSampling2D())
        self.add(Merge(up2, max4))
        self.convolution_block(512)
        self.convolution_block(512)
        self.convolution_block(256)
        self.add(Dropout(0.2))

        up3 = self.add(UpSampling2D())
        self.add(Merge(up3, max3))
        self.convolution_block(256)
        self.convolution_block(256)
        self.convolution_block(128)
        self.add(Dropout(0.2))

        up4 = self.add(UpSampling2D())
        self.add(Merge(up4, max2))
        self.convolution_block(128)
        self.convolution_block(64)
        self.add(Dropout(0.2))

        up5 = self.add(UpSampling2D())
        self.add(Merge(up5, max1))
        self.convolution_block(64)
        self.add(Dropout(0.2))

    def encoder(self):
        self.convolution_block(64)
        max1 = self.add(MaxPooling2D())

        self.convolution_block(128)
        self.convolution_block(128)
        max2 = self.add(MaxPooling2D())

        self.convolution_block(256)
        self.convolution_block(256)
        self.convolution_block(256)
        max3 = self.add(MaxPooling2D())

        self.convolution_block(512)
        self.convolution_block(512)
        self.convolution_block(512)
        max4 = self.add(MaxPooling2D())

        self.convolution_block(512)
        self.convolution_block(512)
        self.convolution_block(512)
        max5 = self.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        return max1, max2, max3, max4, max5

    def convolution_block(self, filters):
        # Apply Convoution, Batch Normalization, ReLU
        self.add(Conv2D(filters=filters, kernel_size=(3, 3), padding='same'))
        self.add(BatchNormalization())
        self.add(Activation(activation='relu'))
        # self.add(Dropout(rate=0.1))

    # def convolution_block(self, filters):
    #     # Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    #     self.add(Conv2D(filters=filters, kernel_size=(3,3), padding='same'))
    #     self.add(BatchNormalization())
    #     self.add(Activation(activation='relu'))
    #     # self.add(Dropout(rate=0.2))


    
