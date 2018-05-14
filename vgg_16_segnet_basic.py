from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.applications.vgg16 import VGG16

class VGG16SegNetBasic(Sequential):
    def __init__(self, no_of_classes, height, width):
        self.layers = []
        input_layer = Input(shape=(height, width, 3))
        output = self.encoder(kernel=3, input=input_layer)
        output = self.decoder(kernel=3, input=output)
        output = Conv2D(no_of_classes, (1, 1), padding="valid")(output)
        output = BatchNormalization()(output)
        output = Activation('softmax')(output)
        self.layers.append([input_layer, output])
        super(VGG16SegNetBasic, self).__init__(inputs=input_layer, outputs=output, name="VGG16SegNetBasic")

    def encoder(self):
        vgg16 = VGG16(weights='imagenet', include_top=False)
        return vgg16.layers[-1].output
   
    def decoder(self, max1, max2, max3, max4, max5):
        up_sample_1 = UpSampling2D(size=(2, 2))(pool_5)
        conv_6 = self.convolution_block(up_sample_1, 512, kernel)
        conv_6 = self.convolution_block(conv_6, 512, kernel)
        conv_6 = self.convolution_block(conv_6, 512, kernel)

        up_sample_2 = UpSampling2D(size=(2, 2))(conv_6)
        conv_7 = self.convolution_block(up_sample_2, 512, kernel)
        conv_7 = self.convolution_block(conv_7, 512, kernel)
        conv_7 = self.convolution_block(conv_7, 256, kernel)

        up_sample_3 = UpSampling2D(size=(2, 2))(conv_7)
        conv_8 = self.convolution_block(up_sample_3, 256, kernel)
        conv_8 = self.convolution_block(conv_8, 256, kernel)
        conv_8 = self.convolution_block(conv_8, 128, kernel)

        up_sample_4 = UpSampling2D(size=(2, 2))(conv_8)
        conv_9 = self.convolution_block(up_sample_4, 128, kernel)
        conv_9 = self.convolution_block(conv_9, 64, kernel)

        up_sample_5 = UpSampling2D(size=(2, 2))(conv_9)
        output = self.convolution_block(up_sample_5, 64, kernel)
        return output

    def convolution_block(self, input, filter, kernel):
        out = Conv2D(filter, (kernel, kernel), padding="same")(input)
        out = BatchNormalization()(out)
        out = Activation("relu")(out)
        return out
