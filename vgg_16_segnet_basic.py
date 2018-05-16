from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D,Input, Activation, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.applications.vgg16 import VGG16

class VGG16SegNetBasic(Model):
    def __init__(self, no_of_classes, height, width):
        self.layers = []
        input_layer = Input(shape=(height, width, 3))
        output = self.encoder(input=input_layer, no_of_classes=no_of_classes)
        output = self.decoder(input=output, kernel=3)
        output = Conv2D(no_of_classes, (1, 1), padding="valid")(output)
        output = BatchNormalization()(output)
        output = Activation('softmax')(output)
        self.layers.append([input_layer, output])
        super(VGG16SegNetBasic, self).__init__(inputs=input_layer, outputs=output, name="VGG16SegNetBasic")

    def encoder(self, input, no_of_classes):
        vgg16 = VGG16(weights='imagenet', classes=no_of_classes, include_top=False, input_tensor=input)
        for layer in vgg16.layers:
            layer.trainable = False
        return vgg16.layers[-1].output
   
    def decoder(self, input, kernel):
        up_sample_1 = UpSampling2D(size=(2, 2))(input)
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
