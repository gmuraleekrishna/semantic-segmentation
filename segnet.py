from keras.models import Model
from keras.layers import Conv2D,  BatchNormalization, Activation, Input, Reshape, UpSampling2D, MaxPooling2D, concatenate, Permute


class SegNet(Model):

    def __init__(self, height, width, no_of_classes):
        self.layers = []
        input_layer = Input(shape=(height, width, 3))
        conv_1, conv_2, conv_3, conv_4, conv_5, pool_5 = self.encoder(kernel=3, input=input_layer)
        output = self.decoder(kernel=3, conv_1=conv_1, conv_2=conv_2, conv_3=conv_3, conv_4=conv_4, conv_5=conv_5, pool_5=pool_5)
        output = Conv2D(no_of_classes, (1, 1), padding="valid")(output)
        output = BatchNormalization()(output)
        output = Activation('softmax')(output)
        super(SegNet, self).__init__(inputs=input_layer, outputs=output, name="SegNet")

    def convolution_block(self, input, filter, kernel):
        out = Conv2D(filter, (kernel, kernel), padding="same")(input)
        out = BatchNormalization()(out)
        out = Activation("relu")(out)
        return out

    def encoder(self, kernel, input):
        conv_1 = self.convolution_block(input, 64, kernel)
        conv_1 = self.convolution_block(conv_1, 64, kernel)
        pool_1 = MaxPooling2D()(conv_1)

        conv_2 = self.convolution_block(pool_1, 128, kernel)
        conv_2 = self.convolution_block(conv_2, 128, kernel)
        pool_2 = MaxPooling2D()(conv_2)

        conv_3 = self.convolution_block(pool_2, 256, kernel)
        conv_3 = self.convolution_block(conv_3, 256, kernel)
        conv_3 = self.convolution_block(conv_3, 256, kernel)
        pool_3 = MaxPooling2D()(conv_3)

        conv_4 = self.convolution_block(pool_3, 512, kernel)
        conv_4 = self.convolution_block(conv_4, 512, kernel)
        conv_4 = self.convolution_block(conv_4, 512, kernel)
        pool_4 = MaxPooling2D()(conv_4)

        conv_5 = self.convolution_block(pool_4, 512, kernel)
        conv_5 = self.convolution_block(conv_5, 512, kernel)
        conv_5 = self.convolution_block(conv_5, 512, kernel)
        pool_5 = MaxPooling2D()(conv_5)

        return conv_1, conv_2, conv_3, conv_4, conv_5, pool_5

    def decoder(self, kernel, conv_1, conv_2, conv_3, conv_4, conv_5, pool_5):
        up_sample_1 = UpSampling2D(size=(2, 2))(pool_5)
        merge_1 = concatenate(inputs=[conv_5, up_sample_1], axis=3)
        conv_6 = self.convolution_block(merge_1, 512, kernel)
        conv_6 = self.convolution_block(conv_6, 512, kernel)
        conv_6 = self.convolution_block(conv_6, 512, kernel)

        up_sample_2 = UpSampling2D(size=(2, 2))(conv_6)
        merge_1 = concatenate(inputs=[conv_4, up_sample_2], axis=3)
        conv_7 = self.convolution_block(merge_1, 512, kernel)
        conv_7 = self.convolution_block(conv_7, 512, kernel)
        conv_7 = self.convolution_block(conv_7, 256, kernel)

        up_sample_3 = UpSampling2D(size=(2, 2))(conv_7)
        merge_2 = concatenate(inputs=[conv_3, up_sample_3], axis=3)
        conv_8 = self.convolution_block(up_sample_3, 256, kernel)
        conv_8 = self.convolution_block(conv_8, 256, kernel)
        conv_8 = self.convolution_block(conv_8, 128, kernel)

        up_sample_4 = UpSampling2D(size=(2, 2))(conv_8)
        merge_2 = concatenate(inputs=[conv_2, up_sample_4], axis=3)
        conv_9 = self.convolution_block(up_sample_4, 128, kernel)
        conv_9 = self.convolution_block(conv_9, 64, kernel)

        up_sample_5 = UpSampling2D(size=(2, 2))(conv_9)
        merge_2 = concatenate(inputs=[conv_1, up_sample_5], axis=3)
        output = self.convolution_block(up_sample_5, 64, kernel)
        return output


