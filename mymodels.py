import os
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers import Input,Activation,BatchNormalization,Concatenate,Add
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.utils import plot_model


class ARCNN:
    def __init__(self, img_height, img_width, img_channels=3):

        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels


    def create_model(self):
        ip = Input(shape=(self.img_height, self.img_width,self.img_channels))
        feature1 = Conv2D(64,(9,9),padding='same',activation='relu')(ip)
        enhance_feature = Conv2D(32,(7,7),padding='same',activation='relu')(feature1)
        conv3 = Conv2D(16,(1,1),padding='same',activation='relu')(enhance_feature)
        op = Conv2D(self.img_channels,(5,5),padding='same',activation='relu')(conv3)

        model =Model(ip,op)
        return model



class L8:
    def __init__(self, img_height, img_width, img_channels=3):

        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels

    def create_model(self):
        ip = Input(shape=(self.img_height, self.img_width,self.img_channels))
        L1 = Conv2D(32, (11, 11), padding='same', activation='relu', kernel_initializer='glorot_uniform')(ip)
        L2 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L1)
        L3 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L2)
        L4 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L3)
        L4=Concatenate(axis=-1)([L4,L1])
        L5 = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L4)
        L6 = Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L5)
        L6=Concatenate(axis=-1)([L6,L1])
        L7 = Conv2D(128, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L6)
        op = Conv2D(self.img_channels, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L7)

        model = Model(ip, op)
        return model



class ResNet:
    def __init__(self, img_height, img_width, img_channels=3):

        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels

    def residual_block(self,inputs,filters,kernel_size):
        conv_1 = Conv2D(filters, (kernel_size, kernel_size), padding='same',kernel_initializer='glorot_uniform')(inputs)
        norm_1 = BatchNormalization(axis=-1)(conv_1)
        relu_1 = LeakyReLU(alpha=0.25)(norm_1)
        #relu_1 = Activation('relu')(norm_1)
        conv_2 = Conv2D(filters, (kernel_size, kernel_size), padding='same',kernel_initializer='glorot_uniform')(relu_1)
        norm_2 = BatchNormalization(axis=-1)(conv_2)
        #relu_2 = LeakyReLU(alpha=0.25)(norm_2)#not sure maybe wrong
        return Add()([inputs, norm_2])

    def create_model(self):
        ip = Input(shape=(self.img_height, self.img_width,self.img_channels))
        x = Conv2D(64, (9, 9), padding='same', activation='linear',  kernel_initializer='glorot_uniform')(ip)
        x = BatchNormalization(axis= -1)(x)
        x = LeakyReLU(alpha=0.25)(x)
        #x = Activation('relu')(x)
        for i in range(5):
            x = self.residual_block(x, 64, 3)

        x = Conv2D(64, (3, 3), padding='same',kernel_initializer='glorot_uniform')(x)
        x = BatchNormalization(axis=-1)(x)
        x=Conv2D(64,(3, 3),padding='same',activation='relu')(x)
        op=Conv2D(self.img_channels,(9,9),padding='same',activation='tanh',kernel_initializer='glorot_uniform')(x)

        model =Model(inputs=ip,outputs= op)
        return model



class DenseNet:
    def __init__(self, img_height, img_width,img_channels=3):

        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels

    def _conv_factory(self, x, nb_filter, weight_decay=1E-4, concat_axis=-1):
        #Apply BatchNorm, Relu 1x1Conv2D, BatchNorm, Relu 3x3Conv2D
        
        x = BatchNormalization(axis=concat_axis,momentum=0.9,gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_filter*4, (1, 1),kernel_initializer="he_uniform",padding="same", kernel_regularizer=l2(weight_decay))(x)#Bottleneck;use_bias=False or True?

        x = BatchNormalization(axis=concat_axis,momentum=0.9,gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_filter, (3, 3),kernel_initializer="he_uniform", padding="same",kernel_regularizer=l2(weight_decay))(x)

        return x

    def _denseblock(self, x, nb_layers, nb_filter, growth_rate, weight_decay=1E-4,concat_axis=-1):

        list_feat = [x]

        for i in range(nb_layers):
            x = self._conv_factory(x, growth_rate)
            list_feat.append(x)
            x = Concatenate(axis=concat_axis)(list_feat)
            nb_filter += growth_rate

        return x, nb_filter

    #not sure
    def _transition(self, x, nb_filter, weight_decay=1E-4, concat_axis=-1):

        x = BatchNormalization(axis=concat_axis,momentum=0.9, gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_filter, (1, 1),kernel_initializer="he_uniform",padding="same",kernel_regularizer=l2(weight_decay))(x)

        return x

    def create_model(self):
        weight_decay=1E-4
        ip = Input(shape=(self.img_height, self.img_width,self.img_channels))
        # Initial convolution
        x = Conv2D(16, (9, 9),kernel_initializer="he_uniform",padding="same",kernel_regularizer=l2(weight_decay))(ip)

        # Add dense blocks
        x, nb_filter = self._denseblock(x, 6, 16, 12)

        # add transition
        x = self._transition(x, nb_filter/2)

        # The last denseblock does not have a transition
        x, nb_filter = self._denseblock(x, 6, nb_filter/2, 12)

        #output
        x = BatchNormalization(axis=-1,momentum=0.9, gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)

        op = Conv2D(self.img_channels,(5, 5),kernel_initializer="he_uniform",padding="same",kernel_regularizer=l2(weight_decay))(x)

        model = Model(inputs=ip, outputs=op)
        return model



class DenseNet_shallow:
    def __init__(self, img_height, img_width, img_channels=3):

        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels

    def _conv_factory(self, x, nb_filter, weight_decay=1E-4, concat_axis=-1):
        #Apply BatchNorm, Relu 1x1Conv2D, BatchNorm, Relu 3x3Conv2D
        
        x = BatchNormalization(axis=concat_axis,momentum=0.9,gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_filter*4, (1, 1),kernel_initializer="he_uniform",padding="same", kernel_regularizer=l2(weight_decay))(x)#Bottleneck;use_bias=False or True?

        x = BatchNormalization(axis=concat_axis,momentum=0.9,gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_filter, (3, 3),kernel_initializer="he_uniform", padding="same",kernel_regularizer=l2(weight_decay))(x)

        return x

    def _denseblock(self, x, nb_layers, nb_filter, growth_rate, weight_decay=1E-4,concat_axis=-1):

        list_feat = [x]

        for i in range(nb_layers):
            x = self._conv_factory(x, growth_rate)
            list_feat.append(x)
            x = Concatenate(axis=concat_axis)(list_feat)
            nb_filter += growth_rate

        return x, nb_filter


    def create_model(self):
        weight_decay=1E-4
        ip = Input(shape=(self.img_height, self.img_width,self.img_channels))
        # Initial convolution
        x = Conv2D(32, (9, 9),kernel_initializer="he_uniform",padding="same",kernel_regularizer=l2(weight_decay))(ip)

        # Add dense blocks
        x, nb_filter = self._denseblock(x, 6, 32, 32)

        #output
        x = BatchNormalization(axis=-1,momentum=0.9, gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)

        op = Conv2D(self.img_channels,(5, 5),kernel_initializer="he_uniform",padding="same",kernel_regularizer=l2(weight_decay))(x)

        model = Model(inputs=ip, outputs=op)
        return model




def visialization():
    model =DenseNet(42,42,3)
    model = model.create_model()
    model.summary()
    print(len(model.layers))
    root_dir=os.getcwd()

    plot_model(model, to_file=os.path.join(root_dir,'test.png'), show_shapes=True, show_layer_names=True)



if __name__=='__main__':
    visialization()
