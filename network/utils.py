from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.platform import flags
from tensorflow.python.platform import app
from tensorflow.python import pywrap_tensorflow
import numpy as np
import tensorflow as tf
import tflearn
import re
from keras import backend as K
from keras.layers import GlobalAveragePooling3D, GlobalMaxPooling3D, Reshape, Dense, Add, Activation

def cbam_block(input_feature, name, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    with tf.variable_scope(name):
        attention_feature = channel_attention(input_feature, 'ch_at', ratio)
        attention_feature = spatial_attention(attention_feature, 'sp_at')
    return attention_feature

def channel_attention(input_feature, ratio=8):
        channel = input_feature.shape.as_list()[-1]

        shared_layer_one = Dense(channel//ratio,
                                                         activation='relu',
                                                         kernel_initializer='he_normal',
                                                         use_bias=True,
                                                         bias_initializer='zeros')
        shared_layer_two = Dense(channel,
                                                         kernel_initializer='he_normal',
                                                         use_bias=True,
                                                         bias_initializer='zeros')

        avg_pool = GlobalAveragePooling3D()(input_feature)
        avg_pool = Reshape((1,1,1,channel))(avg_pool)
        #print(avg_pool.shape.as_list()[1:],channel)
        #assert avg_pool.shape.as_list()[1:] == (1,1,1,channel)
        avg_pool = shared_layer_one(avg_pool)
        #assert avg_pool.shape.as_list()[1:] == (1,1,1,channel//ratio)
        avg_pool = shared_layer_two(avg_pool)
        #assert avg_pool.shape.as_list()[1:] == (1,1,1,channel)

        max_pool = GlobalMaxPooling3D()(input_feature)
        max_pool = Reshape((1,1,1,channel))(max_pool)
        #assert max_pool.shape.as_list()[1:] == (1,1,1,channel)
        max_pool = shared_layer_one(max_pool)
        #assert max_pool.shape.as_list()[1:] == (1,1,1,channel//ratio)
        max_pool = shared_layer_two(max_pool)
        #assert max_pool.shape.as_list()[1:] == (1,1,1,channel)

        cbam_feature = Add()([avg_pool,max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)

        return cbam_feature



def spatial_attention(input_feature, name):
    kernel_size = 7
    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[4], keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[4], keepdims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool,max_pool], 4)
        assert concat.get_shape()[-1] == 2
        
        '''concat = tf.layers.conv2d(concat,
                                filters=1,
                                kernel_size=[kernel_size,kernel_size],
                                strides=[1,1],
                                padding="same",
                                activation=None,
                                kernel_initializer=kernel_initializer,
                                use_bias=False,
                                name='conv')
        '''

        concat = tflearn.layers.conv_3d(concat, 1, kernel_size, strides=1,
                                  padding='same', activation='linear', bias=True, scope='conv3d_spatial_AT', reuse=False, weights_init='uniform_scaling')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')
        
    return input_feature * concat


def ReLU(target, name=None):
    return tflearn.activations.relu(target)


def LeakyReLU(target, alpha=0.1, name=None):
    return tflearn.activations.leaky_relu(target, alpha=alpha, name=name)

def Softmax(target,  name=None):
    return tflearn.activations.softmax(target)

def Sigmoid(target,  name=None):
    return tflearn.activations.sigmoid(target)

def set_tf_keys(feed_dict, **kwargs):
    ret = dict([(k + ':0', v) for k, v in feed_dict.items()])
    ret.update([(k + ':0', v) for k, v in kwargs.items()])
    return ret


class Network:
    def __init__(self, name, trainable=True, reuse=None):
        self._built = reuse
        self._name = name
        self.trainable = trainable

    @property
    def name(self):
        return self._name

    def __call__(self, *args, **kwargs):
        with tf.variable_scope(self.name, reuse=self._built) as self.scope:
            self._built = True
            return self.build(*args, **kwargs)

    @property
    def trainable_variables(self):
        if isinstance(self.trainable, str):
            var_list = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name)
            return [var for var in var_list if re.fullmatch(self.trainable, var.name)]
        elif self.trainable:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name)
        else:
            return []

    @property
    def data_args(self):
        return dict()


class ParallelLayer:
    inputs = {}
    replicated_inputs = None

#Separately learning the reg-seg
class MultiGPUs:
    def __init__(self, num):
        self.num = num

    def __call__(self, net, args, opt=None, scheme=None):
        args = [self.reshape(arg) for arg in args]
        results = []
        grads = []
        self.current_device = None
        for i in range(self.num):
            def auto_gpu(opr):
                # if opr.name.find('stack') != -1:
                #     print(opr)
                if opr.type.startswith('Gather') or opr.type in ('L2Loss', 'Pack', 'Gather', 'Tile', 'ReconstructionWrtImageGradient', 'Softmax', 'FloorMod', 'MatMul'):
                    return '/cpu:0'
                else:
                    return '/gpu:%d' % i
            with tf.device(auto_gpu):
                self.current_device = i
                net.controller = self
                result = net(*[arg[i] for arg in args])
                results.append(result)
                if opt is not None:
                    var_segment =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gaffdfrm/seg_stem")
                    var_deform =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gaffdfrm/deform_stem_0")
                    #registration network optimization
                    if scheme == 'reg' or scheme == 'reg_supervise' or scheme == 'reg_unsupervise':
                        grads.append(opt.compute_gradients(
                            result['reg_loss'],var_list = var_deform))#result['loss'],var_list = net.trainable_variables))
                    #segmentation network optimization
                    else:
                        grads.append(opt.compute_gradients(
                            result['seg_loss'], var_list=var_segment))#previous    result['loss'],var_list = net.trainable_variables

        with tf.device('/gpu:0'):
            concat_result = {}
            for k in results[0]:
                if len(results[0][k].shape) == 0:
                    concat_result[k] = tf.stack(
                        [result[k] for result in results])
                else:
                    concat_result[k] = tf.concat(
                        [result[k] for result in results], axis=0)

            if grads:
                op = opt.apply_gradients(self.average_gradients(grads))
                return concat_result, op
            else:
                return concat_result

    def call(self, net, kwargs):
        if net.replicated_inputs is None:
            with tf.device('/gpu:0'):
                net.replicated_inputs = dict(
                    [(k, self.reshape(v)) for k, v in net.inputs.items()])
        for k, v in net.replicated_inputs.items():
            kwargs[k] = v[self.current_device]
        return net(**kwargs)

    @staticmethod
    def average_gradients(grads):
        ret = []
        for grad_list in zip(*grads):
            grad, var = grad_list[0]
            if grad is None:
                ret.append((None, var))
            else:
                print(var, var.device)
                ret.append(
                    (tf.add_n([grad for grad, _ in grad_list]) / len(grad_list), var))
        return ret

    def reshape(self, tensor):
        return tf.reshape(tensor, tf.concat([tf.stack([self.num, -1]), tf.shape(tensor)[1:]], axis = 0))



