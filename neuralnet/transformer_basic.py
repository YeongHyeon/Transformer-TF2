import os
import numpy as np
import tensorflow as tf
import source.utils as utils
import whiteboxlayer.layers as wbl
import whiteboxlayer.extensions.attention as wblat

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

class Agent(object):

    def __init__(self, **kwargs):

        print("\nInitializing Neural Network...")
        self.dim_s = kwargs['dim_s']
        self.dim_f = kwargs['dim_f']
        self.dim_model = kwargs['dim_model']
        self.dim_ff = kwargs['dim_ff']
        self.depth = kwargs['depth']
        self.num_head = kwargs['num_head']

        self.learning_rate = kwargs['learning_rate']
        self.path_ckpt = kwargs['path_ckpt']

        self.variables = {}

        self.__model = Neuralnet(dim_s=self.dim_s, dim_f=self.dim_f, dim_model=self.dim_model, dim_ff=self.dim_ff, depth=self.depth, num_head=self.num_head)
        self.__model.forward(x=tf.zeros((1, self.dim_s, self.dim_f), dtype=tf.float32), verbose=True)

        self.__init_propagation(path=self.path_ckpt)

    def __init_propagation(self, path):

        self.summary_writer = tf.summary.create_file_writer(self.path_ckpt)

        self.variables['trainable'] = []
        for key in list(self.__model.layer.parameters.keys()):
            trainable = self.__model.layer.parameters[key].trainable
            if(trainable):
                self.variables['trainable'].append(self.__model.layer.parameters[key])

        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.save_params()

    def __loss(self, y, y_hat):

        restore_error = self.loss_l2(x=y-y_hat, reduce=(1))

        loss_b = restore_error #+ energy_term1
        loss = tf.math.reduce_mean(loss_b)

        return {'loss_batch': loss_b, 'loss_mean': loss}

    @tf.autograph.experimental.do_not_convert
    def step(self, minibatch, iteration=0, training=False):

        x, y = minibatch['x'], minibatch['x'][:, ::-1]

        with tf.GradientTape() as tape:
            outputs = self.__model.forward(x=x, verbose=False)
            y_hat = outputs['y_hat']
            enc_attn = tf.math.reduce_mean(outputs['enc']['attention'], axis=0)
            dec_attn = tf.math.reduce_mean(outputs['dec2']['attention'], axis=0)
            losses = self.__loss(y=y, y_hat=y_hat)

        if(training):
            gradients = tape.gradient(losses['loss_mean'], self.variables['trainable'])
            self.optimizer.apply_gradients(zip(gradients, self.variables['trainable']))

            with self.summary_writer.as_default():
                tf.summary.scalar('%s/loss_mean' %(self.__model.who_am_i), losses['loss_mean'], step=iteration)

        return {'y':y, 'y_hat':y_hat.numpy(), 'losses':losses, \
            'enc_attn':enc_attn.numpy(), 'dec_attn':dec_attn.numpy()}

    def save_params(self, model='base'):

        vars_to_save = self.__model.layer.parameters.copy()
        vars_to_save["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_save)
        ckptman = tf.train.CheckpointManager(ckpt, directory=os.path.join(self.path_ckpt, model), max_to_keep=1)
        ckptman.save()

    def load_params(self, model):

        vars_to_load = self.__model.layer.parameters.copy()
        vars_to_load["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(os.path.join(self.path_ckpt, model))
        status = ckpt.restore(latest_ckpt)
        status.expect_partial()

    def loss_l2(self, x, reduce=None):

        distance = tf.math.reduce_mean(\
            tf.math.sqrt(\
            tf.math.square(x) + 1e-30), axis=reduce)

        return distance

class Neuralnet(tf.Module):

    def __init__(self, **kwargs):
        super(Neuralnet, self).__init__()

        self.who_am_i = "Transformer"
        self.dim_s = kwargs['dim_s']
        self.dim_f = kwargs['dim_f']
        self.dim_model = kwargs['dim_model']
        self.dim_ff = kwargs['dim_ff']
        self.depth = kwargs['depth']
        self.num_head = kwargs['num_head']

        self.layer = wbl.Layers()

        self.pos_enc = wblat.positional_encoding(self.dim_s, self.dim_model)
        self.forward = tf.function(self.__call__)

    @tf.function
    def __call__(self, x, verbose=False):

        x_seq_zero = tf.math.reduce_sum(x, axis=(1), keepdims=True)
        x_mask = tf.where(x_seq_zero > 0, x_seq_zero, 0)

        dict_enc = self.__encoder(x=x, depth=self.depth, num_head=self.num_head, name='enc', verbose=verbose)
        dict_dec1, dict_dec2 = self.__decoder(x, y_enc=dict_enc['ln1'], depth=self.depth, num_head=self.num_head, name='dec', verbose=verbose)

        y_hat = self.__linear(x=dict_dec2['ln2'], dim_model=self.dim_model, name='lin_out', verbose=verbose)
        y_hat = tf.math.add(y_hat, 0, name='y_hat')

        return {'y_hat':y_hat, 'enc':dict_enc, 'dec1':dict_dec1, 'dec2':dict_dec2}

    def __encoder(self, x, depth=3, num_head=1, mask_idx=-1, udmask=False, name='enc', verbose=True):

        emb_in = wblat.embedding(layer=self.layer, x=x, dim_model=self.dim_model, name='%s_emb' %(name), verbose=verbose)
        x = emb_in + self.pos_enc

        for idx_depth in range(depth):
            try: del dict_enc
            except: pass
            dict_enc = wblat.self_attention(layer=self.layer, x_query=x, x_key=x, x_value=x, num_head=num_head, mask_idx=mask_idx, udmask=udmask, \
                name='%s_atn_0' %(name), verbose=verbose)
            attention_drop = self.layer.dropout(x=dict_enc['output'], rate=0.1, \
                name='%s_atn_dout_0_%d' %(name, depth))
            y_0 = self.layer.layer_normalization(x=x+attention_drop, trainable=True, \
                name='%s_atn_ln_0_%d' %(name, depth), verbose=verbose)
            dict_enc['ln0'] = y_0

            fnn_out = wblat.feed_forward_network(layer=self.layer, x=y_0, dim_ff=self.dim_ff, dim_model=self.dim_model, \
                name='%s_ffn_%d' %(name, depth), verbose=verbose)
            dict_enc['ffn'] = fnn_out
            fnn_drop = self.layer.dropout(x=fnn_out, rate=0.1, \
                name='%s_ffn_dout_0_%d' %(name, depth))
            y_1 = self.layer.layer_normalization(x=y_0+fnn_drop, trainable=True, \
                name='%s_ffn_ln_0_%d' %(name, depth), verbose=verbose)
            dict_enc['ln1'] = y_1
            x = y_1

        return dict_enc

    def __decoder(self, x, y_enc, depth=3, num_head=1, mask_idx=-1, udmask=False, name='dec', verbose=True):

        emb_in = wblat.embedding(layer=self.layer, x=x, dim_model=self.dim_model, name='%s_emb' %(name), verbose=verbose)
        x = emb_in + self.pos_enc

        for idx_depth in range(depth):
            try: del dict_dec1, dict_dec2
            except: pass
            dict_dec1 = wblat.self_attention(layer=self.layer, x_query=x, x_key=x, x_value=x, num_head=num_head, mask_idx=mask_idx, udmask=udmask, \
                name='%s_atn_0' %(name), verbose=verbose)
            attention_drop0 = self.layer.dropout(x=dict_dec1['output'], rate=0.1, \
                name='%s_atn_dout_0' %(name))
            y_0 = self.layer.layer_normalization(x=x+attention_drop0, trainable=True, \
                name='%s_atn_ln_0' %(name), verbose=verbose)
            dict_dec1['ln0'] = y_0

            dict_dec2 = wblat.self_attention(layer=self.layer, x_query=x, x_key=y_enc, x_value=y_enc, num_head=num_head, mask_idx=mask_idx, udmask=udmask, \
                name='%s_atn_1' %(name), verbose=verbose)
            attention_drop1 = self.layer.dropout(x=dict_dec2['output'], rate=0.1, \
                name='%s_atn_dout_1' %(name))
            y_1 = self.layer.layer_normalization(x=y_0+attention_drop1, trainable=True, \
                name='%s_atn_ln_1' %(name), verbose=verbose)
            dict_dec2['ln1'] = y_1

            fnn_out = wblat.feed_forward_network(layer=self.layer, x=y_1, dim_ff=self.dim_ff, dim_model=self.dim_model, \
                name='%s_ffn' %(name), verbose=verbose)
            dict_dec2['ffn'] = fnn_out
            fnn_drop = self.layer.dropout(x=fnn_out, rate=0.1, \
                name='%s_ffn_dout_0' %(name))
            y_2 = self.layer.layer_normalization(x=y_1+fnn_drop, trainable=True, \
                name='%s_ffn_ln_0' %(name), verbose=verbose)
            dict_dec2['ln2'] = y_2
            x = y_2

        return dict_dec1, dict_dec2

    def __linear(self, x, dim_model, name='emb', verbose=True):

        y = self.layer.fully_connected(x=x, c_out=dim_model, \
            batch_norm=False, activation='sigmoid', name="%s" %(name), verbose=verbose)

        return y
