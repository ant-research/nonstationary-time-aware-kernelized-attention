import tensorflow as tf
import numpy as np
import math
from tensorflow.keras import regularizers

"""
Transformers with the Generalized Spectral Mixture Kernel (GSMK)

How to use:
    encoder_model = EncoderWithGSMK()
    enc_outputs = encoder_model(enc_event_inputs, time_inputs=(enc_time_inputs, enc_time_inputs))

    decoder_model = DecoderWithGSMK()
    dec_outputs = decoder_model(dec_event_inputs, dec_time_inputs=(dec_time_inputs, dec_time_inputs),
        cross_time_inputs=(dec_time_inputs, enc_time_inputs), enc_output=enc_outputs)
"""


class EncoderWithGSMK(tf.keras.layers.Layer):
    def __init__(self, num_layers=2, d_model=32, num_heads=1, dff=32, maximum_position_encoding=None,
                 time_dim=4, rate=0.1, l1=0., l2=0.):
        """

        Args:
            num_layers: how many EncoderLayer are used
            d_model: dimensionality of the hidden embedding
            num_heads: the number of heads in the EncoderLayer
            dff: dimensionality of feed-forward network in the EncoderLayer
            maximum_position_encoding: the length of position encoding if position encoding is used,
                usually equals to the length of the input
            time_dim: params for building the TimeGSMKernel (see details in the definition of TimeGSMKernel)
            rate: dropout rate
            l1: weight for l1 reg
            l2: weight for l2 reg
        """
        super(EncoderWithGSMK, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        if maximum_position_encoding is not None:
            self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                    self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, time_dim, rate, l1, l2)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, time_inputs=None, mask=None, pos_enc_flag=None):
        """

        Args:
            inputs: a tensor
            time_inputs: (t_q, t_k), i.e., the time embedding of the query and key of the encoder
            mask: for masking out the input
            pos_enc_flag: whether to use position encoding

        Returns: a tensor

        """

        seq_len = tf.shape(inputs)[1]
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if pos_enc_flag:
            inputs += self.pos_encoding[:, :seq_len, :]

        inputs = self.dropout(inputs)
        for i in range(self.num_layers):
            with tf.variable_scope('transformer_encoder_' + str(i)):
                inputs = self.enc_layers[i](inputs, time_inputs, mask)

        return inputs  # (batch_size, input_seq_len, d_model)


class DecoderWithGSMK(tf.keras.layers.Layer):
    def __init__(self, num_layers=2, d_model=32, num_heads=1, dff=32,
                 maximum_position_encoding=None, time_dim=None, rate=0.1, l1=0., l2=0.):
        """
        Args:
            num_layers: how many DecoderLayers are used
            d_model: dimensionality of the hidden embedding
            num_heads: the number of heads in the DecoderLayer
            dff: dimensionality of feed-forward network in the DecoderLayer
            maximum_position_encoding: the length of position encoding if position encoding is used,
                usually equals to the length of the input
            time_dim: params for building the TimeGSMKernel (see details in the definition of TimeGSMKernel)
            rate: dropout rate
            l1: weight for l1 reg
            l2: weight for l2 reg
        """
        super(DecoderWithGSMK, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        if maximum_position_encoding is not None:
            self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, time_dim, rate, l1, l2)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, dec_time_inputs=None, cross_time_inputs=None, target_attention=True, enc_output=None,
             look_ahead_mask=None, padding_mask=None, pos_enc_flag=None):
        """

        Args:
            inputs: a tensor
            dec_time_inputs: (t_q, t_k), i.e., the time embedding of the query and key of the decoder
            cross_time_inputs: (t_e, t_d), i.e., the time embedding of the encoder and decoder
            target_attention: whether to use a target attention on the input
            enc_output: outputs from the encoder
            look_ahead_mask: for masking out the look-ahead part
            padding_mask: for masking out the input
            pos_enc_flag: whether to use position encoding

        Returns: a tensor

        """

        if inputs is not None and target_attention:
            seq_len = tf.shape(inputs)[1]
            inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            if pos_enc_flag:
                inputs += self.pos_encoding[:, :seq_len, :]

            inputs = self.dropout(inputs)

        attention_weights = {}
        for i in range(self.num_layers):
            inputs, block1, block2 = self.dec_layers[i](inputs, dec_time_inputs, cross_time_inputs, target_attention,
                                                        enc_output, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return inputs


class TimeGSMKernel(tf.keras.layers.Layer):
    """Build a GSM kernel"""
    def __init__(self, time_dim=4, l1=0., l2=0.):
        """

        Args:
            time_dim: dimensionality of the hidden embedding of ffn
            l1: weight for l1 reg
            l2: weight for l2 reg
        """
        super(TimeGSMKernel, self).__init__()

        self.period_var = gsm_ffn(time_dim, l1, l2)
        self.sigma_var = gsm_ffn(time_dim, l1, l2)
        self.basis_expan_var = gsm_ffn(time_dim, l1, l2)

        self.time_dim = time_dim

    def call(self, inputs, scope='time_gsm_kernel'):
        """

        Args:
            inputs: [t_q, t_k], i.e., inputs for building the GSM kernel
            scope: scope name

        Returns: the GSM kernel

        """

        q, k = inputs  # batch * Lq * 1
        lq = q.get_shape().as_list()[2]
        lk = k.get_shape().as_list()[2]

        period_var_q = self.period_var(q)  # batch * lq * time_dim
        period_var_k = self.period_var(k)

        q_period_q = 2 * math.pi * tf.multiply(period_var_q, q)  # batch * lq * time_dim
        k_period_k = 2 * math.pi * tf.multiply(period_var_k, k)  # batch * lk * time_dim

        q_period_q = tf.tile(tf.expand_dims(q_period_q, axis=2), [1, 1, lk, 1])  # batch * lq * lk * time_dim
        k_period_k = tf.tile(tf.expand_dims(k_period_k, axis=1), [1, lq, 1, 1])  # batch * lq * lk * time_dim
        qk_period_diff = tf.keras.layers.Add()([q_period_q, -1. * k_period_k])  # batch * lq * lk * time_dim

        cos_enc = tf.cos(qk_period_diff)

        sigma_q = self.sigma_var(q)  # batch * lq * time_dim
        sigma_q = tf.tile(tf.expand_dims(sigma_q, axis=2), [1, 1, lk, 1])  # batch * lq * lk * time_dim
        sigma_q += 1e-6  # add an epsilon to avoid zeros
        sigma_k = self.sigma_var(k)  # batch * lk * time_dim
        sigma_k = tf.tile(tf.expand_dims(sigma_k, axis=1), [1, lq, 1, 1])  # batch * lq * lk * time_dim
        sigma_k += 1e-6  # add an epsilon to avoid zeros

        qk_diff = tf.keras.layers.Add()(
            [tf.tile(q, [1, 1, lk]), -1. * tf.transpose(tf.tile(k, [1, 1, lq]), perm=[0, 2, 1])]
        )  # batch * lq * lk
        qk_diff = tf.expand_dims(qk_diff, axis=-1)  # batch * lq * lk *1
        qk_diff = tf.tile(qk_diff, [1, 1, 1, self.time_dim])  # batch*lq*lk*time_dim

        exp_enc = tf.exp(-1.*tf.divide(qk_diff**2, tf.add(sigma_q**2, sigma_k**2)))
        local_enc = (tf.divide(2 * tf.multiply(sigma_q, sigma_k), tf.add(sigma_q**2, sigma_k**2)))**0.5

        gibbs_enc = tf.multiply(local_enc, exp_enc)
        basis_expan_q = self.basis_expan_var(q)
        basis_expan_k = self.basis_expan_var(k)

        basis_expan_q = tf.tile(tf.expand_dims(basis_expan_q, axis=2), [1, 1, lk, 1])
        basis_expan_k = tf.tile(tf.expand_dims(basis_expan_k, axis=1), [1, lq, 1, 1])

        basis_expan_qk = tf.multiply(basis_expan_q, basis_expan_k)

        time_gsm_kernel = tf.multiply(basis_expan_qk, tf.multiply(gibbs_enc, cos_enc))
        time_gsm_kernel = tf.reduce_sum(time_gsm_kernel, axis=-1)

        return time_gsm_kernel


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, time_dim=None, l1=0., l2=0.):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

        if time_dim is not None:
            self.time_kernel = TimeGSMKernel(time_dim, l1, l2)
        else:
            self.time_kernel = None

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, time_inputs=None, mask=None):

        v, k, q = inputs
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        time_kernel_vals = None

        if time_inputs is not None and self.time_kernel is not None:
            time_kernel_vals = self.time_kernel(time_inputs)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, time_kernel_vals, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class LayerNorm(tf.keras.layers.Layer):
    """Layer normalization following with a linear transformation"""

    def __init__(self, l2_reg=0.001, eps=1.e-6, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.eps = eps
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.hidden_units = int(input_shape[-1])
        self.kernel = self.add_weight(
            name='weight_ln',
            shape=(self.hidden_units,),
            initializer=tf.keras.initializers.Ones(),
            regularizer=tf.keras.regularizers.L1L2(0, self.l2_reg),
            trainable=True)
        self.bias = self.add_weight(
            name='bias_ln',
            shape=(self.hidden_units,),
            initializer=tf.keras.initializers.Zeros(),
            regularizer=tf.keras.regularizers.L1L2(0, self.l2_reg),
            trainable=True)
        super(LayerNorm, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # normalization
        mean = tf.keras.backend.mean(inputs, axis=-1, keepdims=True)
        var = tf.keras.backend.mean(tf.keras.backend.square(inputs - mean), axis=-1, keepdims=True)
        output = tf.keras.layers.Lambda(lambda x: (x[0] - x[1]) / tf.sqrt(x[2] + self.eps))([inputs, mean, var])

        # linear transformation
        output = tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))([output, self.kernel])
        output = tf.keras.backend.bias_add(output, self.bias)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'eps': self.eps, 'l2_reg': self.l2_reg, 'hidden_units': self.hidden_units}
        base_config = super(LayerNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, time_dim=None, rate=0.1, l1=0., l2=0.):
        super(EncoderLayer, self).__init__()

        self.multi_head_attn = MultiHeadAttention(d_model, num_heads, time_dim, l1, l2)
        self.ffn = point_wise_feed_forward_network(d_model, dff, l1, l2)

        self.layernorm1 = LayerNorm(eps=1e-6)
        self.layernorm2 = LayerNorm(eps=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, time_inputs=None, mask=None):
        attn_output, _ = self.multi_head_attn((inputs, inputs, inputs), time_inputs, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, time_dim=None, rate=0.1, l1=0., l2=0.):
        super(DecoderLayer, self).__init__()

        self.multi_head_attn1 = MultiHeadAttention(d_model, num_heads, time_dim, l1, l2)
        self.multi_head_attn2 = MultiHeadAttention(d_model, num_heads, time_dim, l1, l2)

        self.ffn = point_wise_feed_forward_network(d_model, dff, l1, l2)

        self.layernorm1 = LayerNorm(eps=1e-6)
        self.layernorm2 = LayerNorm(eps=1e-6)
        self.layernorm3 = LayerNorm(eps=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, dec_time_inputs=None, cross_time_inputs=None, target_attention=True, enc_output=None,
             look_ahead_mask=None, padding_mask=None):

        attn_weights_block1 = None
        if inputs is not None:
            if target_attention:
                if look_ahead_mask is None:
                    look_ahead_mask = tri_causal_mask(inputs.shape[1])

                attn1, attn_weights_block1 = self.multi_head_attn1(
                    (inputs, inputs, inputs), dec_time_inputs, look_ahead_mask
                )
                attn1 = self.dropout1(attn1)
                out1 = self.layernorm1(attn1 + inputs)
            else:
                out1 = self.layernorm1(inputs)

            attn2, attn_weights_block2 = self.multi_head_attn2(
                (enc_output, enc_output, out1), cross_time_inputs, padding_mask
            )
            attn2 = self.dropout2(attn2)
            out2 = self.layernorm2(attn2 + out1)
        else:
            attn2, attn_weights_block2 = self.multi_head_attn2(
                (enc_output, enc_output, enc_output), None, padding_mask)
            attn2 = self.dropout2(attn2)
            out2 = self.layernorm2(attn2 + enc_output)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    return pos_encoding


def scaled_dot_product_attention(q, k, v, time_kernel, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch, heads, seq_len_q, seq_len_k)
    num_heads = q.get_shape().as_list()[1]

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if time_kernel is not None:
        time_kernel = tf.tile(tf.expand_dims(time_kernel, axis=1), [1, num_heads, 1, 1])
        time_kernel = time_kernel / tf.exp(tf.math.sqrt(dk))
        scaled_attention_logits += time_kernel

    if mask is not None:
        scaled_attention_logits += (mask * (-1e9))

    attention_weights = tf.nn.softmax(scaled_attention_logits)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff, l1=0., l2=0.):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
        tf.keras.layers.Dense(d_model, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))
        ])


def gsm_ffn(time_dim, l1=0., l2=0.):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(time_dim, activation='relu', kernel_initializer='glorot_uniform',
                              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
        tf.keras.layers.Dense(time_dim, activation='relu', kernel_initializer='glorot_uniform',
                              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
    ])


def tri_causal_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask
