import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class Layer(object):

    def __init__(self, args, act, name, layer_norm=True):
        self.n_route = args.n_route
        self.n_his = args.n_his
        self.n_pre = args.n_pre
        self.encode_dim = args.encode_dim
        self.act = act
        self.name = name
        self.layer_norm = layer_norm

    def l_norm(self, x, name):
        _, _, N, C = x.get_shape().as_list()
        mu, sigma = tf.nn.moments(x, axes=[2, 3], keep_dims=True)

        gamma = tf.get_variable(f'gamma_{name}', initializer=tf.ones([1, 1, N, C]))
        beta = tf.get_variable(f'beta_{name}', initializer=tf.zeros([1, 1, N, C]))
        _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
        return _x

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.variable_scope(self.name):
            outputs = self._call(inputs)

            return outputs


class MLP(Layer):

    def __init__(self, args, act, name, input_dim, output_dim, layer_norm=True):
        Layer.__init__(self, args, act, name, layer_norm)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _conv_full(self, inputs, input_dim, output_dim, name):
        wt1 = tf.get_variable(name=f'wt1_{name}', shape=[1, 1, input_dim, output_dim], dtype=tf.float32)
        bt1 = tf.get_variable(name=f'bt1_{name}', initializer=tf.zeros([output_dim]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(inputs, wt1, strides=[1, 1, 1, 1], padding='SAME') + bt1
        return x_conv

    def _call(self, inputs):
        _, T, _, _ = inputs.get_shape().as_list()

        mul_input = self._conv_full(inputs, self.input_dim, self.output_dim, self.name)
        # layer_norm
        if self.layer_norm:
            mul_input = self.l_norm(mul_input, self.name)

        return self.act(mul_input)


class TimeConvolution(Layer):

    def __init__(self, args, act, name, input_dim, output_dim, T_O, layer_norm=True):
        Layer.__init__(self, args, act, name, layer_norm)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.T_O = T_O

    def _conv_full(self, inputs, input_dim, output_dim, name):

        wt1 = tf.get_variable(name=f'wt1_{name}', shape=[1, 1, input_dim, output_dim], dtype=tf.float32)
        bt1 = tf.get_variable(name=f'bt1_{name}', initializer=tf.zeros([output_dim]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(inputs, wt1, strides=[1, 1, 1, 1], padding='SAME') + bt1
        return x_conv

    def _time_att(self, inputs, T_O):
        _, T, N, C = inputs.get_shape().as_list()

        # temporal matrix
        W = tf.get_variable(name='W_1', shape=[T, T_O], dtype=tf.float32)
        # B I N C - B O N C
        outputs = tf.einsum('btnc,tm->bmnc', inputs, W)

        return outputs

    def _call(self, inputs):

        if self.input_dim == self.output_dim:
            res_inputs = inputs
        else:
            res_inputs = self._conv_full(inputs, self.input_dim, self.output_dim, name='res')

        inputs_con_att = self._conv_full(inputs, self.input_dim, self.output_dim, name='mul')
        inputs_con_att = self._time_att(inputs_con_att, self.T_O)

        if self.layer_norm:
            outputs = self.l_norm(inputs_con_att + res_inputs[:, -self.T_O:, :, :], name=self.name)
        else:
            outputs = inputs_con_att + res_inputs

        return self.act(outputs)


class Graph_base_conv(Layer):

    def __init__(self, args, act, name, input_dim, output_dim, layer_norm=True):

        Layer.__init__(self, args, act, name, layer_norm)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _gconv(self, inputs, inputs_res, theta, bias):
        # BTNI -> BTNO
        inputs = tf.einsum('btni,io->btno', inputs, theta) + bias

        _, T, N, C = inputs.get_shape().as_list()
        node_en = tf.get_variable(name='embedding_1', shape=[N, self.encode_dim], dtype=tf.float32)

        kernel = tf.matmul(node_en, tf.transpose(node_en, [1, 0]))
        # spatial matrix
        W = tf.nn.softmax(tf.nn.relu(kernel), axis=0)
        # BTNC
        outputs = tf.einsum('btnc,nm->btmc', inputs, W)

        inputs_gconv = outputs + inputs_res
        if self.layer_norm:
            inputs_gconv = self.l_norm(inputs_gconv, name=self.name)

        return inputs_gconv

    def _call(self, inputs):

        """
        Graph convolution
        :param inputs: [B, T, N, C_IN]
        :return: [B, T, N, C_OUT]
        """

        if self.input_dim == self.output_dim:
            inputs_res = inputs

        else:
            wt_res = tf.get_variable(name='wt_res', shape=[1, 1, self.input_dim, self.output_dim], dtype=tf.float32)
            inputs_res = tf.nn.conv2d(inputs, wt_res, strides=[1, 1, 1, 1], padding='SAME')

        ws_1 = tf.get_variable(name='ws', shape=[self.input_dim, self.output_dim], dtype=tf.float32)
        bs_1 = tf.get_variable(name='bs', initializer=tf.zeros([self.output_dim]), dtype=tf.float32)

        gconv_output = self._gconv(inputs=inputs, inputs_res=inputs_res, theta=ws_1, bias=bs_1)

        return self.act(gconv_output)


