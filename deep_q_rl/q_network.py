"""
Code for deep Q-learning as described in:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013

and

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015


Author of Lasagne port: Nissan Pow
Modifications: Nathan Sprague
"""
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks import Linear, Tanh
from blocks import initialization

import lasagne
import numpy as np
import theano
import theano.tensor as T
from updates import deepmind_rmsprop

theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'


class DeepQLearner:
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self, input_width, input_height, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, input_scale=255.0):

        self.input_width = input_width
        self.input_height = input_height
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.discount = discount
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self.rng = rng
        self.RAM_SIZE = 128
        self.network_type = network_type
        np.set_printoptions(threshold='nan')

        lasagne.random.set_rng(self.rng)

        self.update_counter = 0

        self.l_in = None
        self.l_ram_in = None

        self.l_out = self.build_network(network_type, input_width, input_height,
                                        num_actions, num_frames, batch_size)
        if self.freeze_interval > 0:
            self.next_l_out = self.build_network(network_type, input_width,
                                                 input_height, num_actions,
                                                 num_frames, batch_size)
            self.reset_q_hat()

        states = T.tensor4('states')
        next_states = T.tensor4('next_states')
        ram_states = T.matrix('ram_states')
        next_ram_states = T.matrix('next_ram_states')
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')

        self.states_shared = theano.shared(
            np.zeros((batch_size, num_frames, input_height, input_width),
                     dtype=theano.config.floatX))

        self.next_states_shared = theano.shared(
            np.zeros((batch_size, num_frames, input_height, input_width),
                     dtype=theano.config.floatX))

        self.ram_states_shared = theano.shared(
            np.zeros((batch_size, self.RAM_SIZE), dtype=theano.config.floatX))
        self.next_ram_states_shared = theano.shared(
            np.zeros((batch_size, self.RAM_SIZE), dtype=theano.config.floatX))

        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        q_vals = lasagne.layers.get_output(self.l_out,
            {
                self.l_in: (states / input_scale),
                # self.l_ram_in: (ram_states / 256.0)
            }
        )
        
        if self.freeze_interval > 0:
            next_q_vals = lasagne.layers.get_output(self.next_l_out,
                {
                  self.l_in: (next_states / input_scale),
                  # self.l_ram_in: (next_ram_states / 256.0)
            }
            )
        else:
            next_q_vals = lasagne.layers.get_output(self.l_out,
                {
                  self.l_in: (next_states / input_scale),
                  # self.l_ram_in: (next_ram_states / 256.0),
                }
                )
            next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        target = (rewards +
                  (T.ones_like(terminals) - terminals) *
                  self.discount * T.max(next_q_vals, axis=1, keepdims=True))
        diff = target - q_vals[T.arange(batch_size),
                               actions.reshape((-1,))].reshape((-1, 1))

        if self.clip_delta > 0:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            # 
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        if batch_accumulator == 'sum':
            loss = T.sum(loss)
        elif batch_accumulator == 'mean':
            loss = T.mean(loss)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))

        params = lasagne.layers.helper.get_all_params(self.l_out)
        givens = {
            states: self.states_shared,
            next_states: self.next_states_shared,
            # ram_states: self.ram_states_shared,
            # next_ram_states: self.next_ram_states_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }
        if update_rule == 'deepmind_rmsprop':
            updates = deepmind_rmsprop(loss, params, self.lr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho,
                                              self.rms_epsilon)
        elif update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, self.lr)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)

        def inspect_inputs(i, node, fn):
            if ('maxand' not in str(node).lower()):
                return
            print i, node, "input(s) value(s):", fn.inputs #[input[0] for input in fn.inputs],
            raw_input('press enter')

        def inspect_outputs(i, node, fn):
            if ('maxand' not in str(node).lower()):
                return
            print "output(s) value(s):", fn.outputs #[output[0] for output in fn.outputs]
            raw_input('press enter')

        self._train = theano.function([], [loss, q_vals], updates=updates,
                                      givens=givens,
                                      mode="DebugMode"
                                      # mode=theano.compile.MonitorMode(
                                      #     pre_func=inspect_inputs,
                                      #     post_func=inspect_outputs),
                                      # on_unused_input='raise'
                                      )

        self._q_vals = theano.function([], q_vals,
                                       givens={
                                           states: self.states_shared,
                                           # ram_states: self.ram_states_shared,
                                       },
                                       mode="DebugMode"
                                       # mode=theano.compile.MonitorMode(
                                       #     pre_func=inspect_inputs,
                                       #     post_func=inspect_outputs),
                                       # on_unused_input='raise'
                                       )

    def build_network(self, network_type, input_width, input_height,
                      output_dim, num_frames, batch_size):
        if network_type == "nature_cuda":
            return self.build_nature_network(input_width, input_height,
                                             output_dim, num_frames, batch_size)
        if network_type == "nature_dnn":
            return self.build_nature_network_dnn(input_width, input_height,
                                                 output_dim, num_frames,
                                                 batch_size)
        elif network_type == "nips_cuda":
            return self.build_nips_network(input_width, input_height,
                                           output_dim, num_frames, batch_size)
        elif network_type == "nips_dnn":
            return self.build_nips_network_dnn(input_width, input_height,
                                               output_dim, num_frames,
                                               batch_size)
        elif network_type == "linear":
            return self.build_linear_network(input_width, input_height,
                                             output_dim, num_frames, batch_size)
        elif network_type == "mixed_ram":
            return self.build_mixed_ram_network(input_width, input_height,
                                           output_dim, num_frames, batch_size)
        elif network_type == "just_ram":
            return self.build_ram_network(input_width, input_height, output_dim,
                                          num_frames, batch_size)
        elif network_type == "big_mixed_ram":
            return self.build_big_joint_network(input_width, input_height,
                                                output_dim, num_frames, batch_size)
        elif network_type == "rnn":
            return self.build_recurent_network(input_width, input_height,
                                                output_dim, num_frames,
                                               batch_size)
        elif network_type == "ram_dropout":
            return self.build_ram_dropout_network(input_width, input_height,
                    output_dim, num_frames, batch_size)
        elif network_type == "big_ram":
            return self.build_big_ram_network(input_width, input_height,
                    output_dim, num_frames, batch_size)
        else:
            raise ValueError("Unrecognized network: {}".format(network_type))



    def train(self, states, actions, rewards, next_states, terminals, ram_states, next_ram_states):
        """
        Train one batch.

        Arguments:

        states - b x f x h x w numpy array, where b is batch size,
                 f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        next_states - b x f x h x w numpy array
        terminals - b x 1 numpy boolean array (currently ignored)
        ram_states - b x R numpy array, R - ram size
        next_ram_states - b x R numpy array

        Returns: average loss
        """

        self.states_shared.set_value(states)
        self.next_states_shared.set_value(next_states)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
        self.ram_states_shared.set_value(ram_states)
        self.next_ram_states_shared.set_value(next_ram_states)
        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            self.reset_q_hat()
        loss, _ = self._train()
        self.update_counter += 1
        return np.sqrt(loss)

    def q_vals(self, state, ram_state):
        """
        To predict the q-values of the moves, it needs to push the states in a form of a batch to the network, and return the first element of the result.
        """
        states = np.zeros((self.batch_size, self.num_frames, self.input_height,
                           self.input_width), dtype=theano.config.floatX)
        ram_states = np.zeros((self.batch_size, self.RAM_SIZE), dtype=theano.config.floatX)
        states[0, ...] = state
        ram_states[0, ...] = ram_state

        self.states_shared.set_value(states)
        self.ram_states_shared.set_value(ram_states)
        return self._q_vals()[0]

    def choose_action(self, state, epsilon, ram_state):
        """
        Choosing action to perform when in testing mode.
        """
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.q_vals(state, ram_state)
        return np.argmax(q_vals)

    def reset_q_hat(self):
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)


    def build_recurent_network(self, input_width, input_height, output_dim,
                                num_frames, batch_size):
        """
        NIPS + deeper ram
        """
        self.l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )

        l_conv1 = lasagne.layers.Conv2DLayer(
            self.l_in,
            num_filters=16,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1),
        )

        l_conv2 = lasagne.layers.Conv2DLayer(
            l_conv1,
            num_filters=32,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1),
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        recurrent_layer = RecurrentLayer(l_hidden1)

        l_hidden_joined = lasagne.layers.DenseLayer(
            recurrent_layer,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden_joined,
            num_units=output_dim,
            nonlinearity=None,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        return l_out



    def build_big_joint_network(self, input_width, input_height, output_dim,
                                num_frames, batch_size):
        """
        NIPS + deeper ram
        """
        self.l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )

        self.l_ram_in = lasagne.layers.InputLayer( shape=(batch_size, self.RAM_SIZE)
        )

        l_conv1 = lasagne.layers.Conv2DLayer(
            self.l_in,
            num_filters=16,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1),
        )

        l_conv2 = lasagne.layers.Conv2DLayer(
            l_conv1,
            num_filters=32,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1),
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )


        l_hidden_ram1 = lasagne.layers.DenseLayer(
            self.l_ram_in,
            num_units=self.RAM_SIZE,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_hidden_ram2 = lasagne.layers.DenseLayer(
            l_hidden_ram1,
            num_units=self.RAM_SIZE,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_joined = lasagne.layers.ConcatLayer(
            [l_hidden1, l_hidden_ram2],
            axis=1  # 0-based
        )

        l_hidden_joined = lasagne.layers.DenseLayer(
            l_joined,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden_joined,
            num_units=output_dim,
            nonlinearity=None,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        return l_out

    def build_nips_network(self, input_width, input_height, output_dim,
                           num_frames, batch_size):
        """
        Build a network consistent with the 2013 NIPS paper.
        """
        from lasagne.layers import cuda_convnet
        self.l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )

        l_conv1 = cuda_convnet.Conv2DCCLayer(
            self.l_in,
            num_filters=16,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_conv2 = cuda_convnet.Conv2DCCLayer(
            l_conv1,
            num_filters=32,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        return l_out



class RecurrentLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(RecurrentLayer, self).__init__(incoming, **kwargs)

        self.linear = Linear(
            input_dim=256,
            output_dim=256
        )

        self.rnn = SimpleRecurrent(
            dim=256,
            activation=Tanh()
        )

        for brick in [self.linear, self.rnn]:
            brick.weights_init = initialization.Uniform(width=0.08)
            brick.biases_init = initialization.Constant(0)
            brick.initialize()

    def get_output_for(self, input, **kwargs):
        # return self.linear.apply(input)
        return self.rnn.apply(
            self.linear.apply(
                input
            )
        )

    # def get_output_shape_for(self, input_shape):
    #     return (input_shape[0], 256)


def main():
    net = DeepQLearner(84, 84, 16, 4, .99, .00025, .95, .95, 10000,
                       32, 'rnn')


if __name__ == '__main__':
    main()
