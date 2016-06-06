import numpy as np
import theano
from theano import tensor
from blocks import initialization
from blocks.bricks import Identity, Linear
from blocks.bricks.recurrent import SimpleRecurrent


x = tensor.tensor3('x')

rnn = SimpleRecurrent(
    dim=3,
    activation=Identity(),
    weights_init=initialization.Identity()
)

rnn.initialize()

h = rnn.apply(x)

argz = np.ones((3, 1, 3), dtype=theano.config.floatX)


doubler = Linear(
    input_dim=3,
    output_dim=3,
    weights_init=initialization.Identity(2),
    biases_init=initialization.Constant(0)
)

doubler.initialize()

h_doubler = rnn.apply(doubler.apply(x))
f = theano.function([x], h_doubler)

h0 = tensor.matrix('h0')
h = rnn.apply(inputs=x, states=h0)
