"""
Bricks are parametrized Theano operations.
They have:
    * a set of attributes - such as number of input and output units
    * a set of parameters - such as weights and biases of NN layer

The lifecycle of a given brick is as following:
    * Configuration: setting attributes of the brick
    * Allocation: allocates Theano shared variables for the parameters of
                  the grid. When Brick.allocate() is called, Theano variables
                  are allocated and intialized by default to NaN
    * Application: Link-in block as a part of the Theano computational graph,
                   linking the inputs and outputs of the brick through its
                   parameters and according to attributes.
    * Initialization: Sets the numerical values of the Theano variables that
                      store the parameters of the Brick. The user provided
                      value will always replace the default initialization
                      value.


Side note:
If Theano variables of the brick object have not been allocated when apply() is
called, Blocks will quitely call Brick.allocate()
"""

# Example

# Bricks take Theano variables as inputs and provide Theano variables
# as outputs

import theano
from theano import tensor
from blocks.bricks import Tanh
x = tensor.vector('x')

from blocks.bricks import Linear
from blocks.initialization import IsotropicGaussian, Constant

linear = Linear(
    input_dim=10,
    output_dim=5,
    weights_init=IsotropicGaussian(),
    biases_init=Constant(0.01)
)

y = linear.apply(x)

print linear.parameters

print linear.parameters[1].get_value()

linear.initialize()

print linear.parameters[1].get_value()


linear2 = Linear(output_dim=10)
print(linear2.input_dim)

linear2.input_dim = linear.output_dim
print linear2.apply(x)

from blocks.bricks import MLP, Logistic

mlp = MLP(
    activations=[
        Logistic(name='sigmoid_0'),
        Logistic(name='sigmoid_1')
    ],
    dims=[16, 8, 4],
    weights_init=IsotropicGaussian(),
    biases_init=Constant(0.01)
)

print [child.name for child in mlp.children]

y = mlp.apply(x)
mlp.children[0].parameters[0].get_value()
