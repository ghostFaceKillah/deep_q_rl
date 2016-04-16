import numpy as np


x = np.array([1, 2, 4], dtype=np.uint8)

x_bits = np.unpackbits(x)

print len(x)
print x

print len(x_bits)
print x_bits


