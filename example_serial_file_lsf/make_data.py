import numpy as np

a = +3
b = -3
c = +4
d = -1

n = 100

x = np.random.uniform(-1.,0.7,n)
y = a*x**3 + b*x**2 + c*x + d + np.random.uniform(-0.2,0.2,n)

np.savetxt('data_example_serial_file', zip(x,y), fmt="%.3f")

