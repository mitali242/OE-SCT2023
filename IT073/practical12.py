import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.2)

a, b, c = 2, 5, 8
mf_triangular = fuzz.trimf(x, [a, b, c])

a, b, c, d = 2, 4, 6, 8
mf_trapezoidal = fuzz.trapmf(x, [a, b, c, d])

plt.plot(x, mf_triangular, label='Triangular')
plt.plot(x, mf_trapezoidal, label='Trapezoidal')
plt.legend()
plt.xlabel('X')
plt.ylabel('Membership')
plt.title('Fuzzy Membership Functions')
plt.show()
