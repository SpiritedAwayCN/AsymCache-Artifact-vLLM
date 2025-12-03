import matplotlib.pyplot as plt
import numpy as np

data_text = "40 0.02 90 1.695650e+04 1.695650e+02 0.0 -1.768680e+03"
data = list(map(float, data_text.split()))
alpha, beta, A, B = data[3:]


x = np.linspace(0, 1e3, 10000)

y1 = np.exp(-(x+A)/alpha)
y2 = np.exp(-(x+B)/beta)
y = np.minimum(y1, y2)

print(y2[0])

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of minimum of two exponentials')
plt.savefig('freq_plot.png')