#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

x = np.arange(0, 11)  # Generate x values from 0 to 10

plt.plot(x, y, 'r-')  # Plot y as a red line

plt.xlabel('x')
plt.ylabel('y')
plt.title('Line Graph of y')
plt.grid(True)

plt.show()
