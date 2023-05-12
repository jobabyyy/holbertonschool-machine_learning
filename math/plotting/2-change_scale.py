#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

plt.plot(x, y, 'r-')  # Plot x against y as a red line

plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of C-14')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.xlim(0, 28650)  # Set x-axis limits

plt.grid(True)
plt.show()
