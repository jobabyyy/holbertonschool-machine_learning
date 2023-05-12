#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

plt.plot(x, y1, 'r--', label='C-14')  # Plot x against y1 as a dashed red line
plt.plot(x, y2, 'g-', label='Ra-226')  # Plot x against y2 as a solid green line

plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of Radioactive Elements')
plt.xlim(0, 20000)  # Set x-axis limits
plt.ylim(0, 1)  # Set y-axis limits

plt.legend(loc='upper right')  # Add legend to the upper right corner

plt.grid(True)
plt.show()
