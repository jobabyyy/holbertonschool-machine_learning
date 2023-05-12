#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

fruit_labels = ['Farrah', 'Fred', 'Felicia']
fruit_names = ['Apples', 'Bananas', 'Oranges', 'Peaches']
fruit_colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

fig, ax = plt.subplots()

for i in range(fruit.shape[0]):
    ax.bar(fruit_labels, fruit[i], bottom=np.sum(fruit[:i], axis=0),
           color=fruit_colors[i], label=fruit_names[i])

# Replace the 'Fred person' with 'Fred'
ax.set_xticklabels(fruit_labels)

ax.set_xlabel('')
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit')
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))
ax.legend()

plt.show()
