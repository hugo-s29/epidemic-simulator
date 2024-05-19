import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv("./lyon.csv")

# Extract the coordinates of points A and B
x_a = data['x_a']
y_a = data['y_a']
x_b = data['x_b']
y_b = data['y_b']

# Plot the lines
plt.plot([x_a, x_b], [y_a, y_b], marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Lines A and B')
plt.grid(True)
plt.show()
