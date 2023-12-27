import sys
import matplotlib.pyplot as plt
import numpy as np

# Check if the filename is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python script.py <filename>")
    sys.exit(1)

# Get the filename from the command-line argument
filename = sys.argv[1]

# Load data from the specified file
try:
    data = np.loadtxt(filename)
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error: Unable to load data from '{filename}'.")
    print(f"Details: {e}")
    sys.exit(1)

# Sort data based on the second column (index 1)
sorted_indices = np.argsort(data[:, 1])
sorted_data = data[sorted_indices]

# Extracting columns
values = data[:, 0]
radii = data[:, 1]

# Create a 2D plot
plt.plot(radii, values, label='Radial Profile')
plt.title('2D Plot of Radial Profile')
plt.xlabel('Radius')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
