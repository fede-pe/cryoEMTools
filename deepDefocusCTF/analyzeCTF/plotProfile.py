import matplotlib.pyplot as plt
import numpy as np

# Load data from the text files
data_ctf_defocalized = np.loadtxt('ctfDefocalizedProfile.txt')
data_ctf = np.loadtxt('ctfProfile.txt')

# Create arrays of indices for the x-axes
x_ctf_defocalized = np.arange(len(data_ctf_defocalized))
x_ctf = np.arange(len(data_ctf))

# Plot the data
plt.plot(x_ctf_defocalized, data_ctf_defocalized, label='CTF Defocalized Profile')
plt.plot(x_ctf, data_ctf, label='CTF Profile')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('CTF Profiles')

# Show the legend
plt.legend()

# Show the plot
plt.show()

