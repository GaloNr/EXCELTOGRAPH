"""import numpy.fft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Default values

file_data_y = list(np.arange(0, 350, 1)) + list(np.arange(350, 0, -1))
file_data_x = list(np.arange(0, 700, 1))


# Parameters
frames = 700
xtiks_interval = 100
total_time = "200ms"

# Get the requested file path
file_path = input()
filepath = file_path[1:-1]
print(filepath)
"C:\Users\Comp\Desktop\GRATTEN_GA_1102CAL_TEST_DATA_CSV\0V FLAT.CSV"
# Get the file of the path
file = pd.read_csv(filepath)

# Dataframe to Numpy conversion
file = file.to_numpy()

# Exclude the data parameters
file_param = list(filter(lambda x: not x[0].isnumeric(), file.copy()))

# Exclude the data itself
file_data = list(filter(lambda x: x[0].isnumeric(), file.copy()))
file_data = list(map(lambda x: list(x) if isinstance(x, type(np.empty(shape=1))) else x, file_data))
file_data = list(map(lambda x: [float(x[0]) - 1, float(x[1])], file_data))
file_data_y = [i[1] for i in file_data]
file_data_x = [i[0] for i in file_data]
print(file_data[0: 50])
max_y = np.max(file_data_y)
min_y = np.min(file_data_y)

# Set the ticks values
ytiks = np.arange(min_y - (max_y - min_y) / 25 * 2, max_y + (max_y - min_y) / 25 * 2, ((max_y - min_y) / 25))
xtiks = np.arange(0, frames, 100)

# Get the area under the graph
area_xlim = (0, 700)  # Edit these frames to bound the area box
area_ylim_data = file_data_y[area_xlim[0]:area_xlim[1]]
area_xlim_data = np.arange(area_xlim[0], area_xlim[1], 1)
plt.fill_between(area_xlim_data, area_ylim_data, color='blue')
area = 0
for i in range(len(area_xlim_data) - 1):
    area += abs(np.trapz([area_ylim_data[i], area_ylim_data[i+1]], [area_xlim_data[i], area_xlim_data[i+1]]))
print(area)

# Create the graph (voltage/time)
plt.plot(file_data_x, file_data_y, color='green')
plt.ylabel('voltage (V)')
plt.xlabel('frames (s)')
plt.title("idk a graph")
plt.xlim((0, 700))
plt.ylim((min_y - (max_y - min_y) / 25 * 2, max_y + (max_y - min_y) / 25 * 2))
plt.yticks(ytiks)
plt.show()

# Generate a Fourier spectrum of the graph
fft_signal = np.abs(np.fft.fft(file_data_y))
plt.plot(file_data_x, fft_signal)
plt.show()

# TODO change for the same amplitude/voltage, not the time
"""

