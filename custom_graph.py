"""

"""
import matplotlib
import numpy.fft as fft
import pandas as pd
from pandas import DataFrame as df
import skimage.restoration as rest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.signal as sig
import time

places = ["Haileybury Almaty", "Medeu", "Shymbulak Resort", "Mountain Yurt"]
heights = [912, 1691, 2260, 3200]

circuit_light = [6.5, 15.2, 29, 61]
oscilloscope = [24, 46, 76, 56]

circuit_regression = np.polyfit(heights.copy(), circuit_light.copy(), 1)
oscilloscope_regression = np.polyfit(heights.copy(), oscilloscope.copy(), 1)
circuit_regression_y = np.poly1d(circuit_regression)
oscilloscope_regression_y = np.poly1d(oscilloscope_regression)

fig = plt.figure(num=1, figsize=(8, 6), dpi=200)

fig.suptitle("Cosmic ray flux at different altitudes", ha="right", x=0.7, color="#8a8a8a", size=20, weight=300)

print(circuit_light, oscilloscope)

plt.plot(heights, circuit_light, ".b", ms=8, label="Flux (min^-1) (measured by circuit light)")
plt.plot(heights, oscilloscope, ".r", ms=8, label="Flux (min^-1) (measured by oscilloscope trace)")
plt.plot(heights, circuit_light, "_b", ms=8)
plt.plot(heights, oscilloscope, "_r", ms=8)

plt.plot(heights, circuit_regression_y(heights), "-b", linewidth=1.5, alpha=0.5)
plt.plot(heights, oscilloscope_regression_y(heights), "-r", linewidth=1.5, alpha=0.5)

plt.ylabel("Readings (cpm)", labelpad=12)
plt.xlabel("Altitude (m)", labelpad=12)

plt.grid(color="black", alpha=0.25)

plt.legend(loc=0)

plt.savefig("Algraph.png")

plt.show()
