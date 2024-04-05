"""
NOTE: The FFT function works correctly, the sine wave generation is flawed due to
the limit of the sampling rates therefore displaying the magnitude incorrectly by
a small (potentially insignificant margin) in the graph

TODO: DENOISING METHOD: INVERSE FFT WITH LOWER MAGNITUDES ZEROED
TODO: LOCATE THE POSITION OF PEAKS ()
TODO: CALCULATE THE AREA UNDER THE GRAPH BETWEEN THE PEAKS
TODO: CALCULATE THE DURATION OF EACH PEAK ()
"""

import numpy.fft as fft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy as sp


class Root:
    def __init__(self):  # Parameters for graphs(preset + actual)
        self.frequency = 1  # Generates this many waveforms (USED IN PRESET ONLY)
        self.amplitude = -1  # volts (USED IN PRESET ONLY)
        self.sample_count = 100  # total
        self.duration = 1  # in seconds
        self.threshold = 3e-2  # Used in IFFT method to reduce the noise
        self.filepath = None  # File doesn't exist unless said so
        self.file_data = None  # Depends on if the file is used
        self.y_axis = None
        self.x_axis = None  # Axis are overwritten in the code below

    def filepath_exists(self, filepath: str):  # Checks if the mentioned filepath exists, otherwise use\
        # presets
        filepath = filepath[1: -1] if len(
            filepath) > 2 else filepath  # Transform due to the File Explorer copying the filepath with the double quot. marks
        print(filepath)

        # Get the file of the path
        file = Path(filepath)
        if file.exists():
            self.file_data = pd.read_csv(filepath,
                                         low_memory=False)  # If file successfully found, parameters are overridden (
            # IMPLEMENT) (pandas dataframe)
            self.filepath = filepath
            print("Valid filepath, custom file is used")
            return True
        else:
            self.filepath = None
            self.file_data = None  # Overwrite the filepath and filedata
            print("Invalid filepath, preset is used")
            return False

    def fourier_transform(self, y_axis: np.array or list, sample_count: int):
        y_axis_fft = fft.fft(y_axis)  # Perform Fast Fourier Transform on y values
        y_axis_neg_included = y_axis_fft.copy()  # Same value count as x axis
        y_axis_fft = np.abs(y_axis_fft)  # Compute to get the magnitude (not amplitude)
        y_axis_fft = np.multiply(y_axis_fft[:len(y_axis_fft) // 2],
                                 2 / sample_count)  # Include magnitudes for only positive frequencies
        x_axis_fft = fft.fftfreq(sample_count, 1 / sample_count)  # Compute a frequency array
        x_axis_neg_included = x_axis_fft.copy()  # For inclusion of negative frequencies
        x_axis_fft = x_axis_fft[:len(x_axis_fft) // 2]  # Include only positive frequencies
        return x_axis_fft, y_axis_fft, y_axis_neg_included, x_axis_neg_included

    def use_sine(self):  # Uses a preset sine wave
        self.x_axis = np.linspace(0, self.duration, self.sample_count)  # Generates an x-axis timeline
        self.y_axis = self.amplitude * np.sin(
            2 * np.pi * self.frequency * self.x_axis)  # Generates a list of y-axis values

    def use_square_sine(self):  # Uses a transformed sine wave to create a square sine wave mapped using threshold
        self.use_sine()
        threshold = 0
        self.y_axis = list(map(lambda x: 1 if x > threshold else -1, self.y_axis))  # Transforms using threshold

    def oscilloscope_process(self):
        self.file_data = ([list(
            self.file_data)] + self.file_data.values.tolist())  # Extract the values and convert pandas dataframe to
        # python list
        print(self.file_data)
        self.y_axis = list(
            map(lambda y: float(y[1]), filter(lambda x: x[0].isnumeric(), self.file_data)))  # Compute the y-axis
        self.sample_count = int(self.file_data[0][1])  # Extract the sample count
        self.duration = self.sample_count * float(self.file_data[1][1]) * 10  # Compute the duration, *10 due to error xaxis
        self.x_axis = np.linspace(0, self.duration, self.sample_count)  # Generates an x-axis timeline

    def denoise(self, y_axis_neg_included: np.array or list):
        """y_axis = list(map(lambda x: np.real(x), y_axis_neg_included))  # Transform to real numbers
         highest_peak = max(y_axis)  # Locate the highest peak
         y_axis = list(map(lambda x: 0 if np.abs(x) < np.abs(highest_peak) * 0.03 else x, y_axis))  # Filter using threshold
         # y_axis = np.pad(y_axis, (0, 350), mode="constant")
         return y_axis
         # print(y_axis)"""


    def display_data(self):  # Method to display the data gathered, add the next method here
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)  # Show both graphs at the same time
        ax1.plot(self.x_axis, self.y_axis, "-k")
        ax1.set_ylabel("original (VOLTAGE)")
        ax1.set_xlabel("Time (S)")

        # FFT SPECTRUM
        x_axis_fft, y_axis_fft, y_axis_neg_included, x_axis_neg_included = self.fourier_transform(self.y_axis,
                                                                             self.sample_count)  # Fourier Transform the sine
        print(x_axis_neg_included)
        print(y_axis_neg_included)
        ax2.plot(x_axis_neg_included, y_axis_neg_included, "-b")
        ax2.set_ylabel("DFT transformed (MAGNITUDE)")
        ax2.set_xlabel("Frequency (HZ)")

        # FFT SPECTRUM DENOISED (DEBUG)
        '''y_axis_fft = self.denoise(y_axis_fft)
        ax3.plot(x_axis_fft, y_axis_fft, "-c")
        ax3.set_ylabel("DFT SPECTRUM DENOISED")
        ax3.set_xlabel("Frequency (HZ)")'''

        # DEBUG GRAPH / DENOISED GRAPH
        '''self.y_axis = np.pad(self.denoise(y_axis_fft), (0, 350), mode="constant")
        ax4.plot(self.x_axis, fft.ifft(self.y_axis[::-1]+self.y_axis), "-r")
        ax4.set_ylabel("IFFT denoised")
        ax4.set_xlabel("Time(S)")'''
        plt.show()  # Show the graphs

    def main(self):
        filepath = input("Input custom filepath otherwise preset is used...\n")
        if not self.filepath_exists(filepath):  # Check if the preset ot custom file is used
            self.use_sine()  # Use the sine preset wave
            self.display_data()
        else:
            self.oscilloscope_process()
            self.display_data()


a = Root()
a.main()
