"""
NOTE: The FFT function works correctly, the sine wave generation is flawed due to
the limit of the sampling rates therefore displaying the magnitude incorrectly by
a small (potentially insignificant margin) in the graph
"""

import numpy.fft as fft
import pandas as pd
from pandas import DataFrame as df
import skimage.restoration as rest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.signal as sig


class Root:
    def __init__(self):  # Parameters for graphs(preset + actual)
        self.frequency = 1  # Generates this many waveforms (USED IN PRESET ONLY)
        self.amplitude = -1  # volts (USED IN PRESET ONLY)
        self.sample_count = 100  # total
        self.model = None  # Either UTD (UNI-T) or GRA (Gratten)
        self.duration = 1  # in seconds
        self.threshold = 99e-2  # Used in IFFT method to reduce the noise
        self.filepath = None  # File doesn't exist unless said so
        self.file_data = None  # Depends on if the file is used
        self.y_axis = None
        self.x_axis = None  # Axis are overwritten in the code below
        self.y_denoised = None  # Just in case

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
            self.model = "GRA" if "GRATTEN" in filepath else "UTD"
            return True
        else:
            self.filepath = None
            self.file_data = None  # Overwrite the filepath and filedata
            print("Invalid filepath, preset is used")
            return False

    def fourier_transform(self, y_axis: df or np.array or list, sample_count: int):
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
        if self.model == "GRA":
            self.file_data = ([list(
                self.file_data)] + self.file_data.values.tolist())  # Extract the values and convert pandas dataframe to
            # python list
            self.y_axis = list(
                map(lambda y: float(y[1]), filter(lambda x: x[0].isnumeric(), self.file_data)))  # Compute the y-axis
            self.sample_count = int(self.file_data[0][1])  # Extract the sample count
            self.duration = self.sample_count * float(
                self.file_data[1][1]) * 10  # Compute the duration, *10 due to error xaxis
            self.x_axis = np.linspace(0, self.duration, self.sample_count)  # Generates an x-axis timeline
        elif self.model == "UTD":
            self.file_data = ([list(
                self.file_data)] + self.file_data.values.tolist())
            self.y_axis = list(map(lambda x: float(x[0]), [x[1] for x in list(enumerate(self.file_data[6:]))]))
            self.x_axis = np.linspace(0, 63999, 64000)
            self.sample_count = 64000  # Leave for now, could be an error
            self.duration = None  # None mentioned

    def denoise(self, y_axis: df or np.array or list):
        if type(y_axis) == list:
            y_axis = np.array(y_axis)
        y_denoised = rest.denoise_wavelet(y_axis, method="VisuShrink", rescale_sigma=True, wavelet="db20", mode="soft")
        return y_denoised
        # Transform only real parts
        # Filter using the maximum (real part of complex values)

    def find_peaks(self, y_axis: df or np.array or list):
        peaks = None
        peaks_prominences = None
        if self.model == None or self.model == "GRA":
            peaks = sig.find_peaks(y_axis, prominence=1)  # Unfinished
        elif self.model == "UTD":
            peaks = sig.find_peaks(y_axis, prominence=100, height=500)
            peaks = peaks[0]
            peaks_prominences = sig.peak_prominences(y_axis, peaks)
        return peaks, peaks_prominences

    def find_area(self, y_axis: df or np.array or list, peaks: df or np.array or list):
        total_area = 0  # Area under all the peaks
        for i in range(0, len(peaks) - 1):
            total_area += np.trapz(y_axis[peaks[i]: peaks[i + 1]])
        return total_area

    def display_data(self):  # Method to display the data gathered, add the next method here
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(7, 5),
                                                 dpi=120)  # Show both graphs at the same time
        ax1.plot(self.x_axis, self.y_axis, "-k")
        ax1.set_ylabel("original (VOLTAGE)")
        ax1.set_xlabel("Time (S)")

        # FFT SPECTRUM
        x_axis_fft, y_axis_fft, y_axis_neg_included, x_axis_neg_included = self.fourier_transform(self.y_axis,
                                                                                                  self.sample_count)  # Fourier Transform the sine
        ax2.plot(x_axis_neg_included, y_axis_neg_included, "-b")
        ax2.set_ylabel("DFT transformed (MAGNITUDE)")
        ax2.set_xlabel("Frequency (HZ)")

        # denoised graph hehe
        self.y_denoised = self.denoise(self.y_axis)

        # useless shit under construction
        max_y_denoised = max(self.y_denoised)
        min_y_denoised = min(self.y_denoised)
        self.y_axis = list(map(lambda x: x * -1 + max_y_denoised + min_y_denoised, self.y_denoised))
        peaks, peaks_prominences = self.find_peaks(self.y_axis)
        print(peaks)
        print(self.find_area(self.y_axis, peaks))
        ax4.plot(self.x_axis, self.y_axis, "-r")
        ax4.plot(peaks, np.take(self.y_axis, peaks), ".k")
        peak_first = peaks[0]
        peak_last = peaks[-1]
        ax4.fill_between(self.x_axis[peak_first: peak_last+1], self.y_axis[peak_first: peak_last+1], 0)
        ax4.set_ylabel("denoised with peaks")
        ax4.set_xlabel("Time(S)")

        # the rest of the graph here to get the peaks from the last graph
        ax3.plot(self.x_axis, self.y_denoised, "-c")
        ax3.plot(peaks, np.take(self.y_denoised, peaks), ".k")
        ax3.set_ylabel("DENOISED GRAPH")
        ax3.set_xlabel("Time")

        plt.show()  # Show the graphs

    def main(self):
        filepath = input("Input custom filepath otherwise preset is used...\n")
        if not self.filepath_exists(filepath):  # Check if the preset ot custom file is used
            self.use_sine()  # Use the sine preset wave
            self.display_data()
        else:
            self.oscilloscope_process()
            self.display_data()


if __name__ == "__main__":
    a = Root()
    a.main()
