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
import time

class Root:
    def __init__(self):  # Parameters for graphs(preset + actual)
        self.frequency = 1  # Generates this many waveforms (USED IN PRESET ONLY)
        self.amplitude = -1  # volts (USED IN PRESET ONLY)
        self.sample_count = 100  # total
        self.model = None  # Either UTD (UNI-T) or GRA (Gratten)
        self.duration = 1  # in seconds
        self.threshold = 5e-2  # Used in IFFT method to reduce the noise
        self.filepath = None  # File doesn't exist unless said so
        self.file_data = None  # Depends on if the file is used
        self.y_axis = None
        self.x_axis = None  # Axis are overwritten in the code below
        self.y_denoised = None  # Just in case
        self.start = 0
        self.end = 0

    def filepath_exists(self, filepath: str):  # Checks if the mentioned filepath exists, otherwise use\
        # presets
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
        y_denoised = rest.denoise_wavelet(y_axis, method="VisuShrink", rescale_sigma=False, wavelet="db20", mode="soft")
        return y_denoised
        # Transform only real parts
        # Filter using the maximum (real part of complex values)

    def find_peaks(self, y_axis: df or np.array or list):
        peaks = None
        peaks_prominences = None
        if self.model is None or self.model == "GRA":
            peaks = sig.find_peaks(y_axis, prominence=1)  # Unfinished
        elif self.model == "UTD":
            peaks, _ = sig.find_peaks(y_axis)
            peaks_prominences, _, _ = sig.peak_prominences(y_axis, peaks)  # noqa
        return peaks, peaks_prominences

    def find_area(self, y_axis: df or np.array or list, peaks: df or np.array or list):  # Peaks are a list of x values where peaks appear
        if peaks.size >= 2:
            '''total_area = 0  # Area under all the peaks
            distance_furthest = peaks[-1] - peaks[0]
            wave_start = int(round(peaks[0] - distance_furthest * 0.1, 0)) if round(peaks[0] - distance_furthest * 0.1, 0) >= 0 else 0
            wave_end = int(round(peaks[-1] + distance_furthest * 0.1, 0)) if round(peaks[-1] + distance_furthest * 0.1, 0) <= self.sample_count else 0
            total_area += round(np.trapz(y_axis[wave_start: wave_end+1]), 2)
            return total_area, wave_start, wave_end'''
            peaks_enum = np.array(list(zip(np.linspace(0, len(peaks) - 1, len(peaks)), peaks)))  # Enum list of indexes
            peaks_values_enum = np.array(list(zip(np.linspace(0, len(peaks) - 1, len(peaks)), np.take(y_axis, peaks))))  # Enum list of values for the peaks
            peaks_values_max = np.amax(peaks_values_enum)  # Maximum peak
            peaks_values_filter = np.where(peaks_values_enum[:,1] > peaks_values_max * self.threshold)  # Filter to locate the actual peaks indexes
            peaks_start = peaks_values_filter[0][0] - 1
            peaks_end = peaks_values_filter[0][-1] + 1
            peaks_start_x = int(peaks_enum[peaks_start, 1])
            peaks_end_x = int(peaks_enum[peaks_end, 1])
            total_area = np.abs(round(np.trapz(y_axis[peaks_start_x: peaks_end_x+1]), 2))
            peaks_values_max = np.where(peaks_values_enum[:,1] == peaks_values_max)
            peaks_values_max_x = int(peaks_enum[peaks_values_max][0][1])
            return total_area, peaks_start_x, peaks_end_x, peaks_values_max_x
        else:
            return 0, 0, 0

    def runtime(self):
        print(round(self.end - self.start, 2), "s")

    def display_data(self):  # Method to display the data gathered, add the next method here
        fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(8,6),
                                                 dpi=200)  # Show both graphs at the same time
        ax1.plot(self.x_axis, self.y_axis, "-k")
        ax1.set_ylabel("VOLTAGE (mV)", labelpad=12)
        ax1.set_xlabel("Time (uS * 1e-1)", labelpad=12)

        # FFT SPECTRUM
        '''x_axis_fft, y_axis_fft, y_axis_neg_included, x_axis_neg_included = self.fourier_transform(self.y_axis,
                                                                                                  self.sample_count)  # Fourier Transform the sine
        ax2.plot(x_axis_neg_included, y_axis_neg_included, "-b")
        ax2.set_ylabel("FFT")
        ax2.set_xlabel("Frequency (HZ)")'''

        # denoised graph hehe
        self.y_denoised = self.denoise(self.y_axis)

        # reverse magic
        max_y_denoised = max(self.y_denoised)
        min_y_denoised = min(self.y_denoised)
        self.y_axis = list(map(lambda x: x * -1 + max_y_denoised + min_y_denoised, self.y_denoised))
        peaks, peaks_prominences = self.find_peaks(self.y_axis)

        if peaks.size >= 2:
            '''ax4.plot(self.x_axis, self.y_axis, "-r",)
            ax4.plot(start, np.take(self.y_axis, start), ".k", markersize=8, label="Time period of the wave: {calc} microseconds".format(calc = (end-start) / 10))
            ax4.plot(end, np.take(self.y_axis, end), ".k", markersize=8)
            ax4.plot(max1, np.take(self.y_axis, max1), ".y", markersize=8, label="Area under the wave: {total_area:.3e}".format(total_area=total_area))
            ax4.fill_between(self.x_axis[start: end+1], self.y_axis[start: end+1], self.y_axis[start], alpha=0.3, color="red")'''

            total_area, start, end, max1 = self.find_area(self.y_axis, peaks)
            # the rest of the graph here to get the peaks from the last graph
            ax3.plot(self.x_axis, self.y_denoised, "-c")
            # ax3.plot(peaks, np.take(self.y_denoised, peaks), ".k", markersize=5)
            y_max = max(self.y_axis)
            y_min = min(self.y_axis)
            self.y_axis = list(map(lambda x: x * -1 + y_max + y_min, self.y_axis))
            ax3.fill_between(self.x_axis[start: end+1], self.y_axis[start: end+1], self.y_axis[start], alpha=0.3, color="cyan")
            ax3.plot(start, np.take(self.y_axis, start), "xk", markersize=4,
                     label="Time period of the wave: {calc} microseconds".format(calc = (end-start) / 10))
            ax3.plot(end, np.take(self.y_axis, end), "xk", markersize=4)
            ax3.plot(max1, np.take(self.y_axis, max1), "xy", markersize=4,
                     label="Area under the wave: {total_area:.3e}".format(total_area=total_area))
        else:
            '''ax4.plot(self.x_axis, self.y_axis, "-r")'''
            ax3.plot(self.x_axis, self.y_denoised, "-c")
            print("no peaks detected :(")

        '''ax4.set_ylabel("INVERSED")
        ax4.set_xlabel("Time(S)")'''
        ax3.set_ylabel("VOLTAGE (mV)", labelpad=12)
        ax3.set_xlabel("Time (uS * 1e-1)", labelpad=12)

        plt.subplots_adjust(hspace=1)
        self.end = time.time()
        '''ax4.legend()'''
        ax3.legend()
        fig.suptitle("Graph", size=20)
        plt.savefig("Singraph.png")
        plt.show()  # Show the graph

    def main(self):
        self.start = time.time()
        filepath = input("Input custom filepath otherwise preset is used...\n")
        if not self.filepath_exists(filepath):  # Check if the preset ot custom file is used
            self.use_sine()  # Use the sine preset wave
            self.display_data()
        else:
            self.oscilloscope_process()
            self.display_data()
        self.runtime()


if __name__ == "__main__":
    start = time.time()
    a = Root()
    a.main()
