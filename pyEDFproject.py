import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class RawData:
    """
    A class to parse, process, and visualize EEG data from EDF (European Data Format) files.
    """

    def __init__(self, file_path):
        """
        Initializes the RawData object with the path to the EDF file.
        """
        self.file_path = file_path

    def parse_header(self): 
        """
        Reads the fixed-length and variable-length header information from the EDF file.
        Extracts patient info, recording metadata, and channel-specific parameters.
        """
        with open(self.file_path, "rb") as f:
            self.version = f.read(8).decode("ascii").strip()
            self.patient_id = f.read(80).decode("ascii").strip()
            self.recording_id = f.read(80).decode("ascii").strip()
            self.start_date = f.read(8).decode("ascii").strip()
            self.start_time = f.read(8).decode("ascii").strip()
            self.header_size = (f.read(8).decode("ascii").strip())
            self.header_size_int = int(self.header_size)
            self.reserved_area = f.read(44).decode("ascii").strip()

            self.ndr = f.read(8).decode("ascii").strip() # ndr: Number of Data Records
            self.ndr_int = int(self.ndr)

            self.ddr = f.read(8).decode("ascii").strip() # ddr: Duration of a Data Record
            self.ddr_int = float(self.ddr)

            self.nc = f.read(4).decode("ascii").strip() # nc: Number of Channels
            self.nc_int = int(self.nc)

            self.labels = []
            self.transcuder_type = []
            self.physical_dimension = []
            self.physical_min = []
            self.physical_max = []
            self.digital_min = []
            self.digital_max = []
            self.prefiltering = []
            self.ns = [] # ns: Number of Samples
    
            for i in range(self.nc_int):
                self.labels.append(f.read(16).decode("ascii").strip())

            for i in range(self.nc_int):
                self.transcuder_type.append(f.read(80).decode("ascii").strip())

            for i in range(self.nc_int):
                self.physical_dimension.append(f.read(8).decode("ascii").strip())

            for i in range(self.nc_int):
                self.physical_min.append(float(f.read(8).decode("ascii").strip()))

            for i in range(self.nc_int):
                self.physical_max.append(float(f.read(8).decode("ascii").strip()))

            for i in range(self.nc_int):
                self.digital_min.append(int(f.read(8).decode("ascii").strip()))

            for i in range(self.nc_int):
                self.digital_max.append(int(f.read(8).decode("ascii").strip()))

            for i in range(self.nc_int):
                self.prefiltering.append(f.read(80).decode("ascii").strip())

            for i in range(self.nc_int):
                self.ns.append(int(f.read(8).decode("ascii").strip()))

            f.read(32*self.nc_int) # Skip 32 bytes for each channel (reserved area)
    
    def read_data(self): 
        """
        Reads the raw binary signal data from the EDF file and converts it into 
        a NumPy array of 16-bit integers.
        """
        with open(self.file_path, "rb") as f:
            f.seek(self.header_size_int)
            self.raw_eeg_bytes = f.read()
            f.close()

        self.raw_eeg_data = np.frombuffer(self.raw_eeg_bytes, dtype="<i2")

        assert sum(self.ns) * self.ndr_int == len(self.raw_eeg_data), "Data size mismatch"

    def convert_to_matrix(self): 
        """
        Reshapes the raw signal data into a matrix and maps each channel's 
        data to its corresponding label in a dictionary.
        """
        self.raw_eeg_data_reshaped = self.raw_eeg_data.reshape(self.ndr_int, sum(self.ns))

        self.raw_eeg_data_reshaped_dictionary = {}

        offset = 0
        for i in range(self.nc_int):
            current_channel_ns = self.ns[i]
            sliced_matris = self.raw_eeg_data_reshaped[:, offset:offset + current_channel_ns]
            sliced_matris_flattened = sliced_matris.flatten()
            self.raw_eeg_data_reshaped_dictionary[self.labels[i]] = sliced_matris_flattened
            offset = offset + current_channel_ns

    def convert_digital_to_physical(self): 
        """
        Converts the raw digital values into physical units (e.g., microvolts) 
        using the gains calculated from header information.
        """
        self.physical_signals = {}

        for i in range(self.nc_int):
            physical_min = self.physical_min[i]
            physical_max = self.physical_max[i]
            digital_min = self.digital_min[i]
            digital_max = self.digital_max[i]

            if digital_max == digital_min:
                gain = 0
            else:
                gain = (physical_max - physical_min) / (digital_max - digital_min)

            channel_values = self.raw_eeg_data_reshaped_dictionary[self.labels[i]]
            channel_values = channel_values.astype(np.float64)

            physical_values = (channel_values - digital_min) * gain + physical_min

            self.physical_signals[self.labels[i]] = physical_values

    def apply_bandpass_filter(self, channel_name, low_cut=0.5, high_cut=40.0, order=4):
        """
        Applies a Butterworth bandpass filter to a specific channel or all channels.
        Useful for removing noise outside the relevant frequency bands (e.g., 0.5-40 Hz).
        """
        if channel_name == "all":
            self.filtered_datas_dictionary = {}
            for i in range(len(self.labels)):
                channel_name = self.labels[i]
                raw_data = self.physical_signals[channel_name]
                idx = self.labels.index(channel_name)
                fs = self.ns[idx] / self.ddr_int
                nyquist = 0.5 * fs
                low = low_cut / nyquist
                high = high_cut / nyquist
                b, a = signal.butter(order, [low, high], btype='band')
                self.filtered_data = signal.filtfilt(b, a, raw_data)
                print(f"Channel \"{channel_name}\" filtered: {low_cut}-{high_cut} Hz (Butterworth Order {order})")
                self.filtered_datas_dictionary[channel_name] = self.filtered_data
        else:
            raw_data = self.physical_signals[channel_name]

            idx = self.labels.index(channel_name)
            fs = self.ns[idx] / self.ddr_int

            nyquist = 0.5 * fs

            low = low_cut / nyquist
            high = high_cut / nyquist
            
            b, a = signal.butter(order, [low, high], btype='band')
            
            self.filtered_data = signal.filtfilt(b, a, raw_data)

            print(f"Channel \"{channel_name}\" filtered: {low_cut}-{high_cut} Hz (Butterworth Order {order})")

    def plot_channel(self, channel_name):
        """
        Plots the time-domain signal for a single specified channel.
        """
        number_of_this_channel = self.labels.index(channel_name)

        ns_of_this_channel = self.ns[number_of_this_channel]
        ddr_of_this_channel = self.ddr_int

        self.frequency_of_channel = ns_of_this_channel / ddr_of_this_channel

        data_of_this_channel = self.physical_signals[channel_name]

        number_of_points = len(data_of_this_channel)

        total_time = number_of_points / self.frequency_of_channel

        time_axis = np.linspace(0, total_time, number_of_points, endpoint=True)

        plt.plot(time_axis, data_of_this_channel)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Voltage)")
        plt.title(f"Channel: {channel_name}")
        plt.show()

    def plot_all_channels(self):
        """
        Visualizes all EEG channels in a grid format (5x8 subplot).
        """
        total_number_of_channels = len(self.labels)
        fig, ax = plt.subplots(5, 8)

        for i in range(len(self.labels)):
            channel_name = self.labels[i]
            number_of_this_channel = self.labels.index(channel_name)
            ns_of_this_channel = self.ns[number_of_this_channel]
            ddr_of_this_channel = self.ddr_int
            self.frequency_of_channel = ns_of_this_channel / ddr_of_this_channel
            data_of_this_channel = self.physical_signals[channel_name]
            number_of_points = len(data_of_this_channel)
            total_time = number_of_points / self.frequency_of_channel
            time_axis = np.linspace(0, total_time, number_of_points, endpoint=True)

            if i <= 7:
                ax[0,i].plot(time_axis, data_of_this_channel); ax[0,i].set_title(channel_name)
            elif i <= 15:
                ax[1,i-8].plot(time_axis, data_of_this_channel); ax[1,i-8].set_title(channel_name)
            elif i <= 23:
                ax[2,i-16].plot(time_axis, data_of_this_channel); ax[2,i-16].set_title(channel_name)
            elif i <= 31:
                ax[3,i-24].plot(time_axis, data_of_this_channel); ax[3,i-24].set_title(channel_name)
            elif i <= 39:
                ax[4,i-32].plot(time_axis, data_of_this_channel); ax[4,i-32].set_title(channel_name)
        plt.show()

    def filter_and_plot(self, channel_name, low_cut=0.5, high_cut=40.0):
        """
        Filters the signal and plots the result for a single channel.
        """
        self.apply_bandpass_filter(channel_name, low_cut, high_cut)

        number_of_this_channel = self.labels.index(channel_name)
        ns_of_this_channel = self.ns[number_of_this_channel]
        ddr_of_this_channel = self.ddr_int

        self.frequency_of_channel = ns_of_this_channel / ddr_of_this_channel
        data_of_this_channel = self.filtered_data
        number_of_points = len(data_of_this_channel)
        total_time = number_of_points / self.frequency_of_channel
        time_axis = np.linspace(0, total_time, number_of_points, endpoint=True)

        plt.plot(time_axis, data_of_this_channel)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Voltage)")
        plt.title(f"Filtered Channel: {channel_name}")
        plt.show()
            
    def filter_and_plot_all(self, low_cut=0.5, high_cut=40.0):
        """
        Applies filtering to all channels and visualizes them in a grid.
        """
        self.apply_bandpass_filter("all", low_cut, high_cut)
        fig, ax = plt.subplots(5, 8)

        for i in range(len(self.labels)):
            channel_name = self.labels[i]
            number_of_this_channel = self.labels.index(channel_name)
            ns_of_this_channel = self.ns[number_of_this_channel]
            ddr_of_this_channel = self.ddr_int
            self.frequency_of_channel = ns_of_this_channel / ddr_of_this_channel
            data_of_this_channel = self.filtered_datas_dictionary[channel_name]
            number_of_points = len(data_of_this_channel)
            total_time = number_of_points / self.frequency_of_channel
            time_axis = np.linspace(0, total_time, number_of_points, endpoint=True)

            if i <= 7:
                ax[0,i].plot(time_axis, data_of_this_channel); ax[0,i].set_title(channel_name)
            elif i <= 15:
                ax[1,i-8].plot(time_axis, data_of_this_channel); ax[1,i-8].set_title(channel_name)
            elif i <= 23:
                ax[2,i-16].plot(time_axis, data_of_this_channel); ax[2,i-16].set_title(channel_name)
            elif i <= 31:
                ax[3,i-24].plot(time_axis, data_of_this_channel); ax[3,i-24].set_title(channel_name)
            elif i <= 39:
                ax[4,i-32].plot(time_axis, data_of_this_channel); ax[4,i-32].set_title(channel_name)
        plt.show()

    def fourier_transform(self, channel_name):
        """
        Computes and plots the Fast Fourier Transform (FFT) of a raw channel signal 
         to visualize the frequency spectrum.
        """
        data_of_this_channel = self.physical_signals[channel_name]
        number_of_this_channel = self.labels.index(channel_name)
        n = data_of_this_channel.size
        data_of_this_channel = data_of_this_channel - np.mean(data_of_this_channel)

        fourier = np.fft.fft(data_of_this_channel)
        temizlenmis_fourier = np.abs(fourier)[:n//2]
        normalled_fourier= temizlenmis_fourier / n
        fourier_real_amplitudes =  normalled_fourier * 2

        sampling_rate = self.ns[number_of_this_channel] / self.ddr_int
        timestep = 1/sampling_rate
        x_axis = np.fft.fftfreq(n, d=timestep)[:n//2]

        plt.plot(x_axis, fourier_real_amplitudes)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title(f"FFT: {channel_name}")
        plt.show()
    
    def filtered_fourier_transform(self, channel_name):
        """
        Applies a bandpass filter first, then computes and plots the FFT of the signal.
        """
        self.apply_bandpass_filter(channel_name, low_cut=0.5, high_cut=40.0)
        data_of_this_channel = self.filtered_data
        number_of_this_channel = self.labels.index(channel_name)
        n = data_of_this_channel.size
        data_of_this_channel = data_of_this_channel - np.mean(data_of_this_channel)

        fourier = np.fft.fft(data_of_this_channel)
        temizlenmis_fourier = np.abs(fourier)[:n//2]
        normalled_fourier= temizlenmis_fourier / n
        fourier_real_amplitudes =  normalled_fourier * 2

        sampling_rate = self.ns[number_of_this_channel] / self.ddr_int
        timestep = 1/sampling_rate
        x_axis = np.fft.fftfreq(n, d=timestep)[:n//2]

        plt.plot(x_axis, fourier_real_amplitudes)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title(f"Filtered FFT: {channel_name}")
        plt.show()

if __name__ == "__main__":
    # Example usage:
    # read_edf_data = RawData("sample/your_file.edf")
    # ... (other methods)
    pass