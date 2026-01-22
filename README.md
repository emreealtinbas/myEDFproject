# 🧠 pyEDFproject: EEG Data Processing Library

**pyEDFproject** is a pure Python-based library for reading, processing, and analyzing **European Data Format (EDF)** files, developed to enhance my skills in Biomedical Engineering and Neuroscience.

Instead of using ready-made "black box" solutions, this library implements signal processing workflows (Parsing, Filtering, FFT) **from scratch**.

---

## 🚀 Features

* **EDF Parsing:** Reads header information (Patient ID, recording time, number of channels) and technical parameters.
* **Signal Conversion:** Reads raw binary data and converts digital values to physical units (uV / Voltage).
* **Advanced Filtering:**
    * **Butterworth Band-Pass Filter:** (Default: 0.5 - 40.0 Hz)
    * **Zero-Phase Filtering:** Uses the `filtfilt` method to clean signals without phase shift (delay).
* **Frequency Analysis:** Extracts the frequency spectrum of the signal using Fast Fourier Transform (FFT).
* **Visualization:**
    * Single channel time-series plotting.
    * Batch plotting of all channels in a matrix (grid) layout.
    * FFT comparison of raw and filtered signals.

---

## 🛠️ Installation

1.  Clone the project to your computer:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/pyEDFproject.git](https://github.com/YOUR_USERNAME/pyEDFproject.git)
    cd pyEDFproject
    ```

2.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
    *(If the requirements file is missing: `pip install numpy matplotlib scipy`)*

---

## 💻 Usage

Example usage is available in `test.py`. To use it in your own project:

### 1. Initializing the Library and Processing Data

```python
import pyEDFproject

# Initialize the RawData class by specifying the file path
# (Replace the example path with your own file)
file_path = "sample/ma0844az_1-1+.edf"
data = pyEDFproject.RawData(file_path)

# Step 1: Read header information
data.parse_header()
print(f"Detected Channels: {data.labels}")

# Step 2: Read data and convert to Matrix
data.read_data()
data.convert_to_matrix()

# Step 3: Convert digital data to Physical (Voltage) values
data.convert_digital_to_physical()

## ℹ️ Notes

* **Example Data:** A sample EDF file is included in the `sample/` directory for testing purposes.
* **Data Source:** These test files were obtained from [Teunis van Beelen's EDF Test Files](https://www.teuniz.net/edf_bdf_testfiles/).
* **Educational Purpose:** This project was developed to understand biomedical signal processing algorithms (FFT, Filtering) from the ground up.