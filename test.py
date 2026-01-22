import pyEDFproject


veri = pyEDFproject.RawData("sample/ma0844az_1-1+.edf")


veri.parse_header()
print(veri.labels)
veri.read_data()
veri.convert_to_matrix()
veri.convert_digitol_to_physical()
veri.plot_all_channels()
veri.filter_and_plot_all()
veri.fourier_transform("EEG FP2")



