import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import pathlib

"""
Usage
python <plot_title> <y_label> <file_name.csv>
"""

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order, T, title):
    ### Plotting the filter used
    # plot_butter_filter(order,fs,cutoff)

    ### Raw vs. Filtered Plotting
    x_axis = create_raw_axis(data, T)
    fig, axs = plt.subplots(2,1, constrained_layout=True)
    # FFT stem data plots
    axs[0].plot(x_axis, data)
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Channel intensity')

    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)

    # Plotting filtered data
    axs[1].plot(x_axis, y)
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Channel intensity')
    axs[1].legend(['$\omega_c = $' + str(cutoff)])

    file_name = sys.argv[1].split('.')
    channel = "_".join(title.split())
    plt.savefig(get_google_path()+ file_name[0] + '_' + channel + '_filter_plots.png')
    plt.show()
    return y

def create_plot(raw_data, name, y_label):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)

    # Save plots and show them
    # plt.figure(figsize=(20, 8))
    plt.title(name)
    step_raw = np.asarray(raw_data['Step'])
    value_raw = np.asarray(raw_data['Value'])
    print("step raw length: ",len(step_raw))
    step = step_raw[:50000]
    print("step length: ",len(step))
    value = value_raw[:50000]
    value = butter_lowpass_filter(value, 10000, 1, 1, 1, "bla")
    plt.plot(step, value)
    plt.xlabel('Step [n]')
    plt.ylabel(y_label)

    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()



if __name__ == "__main__":
    assert (len(sys.argv) > 3), 'Have to pass a files and y_label as the variable'

    #Load the data:
    raw_data = pd.read_csv(sys.argv[3])
    raw_data['Value']
    #Plot
    create_plot(raw_data, sys.argv[1], sys.argv[2])

