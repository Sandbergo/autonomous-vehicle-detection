import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import pathlib

"""
Usage
python <plot_title> <y_label> <file_name.csv>
"""

def create_plot(raw_data, name, y_label):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)

    # Save plots and show them
    # plt.figure(figsize=(20, 8))
    plt.title(name)
    plt.plot(raw_data['Step'], raw_data['Value'])
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

