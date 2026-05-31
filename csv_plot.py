import csv
# import neuron
# from neuron import h
# from neuron.units import mV, ms
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date, datetime
import re
import pandas as pd

# # Load required NEURON libraries
# h.load_file("stdrun.hoc")
# h.load_file("import3d.hoc")

### add today's date to the folder name
today = date.today()
date_string = today.strftime("%m.%d.%y")

# Format the timestamp (YYYYMMDD_HHMMSS)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

### Get the absolute path of the current script
script_path = os.path.abspath(__file__)

### Extract the directory and filename from the script path
script_dir = os.path.dirname(script_path)        

def plot_neuron_results(list_of_folders):
    """ plot different voltages of the same apic dend in an upper graph along with different resulting soma voltages from csv file with different colors
    Arguments:
    list_of_folders (list): a list of strings of URLs of all folders included.
    """
    output_csv_figs_folder = "/home/mohamed/myprojects/migliore2024/phasecoding/Results/csv_results"
    ## 1. Get the number of csv files that will be plotted across all folders
    apic_label_0= re.compile(r"Stim_Ran")
    apic_label_1= re.compile(r".csv")
    soma_label = re.compile(r"Soma_")

    all_csv_files = []
    all_apic_csv_files = []
    all_soma_csv_files = []

    ### 2. Add an error detection if an apic/soma file does not have a matching soma/apic csv 
    for current_folder in list_of_folders:
        if not os.path.exists(current_folder):
            continue
        for current_file in os.listdir(current_folder):
            if current_file.endswith('.csv') and apic_label_0.search(current_file) or  soma_label.search(current_file) and not current_file.startswith('.~lock.') and not current_file.startswith('~$'):
                # Store full path for loading and file name for the legend
                all_csv_files.append({
                    'full_path': os.path.join(current_folder, current_file),
                    'name': current_file
                })
            if current_file.endswith('.csv') and apic_label_0.search(current_file) and not current_file.startswith('.~lock.') and not current_file.startswith('~$'):
                all_apic_csv_files.append({
                    'full_path': os.path.join(current_folder, current_file),
                    'name': current_file
                })

            if current_file.endswith('.csv') and soma_label.search(current_file) and not current_file.startswith('.~lock.') and not current_file.startswith('~$'):
                all_soma_csv_files.append({
                    'full_path': os.path.join(current_folder, current_file),
                    'name': current_file
                })
            
    total_files = len(all_apic_csv_files)
    if total_files==0:
        print("No CSV files in the folders selected.")
        return

    # 3. Grab a matplotlib color library and sample exactly 'num_files' distinct colors
    cm = plt.get_cmap('turbo')
    color_library = [cm(i) for i in np.linspace(0, 1, total_files)]

    # 4. Pair csv files with colors
    apic_files_with_colors= zip(all_apic_csv_files, color_library)
    soma_files_with_colors= zip(all_soma_csv_files, color_library)

    # 5. Create a single shared figure for all pairs
    fig, (ax_above, ax_below) = plt.subplots(2, 1, figsize=(12, 8))

    # Initialize variables to track global mins/maxs for time >= 550 ms
    above_min, above_max = float('inf'), float('-inf')
    below_min, below_max = float('inf'), float('-inf')
    
    # 6. Loop through zipped pairs and plot onto the shared axes
    for idx, ((apic_file, apic_color), (soma_file, soma_color)) in enumerate(zip(apic_files_with_colors, soma_files_with_colors)):
        # Get data from apic file
        apic_short_label_pattern = re.compile(r'apic\[\d+\](?:_\d+(?:\.\d+)?HCNChDensity)?')
        apic_match = apic_short_label_pattern.search(apic_file['name'])
        apic_short_name = apic_match.group(0)
        apic_df = pd.read_csv(apic_file['full_path'])
        ax_above.plot(apic_df.iloc[:, 0], apic_df.iloc[:, 1], color=apic_color, label=apic_short_name)

        # Track min/max for apic only for Time >= 550
        apic_visible = apic_df[apic_df.iloc[:, 0] >= 550]
        if not apic_visible.empty:
            above_min = min(above_min, apic_visible.iloc[:, 1].min())
            above_max = max(above_max, apic_visible.iloc[:, 1].max())

        # Get data from soma file
        soma_match = apic_short_label_pattern.search(soma_file['name'])
        soma_short_name = soma_match.group(0)
        soma_df = pd.read_csv(soma_file['full_path'])
        ax_below.plot(soma_df.iloc[:, 0], soma_df.iloc[:, 1], color=soma_color, label=f"Soma_{soma_short_name}")
        
        # Track min/max for soma only for Time >= 550
        soma_visible = soma_df[soma_df.iloc[:, 0] >= 550]
        if not soma_visible.empty:
            below_min = min(below_min, soma_visible.iloc[:, 1].min())
            below_max = max(below_max, soma_visible.iloc[:, 1].max())

    # Configure styling, limits, labels, and legends after all datasets are plotted
    ax_above.set_xlabel("Time (ms)")
    ax_above.set_xlim(left=550, right=710) # start recording from 550 ms because no activity before that
    ax_above.set_ylabel("Membrane Potential (mV)")
    #ax_above.autoscale(enable=True, axis='y', tight=True) 
    all_voltages_ax_above = apic_df.iloc[:, 1]
    # ax_above_min_y = np.min(all_voltages_ax_above)
    # ax_above_max_y = np.max(all_voltages_ax_above)
    padding = 0.5
    # ax_above.set_ylim(bottom=ax_above_min_y - padding, top=ax_above_max_y + padding)
    ax_above.set_ylim(bottom=above_min - padding, top=above_max + padding)
    # Define the step size (interval) for the numbers on the axis
    step_size = 0.5  # Put a label/line every 0.5 mV
    # Create a list of tick locations from bottom to top counting by step_size
    y_ticks = np.arange(np.floor(above_min), np.ceil(above_max) + step_size, step_size)
    # Apply the steps to the axis
    ax_above.set_yticks(y_ticks)
    ax_above.legend(loc='upper right')
    ax_above.set_title("Apical Plot")
    ax_below.set_xlabel("Time (ms)")
    ax_below.set_xlim(left=550, right=710) # start recording from 550 ms because no activity before that
    ax_below.set_ylabel("Membrane Potential (mV)")
    #ax_below.autoscale(enable=True, axis='y', tight=True) 
    # all_voltages_ax_below = soma_df.iloc[:, 1]
    # ax_below_min_y = np.min(all_voltages_ax_below)
    # ax_below_max_y = np.max(all_voltages_ax_below)
    # padding = 1
    # ax_below.set_ylim(bottom=ax_below_min_y - padding, top=ax_below_max_y + padding)
    ax_below.set_ylim(bottom=below_min - padding, top=below_max + padding)
    # Define the step size (interval) for the numbers on the axis
    step_size = 0.5  # Put a label/line every 0.5 mV
    # Create a list of tick locations from bottom to top counting by step_size
    y_ticks = np.arange(np.floor(below_min), np.ceil(below_max) + step_size, step_size)
    # Apply the steps to the axis
    ax_below.set_yticks(y_ticks)
    ax_below.legend(loc='upper right')
    ax_below.set_title("Soma Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(output_csv_figs_folder, f'combined_plots_{apic_short_name}_{timestamp}.png'), bbox_inches='tight')






