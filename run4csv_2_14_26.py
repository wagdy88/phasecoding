import csv
from optparse import Values
import neuron
from neuron import h
from neuron.units import mV, ms
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date, datetime

# Load required NEURON libraries
h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")

# ### add today's date to the folder name
# today = date.today()
# date_string = today.strftime("%m.%d.%y")

# # Format the timestamp (YYYYMMDD_HHMMSS)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# ### Get the absolute path of the current script
# script_path = os.path.abspath(__file__)

# ### Extract the directory and filename from the script path
# script_dir = os.path.dirname(script_path)        

with open('/home/mohamed/myprojects/migliore2024/phasecoding/Apic0_Voltage_After_NetStim_Ran_On_20260214_1150/Soma_Voltage_After_NetStim_Ran_On_20260214_1150.csv', 'r') as f:
    reader = csv.reader(f)
    soma_apical0_data = list(reader)
    
t_soma_apical0_array = [float(row[0]) for row in soma_apical0_data]
v_soma_apical0_array = [float(row[1]) for row in soma_apical0_data]


with open('/home/mohamed/myprojects/migliore2024/phasecoding/Apic21_Voltage_After_NetStim_Ran_On_20260214_1150/Soma_Voltage_After_NetStim_Ran_On_20260214_1150.csv', 'r') as f:
    reader = csv.reader(f)
    soma_apical21_data = list(reader)
    
t_soma_apical21_array = [float(row[0]) for row in soma_apical21_data]
v_soma_apical21_array = [float(row[1]) for row in soma_apical21_data]


with open('/home/mohamed/myprojects/migliore2024/phasecoding/Apic152_Voltage_After_NetStim_Ran_On_20260214_1150/Soma_Voltage_After_NetStim_Ran_On_20260214_1150.csv', 'r') as f:
    reader = csv.reader(f)
    soma_apical152_data = list(reader)
    
t_soma_apical152_array = [float(row[0]) for row in soma_apical152_data]
v_soma_apical152_array = [float(row[1]) for row in soma_apical152_data]


with open('/home/mohamed/myprojects/migliore2024/phasecoding/Soma_Voltage_After_NetStim_Ran_On_20260214_1150/Soma_Voltage_After_NetStim_Ran_On_20260214_1150.csv', 'r') as f:
    reader = csv.reader(f)
    soma_data = list(reader)
    
t_soma_array = [float(row[0]) for row in soma_data]
v_soma_array = [float(row[1]) for row in soma_data]

# Plot results
fig, axes = plt.subplots(1, 1, figsize=(15, 10))
                    
## Recording from Apical dendrite
#plt.figure()
axes.plot(t_soma_apical0_array, v_soma_apical0_array, color='red')
axes.plot(t_soma_apical21_array, v_soma_apical21_array, color='blue')
axes.plot(t_soma_apical152_array, v_soma_apical152_array, color='green')
axes.plot(t_soma_array, v_soma_array, color='black')
axes.set_xlabel("Time (ms)")
axes.set_ylabel("Membrane Potential (mV)")
axes.set_title("Soma Membrane Potential")
axes.legend(["Soma w/ Apical Dendrite 0 Stim", "Soma w/ Apical Dendrite 21 Stim", "Soma w/ Apical Dendrite 152 Stim", "Soma"])
plt.show()

# Save results
# Save results
# folder_name = f'Soma2Voltages_Voltage_Ran_On_{timestamp}/'
# output_folder_path_apic = os.path.join(script_dir, folder_name)

# # Create the folder if it doesn't already exist
# try:
#     os.makedirs(output_folder_path_apic, exist_ok=True)
#     print(f"Folder '{folder_name}' created successfully at: {output_folder_path_apic}")
# except OSError as e:
#     print(f"Error creating folder: {e}")


#     plt.savefig(os.path.join(output_folder_path_apic, f'Soma2Voltages_Ran_On_{timestamp}.png'), dpi=300)

