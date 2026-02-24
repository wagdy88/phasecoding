"""
This code is to create one hippocampal cell, plot it, choose 10 apic dend of median distance and 10 of high distance from soma and synapses to them
"""
# Following meeting with Mike and Sherif 2/17/26
# Created by: Mohamed ElSayed, supervised by Mohamed Sherif 


import csv
from optparse import Values
import neuron
from neuron import h
from neuron.units import mV, ms
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date, datetime
import cellM1_ME
from cellM1_ME import CA1_PC_cAC_sig

# Load required NEURON libraries
h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")

### add today's date to the folder name
today = date.today()
date_string = today.strftime("%m.%d.%y")

# Format the timestamp (YYYYMMDD_HHMMSS)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

### Get the absolute path of the current script
script_path = os.path.abspath(__file__)

### Extract the directory and filename from the script path
script_dir = os.path.dirname(script_path)        

# Create a new folder with the date and timestamp in the name
folder_name = f"ten_median_distant_dend_{timestamp}"

n = CA1_PC_cAC_sig(0)