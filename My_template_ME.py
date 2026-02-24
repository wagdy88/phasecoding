"""This code Migliore 2024 Two reciprocally connected CA1 pyramidal cell model. This is the DeepSeek created and author's verified version of My_template.hoc"""
## Created by: Mohamed ElSayed, supervised by: Mohamed Sherif, and in collaboration with Michael Hasselmo

import neuron
from neuron import h, gui
from neuron.units import ms, mV, um
import random
import numpy as np
from cellM1_ME import CA1_PC_cAC_sig

# Load required files
h.load_file("nrngui.hoc")
#h.load_file("cell.hoc") ##### will change to another python file ####
h.cvode.active(1) # CVODE variable time step integrator for faster simulations.

# Initialize data structures similar to hoc file
## Global variables
N = 2  # Number of neurons
Nsyn = 20  # Number of excitatory synapses per neuron
NeuroCell = [] # Main Neuron objects
stim = []  # Stimulus objects
synD = [[None for _ in range(400)] for _ in range(2)] # Synapse objects (400 max per cell, 2 cells)
nsD = [[None for _ in range(400)] for _ in range(2)] # NetStim objects (400 max per cell, 2 cells)
ncD = [[None for _ in range(400)] for _ in range(2)] # NetCon connection objects
nt = [None, None]  # NetCon objects (2 cells)
syn = [None, None, None]  # syn[3] in original
fseq = None  # File sequence for random seed

DistSD = [[0.0 for _ in range(400)] for _ in range(2)] # Stores distances of synapses from soma
DistSDm = [0.0, 0.0]
ds = [0.0, 0.0]

# Initialize variables
aI = 200 # Current amplitude (200 pA)
ws = 5e-4 # Excitatory synaptic weight (0.0005 μS)
wi = 0 # Inhibitory synaptic weight (initially disabled)
ds0, ds1 = 50, 150 # Minimum distances for synapse placement on two cells
Dsd = 40 # Spread/distance range for synaptic placement
FreqInp = 100 # Input frequency (100 Hz)
chnoise = 0 # Noise flag (0=no noise, 1=noisy inputs)

# Create NeuroCell objects
## Instantiates 2 pyramidal cell models from the loaded template (CA1_PC_cAC_sig)
for j in range(N):
    NeuroCell.append(CA1_PC_cAC_sig())
    NeuroCell[j].init()

# List all sections
for sec in h.allsec():
    print(sec.name())

# Set temperature
h.celsius = 34

# GUI panel function - replaced with variable assignments
def update_variables():
    global ws, wi, ds0, ds1, FreqInp, chnoise
    # In Python, we'll use direct variable assignments instead of GUI
    # These would be connected to GUI widgets in a real implementation
    # In HOC, xpanel() creates interactive controls
    # In Python, this would need a proper GUI framework
    # global keyword allows modifying global variables from within function
    pass

# Set Distance Parameters
## Copies ds0 and ds1 into the ds array for cell-specific distances
ds[0] = ds0
ds[1] = ds1

# Read random seed from file
try:
    with open("seq.txt", "r") as fseq:
        xseed = float(fseq.readline().strip())
except:
    xseed = 12345  # Provides default seed (12345) as fallback

# Initialize random generator
## Initializes NEURON's specific random number generator with seed
# Generates first random number for verification
# mcell_ran4 is NEURON's own RNG function

h.mcell_ran4_init(xseed)
rrr = h.mcell_ran4(xseed)
print(rrr)

# Translate NeuroCell positions
## Moves second cell 400 μm in x-direction
##This separates the two cells spatially
for j in range(1, N):
    soma_sec = NeuroCell[j].soma[0]
    n3d = h.n3d() # n3d(): Returns number of 3D points defining the section
    for i in range(int(n3d)):
        x, y, z, diam = h.x3d(i), h.y3d(i), h.z3d(i), h.diam3d(i) # x3d(i), y3d(i), z3d(i), diam3d(i): Get coordinates and diameter
        h.pt3dchange(i, x + 400, y, z, diam) # pt3dchange(): Modifies 3D point coordinates

# Create Input Stimulus (NetStim) - the main stimulus method to the cell rather than IClamp
## NetStim: Creates a spike generator

ns = h.NetStim(0.5) ## Position 0.5 (middle of virtual section)
ns.number = 1000000 ## number: Very large number of spikes (essentially infinite)
ns.interval = 1000.0 / FreqInp  ## interval: 10 ms between spikes (100 Hz frequency)
ns.start = 10 ## start: First spike at 10 ms
ns.noise = chnoise ## noise: Adds randomness to intervals if enabled

# Create inhibitory synapses
## Creates exponential synapses on each soma
for m in range(N):
    soma_sec = NeuroCell[m].soma[0]
    syn[m] = h.ExpSyn(0.5, sec=soma_sec) # ExpSyn: Single exponential synapse model
    syn[m].tau = 30 # Decay time constant (30 ms)
    syn[m].e = -80 # Reversal potential (-80 mV, inhibitory)

# Create Recurrent (NetCon) connections
## Creates reciprocal inhibitory connections:
### Cell 0 → Cell 1
### Cell 1 → Cell 0
for m in range(N):
    k = m + 1
    if m == (N - 1):
        k = 0
    soma_sec = NeuroCell[m].soma[0]
    nt[m] = h.NetCon(soma_sec(0.5)._ref_v, syn[k], 0, 0, wi) 
    # NetCon: Connects a source to a target, Monitors voltage at soma(0.5)
    # Triggers when voltage crosses 0 mV (threshold)
    # Weight wi determines connection strength
    # ._ref_v: Reference to voltage variable for monitoring
    
def wca():
    """Update connection weights and parameters"""
    ## Updates all connection weights dynamically
    ## Modifies both excitatory (ws) and inhibitory (wi) weights
    ## Updates noise and frequency parameters
    ## Called when GUI controls change values
    ## Note: weight[0] not weight - Python NEURON uses array
    for j in range(N):
        soma_sec = NeuroCell[j].soma[0]
        for ll in range(Nsyn):
            if ncD[j][ll] is not None:
                ncD[j][ll].weight[0] = ws
        if nt[j] is not None:
            nt[j].weight[0] = wi
    
    ds[0] = ds0
    ds[1] = ds1
    ns.noise = chnoise
    ns.interval = 1000.0 / FreqInp
    
def CreateShuffleSyn():
    """Create shuffled synapses"""
    ## CreateShuffleSyn() Function: Distributed Synapses
    ## Main function for placing synapses randomly on dendrites
    ## global: Modifies global variables xseed and Dsd
    ## Ensures minimum distance spread of 50 μm
    
    global xseed, Dsd
    
    if Dsd == 0:
        Dsd = 50  # max 400
    
    for m in range(N):
        # Access soma to set distance()
        ## h.distance(): Sets reference point for distance measurements
        ## Reference is soma(0.5) - distances measured from here
        ## dm: Accumulator for average distance calculation
        h.distance(0, 0.5, sec=NeuroCell[m].soma[0])
        dm = 0
        
        for j in range(Nsyn):
            ## Random location search loop:
            ## Selects random apical dendrite (indAD: 0-188)
            ## Selects random position along it (rrr_val: 0-1)
            ## Checks if distance from soma is within allowed range
            ## Repeats until valid location found
            ## Ensures synapses are placed within specific distance band
            
            k = 0
            indAD = 0
            rrr_val = 0
            
            while k == 0:
                indAD = int(189 * h.mcell_ran4(xseed))
                rrr_val = h.mcell_ran4(xseed)
                
                # Access apical dendrite
                apic_sec = NeuroCell[m].apic[indAD]
                dist = h.distance(rrr_val, sec=apic_sec)
                
                if ds[m] < dist < (ds[m] + Dsd):
                    k = 1
            
            # Create synapse at random location
            # Creates double-exponential synapse at selected location
            apic_sec = NeuroCell[m].apic[indAD]
            synD[m][j] = h.Exp2Syn(rrr_val, sec=apic_sec) # Exp2Syn: Dual time constant synapse (rise and decay)
            synD[m][j].tau1 = 0.5 # Rise time constant (fast activation)
            synD[m][j].tau2 = 3 # Decay time constant (slower deactivation)
            synD[m][j].e = 0 # Excitatory synapse (0 mV reversal potential)
            ncD[m][j] = h.NetCon(ns, synD[m][j], 0, 0, ws) # Connects to ns (NetStim) with weight ws
            
            # Records distance of each synapse from soma
            # Calculates average synapse distance per cell
            dist_loc = h.distance(rrr_val, sec=apic_sec)
            DistSD[m][j] = dist_loc
            dm += dist_loc
            
            print(m, j, indAD, synD[m][j].get_loc(), rrr_val)
        
        DistSDm[m] = dm / Nsyn
    
    # Prints summary statistics of synapse distribution
    print(f"      Synapsis shuffling - dSm[0] = {DistSDm[0]} - dSm[1] = {DistSDm[1]}")
    print(f"                   Frequency (Hz) = {1000/ns.interval} - ISI (ms) = {ns.interval}")
    print(f"                               ws = {ws} - wi = {wi}")

# Create initial shuffled synapses
# Creates initial set of shuffled synapses
CreateShuffleSyn()

# Load session file
# Loads previously saved NEURON session
h.load_file("Ses.ses")

