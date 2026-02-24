"""This code is DeepSeek created and author's verified version of Synch_DELAY_M1.hoc"""
# Created by: Mohamed ElSayed, supervised by: Mohamed Sherif, and in collaboration with Michael Hasselmo

### This is a synchronization study between two CA1 pyramidal neurons that receive:
## Common excitatory inputs (shared NetStim → dendritic synapses)
## Mutual inhibition (soma → soma connections)
## Distance-dependent synapse placement (shuffled locations)

#### Experimental Setup ####
# 2 neurons with identical morphology (second neuron shifted by 400 μm)
# Nexp experiments with different synapse shufflings
# Multiple output files for spikes, voltages, correlations, phases

#### Input Structure ####
# 1. One Common Input (NetStim, Freq= FreqInp) -> Dendritic synapses (Nsyn per neuron, shuffled locations) [Excitatory: ws weight, Exp2Syn kinetics]

# 2. Mutual inhibition (soma → soma connections) [Inhibitory: wi, delay=del]

#### Synchronization Measures ####


from numbers import Real
import neuron
from neuron import h, gui
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import signal
import os
import cellM1_ME

class TwoNeuronSynchronization:
    """Study synchronization between two CA1 pyramidal neurons"""
    
    def __init__(self):
        # Load required NEURON libraries
        h.load_file("nrngui.hoc")
        h.load_file("correl.hoc") ## not provided in the github page?
        h.cvode_active(1) # Variable time step integration (faster for stiff equations)
        
        # Set physiological temperature
        h.celsius = 34
        
        # Initialize object references
        self.init_objects()
        
        # Initialize variables
        self.init_variables()
        
        # Read input parameters from file
        self.read_input_parameters()
        
        # Create neurons
        self.create_neurons()
        
        # Setup GUI
        self.setup_gui()
        
        # Load session
        h.load_file("Ses_t1000.ses")
    
    #### Three-Tier Object Structure ####
    ## Tier 1: Neuron and stimulation objects (Python object references)
    ## Tier 2: NEURON SectionLists (Managed internally by cell objects)
    ## Tier 3: HOC object references (Created and managed by HOC code)
    
    def init_objects(self):
        """Initialize all object references"""
        # Neuron and stimulation objects
        self.NC = []  # list of Neuron objects/cells
        self.stim = [None, None]
        self.syn = [None, None]  # Somatic inhibitory synapses
        self.apc = [None, None]  # Action potential counters
        
        # Synaptic connections
        self.synD = [[None for _ in range(400)] for _ in range(2)]  # Dendritic synapses
        self.nt = [[None for _ in range(2)] for _ in range(2)]  # Inter-neuron connections
        self.nsD = [[None for _ in range(400)] for _ in range(2)]
        self.ncD = [[None for _ in range(400)] for _ in range(2)]
        self.ns = None  # NetStim (common input)
        self.ncS = [None, None]  # Current clamps (if used)
        self.randDen = None
        
        # File objects
        self.fseq = None
        self.fv = None
        self.finp = None
        self.fout = None
        self.fphi = None
        self.fspk = None
        self.fcorr = None
        self.fc = None
        
        # GUI objects
        self.b = None
        self.c = None
        self.d = None
        self.e = None
        self.b2 = None
        self.c2 = None
        self.d2 = None
        self.e2 = None
        
        # Cross-correlation objects
        self.correl = None
        self.C1 = None
        self.C2 = None
        self.tC1 = None
        self.tC2 = None
    
    def init_variables(self):
        """Initialize all variables"""
        # Distance arrays
        self.vecAD = [[0.0 for _ in range(400)] for _ in range(2)]
        self.DistSD = [[0.0 for _ in range(400)] for _ in range(2)]
        self.DistSDm = [0.0, 0.0]
        self.dSDm = [0.0, 0.0]
        self.ds = [0.0, 0.0]
        
        # Spike data
        self.tspk = [[0.0 for _ in range(1000)] for _ in range(2)] # Spike time storage in a structure of [0.0, 0.0, .... 1000 total], first index is Neuron 0 and second index is Neuron 1
        
        self.nspk = [0, 0] # Spike count for each neuron 
        # nspk[0]: How many spikes neuron 0 has fired in current experiment
        # nspk[1]: How many spikes neuron 1 has fired in current experiment
        # Starts at 0, increments with each spike
        # Also serves as index for tspk array  
         
        self.nspkm = [0, 0] # Accumulated Spike Counts
        # nspkm[0]: Total spikes from neuron 0 across all experiments
        # nspkm[1]: Total spikes from neuron 1 across all experiments
        # Used for averages: nspkm[0]/Nexp = mean spikes per experiment
        
        self.f = [0.0, 0.0]  # FiringFrequencies, Calculated as: (spike_count / simulation_time) × 1000
        
        self.Esti = [0.0 for _ in range(1000)]  # External/Extra spike times
        ## Poisson background inputs: Random extra spikes
        ## Experimentally recorded spike times: Playback of real data
        ## Control stimuli: Precisely timed test pulses
        
        ### Difference between tspk and Esti
        # tspk: Recorded spikes (output of the neurons)
        # Esti: Input spikes (extra stimulation to the neurons)
        
        self.Esi = [0.0 for _ in range(1000)]  # External spike indices/intervals
        
        # Correlation data
        self.corr = [0.0 for _ in range(10000000)]
        self.corrvi = [0.0 for _ in range(10000000)]
        self.tt = [0.0 for _ in range(10000000)]
        
        # Phase calculation
        self.tspkprec = [0.0, 0.0]
        self.tspknext = [0.0, 0.0]
        self.i_spk = [0, 0]
        self.phi = [0.0, 0.0]
        
        # Correlation calculation variables
        self.sum1 = 0.0
        self.sum2 = 0.0
        self.sum12 = 0.0
        self.sumI = 0.0
        self.sumVI = 0.0
        self.cc = 0.0
        self.ccvi = 0.0
        self.Corr_me = 0.0
        self.CorrVI_me = 0.0
        self.corr_med = 0.0
        self.corrvi_med = 0.0
    
    def read_input_parameters(self):
        """Read parameters from input file"""
        try:
            with open("Input.inp", "r") as self.finp:
                # Nexp: Number of experiments
                self.Nexp = float(self.finp.readline().strip())
                # wi: Inhibitory weight (μS)
                self.wi = float(self.finp.readline().strip())
                # ws: Excitatory weight (μS)
                self.ws = float(self.finp.readline().strip())
                # Nsyn: Synapses per neuron
                self.Nsyn = int(self.finp.readline().strip())
                # ds0: Min distance for neuron 0 (μm)
                self.ds0 = float(self.finp.readline().strip())
                # ds1: Min distance for neuron 1 (μm)
                self.ds1 = float(self.finp.readline().strip())
                # Dsd: Distance spread (μm)
                self.Dsd = float(self.finp.readline().strip())
                # chnoise: Noise on/off
                self.chnoise = float(self.finp.readline().strip())
                # FreqInp: Input frequency (Hz)
                self.FreqInp = float(self.finp.readline().strip())
                # del_val: Inhibition Delay value (ms)
                self.del_val = float(self.finp.readline().strip())  # del is reserved in Python
            
            print(f"Nsyn={self.Nsyn}, wi={self.wi}, ws={self.ws}, ds0={self.ds0}, "
                  f"ds1={self.ds1}, Dsd={self.Dsd}, Noise={self.chnoise}, "
                  f"f={self.FreqInp}, Del={self.del_val}")
        except Exception as e:
            print(f"Error reading Input.inp: {e}")
            # Default values
            self.Nexp = 10
            self.wi = 0.0
            self.ws = 5e-4
            self.Nsyn = 20
            self.ds0 = 50
            self.ds1 = 150
            self.Dsd = 40
            self.chnoise = 0
            self.FreqInp = 100
            self.del_val = 0.0
        
        # Other parameters
        self.SynLoc = 0.5  # Synapse location along dendrite
        self.tstop = 1000  # Simulation duration
        h.cvode.event(self.tstop)
        
        # Read random seed
        try:
            with open("seq.txt", "r") as self.fseq:
                self.xseed = float(self.fseq.readline().strip())
            print(f"Random seed: {self.xseed}")
        except:
            self.xseed = 12345
            print(f"Using default random seed: {self.xseed}")
    
    def create_neurons(self):
        """Create and initialize the two neurons"""
        self.N = 2  # Number of neurons
        
        # Create neurons
        for j in range(self.N):
            self.NC.append(h.CA1_PC_cAC_sig())
            self.NC[j].init()
        
        # Number of apical dendrites (M1 morphology)
        self.Napic = 189
        
        # Translate second neuron position
        for j in range(1, self.N):
            soma_sec = self.NC[j].soma[0]
            n3d = h.n3d()
            for i in range(int(n3d)):
                x, y, z, diam = h.x3d(i), h.y3d(i), h.z3d(i), h.diam3d(i)
                h.pt3dchange(i, x + 400, y, z, diam)
        
        ### Create common input (NetStim)
        ## Inputs could be from entorhinal cortex or CA3 inputs to CA1
        self.ns = h.NetStim(0.5)
        self.ns.number = 1000000  # Essentially infinite
        self.ns.interval = 1000.0 / self.FreqInp  # Convert Hz to ms interval
        self.ns.start = 10 # start time of the stimulus at 10 ms
        self.ns.noise = self.chnoise # Randomize intervals if chnoise =1 
        
        # Create somatic inhibitory synapses and spike counters
        ## GABA-A like
        for m in range(self.N):
            soma_sec = self.NC[m].soma[0]
            
            # Inhibitory synapse (GABA-A like)
            self.syn[m] = h.ExpSyn(0.5, sec=soma_sec)
            self.syn[m].tau = 30 # Fast inhibition (GABA_A ~ 10-30ms)
            self.syn[m].e = -80 # Cl- reversal potential
            
            # Spike counter
            self.nspk[m] = 0
            self.nspkm[m] = 0
            
            # Action potential counter
            self.apc[m] = h.APCount(0.5, sec=soma_sec)
            self.apc[m].thresh = 0
        
        # Create inter-neuron connections (mutual inhibition)
        for m in range(self.N):
            for k in range(self.N):
                soma_sec = self.NC[m].soma[0]
                self.nt[m][k] = h.NetCon(soma_sec(0.5)._ref_v, 
                                        self.syn[k], 0, self.del_val, self.wi)
        
        # Initialize random number generator
        h.mcell_ran4_init(self.xseed)
        rrr = h.mcell_ran4(self.xseed)
        print(f"First random number: {rrr}")
        
        # Set distance parameters
        self.ds[0] = self.ds0
        self.ds[1] = self.ds1
        
        # Initialize spike arrays
        for l in range(self.N):
            self.tspkprec[l] = 0
            self.tspknext[l] = 0
            self.i_spk[l] = 0
            
            # Clear spike times
            l2 = 1
            while l2 < len(self.tspk[l]) and self.tspk[l][l2] > 0:
                self.tspk[l][l2] = 0
                l2 += 1
        
        # Initialize Esti and tspk arrays
        for i in range(1000):
            self.Esti[i] = 0.0
            for j in range(self.N):
                if i < len(self.tspk[j]):
                    self.tspk[j][i] = 0.0
        
        # Initialize distance arrays
        for i in range(400):
            for j in range(self.N):
                self.vecAD[j][i] = 0
                self.DistSD[j][i] = 0
        
        # Global counters
        self.kcommon = 0
        self.mincommon = 0
        self.NspikesTot = 0
        self.deltats = 0.0
    
    def setup_gui(self):
        """Setup GUI for visualization and control"""
        # Main plotting window
        self.b = h.VBox()
        self.b.intercept(1)
        
        # Graph 1: Correlation between neurons
        self.c = h.Graph()
        self.c.size(0, 2000, -1, 1)
        self.c.addvar("Correlation V1-V2")
        self.c.label(0.43, 0.93, "Corr V1-V2")
        self.c.label(0.49, 0.0, "t")
        self.c.color(2)
        
        # Graph 2: Synchronization measure S(t)
        self.d = h.Graph()
        self.d.size(0, 2000, 0, 1)
        self.d.addvar("Synchronization S(t)")
        self.d.label(0.43, 0.93, "S(t)")
        self.d.label(0.49, 0.0, "t")
        self.d.color(3)
        
        # Graph 3: Spike counts
        self.e = h.Graph()
        self.e.size(0, 2000, 0, 50)
        self.e.addvar("Spike counts")
        self.e.label(0.45, 0.93, "N_spikes")
        self.e.label(0.49, 0.0, "t")
        self.e.color(5)
        
        # Control panel
        h.xpanel("Controls")
        h.xbutton("Shuffle syn. & reRun", self.runm)
        h.xpanel()
        
        # Parameter controls
        h.xpanel("Variables")
        h.xvalue("Inter-soma Synaptic Inhibition - wi", "wi", 0, self.wca)
        h.xvalue("Dendr Net Synaptic Conductance - ws", "ws", 0, self.wca)
        h.xpanel()
        
        self.b.intercept(0)
        self.b.map()
        
        # Second window for averages
        self.b2 = h.VBox()
        self.b2.intercept(1)
        
        # Average correlation
        self.c2 = h.Graph()
        self.c2.size(0, 1, -1, 1)
        self.c2.addvar("Average Correlation")
        self.c2.label(0.4, 0.93, "<Corr V1-V2>")
        self.c2.label(0.49, 0.0, "Parameter")
        self.c2.color(2)
        
        # Average synchronization
        self.d2 = h.Graph()
        self.d2.size(0, 1, 0, 1)
        self.d2.addvar("Average Synchronization")
        self.d2.label(0.4, 0.93, "<S(t)>")
        self.d2.label(0.49, 0.0, "Parameter")
        self.d2.color(3)
        
        # Average frequency
        self.e2 = h.Graph()
        self.e2.size(0, 1, 0, 55)
        self.e2.addvar("Average Frequency")
        self.e2.label(0.42, 0.93, "<Freq.>")
        self.e2.label(0.49, 0.0, "Parameter")
        self.e2.color(5)
        
        self.b2.intercept(0)
        self.b2.map()
    
    def advance(self):
        """Custom advance procedure for spike detection and correlation
        This function gets called every time step (≈0.025ms)"""
        # Call standard fadvance
        h.fadvance()
        ## APCount object counts spikes crossing 0mV
        ## Spike times stored in tspk[neuron][spike_index]
        ## Real-time plotting of spike rasters
        
        # Check for new spikes in each neuron
        for l in range(self.N):
            if self.apc[l].n > self.nspk[l]:
                # New spike detected
                self.nspk[l] = self.apc[l].n
                
                # Store spike time
                if self.nspk[l] < len(self.tspk[l]):
                    self.tspk[l][self.nspk[l]] = h.t
                
                self.NspikesTot += 1
                
                # Record spike to file
                if l == 0:
                    self.fspk.printf(f"{self.Ne} {self.Nsyn} {self.ws} {self.wi}   "
                                    f"{self.nspk[0]} {h.t}   {self.nspk[1]} 0\n")
                else:
                    self.fspk.printf(f"{self.Ne} {self.Nsyn} {self.ws} {self.wi}   "
                                    f"{self.nspk[0]} 0   {self.nspk[1]} {h.t}\n")
                
                # Mark spike on graph
                if l == 0:
                    self.e.mark(h.t, self.nspk[l], "o", 8, 1, 1)
                else:
                    self.e.mark(h.t, self.nspk[l], "t", 8, 1, 1)
                self.e.flush()
        
        # Update correlation calculation
        self.istep += 1
        v0 = self.NC[0].soma[0].v
        v1 = self.NC[1].soma[0].v
        
        self.sum12 += v0 * v1 * h.dt
        self.sum1 += v0**2 * h.dt
        self.sum2 += v1**2 * h.dt
        
        # Calculate correlation over windows
        if self.istep == self.NstepsCorr: # NstepsCorr=50: Correlation window size, Updates plots in real-time
            denom = np.sqrt(abs(self.sum1 * self.sum2))
            if denom > 0:
                self.cc = self.sum12 / denom
            else:
                self.cc = 0
            
            # Store correlation
            if self.ic < len(self.corr):
                self.corr[self.ic] = self.cc
                self.tt[self.ic] = h.t
            
            # Plot correlation
            self.c.line(self.ic, self.cc)
            self.c.flush()
            
            # Update counters
            self.ic += 1
            self.istep = 0
            self.sum1 = 0
            self.sum2 = 0
            self.sum12 = 0
            self.Nwindowscorr += 1
            self.corr_med += self.cc
        
        # Record voltages (for debugging)
        if self.Ne == 1:
            indAD = getattr(self, 'indAD', 0)
            apic_v = self.NC[1].apic[indAD].v if hasattr(self.NC[1], 'apic') and indAD < len(self.NC[1].apic) else 0
            self.fv.printf(f"{h.t} {v0} {v1} {apic_v}\n")
    
    def wca(self):
        """Update connection weights"""
        for k in range(self.N):
            for m in range(self.N):
                if self.nt[k][m]:
                    self.nt[k][m].weight[0] = self.wi
            # Update stimulus amplitude if used
            # if self.stim[k]:
            #     self.stim[k].amp = self.sta
    
    def CreateShuffleSyn(self):
        """Create shuffled excitatory synapses on dendrites"""
        if self.Dsd == 0:
            self.Dsd = 50  # Maximum spread
        
        for m in range(self.N):
            # Set distance reference point (soma center)
            h.distance(0, 0.5, sec=self.NC[m].soma[0])
            dm = 0
            
            for j in range(self.Nsyn):
                k = 0
                indAD = 0
                rrr_val = 0
                
                # Find valid location
                while k == 0:
                    indAD = int(self.Napic * h.mcell_ran4(self.xseed))
                    rrr_val = h.mcell_ran4(self.xseed)
                    
                    # Access selected apical dendrite
                    if hasattr(self.NC[m], 'apic') and indAD < len(self.NC[m].apic):
                        apic_sec = self.NC[m].apic[indAD]
                        dist = h.distance(rrr_val, sec=apic_sec)
                        diam_at_loc = apic_sec.diam(rrr_val)
                        
                        # Check criteria: within distance range and thin enough
                        ## Distance range: ds[m] to ds[m] + Dsd from soma
                        ## Distance limit: 1.2 um (thin dendrites only)
                        ## Random apical dendrite selection: 0 to Napic-1
                        ## Random position 0-1 along branch
                        if (self.ds[m] < dist < (self.ds[m] + self.Dsd) and 
                            diam_at_loc < 1.2):
                            k = 1
                
                # Create synapse at valid location
                # Dendritic excitatory synapses (AMPA/NMDA-like)
                apic_sec = self.NC[m].apic[indAD]
                self.synD[m][j] = h.Exp2Syn(rrr_val, sec=apic_sec)
                self.synD[m][j].tau1 = 0.5  # Activation time constant - Fast rise (AMPA-like)
                self.synD[m][j].tau2 = 3    # Deactivation time constant - Slower decay
                self.synD[m][j].e = 0       # Excitatory (0 mV reversal) - Glutamate reversal
                
                # Connect to common input
                self.ncD[m][j] = h.NetCon(self.ns, self.synD[m][j], 0, 0, self.ws)
                
                # Record distance
                self.DistSD[m][j] = h.distance(rrr_val, sec=apic_sec)
                dm += self.DistSD[m][j]
                
                # Store which apical dendrite was used
                self.vecAD[m][j] = indAD
            
            # Calculate average distance
            self.DistSDm[m] = dm / self.Nsyn
        
        print(f"      Synapsis shuffling - dSm[0] = {self.DistSDm[0]}, "
              f"dSm[1] = {self.DistSDm[1]}")
        print(f"                   Frequency (Hz) = {1000/self.ns.interval}, "
              f"ISI (ms) = {self.ns.interval}")
        print(f"                               ws = {self.ws}, wi = {self.wi}, "
              f"delay = {self.del_val}")
    
    def runm(self):
        """Wrapper for runMain to use with GUI button"""
        self.runMain()
    
    def runMain(self):
        """Main simulation loop with multiple experiments"""
        # Create synapses
        self.CreateShuffleSyn()
        
        # Setup recording vectors
        self.C1 = h.Vector()
        self.tC1 = h.Vector()
        self.C2 = h.Vector()
        self.tC2 = h.Vector()
        
        # Record voltages
        self.NC[0].soma[0].push()
        h.cvode.record(self.NC[0].soma[0](0.5)._ref_v, self.C1, self.tC1)
        h.pop_section()
        
        self.NC[1].soma[0].push()
        h.cvode.record(self.NC[1].soma[0](0.5)._ref_v, self.C2, self.tC2)
        h.pop_section()
        
        # Initialize counters
        self.NspikesTotm = 0
        self.Corr_me = 0.0
        self.S_me = 0.0
        self.CS_me = 0.0
        self.S2_me = 0.0
        self.CS2_me = 0.0
        self.NstepsCorr = 50
        
        self.ic = 0
        self.cc = 0
        
        # Reset weights and spike counters
        # 2x2 Connectivity
        ### Features: 
        ##### All-to-all connectivity (both neurons inhibit each other)
        ##### Same weight (wi) and delay (del) for both directions
        ##### Voltage threshold = 0mV (spike detection)
        
        for k in range(self.N):
            for j in range(self.Nsyn):
                if self.ncD[k][j]:
                    self.ncD[k][j].weight[0] = self.ws
            
            for m in range(self.N):
                if self.nt[k][m]:
                    self.nt[k][m].weight[0] = self.wi
            
            self.apc[k].n = 0
            self.f[k] = 0.0
            self.nspkm[k] = 0
        
        # Setup output files
        fphin = f"Synchro_Phi_wi{self.wi}_dl{self.del_val}.out"
        self.fphi = open(fphin, 'w')
        self.fphi.write("# Ne,Nsyn,ws,wi,nspk[0],i_spk[0],tspkprec[0],tspknext[0],phi[0],"
                       "nspk[1],i_spk[1],tspkprec[1],tspknext[1],phi[1],t,s_med,cs_med\n")
        
        fvn = f"Synchro_V_wi{self.wi}_dl{self.del_val}.out"
        self.fv = open(fvn, 'w')
        self.fv.write("# t    NC[0].soma.v, NC[1].soma.v  NC[1].apic[indAD].v\n")
        
        if self.Ne == 1:
            self.fv.write(f"# Nsyn {self.Nsyn}  - d0 {self.DistSDm[0]}  - d1 {self.DistSDm[1]}  "
                         f"- ws {self.ws}  - wi {self.wi}  - dli {self.del_val}\n")
            
            fcn = f"Synchro_C_wi{self.wi}_dl{self.del_val}.out"
            self.fc = open(fcn, 'w')
            self.fc.write("# Sm     Cm      C0      Dphim\n")
            self.fc.write(f"# Nsyn {self.Nsyn}  - d0 {self.DistSDm[0]}  - d1 {self.DistSDm[1]}  "
                         f"- ws {self.ws}  - wi {self.wi}  - dli {self.del_val}\n")
        
        print("Ne Nsyn N0 N1 - ws  wi  f1  f2 Crm Sm dPhi0 dPhi1 CSm")
        
        # Initialize averages
        for k in range(self.N):
            self.dSDm[k] = 0
        
        self.DPhim = 0.0
        self.DPhim2 = 0.0
        
        # Run multiple experiments
        for ke in range(int(self.Nexp)):
            self.Ne = ke + 1
            
            # Reset correlation counters
            self.Nwindowscorr = 0
            self.istep = 0
            self.sum1 = 0.0
            self.sum2 = 0.0
            self.sum12 = 0.0
            
            # Shuffle synapses for each experiment after the first
            if ke > 0:
                for k in range(self.N):
                    self.DistSDm[k] = 0
                self.CreateShuffleSyn()
            
            # Run simulation
            self.run_simulation()
            
            # Process results
            for k in range(self.N):
                self.dSDm[k] += self.DistSDm[k]
                if self.apc[k].time > 0:
                    self.f[k] += self.nspk[k] / self.apc[k].time * 1000
                self.nspkm[k] += self.nspk[k]
                self.NspikesTotm += self.nspk[k]
            
            if self.Nwindowscorr > 0:
                self.Corr_me += self.corr_med / self.Nwindowscorr
            
            # Phase calculation
            self.calculate_phase()
            
            # Update statistics
            self.update_statistics(ke)
            
            # Reset for next experiment
            self.reset_for_next_experiment()
            
            # Save random seed state
            rrs = h.mcell_ran4_init(self.xseed)
            print(f"Random seed state: {rrs}")
        
        # Finalize results
        self.finalize_results()
    
    def run_simulation(self):
        """Run a single simulation"""
        # Reset spike counters
        for l in range(self.N):
            self.nspk[l] = 0
            self.apc[l].n = 0
        
        # Setup custom advance
        h.finitialize(-70)
        h.frecord_init()
        
        # Run simulation
        while h.t < self.tstop:
            self.advance()
    
    def calculate_phase(self):
        """Calculate phase synchronization between neurons"""
        # Find maximum spike time
        tmax = self.tspk[0][self.nspk[0]] if self.nspk[0] > 0 else 0
        if self.nspk[1] > 0 and self.tspk[1][self.nspk[1]] > tmax:
            tmax = self.tspk[1][self.nspk[1]]
        
        ttt = 0
        ddt = 0.1
        Ntsteps = 0
        DeltaPhim = 0
        
        # Initialize spike indices
        for l in range(self.N):
            self.i_spk[l] = 0
            if self.nspk[l] > 0:
                self.tspknext[l] = self.tspk[l][self.i_spk[l] + 1]
        
        # Random initial phases
        self.phi[0] = h.mcell_ran4(self.xseed) * 6.283184
        self.phi[1] = h.mcell_ran4(self.xseed) * 6.283184
        
        s_med = 0.0
        cs_med = 0.0
        
        # Calculate phase over time
        while ttt < tmax:
            # Update phase based on spikes
            if self.nspk[0] > 1 and self.nspk[1] > 1:
                for l in range(self.N):
                    if ttt >= self.tspk[l][self.i_spk[l] + 1]:
                        self.i_spk[l] += 1
                        self.tspkprec[l] = self.tspk[l][self.i_spk[l]]
                        self.tspknext[l] = self.tspk[l][self.i_spk[l] + 1]
                    
                    if self.tspkprec[l] != self.tspknext[l] and ttt < self.tspknext[l]:
                        self.phi[l] = 2 * np.pi * (ttt - self.tspkprec[l]) / (
                            self.tspknext[l] - self.tspkprec[l])
            
            # Update measures
            DeltaPhim += (self.phi[1] - self.phi[0]) * ddt
            s_med += np.sin((self.phi[1] - self.phi[0]) / 2)**2 * ddt
            cs_med += np.cos(self.phi[1] - self.phi[0]) * ddt
            
            ttt += ddt
            Ntsteps += 1
            
            # Record for first experiment
            if self.Ne == 1 and Ntsteps % 10 == 0:
                self.fphi.write(
                    f"{self.Ne} {self.Nsyn} {self.ws} {self.wi} "
                    f"{self.nspk[0]} {self.i_spk[0]} {self.tspkprec[0]} "
                    f"{self.tspknext[0]} {self.phi[0]}    "
                    f"{self.nspk[1]} {self.i_spk[1]} {self.tspkprec[1]} "
                    f"{self.tspknext[1]} {self.phi[1]}   {ttt} "
                    f"{s_med/ttt if ttt>0 else 0} {cs_med/ttt if ttt>0 else 0}\n"
                )
                self.d.line(ttt, s_med/ttt if ttt>0 else 0)
                self.d.flush()
        
        # Handle case with no spikes
        if tmax == 0:
            ttt = 1
            s_med = np.sin((self.phi[1] - self.phi[0]) / 2)**2
            # S=0: Perfect synchrony (φ1 = φ2)
            # S=1: Perfect anti-synchrony (φ1 = φ2 + π)
            # 0<S<1: Partial synchrony

            cs_med = np.cos(self.phi[1] - self.phi[0])
            # CS=1: In-phase
            # CS=-1: Anti-phase
            # CS=0: Quadrature (90° out)
            
            DeltaPhim = self.phi[1] - self.phi[0] # Mean phase difference
            # Positive: Neuron 1 leads
            # Negative: Neuron 0 leads
            # Near 0: Synchronous
        
        # Store results
        self.S_me += s_med / ttt if ttt > 0 else 0
        self.CS_me += cs_med / ttt if ttt > 0 else 0
        self.DPhim += DeltaPhim / ttt if ttt > 0 else 0
        
        self.S2_me += (s_med / ttt if ttt > 0 else 0)**2
        self.CS2_me += (cs_med / ttt if ttt > 0 else 0)**2
        self.DPhim2 += (DeltaPhim / ttt if ttt > 0 else 0)**2
    
    def update_statistics(self, ke):
        """Update statistics after each experiment - means and standard dev"""
        Ne = ke + 1
        
        # Calculate variances
        varS = abs(self.S2_me/Ne - (self.S_me/Ne)**2)
        varCS = abs(self.CS2_me/Ne - (self.CS_me/Ne)**2)
        varPH = abs(self.DPhim2/Ne - (self.DPhim/Ne)**2)
        
        S_sig = np.sqrt(varS) if varS > 0 else 0
        CS_sig = np.sqrt(varCS) if varCS > 0 else 0
        DPhim_sig = np.sqrt(varPH) if varPH > 0 else 0
        
        # Print results
        print(f"{Ne} {self.Nsyn} {self.nspkm[0]/Ne:.1f} {self.nspkm[1]/Ne:.1f} "
              f"{self.ws} {self.wi} {self.f[0]/Ne:.1f} {self.f[1]/Ne:.1f} "
              f"{self.Corr_me/Ne:.3f} {self.S_me/Ne:.3f} {self.dSDm[0]/Ne:.1f} "
              f"{self.dSDm[1]/Ne:.1f} {self.DPhim/Ne:.3f} {self.CS_me/Ne:.3f} "
              f"{S_sig:.3f} {CS_sig:.3f} {DPhim_sig:.3f} {self.del_val}")
        
        print(f"           Sm = {self.S_me/Ne:.3f}, Cm = {self.CS_me/Ne:.3f}, "
              f"C(0) = {self.Corr_me/Ne:.3f}")
        
        # Cross-correlation for first experiment
        if Ne == 1:
            self.correl = h.CrossCorrelation(self.C1, self.tC1, self.C2, self.tC2)
            
            fcrn = f"Synchro_Corr_wi{self.wi}_dl{self.del_val}.out"
            with open(fcrn, 'w') as fcorr:
                fcorr.write(f"# Nintervals {self.correl.cor.size-1}\n")
                for i in range(self.correl.cor.size):
                    fcorr.write(f"{i} {self.correl.cor.x[i]}\n")
        
        # Save results to file
        foutn = f"Synchro_wi{self.wi}_dl{self.del_val}.out"
        with open(foutn, 'w') as fout:
            fout.write("# ds1 NspkT nspk[0] nspk[1] Nsyn ws wi f1m f2m corr_vv "
                      "S_me dS0 dS1 Dphinm CS_me S_sig CS_sig DPhim_sig Nexp del\n")
            fout.write(
                f"{self.ds1} {self.NspikesTotm/Ne:.1f} "
                f"{self.nspkm[0]/Ne:.1f} {self.nspkm[1]/Ne:.1f} {self.Nsyn} "
                f"{self.ws} {self.wi} {self.f[0]/Ne:.1f} {self.f[1]/Ne:.1f} "
                f"{self.Corr_me/Ne:.3f} {self.S_me/Ne:.3f} {self.dSDm[0]/Ne:.1f} "
                f"{self.dSDm[1]/Ne:.1f} {self.DPhim/Ne:.3f} {self.CS_me/Ne:.3f} "
                f"{S_sig:.3f} {CS_sig:.3f} {DPhim_sig:.3f} {Ne} {self.del_val}\n"
            )
    
    def reset_for_next_experiment(self):
        """Reset variables for next experiment"""
        self.NspikesTot = 0
        
        for l in range(self.N):
            self.nspk[l] = 0
            self.tspkprec[l] = 0
            self.tspknext[l] = 0
            
            # Clear spike times
            l2 = 1
            while l2 < len(self.tspk[l]) and self.tspk[l][l2] > 0:
                self.tspk[l][l2] = 0
                l2 += 1
            
            self.apc[l].n = 0
        
        self.ic = 0
        self.corr_med = 0.0
    
    def finalize_results(self):
        """Finalize and plot results after all experiments"""# Close files
        if self.fphi:
            self.fphi.close()
        if self.fc:
            self.fc.close()
        if self.fspk:
            self.fspk.close()
        if self.fout:
            self.fout.close()
        if self.fv and self.Ne == 1:
            self.fv.close()
        
        # Update average plots
        xval = self.ds1
        
        self.c2.mark(xval, self.Corr_me/self.Nexp, "o", 8, 1, 1.5)
        self.c2.flush()
        
        self.d2.mark(xval, self.S_me/self.Nexp, "o", 8, 1, 1.5)
        self.d2.flush()
        
        self.e2.mark(xval, self.f[0]/self.Nexp, "o", 8, 1, 1.5)
        self.e2.mark(xval, self.f[1]/self.Nexp, "t", 8, 1, 1.5)
        self.e2.flush()

# Main execution
if __name__ == "__main__":
    # Create synchronization study
    study = TwoNeuronSynchronization()
    
    # Run main simulation
    study.runMain()