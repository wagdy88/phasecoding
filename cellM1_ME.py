"""This code Migliore 2024 Two reciprocally connected CA1 pyramidal cell model. This is the DeepSeek created and author's verified version of cellM1.hoc"""
## Created by: Mohamed ElSayed, supervised by: Mohamed Sherif, and in collaboration with Michael Hasselmo

## This is a NEURON cell template for a CA1 pyramidal cell model created by BluePyOpt (an optimization tool for neuron models).

#### Compartment-Specific Properties: ####
# Soma: High Na+ density, moderate K+ channels
# Apical dendrites: Ca2+ channels, distance-dependent gradients
# Basal dendrites: Similar to apical but different parameters
# Axon: High Na+ density for reliable spike propagation

#### Optimization Context ####
# Parameters were optimized by BluePyOpt to match:
# Resting membrane potential
# Input resistance
# Spike shape and timing
# Frequency-current relationships
# Uses experimental data from real CA1 pyramidal cells

import csv
from optparse import Values
import neuron
from neuron import h
from neuron.units import mV, ms
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date, datetime
import random 
import hashlib

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

class CA1_PC_cAC_sig:
    """CA1 Pyramidal Cell template converted from HOC to Python"""
    
    def __init__(self, gid, name="CA1", seed=42):
        """
        Initialize the CA1 pyramidal cell model and sets up compartmental organization
        name: str: cell name (default "CA1")
        gid: int or None: global identifier for the cell
        seed: int: random seed for channel randomization (default 42)        
        """
        self.script_path = script_path
        self.script_dir = script_dir
        self._gid = gid
        self.name = name
        self.seed = seed
        self.load_morphology()
        
         # Set random seeds for Python's random module
        random.seed(seed)
        
        # Set numpy random seed if you're using numpy anywhere
        np.random.seed(seed)
        
        # Create nseg = 3 for each section
        for sec in h.allsec():
            sec.nseg = 3

        # Change soma reference to for easier access
        self.soma = self.soma[0]
        
        # Create section lists for organization (grouping similar section types together)
        self.axonal = h.SectionList()
        for sec in list(self.axon):
            self.axonal.append(sec)
        
        self.dendrites = h.SectionList()
        for sec in list(self.dend):
            self.dendrites.append(sec)
        
        self.apical = h.SectionList()
        for sec in list(self.apic):
            self.apical.append(sec)
    
        # For compatibility with BBP (Blue Brain Project) cells
        ## in BBP, cells have a CellRef attribute 
        self.CellRef = self
        
        # Random seed for channel randomization
        self.gid = gid if gid is not None else 0  # Used for RNG seeding
        self.channel_seed = seed 
        self.channel_seed_set = True # Mark as set immediately to ensure reproducibility
 
        ## Post-morphology set up
        self.replace_axon()
        self.insertChannel() 
        self.biophys()

        # Initialize random number generators with seed
        self.re_init_rng(seed)

        ## Measure apical dendritic distances for Hasselmo's question
        # apical_dist = []
        # apical_py_list = list(self.apical)
        # shifted_apical_list = apical_py_list[1:] + apical_py_list[:1]
        
        
        # for d1, d2 in zip(apical_py_list, shifted_apical_list):
        #     x = h.distance(d1(1), d2(1))
        #     apical_dist.append(x)
            
        #     #apical_dist.append(x)
        #     print(f"Distance between apical dendrties: {apical_dist}")
        #     print(f"Maximum apical dendritic distance: {max(apical_dist)}")
        #     print(f"Minimum apical dendritic distance: {min(apical_dist)}")

        ## Measure dendritic distances for Hasselmo's question
        # dend_dist = []
        # dend_py_list = list(self.dendrites)
        # shifted_dend_list = dend_py_list[1:] + dend_py_list[:1]
        
        
        # for d1, d2 in zip(dend_py_list, shifted_dend_list):
        #     x = h.distance(d1(1), d2(1))
        #     dend_dist.append(x)
            
        #     #dend_dist.append(x)
        #     print(f"Distance between dendrties: {dend_dist}")
        #     print(f"Maximum basal dendritic distance: {max(dend_dist)}")
        #     print(f"Minimum basal dendritic distance: {min(dend_dist)}")

        # Get the apic dendrite with the shortest and longest distance from the soma
        apic_dist_from_soma = {}
        apic_py_list = list(self.apical)


        for d1 in apic_py_list:
            apic_dist_from_soma[d1] = h.distance(d1(1), self.soma(0.5))
                    #apic_dist_from_soma.append(s)

        median_dist_ten = [dend for dend, dist in apic_dist_from_soma.items() if 200<= dist <= 300]
        

        #for dist in list(dend_dist_from_soma.values()):
        min_dist_from_soma = min(list(apic_dist_from_soma.values()))
        min_key = [k for k, v in apic_dist_from_soma.items() if v == min_dist_from_soma]
        print(f'The apical dendrite with shortest distance from soma is: {min_key} and has a distance of {min_dist_from_soma}')

        max_dist_from_soma = max(list(apic_dist_from_soma.values()))
        max_key = [k for k, v in apic_dist_from_soma.items() if v == max_dist_from_soma]
        print(f'The apical dendrite with longest distance from soma is: {max_key} and has a distance of {max_dist_from_soma}')
        
        
        median_dist_from_soma = np.median(list(apic_dist_from_soma.values()))
        median_key = [k for k, v in apic_dist_from_soma.items() if v == median_dist_from_soma]
        print(f'The apical dendrite with median distance from soma is: {median_key} and has a distance of {median_dist_from_soma}')


    def check_simulator(self):
        """Check simulation parameters match expected values"""
        if h.celsius != 34:
            raise ValueError(f"celsius has different value 34 != {h.celsius}")
        if h.v_init != -70:
            raise ValueError(f"v_init has different value -70 != {h.v_init}")
        
    def __repr__(self):
        # Use getattr to provide a fallback name if self.name doesn't exist yet
        name = getattr(self, 'name', 'Initializing Cell')
        gid = getattr(self, '_gid', 'None')
        return "{}[{}]".format(name, gid)
        
    def load_morphology(self):
        """ loads cell morphology from file """
        # Load morphology
        morph_path = os.path.join('/home/mohamed/myprojects/migliore2024/phasecoding/morphology','mpg141017_a1-2_idC.asc')
        
        # Create reader object
        reader = h.Import3d_Neurolucida3(morph_path)
        
        # Use the reader's .input method to point to the file path
        reader.input(morph_path)
        
        # Pass the reader object (not the path) to the GUI tool
        # The '0' argument suppresses the window for script automation
        import_gui = h.Import3d_GUI(reader, 0)
        import_gui.instantiate(self)
        
        
    def replace_axon(self):
        """
        Replace detailed axon morphology with a simplified stub axon
        Preserves diameter profile from original axon
        Creates an axon that has two sections with 5 segments each and total length of 60 μm
        """
        ### Features of stub axon ###
        ## 1. Preserves Critical Regions:
        # Axon Initial Segment (AIS): Where action potentials are initiated
        # Maintains correct diameter profile from original axon
        # Preserves ion channel densities in critical regions

        ## 2. Eliminates Unnecessary Complexity:
        # Removes long myelinated segments (computationally expensive)
        # Eliminates terminal branches (not relevant for somatic recording)
        # Reduces thousands of compartments to ~10
        
        # Set random seed for any random operations in this method
        random.seed(self.seed)
        
        #### Steps to replace axon
        print("Replacing axon with stub...")
        
        ## 1. Get old axon info 
        old_axon_sections = []
        for sec in self.axonal:
            old_axon_sections.append(sec)
        
        ## 2. Sample old axon diameter and length
        L_target = 60  # Length of stub axon (μm)
        nseg0 = 5      # Number of segments per axon section
        
        nseg_total = nseg0 * 2
        chunk_size = L_target / nseg_total
        
        # Count original axon sections
        nSec = len(old_axon_sections)
        
        if nSec < 2:
            raise ValueError("Less than two axon sections are present! Add an axon to the morphology.")
        
        # Collect diameter and length information from original axon
        diams = h.Vector()
        lens = h.Vector()
        
        count = 0
        for sec in old_axon_sections:
            # Temporarily set nseg to sample diameters
            original_nseg = sec.nseg
            sec.nseg = 1 + int(sec.L / chunk_size / 2.) * 2
            
            # Sample diameters along the section
            # Collect diameter samples from original axon
            for seg_index in range(sec.nseg):
                x = (seg_index + 0.5) / sec.nseg
                if 0 < x < 1:
                    count += 1
                    diams.resize(count)
                    lens.resize(count)
                    diams.x[count-1] = sec(x).diam
                    lens.x[count-1] = sec.L / sec.nseg
                    
                    if count == nseg_total:
                        break
            sec.nseg = original_nseg  # Restore original nseg
            
            if count == nseg_total:
                break
               
        # 2. Create new stub axon with 2 sections using old axon diameter
        print("Now creating the new stub axon...")
        new_axon = []
        L_real = 0
        count = 0
        for i in range(2):
            axon_sec = h.Section(name=f'axon{i}', cell=self)
            axon_sec.L = L_target / 2
            axon_sec.nseg = nseg_total // 2
            # Assign diameters from original axon
            for seg_index in range(axon_sec.nseg):
                x = (seg_index + 0.5) / axon_sec.nseg
                if 0 < x < 1:
                    axon_sec(x).diam= diams.x[count]
                    L_real += lens.x[count]
                    count += 1
            new_axon.append(axon_sec)     
            print(f"Created axon[{i}]")
        print(f"Target stub axon length: {L_target} μm, equivalent length: {L_real} μm")
            
        # 3. Connect axon sections
        if self.soma:   
                new_axon[0].connect(self.soma, 0)
                new_axon[1].connect(new_axon[0], 0)
                print("Connected the new axon to the soma")
                
        # 4. Delete original axon sections and section list
        #print(f"Deleting {len(old_axon_sections)} old axon sections...")
        for sec in old_axon_sections:
            try:
                h.delete_section(sec=sec)
            except:
                pass  # Already deleted
        
        #print(f"old axon sections after deletion: {old_axon_sections}")
        
        # 5. Update data structures
        print("Updating data structures with new axon sections...")
        
        ## a. Cleaning previous axon Section and SectionList
        self.axon.clear()
        self.axonal = h.SectionList()
        
        #print(f"this is self.axon after deleting the old axon sections: {self.axon}")
        #print(f"this is self.axonal after deleting the old axon sections: {list(self.axonal)}")
        
        
        # b. Add to section lists
        for sec in new_axon:
            self.axon.append(sec)
            self.axonal.append(sec)
        
        print(f"this is self.axon after adding the new axon sections: {self.axon}")
        print(f"this is self.axonal after adding the new axon sections: {list(self.axonal)}")
        
        
         # 6. Clean up self.all (remove any deleted sections)
        self.all = self._clean_sectionlist(self.all)
        
        #print(f"this is self.all after cleaning: {self.all}")
    
        # 7. Ensure new axons are in self.all
        for sec in new_axon:
        # Check if already in self.all
            already_there = False
            for existing in self.all:
                if existing == sec:
                    already_there = True
                    break
        
            if not already_there:
                self.all.append(sec)
        #print(f"this is self.all after adding new axon sections: {self.all}")
        
        print(f"Axon replacement complete. New axon has {len(new_axon)} sections")
        return new_axon

    def _clean_sectionlist(self, sec_list):
        """Remove deleted sections from a SectionList"""
        cleaned = h.SectionList()
        for sec in sec_list:
            try:
                _ = h.secname(sec=sec)
                cleaned.append(sec)  # Only add valid sections
            except:
                pass  # Skip deleted sections
        return cleaned
        
       # self.nSecAxonal = 2
        
    def insertChannel(self):
        """Insert ion channels into appropriate sections"""
        
        # Insert basic channels in all sections
        for sec in self.all:
            sec.insert('pas')  # Passive leak
            sec.insert('kdr')  # Delayed rectifier K+
            sec.insert('nax')  # Fast Na+
        
        # Apical dendrites get additional channels
        for sec in self.apical:
            sec.insert('kad')   # A-type K+ (apical)
            sec.insert('hd')    # HCN (hyperpolarization-activated)
            sec.insert('can')   # N-type Ca2+
            sec.insert('cal')   # L-type Ca2+
            sec.insert('cat')   # T-type Ca2+
            sec.insert('kca')   # Ca2+-activated K+
            sec.insert('cagk')  # Ca2+-activated K+ (Gurkiewicz)
            sec.insert('cacum') # Calcium accumulation
        
        # Axonal channels
        for sec in self.axonal:
            sec.insert('kmb')   # M-type K+
            sec.insert('kap')   # A-type K+ (axonal)
        
        # Basal dendrites (same as apical)
        for sec in self.dendrites:
            sec.insert('kad')
            sec.insert('hd')
            sec.insert('can')
            sec.insert('cal')
            sec.insert('cat')
            sec.insert('kca')
            sec.insert('cagk')
            sec.insert('cacum')
        
        # Somachannels
        if self.soma:
            self.soma.insert('kmb')
            self.soma.insert('kap')
            self.soma.insert('hd')
            self.soma.insert('can')
            self.soma.insert('cal')
            self.soma.insert('cat')
            self.soma.insert('kca')
            self.soma.insert('cagk')
            self.soma.insert('cacum')
    
    def biophys(self):
        """Set biophysical parameters for all sections"""
        ## Conductance densities (gbar_*): Maximum channel conductances
        ## Ra: Axial resistivity (Ω·cm) - affects signal propagation
        ## cm: Membrane capacitance (μF/cm²)
        ## ena/ek: Ion reversal potentials (mV)
        ## Values were optimized by BluePyOpt to match experimental data
                
        # Global parameters for all sections
        for sec in self.all:
            sec.cm = 1      # Membrane capacitance (μF/cm²)
            sec.ena = 50    # Na+ reversal potential (mV)
            sec.ek = -90    # K+ reversal potential (mV)
        
        # Apical dendrite parameters
        for sec in self.apical:
            sec.gkdrbar_kdr = 0.0051140701369551591      # Delayed rectifier K+
            sec.gcalbar_cal = 1.3433464444108534e-06     # L-type Ca2+
            sec.gcanbar_can = 9.1375529468386047e-06     # N-type Ca2+
            sec.gcatbar_cat = 5.2627884427712998e-06     # T-type Ca2+
            sec.gbar_kca = 0.00011632706997854596        # Ca2+-activated K+
            sec.gbar_cagk = 3.8813989222985606e-05       # Ca2+-activated K+ (G)
            sec.gbar_nax = 0.023865342299765658          # Fast Na+
            sec.Ra = 102.93574883917653                  # Axial resistance (Ω·cm)
            sec.g_pas = 3.6662171792438072e-05           # Passive conductance
        
        # Axonal parameters
        for sec in self.axonal:
            sec.gbar_nax = 0.15092656523953063
            sec.gkdrbar_kdr = 0.012773439165875373
            sec.gbar_kmb = 0.015708771639181624
            sec.gkabar_kap = 0.17033392959442739
            sec.Ra = 68.659727248098548
            sec.g_pas = 0.00019814434729699748
            sec.e_pas = -60.050622497180008
        
        # Basal dendrite parameters
        for sec in self.dendrites:
            sec.gkdrbar_kdr = 0.0051140701369551591
            sec.gcalbar_cal = 1.3433464444108534e-06
            sec.gcanbar_can = 9.1375529468386047e-06
            sec.gcatbar_cat = 5.2627884427712998e-06
            sec.gbar_kca = 0.00011632706997854596
            sec.gbar_cagk = 3.8813989222985606e-05
            sec.gbar_nax = 0.0050344912419360861
            sec.Ra = 196.53208992869335
            sec.g_pas = 4.8328456800280221e-05
        
        # Soma parameters
        if self.soma:
            self.soma.gkabar_kap = 0.077773803027485822
            self.soma.gbar_nax = 0.03324329826147978
            self.soma.gbar_kmb = 0.0015345761039855402
            self.soma.Ra = 191.49312086794978
            self.soma.g_pas = 6.4696735301537531e-05
            self.soma.gkdrbar_kdr = 0.0051140701369551591
            self.soma.gcalbar_cal = 1.3433464444108534e-06
            self.soma.gcanbar_can = 9.1375529468386047e-06
            self.soma.gcatbar_cat = 5.2627884427712998e-06
            self.soma.gbar_kca = 0.00011632706997854596
            self.soma.gbar_cagk = 3.8813989222985606e-05
        
        # Distance-dependent parameter distributions (controls how many channels are in the membrane)
        # These create gradients of channel densities along dendrites
        ## Creates spatial gradients of channel densities:
        ### HCN channels (hd): Increase linearly with distance from soma (+3% per 100 μm)
        ### A-type K+ (kad): Sigmoidal gradient, increases with distance
        ### Passive reversal (e_pas): Becomes more depolarized with distance
        ### These gradients are biologically realistic and affect signal integration
        ### the formula to calculate the density is: 1 + 0.03 * distance (μm) * gradient at the soma 
        # %.17g means distance value with 17-digit precision: divided to % (placeholder), .17 (17 digits of precision), and g (general format)
        
             
        # Apical dendrites: HCN channel gradient
        self.distribute_distance(self.apical, "ghdbar_hd", 
                                "(1. + 3./100. * %.17g)*1.9139190469809671e-05")
        # Apical: Passive reversal potential gradient
        self.distribute_distance(self.apical, "e_pas",
                                "(-66.279120044807371-5*%.17g/150)")
        # Apical: A-type K+ gradient
        
        #apical_value = np.exp((300- %.17g)/50)
        #self.distribute_distance(self.apical, "gkabar_kad",
        #                        f"(15./(1. + {apical_value}))*0.013537136988629043")
        
        
        self.distribute_distance(self.apical, "gkabar_kad",
                                "(15./(1. + exp((300-%.17g)/50)))*0.013537136988629043")
        
        # Basal dendrites (similar gradients)
        self.distribute_distance(self.dendrites, "ghdbar_hd",
                                "(1. + 3./100. * %.17g)*1.9139190469809671e-05")
        self.distribute_distance(self.dendrites, "e_pas",
                                "(-67.859607336867882-5*%.17g/150)")
        self.distribute_distance(self.dendrites, "gkabar_kad",
                                "(15./(1. + exp((300-%.17g)/50)))*0.013537136988629043")
        
        # Soma gradients
        self.distribute_distance(self.soma, "e_pas",
                                "(-71.397169317070947-5*%.17g/150)")
        self.distribute_distance(self.soma, "ghdbar_hd",
                                "(1. + 3./100. * %.17g)*1.9139190469809671e-05")
    
    def distribute_distance(self, section_list, mech_name, dist_func_str):
        """
        Distribute mechanism parameters based on distance from soma
        
        Args:
            section_list: SectionList or single Section object 
            mech_name: Name of mechanism parameter (e.g., 'ghdbar_hd')
            dist_func_str: String defining distance-dependent function
                          Contains %.17g placeholder for distance
        """
        # Set reference point for distance measurements
        # 0 means Distance origin at soma START (0)
        # 0.5 means measure TO the MIDDLE of each section 
        
        h.distance(0, 0.5, sec=self.soma)
        
        # Handle both SectionList and single Section
        # The code first checks if the input is a SectionList or a single Section.
        # If the input is single section, the code wraps it in a list to unify processing
        if hasattr(section_list, '__iter__'):
            sections = section_list
        else:
            sections = [section_list]
        
        # Iterate over sections
        if not self.soma:
            for sec in sections:
            # Check if mechanism exists in this section before trying to set it
                try:
                    # Iterate over segments in the section
                    nseg = sec.nseg
                    for seg_index in range(nseg):
                        x = (seg_index + 0.5) / nseg  # Segment center
                        dist = h.distance(x, sec=sec)
                
                        # Evaluate the distance function
                        func_value = self._eval_distance_func(dist_func_str, dist)
                
                    # Set the mechanism parameter
                    self._set_mech_param(sec, mech_name, x, func_value)
                except AttributeError as e:
                    # Mechanism doesn't exist in this section, skip it
                    print(f"Warning: {mech_name} not found in {sec.name}, skipping")
        # else:
        #     soma_nseg = self.soma.nseg
        #     for seg_index in range(soma_nseg):
        #         x = (seg_index + 0.5) / soma_nseg  # Segment center
        #         dist = h.distance(x, sec=self.soma)

        #         # Evaluate the distance function
        #         func_value = self._eval_distance_func(dist_func_str, dist)

        #         # Set the mechanism parameter
        #         self._set_mech_param(self.soma, mech_name, x, func_value)
            
    def _eval_distance_func(self, func_str, distance):
        """Safely evaluate distance function string"""
        # Replace %.17g placeholder with distance value
        eval_str = func_str.replace("%.17g", str(distance))
        
        # Use NEURON's exp function for consistency
        # Provide a safe namespace with only necessary functions
        safe_dict = {"exp": h.exp}
        try:
            return eval(eval_str, safe_dict)
        except Exception as e:
            print(f"Error evaluating distance function: {eval_str}")
            print(f"Error: {e}")
            raise
    
    def _set_mech_param(self, section, mech_name, x, value):
        """Set mechanism parameter at specific location"""
        # Map mechanism names to section attributes
        mech_map = {
            'ghdbar_hd': 'ghdbar_hd',
            'e_pas': 'e_pas',
            'gkabar_kad': 'gkabar_kad',
            # Add more mappings as needed
        }
        
        if mech_name in mech_map:
            param_name = mech_map[mech_name]
            # Access segment and set parameter
            # This is complex in NEURON Python - usually done via hoc
            exec_str = f"{section.name()}({x}).{param_name} = {value}"
            h(exec_str)
    
    def hash_str(self, input_str):
        """Create hash from string for RNG seeding"""
         # Use Python's built-in hash with fixed seed
        # But note: Python's hash is randomized by default in some versions
        # Better to use a deterministic algorithm
        return int(hashlib.md5(input_str.encode()).hexdigest()[:8], 16)
        
        # hash_val = 0
        # for char in input_str:
        #     hash_val = (hash_val * 31 + ord(char)) % (2**31 - 1)
        # return hash_val
    
    def re_init_rng(self, seed=None):
        """Reinitialize random number generators"""
        if seed is not None:
            self.channel_seed = seed
            self.channel_seed_set = True
            
            # Set random seed for NEURON's random number generators if needed
            # This depends on how your channels use random numbers
            try:
                # If you have Random objects in NEURON
                for r in h.Random.list():
                    r.Random123_globalindex(seed)
            except:
                pass
        else:
            self.channel_seed_set = False

    def stimIC_soma(self, amp, dur, delay):
        """Inject current into the soma"""
        stim = h.IClamp(self.soma(0.5))
        stim.amp = amp
        stim.dur = dur
        stim.delay = delay
        
        # Set up recording vectors
        v = h.Vector().record(self.soma(0.5)._ref_v)  # Membrane potential vector
        t = h.Vector().record(h._ref_t)  # Time stamp vector

        # Running simulation
        h.finitialize(-65 * mV)
        h.continuerun(40 * ms)
        
        # get results
        v_array = np.array(v)
        t_array = np.array(t)
        
        # Plot results
        plt.figure()
        plt.plot(t_array, v_array)
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential (mV)")
        plt.title("Soma Membrane Potential")
        plt.show()
        
        # Save results
        folder_name = f'SomaVoltage_Ran_On_{timestamp}/'

        ### d. Construct the full path for the new folder
        output_folder_path_soma = os.path.join(self.script_dir, folder_name)

        # Create the folder if it doesn't already exist
        try:
            os.makedirs(output_folder_path_soma, exist_ok=True)
            print(f"Folder '{folder_name}' created successfully at: {output_folder_path_soma}")
        except OSError as e:
            print(f"Error creating folder: {e}") 

        try:
            plt.savefig(os.path.join(output_folder_path_soma, f'SomaVoltage_Ran_On_{timestamp}.png'), dpi=300)
        
            with open(os.path.join(output_folder_path_soma, f"SomaVoltage_Ran_On_{timestamp}.csv"), "a") as f:
                csv.writer(f).writerows(zip(t_array, v_array))
        except OSError as e:
            print(f"Error saving Figure or CSV file: {e}")
        
        return stim, v_array, t_array


    def stimIC_apic(self, amp, dur, delay, aid, loc=0.5):
            """Inject IC current into the apical dendrite
            
            Args:
                amp (float): Amplitude of the IC current
                dur (float): Duration of the IC current
                delay (float): Delay before the IC current starts
                aid (int): Apical dendrite ID
                loc (float): Location along the apical dendrite (default: 0.5)
            
            """
            
            #for i in range(len(self.apic)):
            #if aid in list(self.apic):
            if 0 <= aid < len(self.apic):
                apic_sec = self.apic[aid]
                stim = h.IClamp(apic_sec(loc))
                stim.amp = amp
                stim.dur = dur
                stim.delay = delay

                # Set up recording vectors
                v = h.Vector().record(apic_sec(loc)._ref_v)  # Membrane potential vector
                t = h.Vector().record(h._ref_t)  # Time stamp vector

                # Running simulation
                h.finitialize(-65 * mV)
                h.continuerun(40 * ms)

                # get results
                v_array = np.array(v)
                t_array = np.array(t)

                # Plot results
                plt.figure()
                plt.plot(t_array, v_array)
                plt.xlabel("Time (ms)")
                plt.ylabel("Membrane Potential (mV)")
                plt.title("Apical Dendrite Membrane Potential")
                plt.show()

                # Save results
                # Save results
                folder_name = f'Apic{aid}_Voltage_Ran_On_{timestamp}/'
                output_folder_path_apic = os.path.join(self.script_dir, folder_name)

                # Create the folder if it doesn't already exist
                try:
                    os.makedirs(output_folder_path_apic, exist_ok=True)
                    print(f"Folder '{folder_name}' created successfully at: {output_folder_path_apic}")
                except OSError as e:
                    print(f"Error creating folder: {e}")

                
                try:
                    plt.savefig(os.path.join(output_folder_path_apic, f'Apic{aid}_Voltage_Ran_On_{timestamp}.png'), dpi=300)
                
                    with open(os.path.join(output_folder_path_apic, f"Apic{aid}_Voltage_Ran_On_{timestamp}.csv"), "a") as f:
                        csv.writer(f).writerows(zip(t_array, v_array))
                except OSError as e:
                    print(f"Error saving Figure or CSV file: {e}")

                return stim, v_array, t_array

            else:
                 print(f"Apical dendrite ID: {aid} is not found.")
            
                
    def stimIC_apic(self, amp, dur, delay, aid, loc=0.5):
                """Inject synaptic current into the apical dendrite
                
                Args:
                    amp (float): Amplitude of the synaptic current
                    dur (float): Duration of the synaptic current
                    delay (float): Delay before the synaptic current starts
                    aid (int): Apical dendrite ID
                    loc (float): Location along the apical dendrite (default: 0.5)
                
                """
                
                #for i in range(len(self.apic)):
                #if aid in list(self.apic):
                if 0 <= aid < len(self.apic):
                    apic_sec = self.apic[aid]
                    stim = h.IClamp(apic_sec(loc))
                    stim.amp = amp
                    stim.dur = dur
                    stim.delay = delay

                    # Set up recording vectors
                    v_apic = h.Vector().record(apic_sec(loc)._ref_v)  # Membrane potential vector
                    t_apic = h.Vector().record(h._ref_t)  # Time stamp vector

                    # Record from soma
                    v_soma = h.Vector().record(self.soma(0.5)._ref_v)

                    
                    # Running simulation
                    h.finitialize(-65 * mV)
                    h.continuerun(40 * ms)

                    # get results
                    v_apic_array = np.array(v_apic)
                    t_apic_array = np.array(t_apic)
                    v_soma_array = np.array(v_soma)

                    # Plot results
                    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
                    
                    ## Recording from Apical dendrite
                    #plt.figure()
                    axes[0].plot(t_apic_array, v_apic_array)
                    axes[0].set_xlabel("Time (ms)")
                    axes[0].set_ylabel("Membrane Potential (mV)")
                    axes[0].set_title("Apical Dendrite Membrane Potential")
                    #plt.show()

                    # Record from soma
                    axes[1].plot(t_apic_array, v_soma_array)
                    axes[1].set_xlabel("Time (ms)")
                    axes[1].set_ylabel("Membrane Potential (mV)")
                    axes[1].set_title("Soma Membrane Potential")
                    #plt.show()

                    # Save results
                    # Save results
                    folder_name = f'Apic{aid}_Voltage_Ran_On_{timestamp}/'
                    output_folder_path_apic = os.path.join(self.script_dir, folder_name)

                    # Create the folder if it doesn't already exist
                    try:
                        os.makedirs(output_folder_path_apic, exist_ok=True)
                        print(f"Folder '{folder_name}' created successfully at: {output_folder_path_apic}")
                    except OSError as e:
                        print(f"Error creating folder: {e}")

                    
                    try:
                        plt.savefig(os.path.join(output_folder_path_apic, f'Apic{aid}_Voltage_Ran_On_{timestamp}.png'), dpi=300)
                    
                        with open(os.path.join(output_folder_path_apic, f"Apic{aid}_Voltage_Ran_On_{timestamp}.csv"), "a") as f:
                            csv.writer(f).writerows(zip(t_apic_array, v_apic_array))
                            
                        with open(os.path.join(output_folder_path_apic, f"Soma_Voltage_Ran_On_{timestamp}.csv"), "a") as f:
                            csv.writer(f).writerows(zip(t_apic_array, v_soma_array))
                    
                    except OSError as e:
                        print(f"Error saving Figure or one of the two CSV files: {e}")

                    return stim, v_apic_array, t_apic_array, v_soma_array

                else:
                    print(f"Apical dendrite ID: {aid} is not found.")
                
        
    def netstimSyn_apic(self, aid, interval, number, start, noise, weight, delay, threshold, tstop, loc=0.5):
            """Inject synaptic current into the apical dendrite
            
            Args:
                interval (float): Mean interval between spikes (ms)
                number (int): Number of spikes
                start (float): Start time of first spike (ms)
                noise (float): Randomness (0=deterministic, 1=Poisson)
                aid (int): Apical dendrite ID
                loc (float): Location along the apical dendrite (default: 0.5)
                weight (float): Synaptic weight (μS)
                delay (float): Synaptic delay (ms)
                threshold (float): Threshold for NetStim events
            
            """
            tstop = tstop
            #for i in range(len(self.apic)):
            #if aid in list(self.apic):
            if 0 <= aid < len(self.apic):
                apic_sec = self.apic[aid]
                
                # Create a synapse at that apic 
                apical_syn = h.ExpSyn(apic_sec(loc))
                apical_syn.tau = 2
                apical_syn.e = 0
                
                # Create a NetStim to drive the synapse
                netstim = h.NetStim()
                
                # Configure NetStim parameters
                netstim.interval = interval     # Mean interval between spikes (ms)
                netstim.number = number       # Number of spikes
                netstim.start = start        # Start time of first spike (ms)
                netstim.noise = noise       # Randomness (0=deterministic, 1=Poisson)
                            
                # Connect NetStim to the synapse using NetCon
                nc = h.NetCon(netstim, apical_syn)
                nc.weight[0] = weight      # synaptic weight (μS)
                nc.delay = delay          # synaptic delay (ms)
                nc.threshold = threshold        # NetStim events have value 1, so threshold at 0
                
                # Set up recording vectors
                v_apical = h.Vector().record(apic_sec(loc)._ref_v)  # Membrane potential vector
                v_soma = h.Vector().record(self.soma(0.5)._ref_v)  # Membrane potential vector
                t = h.Vector().record(h._ref_t)  # Time stamp vector

                # Running simulation
                h.finitialize(-65 * mV)
                h.continuerun(tstop)

                # get results
                v_apic_array = np.array(v_apical)
                v_soma_array = np.array(v_soma)
                t_apic_array = np.array(t)
                
                # Plot results
                fig, axes = plt.subplots(2, 1, figsize=(15, 10))
                
                ## Recording from Apical dendrite
                #plt.figure()
                axes[0].plot(t_apic_array, v_apic_array)
                axes[0].set_xlabel("Time (ms)")
                axes[0].set_ylabel("Membrane Potential (mV)")
                axes[0].set_title("Apical Dendrite Membrane Potential After NetStim")
                #plt.show()

                # Record from soma
                axes[1].plot(t_apic_array, v_soma_array)
                axes[1].set_xlabel("Time (ms)")
                axes[1].set_ylabel("Membrane Potential (mV)")
                axes[1].set_title("Soma Membrane Potential After NetStim")
                #plt.show()

                # Save results
                # Save results
                folder_name = f'Apic{aid}_Voltage_After_NetStim_Ran_On_{timestamp}/'
                output_folder_path_apic = os.path.join(self.script_dir, folder_name)

                # Create the folder if it doesn't already exist
                try:
                    os.makedirs(output_folder_path_apic, exist_ok=True)
                    print(f"Folder '{folder_name}' created successfully at: {output_folder_path_apic}")
                except OSError as e:
                    print(f"Error creating folder: {e}")

                
                try:
                    plt.savefig(os.path.join(output_folder_path_apic, f'Apic{aid}_Voltage_After_NetStim_Ran_On_{timestamp}.png'), dpi=300)
                
                    with open(os.path.join(output_folder_path_apic, f"Apic{aid}_Voltage_After_NetStim_Ran_On_{timestamp}.csv"), "a") as f:
                        csv.writer(f).writerows(zip(t_apic_array, v_apic_array))
                        
                    with open(os.path.join(output_folder_path_apic, f"Soma_Voltage_After_NetStim_Ran_On_{timestamp}.csv"), "a") as f:
                        csv.writer(f).writerows(zip(t_apic_array, v_soma_array))
                
                except OSError as e:
                    print(f"Error saving Figure or one of the two CSV files: {e}")


    def netstimSyn_soma(self, interval, number, start, noise, weight, delay, threshold, tstop, loc=0.5):
            """Inject synaptic current into the apical dendrite
            
            Args:
                interval (float): Mean interval between spikes (ms)
                number (int): Number of spikes
                start (float): Start time of first spike (ms)
                noise (float): Randomness (0=deterministic, 1=Poisson)
                aid (int): Apical dendrite ID
                loc (float): Location along the apical dendrite (default: 0.5)
                weight (float): Synaptic weight (μS)
                delay (float): Synaptic delay (ms)
                threshold (float): Threshold for NetStim events
            
            """
            
            #for i in range(len(self.apic)):
            #if aid in list(self.apic):
            #if 0 <= aid < len(self.apic):
             #   apic_sec = self.apic[aid]
            
            tstop = tstop    
            # Create a synapse at that apic 
            soma_syn = h.ExpSyn(self.soma(loc))
            soma_syn.tau = 2
            soma_syn.e = 0
            
            # Create a NetStim to drive the synapse
            netstim = h.NetStim()
            
            # Configure NetStim parameters
            netstim.interval = interval     # Mean interval between spikes (ms)
            netstim.number = number       # Number of spikes
            netstim.start = start        # Start time of first spike (ms)
            netstim.noise = noise       # Randomness (0=deterministic, 1=Poisson)
                        
            # Connect NetStim to the synapse using NetCon
            nc = h.NetCon(netstim, soma_syn)
            nc.weight[0] = weight      # synaptic weight (μS)
            nc.delay = delay          # synaptic delay (ms)
            nc.threshold = threshold        # NetStim events have value 1, so threshold at 0
            
            # Set up recording vectors
            #v_apical = h.Vector().record(apic_sec(loc)._ref_v)  # Membrane potential vector
            v_soma = h.Vector().record(self.soma(0.5)._ref_v)  # Membrane potential vector
            t = h.Vector().record(h._ref_t)  # Time stamp vector

            # Running simulation
            h.finitialize(-65 * mV)
            h.continuerun(tstop)

            # get results
            #v_apic_array = np.array(v_apical)
            v_soma_array = np.array(v_soma)
            t_apic_array = np.array(t)
            
            # Plot results
            #fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            ## Recording from Apical dendrite
            plt.figure()
            plt.plot(t_apic_array, v_soma_array)
            plt.xlabel("Time (ms)")
            plt.ylabel("Membrane Potential (mV)")
            plt.title("Soma Membrane Potential After NetStim")
            #plt.show()

            # Record from soma
            # axes[1].plot(t_apic_array, v_soma_array)
            # axes[1].set_xlabel("Time (ms)")
            # axes[1].set_ylabel("Membrane Potential (mV)")
            # axes[1].set_title("Soma Membrane Potential After NetStim")
            # #plt.show()

            # Save results
            # Save results
            folder_name = f'Soma_Voltage_After_NetStim_Ran_On_{timestamp}/'
            output_folder_path_apic = os.path.join(self.script_dir, folder_name)

            # Create the folder if it doesn't already exist
            try:
                os.makedirs(output_folder_path_apic, exist_ok=True)
                print(f"Folder '{folder_name}' created successfully at: {output_folder_path_apic}")
            except OSError as e:
                print(f"Error creating folder: {e}")

            
            try:
                plt.savefig(os.path.join(output_folder_path_apic, f'Soma_Voltage_After_NetStim_Ran_On_{timestamp}.png'), dpi=300)
            
                #with open(os.path.join(output_folder_path_apic, f"Soma_Voltage_After_NetStim_Ran_On_{timestamp}.csv"), "w") as f:
                    #csv.writer(f).writerows(zip(t_apic_array, v_apic_array))
                    
                with open(os.path.join(output_folder_path_apic, f"Soma_Voltage_After_NetStim_Ran_On_{timestamp}.csv"), "a") as f:
                    csv.writer(f).writerows(zip(t_apic_array, v_soma_array))
            
            except OSError as e:
                print(f"Error saving Figure or one of the two CSV files: {e}")

                
            
#ps = h.PlotShape(False)  # False tells h.PlotShape not to use NEURON's gui
#ps.plot(pyplot)
#pyplot.show()