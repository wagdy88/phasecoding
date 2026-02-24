from optparse import Values
import neuron
from neuron import h
import numpy as np
import os

# Load required NEURON libraries
h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")
        
class CA1_PC_cAC_sig:
    """CA1 Pyramidal Cell template converted from HOC to Python"""
    
    def __init__(self, gid, name="CA1"):
        """
        Initialize the CA1 pyramidal cell model and sets up compartmental organization
        name: str: cell name (default "CA1")
        gid: int or None: global identifier for the cell        
        """
        
        self._gid = gid
        self.name = name
        self.load_morphology()
        
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
        self.gid = 0  # Used for RNG seeding
        self.channel_seed = 0
        self.channel_seed_set = False
 
        ## Post-morphology set up
        self.replace_axon()
        self.insertChannel() 
        self.biophys()

        # Initialize random number generators
        self.re_init_rng()

        ## Measure dendritic distances for Hasselmo's question
        dend_dist = []
        dend_py_list = list(self.dendrites)
        shifted_dend_list = dend_py_list[1:] + dend_py_list[:1]
        
        
        for d1, d2 in zip(dend_py_list, shifted_dend_list):
            x = h.distance(d1(1), d2(1))
            dend_dist.append(x)
            
            #dend_dist.append(x)
            print(f"Distance between dendrties: {dend_dist}")
            print(f"Maximum dendritic distance: {max(dend_dist)}")
            print(f"Minimum dendritic distance: {min(dend_dist)}")

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
        print(f"Deleting {len(old_axon_sections)} old axon sections...")
        for sec in old_axon_sections:
            try:
                h.delete_section(sec=sec)
            except:
                pass  # Already deleted
        
        print(f"old axon sections after deletion: {old_axon_sections}")
        
        # 5. Update data structures
        print("Updating data structures with new axon sections...")
        
        ## a. Cleaning previous axon Section and SectionList
        self.axon.clear()
        self.axonal = h.SectionList()
        
        print(f"this is self.axon after deleting the old axon sections: {self.axon}")
        print(f"this is self.axonal after deleting the old axon sections: {list(self.axonal)}")
        
        
        # b. Add to section lists
        for sec in new_axon:
            self.axon.append(sec)
            self.axonal.append(sec)
        
        print(f"this is self.axon after adding the new axon sections: {self.axon}")
        print(f"this is self.axonal after adding the new axon sections: {list(self.axonal)}")
        
        
         # 6. Clean up self.all (remove any deleted sections)
        self.all = self._clean_sectionlist(self.all)
        
        print(f"this is self.all after cleaning: {self.all}")
    
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
        print(f"this is self.all after adding new axon sections: {self.all}")
        
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
        hash_val = 0
        for char in input_str:
            hash_val = (hash_val * 31 + ord(char)) % (2**31 - 1)
        return hash_val
    
    def re_init_rng(self, seed=None):
        """Reinitialize random number generators"""
        if seed is not None:
            self.channel_seed = seed
            self.channel_seed_set = True
        else:
            self.channel_seed_set = False
    