import BioReactorSubSys as BRS
import CarbonCaptureSubSys as CCSS
import BioSepSubSys as BSS
import numpy as np
from matplotlib.ticker import MaxNLocator
from math import ceil, floor
import numpy as np

def generate_constrained_list(x: int, y: int) -> list[int]:
    """
    Generate a list satisfying the constraints using a simplified approach.
    
    Args:
        x (int): Lower bound (result must have a value < x)
        y (int): Upper bound (result must have a value > y)
    
    Returns:
        list[int]: A list satisfying all constraints
    """
    # Validate inputs
    if x >= y:
        raise ValueError("x must be less than y")
        
    # Find the range bounds that must include 0
    min_val = min(x, 0)
    max_val = max(y, 0)
    
    # Calculate minimum required step size
    min_step = ceil((y - x) / 4)
    
    # Valid step sizes with prime factors {2, 5}
    valid_steps = [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 64, 80, 100]
    
    # Find the first valid step size >= min_step
    step_size = next(s for s in valid_steps if s >= min_step)
    
    # Find starting point by rounding x/step_size to higher magnitude
    start_multiple = np.sign(min_val)*ceil(abs(min_val/step_size)) if min_val < 0 else floor(min_val/step_size)
    start = start_multiple * step_size
    
    # Generate the list
    result = []
    current = start
    while len(result) < 5 and current < max_val+step_size:
        result.append(current)
        current += step_size
        
    return result


Error = 0.5

# Script 1
c_elect = 0.28
lang = 4

CC_VARS_Base = CCSS.CarbonCaptureSubSys()

# Set BioReactor properties
CC_VARS_Base.a = -1250          # Pump Pre-Exponential Const/£
CC_VARS_Base.b = 2400          # Pump Pre-Exponential Factor/£
CC_VARS_Base.n = 0.9          # Pump Exponent
CC_VARS_Base.E0_CO2 = 15.64          # Ideal Gas const*T kWh/Tonne
# CC_VARS_Base.C0 = 0.087517E+6          # 1 Tonne reference Plant cost
CC_VARS_Base.Pressure = 30       # Pumping Presure Req./kPa
CC_VARS_Base.nu_pump = 0.8        # Pump Efficiencey
CC_VARS_Base.nu_conv = 0.85        # Reactor Conversion
CC_VARS_Base.nu_cap  = 0.9        # Capture Efficiencey
CC_VARS_Base.nu_2nd  = 0.2        # 2nd Law efficiencey
CC_VARS_Base.trials = 5000 # Num trials for MC
CC_VARS_Base.ERR = 0.2 # Base Param err
CC_VARS_Base.p = .95 #Outlet purity 
CC_VARS_Base.c_elect = c_elect
CC_VARS_Base.C0 = 8.6E+6
CC_VARS_Base.x0 = 0.14
# Assign errors to variables
ERR_array = [0, 0, 0, 0]
ERR_names = ['E0_CO2', 'x0', 'p', 'n']
CC_VARS_Base.assign_err(value=Error)
CC_VARS_Base.assign_err(ERR_names, ERR_array) 

# Script 2
BR_Vars_Base = BRS.BioReactorSubSys()

# Set BioReactor properties
BR_Vars_Base.a = 129000  # Reactor Pre-Exponential Const/£
BR_Vars_Base.b = 3700    # Reactor Pre-Exponential Factor/£
BR_Vars_Base.n = 0.65    # Reactor Exponent
BR_Vars_Base.A_m = 8    # Membrane Area @ 1 Tonne
BR_Vars_Base.A_E = 40    # Electrode Area @ 1 Tonne
BR_Vars_Base.H_air = 8.76  # Glass air convection coeff kWh/yr m^-2 K^-1
BR_Vars_Base.del_T = 15    # Air Reactor Temperature Difference K
BR_Vars_Base.Is = 0      # Average Solar Insolation kWh/m^2/yr
BR_Vars_Base.E0 = 2.3E+3   # Minimum Energy kWh/Tonne
BR_Vars_Base.PPFD_ENERGY = 0.5 ## Required PPFD in kW/m3
BR_Vars_Base.LED_Cost = 170 ## cost for 5W LED
BR_Vars_Base.trials = 5000
BR_Vars_Base.D = 1  # Reactor Height m
BR_Vars_Base.assign_err(value=Error)
BR_Vars_Base.assign_err('v0',0.25)

# Material costs and penalties
BR_Vars_Base.C_m = 500
BR_Vars_Base.penalty = 0
BR_Vars_Base.C_E = 60
BR_Vars_Base.v0 = 13.3

# Script 3
SS_Vars_Base = BSS.SeparationSubSys()


# Separation Design Params
SS_Vars_Base.C0 = 8.6E+6          # Pump Pre-Exponential Const/£
SS_Vars_Base.Kp = [1.99,2.126,1.760]  # kg CO2/kg product
SS_Vars_Base.E0 = 688.3  # refrence energy in kWh/Mmol
SS_Vars_Base.Mr_product = [376,186.2,25]  # Product Molecular Mass type list
SS_Vars_Base.Mr_solvent = 18  # Solvent Molecular Mass type list
SS_Vars_Base.mass_fraction = 0.1  # Mass fraction of product type list 
SS_Vars_Base.p = [0.99,0.5,0]  # desired product purity type list 
SS_Vars_Base.nu_cap = [0.98,0.9,0.9]  # capture efficiencey of product
SS_Vars_Base.nu_2nd = 0.005  # 2nd Law efficenciey
SS_Vars_Base.yield_ = [0.2,0,0.8]  # yield of product type list sum ==1
SS_Vars_Base.value = [25E+6,1.5E+6,7.5E+3]  # Value £/MMol product type list 
SS_Vars_Base.trials = 5000  # Num trials for MC

# Choices
N0 = 1000
SS_Vars_Base.assign_err(value=Error)
SS_Vars_Base.assign_err(['yield_','p','Mr_product','Mr_solvent','Kp','E0'],value=[0,0,0,0,0,0])

def swap_variables(classes, class_indices, string_inputs, float_values, lang, c_elect, N0):
    # Store copies of original class instances
    original_classes = [cls.__class__() for cls in classes]
    for orig_cls, cls in zip(original_classes, classes):
        orig_cls.__dict__ = cls.__dict__.copy()

    # Swap variables with corresponding string inputs
    for cls_i in range(len(classes)):
        class_i = classes[cls_i]
        for input_idx in range(len(class_indices)):
            class_no = class_indices[input_idx]
            if class_no==cls_i:
                    setattr(class_i, string_inputs[input_idx], float_values[input_idx])

    # Call the function foo with all classes separately
    OPEX, CAPEX,PROFIT, TAU = get_costs(classes[0], classes[1], classes[2], lang, c_elect, N0)

    # Reset classes to their original state
    for cls, orig_cls in zip(classes, original_classes):
        cls.__dict__ = orig_cls.__dict__.copy()

    return OPEX, CAPEX,PROFIT, TAU

# Function foo taking all classes separately
def get_costs(CC_VARS_Base, BR_Vars_Base, SS_Vars_Base, lang, c_elect, N0):
    # Example operation using the inputs
    COSTS = {}
    COSTS['CC'] = CC_VARS_Base.get_COST_SET_MC(lang, N0)
    COSTS['BR'] = BR_Vars_Base.get_COST_SET_MC(lang, N0, 1.2*c_elect)
    COSTS['SS'] = SS_Vars_Base.get_COST_SET_MC(N0, c_elect)

    # Initialize variables to store the summed OPEX and CAPEX lists
    total_opex = np.zeros((1,3))
    total_capex = np.zeros((1,3))


    # Iterate through each dictionary within the costs dictionary
    for entry in COSTS.values():
        # Sum the OPEX and CAPEX lists and add them to the respective variables
        total_opex+= np.percentile(entry['OPEX_RAW'], [25, 50, 75], axis=0)
        total_capex+=  np.percentile(entry['CAPEX_RAW'], [25, 50, 75], axis=0)
        #opex_stdev+= entry[]

    total_revenue = np.percentile(COSTS['SS']['GROSS_RAW'], [75, 50, 25], axis=0)
    # Print the total sum of OPEX and CAPEX
    total_opex = total_opex*1.2+0.1*total_capex
    total_profit = total_revenue - total_opex
    payback_time = total_capex/total_profit
    payback_time[payback_time<0] = 1E+9
    return total_opex, total_capex,total_profit, payback_time


def run_cases(classes, lang, c_elect, N0, cases):
    results = {}

    for case_id, case_info in cases.items():
        # Extract case information
        class_indices = case_info['class_indices']
        string_inputs = case_info['Vars']
        float_values = case_info['Vals']

        # Swap variables & get Costs
        OPEX, CAPEX,PROFIT, TAU = swap_variables(classes, class_indices, string_inputs, float_values, lang, c_elect, N0)
        print(case_id)
        # Store results in the results dictionary
        results[case_id] = {'OPEX': OPEX/1E+6, 'CAPEX': CAPEX/1E+6, 'PROFIT': PROFIT/1E+6, 'TAU': TAU, 'TAC': (OPEX+0.199*CAPEX)/1E+6}

    return results



import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import os

def plot_results(results, y_titles, scale,save_folder):
    num_cases = len(results)
    bar_width = 0.8
    index = np.arange(num_cases)
    colors = cm.Pastel1(np.linspace(0, 1, num_cases))  

    fig, axes = plt.subplots(5, 1, figsize=(10, 15))

    mean_values_dict = {}
    y_min_dict = {}
    y_max_dict = {}

    for idx, metric in enumerate(['CAPEX', 'OPEX', 'PROFIT', 'TAU','TAC']):
        ax = axes[idx]
        ax.set_title(metric, fontsize=28)
        ax.set_ylabel(y_titles[idx], fontsize=24)
        ax.set_xlabel('Case', fontsize=24)

        mean_values = []
        y_min = []
        y_max = []

        for case, data in results.items():
            mean_values.append(np.median(data[metric]))
            y_min.append(np.min(data[metric]))
            y_max.append(np.max(data[metric]))

        mean_values = np.asarray(mean_values)
        y_min = np.asarray(y_min)
        y_max = np.asarray(y_max)

        mean_values_dict[metric] = mean_values
        y_min_dict[metric] = y_min
        y_max_dict[metric] = y_max

        ax.bar(index, mean_values, bar_width, yerr=[mean_values - y_min, y_max - mean_values], color=colors, edgecolor='black', capsize=5, linewidth=1.5)
        ax.set_xticks(index)
        ax.set_xticklabels(results.keys())
        ax.set_yscale(scale[idx])
        # Adjust tick label font size
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=16)
        if metric=='TAU':
            ax.set_ylim(1E-2,10)

    # Adjust spacing between subplots
    plt.tight_layout(pad=3.0)

    # Save plots to the specified folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for idx, metric in enumerate(['CAPEX', 'OPEX', 'PROFIT', 'TAU','TAC']):
        plt.figure(figsize=(5, 5))
        plt.bar(index, mean_values_dict[metric], bar_width, yerr=[mean_values_dict[metric] - y_min_dict[metric], y_max_dict[metric] - mean_values_dict[metric]], color=colors, edgecolor='black', capsize=5, linewidth=1.5)
        #plt.title(metric, fontsize=14)
        plt.ylabel(y_titles[idx], fontsize=36)
        plt.xlabel('Case', fontsize=36)
        plt.xticks(index, results.keys(),fontsize=28)
        #plt.yscale(scale[idx])
        if metric !='TAU':
            y_ticks = generate_constrained_list(np.min(y_min_dict[metric]), np.max(y_max_dict[metric]))
            plt.yticks(y_ticks,fontsize=28)
        plt.tight_layout()
        
        if metric=='TAU':
            plt.ylim((1E-1,1000))
            plt.yticks(fontsize=28)
        plt.savefig(os.path.join(save_folder, f"{metric}.png"))


# Example usage:
# plot_results(results)


# Example usage:
# plot_results(results)


# Example usage:
# plot_results(Results)

# Example usage:
# plot_results(Results)

# Example usage:
# plot_results(Results)



CASES = {
    '0': {
        'class_indices': [],
        'Vars': [],
        'Vals': []
    },
    '1': {
        'class_indices': [0],
        'Vars': ['p'],
        'Vals': [0.2]
    },
        '2': {
        'class_indices': [1],
        'Vars': ['v0'],
        'Vals': [0.14]
    },
        '3': {
        'class_indices': [1],
        'Vars': ['PPFD_ENERGY'],
        'Vals': [0.25]
    },
        '4': {
        'class_indices': [1],
        'Vars': ['C_m'],
        'Vals': [0]
    },

        '5': {
        'class_indices': [0, 1, 1, 1],
        'Vars': ['p','v0','PPFD_ENERGY','C_m'],
        'Vals': [0.2,0.01,0.25,0]
    },
    # Add more cases here as needed
}

results = run_cases([CC_VARS_Base, BR_Vars_Base, SS_Vars_Base], lang, c_elect, N0, CASES)
plot_results(results, ['CAPEX/ £M','OPEX/ £M/yr','PROFIT/ £M/yr','Payback Time/ yr', 'TAC/ £'],['linear','linear','linear','log','linear'], '/home/hs280/TEA/Bin')



# Example usage:
# plot_results(Results)


# Example usage
# swap_variables([CC_VARS_Base, BR_Vars_Base, SS_Vars_Base], [0, 1, 2], ['C0', 'n', 'mass_fraction'], [1.5E+6, 0.75, 0.5], lang, c_elect, N0)


# Functions
def get_COST_SET_MC_CC(lang, N0):
    return CC_VARS_Base.get_COST_SET_MC(lang, N0)

def get_COST_SET_MC_BR(N, c_elect):
    return BR_Vars_Base.get_COST_SET_MC(lang, N, c_elect)

def get_COST_SET_MC_BS(N0, C_electric):
    return SS_Vars_Base.get_COST_SET_MC(N0, C_electric)
