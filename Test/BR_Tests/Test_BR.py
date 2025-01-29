import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ray
import os


# Import the BioReactorSubSys class
sys.path.append('/home/hs280/TEA/Bin')
import BioReactorSubSys as BRS

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
    min_step = np.ceil((y - x) / 3)
    
    # Valid step sizes with prime factors {2, 5}
    valid_steps = [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 64, 80, 100]
    
    # Find the first valid step size >= min_step
    step_size = next(s for s in valid_steps if s >= min_step)
    
    # Find starting point by rounding x/step_size to higher magnitude
    start_multiple = np.sign(min_val)*np.ceil(abs(min_val/step_size)) if min_val < 0 else np.floor(min_val/step_size)
    start = start_multiple * step_size
    
    # Generate the list
    result = []
    current = start
    while len(result) < 4 and current < max_val+step_size:
        result.append(current)
        current += step_size
        
    return result



# Constants
Cur_folder = '/home/hs280/TEA/Test/BR_Tests/'
c_elect = 1.2*0.28
lang = 4

# Initialize BioReactorSubSys instance
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
BR_Vars_Base.PPFD_ENERGY = 0.5 ## Required PPFD in kW/tonne/yr 
BR_Vars_Base.LED_Cost = 170 ## cost for 5W LED
BR_Vars_Base.trials = 1000
BR_Vars_Base.D = 1  # Reactor Height m
BR_Vars_Base.assign_err(value=0.5)

# Design variables
N0 = 1000

# Material costs and penalties
MEMB_COSTS = [500, 17, 10]
MEMB_PENALTIES = [0, 0.2, 0.3]
MEMB_NAMES = ['Nafion', 'BioChar', 'EarthWare']

ELEC_COSTS = [60, 450]
ELEC_V0 = [13.3, 0.14]
ELEC_NAMES = ['CF', 'RVC']


def BioReactMats(BR_Vars_Base, MEMB_COSTS, MEMB_PENALTIES, MEMB_NAMES, ELEC_COSTS, ELEC_V0, ELEC_NAMES, N, c_elect, lang):
    """
    Calculate CAPEX and OPEX for different configurations.
    Args:
        BR_Vars_Base: BioReactorSubSys instance.
        MEMB_COSTS (list): List of membrane costs.
        MEMB_PENALTIES (list): List of membrane penalties.
        MEMB_NAMES (list): List of membrane names.
        ELEC_COSTS (list): List of electrode costs.
        ELEC_V0 (list): List of electrode V0 values.
        ELEC_NAMES (list): List of electrode names.
        N (float): CO2 conversion rate.
        c_elect (float): Electrical cost.
        lang (float): Language variable.
    Returns:
        dict: Dictionary containing CAPEX and OPEX for each configuration.
    """
    num_combin = len(MEMB_COSTS) * len(ELEC_COSTS)
    costs = {'CAPEX': [0] * num_combin, 'OPEX': [0] * num_combin, 'CONFIG': [None] * num_combin}
    index = 0
    for i, memb_cost in enumerate(MEMB_COSTS):
        BR_Vars_Base.C_m = memb_cost
        BR_Vars_Base.penalty = MEMB_PENALTIES[i]
        memb_name = MEMB_NAMES[i]
        for j, elec_cost in enumerate(ELEC_COSTS):
            index += 1
            elec_name = ELEC_NAMES[j]
            BR_Vars_Base.v0 = ELEC_V0[j]
            BR_Vars_Base.C_E = elec_cost
            # Calculate CAPEX and OPEX
            costs['CAPEX'][index - 1] = BR_Vars_Base.get_total_CAPEX(lang, N)
            costs['OPEX'][index - 1] = BR_Vars_Base.get_operating_cost(N, c_elect)
            # Store configuration
            costs['CONFIG'][index - 1] = f"{elec_name}+{memb_name}"
    return costs

def BioReactMats_MC(BR_Vars_Base, MEMB_COSTS, MEMB_PENALTIES, MEMB_NAMES, ELEC_COSTS, ELEC_V0, ELEC_NAMES, N, c_elect, lang, Err):
    """
    Calculate CAPEX and OPEX with Monte Carlo simulation for different configurations.
    Args:
        BR_Vars_Base: BioReactorSubSys instance.
        MEMB_COSTS (list): List of membrane costs.
        MEMB_PENALTIES (list): List of membrane penalties.
        MEMB_NAMES (list): List of membrane names.
        ELEC_COSTS (list): List of electrode costs.
        ELEC_V0 (list): List of electrode V0 values.
        ELEC_NAMES (list): List of electrode names.
        N (float): CO2 conversion rate.
        c_elect (float): Electrical cost.
        lang (float): Language variable.
        Err (float): Error value.
    Returns:
        dict: Dictionary containing CAPEX, OPEX, and raw data for each configuration.
    """
    num_combin = len(MEMB_COSTS) * len(ELEC_COSTS)
    costs = {'CAPEX': np.zeros((num_combin, 4)), 'OPEX': np.zeros((num_combin, 4)), 'CONFIG': [None] * num_combin, 'CAPEX_RAW': np.zeros((num_combin, BR_Vars_Base.trials + 1)), 'OPEX_RAW': np.zeros((num_combin, BR_Vars_Base.trials + 1))}
    index = 0
    for i, memb_cost in enumerate(MEMB_COSTS):
        BR_Vars_Base.C_m = memb_cost
        BR_Vars_Base.penalty = MEMB_PENALTIES[i]
        memb_name = MEMB_NAMES[i]
        for j, elec_cost in enumerate(ELEC_COSTS):
            index += 1
            elec_name = ELEC_NAMES[j]
            BR_Vars_Base.v0 = ELEC_V0[j]
            BR_Vars_Base.C_E = elec_cost
            # Get costs with Monte Carlo simulation
            MC_COSTS = BR_Vars_Base.get_COST_SET_MC(lang, N, c_elect)
            costs['CAPEX'][index - 1, :] = MC_COSTS['CAPEX']
            costs['OPEX'][index - 1, :] = MC_COSTS['OPEX']
            costs['CAPEX_RAW'][index - 1, :] = MC_COSTS['CAPEX_RAW']
            costs['OPEX_RAW'][index - 1, :] = MC_COSTS['OPEX_RAW']
            # Store configuration
            costs['CONFIG'][index - 1] = f"{elec_name}+{memb_name}"
    return costs

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

def plot_violin(data, title, ylabel, filename, type=None):
    data = np.asarray(data) / 1E+6
    plt.figure(figsize=(6, 6))
    parts = plt.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)
    whiskers = np.array([adjacent_values(sorted_array, q1, q3)
                         for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    plt.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    plt.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    modified_labels = [label.replace('+', '+\n') for label in configurations]
    xtick_positions = np.arange(1, len(modified_labels) + 1)  # Adjust the value as needed

    # Apply the xticks with the modified positions and labels
    plt.xticks(xtick_positions, modified_labels, rotation=60, ha='center', fontsize=18)
    
    # Calculate the minimum and maximum values for y-axis
    y_min, y_max = np.min(data), np.max(data)

    if type == 'Log':
        plt.yscale('log')

        # Use LogLocator to set the ticks in log scale with at least 2 major ticks
        if y_min > 0 and y_max > 0:  # Log scale only works for positive values
            # Calculate the smallest integer power of 10 larger than y_max
            log_max = np.ceil(np.log10(y_max))  # Round up to next power of 10
            log_min = np.floor(np.log10(y_min))  # Round down to previous power of 10
            
            # Set limits for y-axis to match these powers of 10
            plt.gca().set_ylim(y_min*0.7, 10**log_max)

            # Use LogLocator to set the ticks in log scale
            plt.yticks(fontsize=20)
        else:
            print("Warning: Log scale requires all values to be positive.")
    
    else:
        # For linear scale, generate at least 3 ticks between min and max
        if y_min != y_max:
            y_ticks = np.linspace(y_min, y_max, max(2, 5))  # Ensure at least 3 ticks, up to 5
        else:
            y_ticks = [y_min]  # Handle the case where min == max

        plt.yticks(y_ticks, fontsize=20)

    plt.xlabel('Configuration', fontsize=28)
    plt.ylabel(ylabel, fontsize=36)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
#    plt.show()


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def plot_line(N0_range, data, labels, title, xlabel, ylabel, filename, type=None):
    plt.figure(figsize=(12, 6))
    i = 0
    modified_labels = []
    
    for label in labels:
        # Split the label after '+' for the new line in legend
        modified_label = label.replace('+', '+\n')
        modified_labels.append(modified_label)
        
        data_config = data[label]
        data_config = (np.asarray(data_config) / 1E+6).tolist()
        
        plt.plot(N0_range, np.median(data_config, axis=1), label=modified_label, color=custom_palette[i])
        plt.fill_between(N0_range, np.percentile(data_config, 25, axis=1), np.percentile(data_config, 75, axis=1), alpha=0.08, color=custom_palette[i])
        plt.plot(N0_range, np.percentile(data_config, 25, axis=1), color=custom_palette[i], linestyle='--', alpha=0.2)  # Add lower quartile line
        plt.plot(N0_range, np.percentile(data_config, 75, axis=1), color=custom_palette[i], linestyle='--', alpha=0.2)  # Add upper quartile line
        i += 1
    
    plt.xlabel(xlabel, fontsize=36)
    plt.ylabel(ylabel, fontsize=36)
    
    # Set x-ticks at the start and end only
    plt.xticks([N0_range[0], N0_range[-1]], fontsize=20)  # x-ticks at the start and end only
    
    # Calculate the minimum and maximum values for y-axis across all data
    y_min = np.min([np.min(np.percentile(data_config, 25, axis=1)) for data_config in data.values()])/ 1E+6
    y_max = np.max([np.max(np.percentile(data_config, 75, axis=1)) for data_config in data.values()])/ 1E+6

    if type == 'Log':
        plt.yscale('log')

        # Ensure y_min and y_max are positive
        if y_min > 0 and y_max > 0:
            # Calculate the smallest integer power of 10 larger than y_max
            log_max = np.ceil(np.log10(y_max))  # Round up to next power of 10
            log_min = np.floor(np.log10(y_min))  # Round down to previous power of 10
            
            # Set limits for y-axis to match these powers of 10
            plt.gca().set_ylim(y_min*0.7, 10**log_max)

            # Use LogLocator to set the ticks in log scale with automatic subs
            plt.yticks(fontsize=20)
        else:
            print("Warning: Log scale requires all values to be positive.")
    else:
        # For linear scale, generate at least 3 ticks between min and max
        if y_min != y_max:
            y_ticks = generate_constrained_list(y_min,y_max) # Ensure at least 3 ticks, up to 5
        else:
            y_ticks = [y_min]  # Handle the case where min == max

        plt.yticks(y_ticks,fontsize=20)

    # Move the legend outside the plot to the right
    plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')

    #plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(right=0.5)  # Adjust the right margin to make space for the legend
    plt.savefig(filename, dpi=600)
#    plt.show()

# Calculate costs
costs = BioReactMats(BR_Vars_Base, MEMB_COSTS, MEMB_PENALTIES, MEMB_NAMES, ELEC_COSTS, ELEC_V0, ELEC_NAMES, N0, c_elect, lang)
MC_costs = BioReactMats_MC(BR_Vars_Base, MEMB_COSTS, MEMB_PENALTIES, MEMB_NAMES, ELEC_COSTS, ELEC_V0, ELEC_NAMES, N0, c_elect, lang, 0.2)

# Extract data
configurations = MC_costs['CONFIG']
CAPEX_raw_data = np.transpose(MC_costs['CAPEX_RAW'])
OPEX_raw_data = np.transpose(MC_costs['OPEX_RAW'])

# Plant scale range
N0_range = np.arange(500, 2000, 50)

# Initialize lists to store mean CAPEX and quartiles for each configuration
CAPEX_scale_dict = {}
OPEX_scale_dict = {}

# Calculate mean, lower quartile, and upper quartile for CAPEX and OPEX
for config_index, config in enumerate(configurations):
    CAPEX_values = []
    OPEX_values = []
    
    for N0_value in N0_range:
        MC_costs = BioReactMats_MC(BR_Vars_Base, MEMB_COSTS, MEMB_PENALTIES, MEMB_NAMES, ELEC_COSTS, ELEC_V0, ELEC_NAMES, N0_value, c_elect, lang, 0.2)
        CAPEX_values_for_scale = MC_costs['CAPEX_RAW'][config_index]
        OPEX_values_for_scale = MC_costs['OPEX_RAW'][config_index]
        CAPEX = CAPEX_values_for_scale
        OPEX = OPEX_values_for_scale
        
        CAPEX_values.append(CAPEX)
        OPEX_values.append(OPEX)
    
    CAPEX_scale_dict[config] = np.array(CAPEX_values)
    OPEX_scale_dict[config] = np.array(OPEX_values)



# Custom color palette
custom_palette = sns.color_palette("tab10", n_colors=len(configurations))

# Violin plots
plot_violin(CAPEX_raw_data, 'Violin Plot of CAPEX for Each Configuration', 'CAPEX/ M£', Cur_folder+'CAPEX_violin_plot.png',type='Log')
plot_violin(OPEX_raw_data, 'Violin Plot of OPEX for Each Configuration', 'OPEX/ M£/yr', Cur_folder+'OPEX_violin_plot.png',type='Log')

# Line plots
plot_line(N0_range, CAPEX_scale_dict, configurations, 'CAPEX vs. Plant Scale for Each Configuration', 'Plant Scale/ Tonnes/yr)', 'CAPEX/ M£', Cur_folder+'CAPEX_vs_Plant_Scale.png',type='Log')
plot_line(N0_range, OPEX_scale_dict, configurations, 'OPEX vs. Plant Scale for Each Configuration', 'Plant Scale/ Tonnes/yr)', 'OPEX/ M£/yr', Cur_folder+'OPEX_vs_Plant_Scale.png',type='Log')

## PPFD sensitivity 
CAPEX_PPFD_dict = {}
OPEX_PPFD_dict = {}
PPFD_Range = np.linspace(0,1,100)
# Calculate mean, lower quartile, and upper quartile for CAPEX and OPEX
for config_index, config in enumerate(configurations):
    CAPEX_values_PPFD = []
    OPEX_values_PPFD = []
    i = 0
    for PPFD in PPFD_Range:
        N0_value = 1000
        BR_Vars_Base.assign_err(property='PPFD_ENERGY',value=0)
        #BR_Vars_Base.assign_err(value=0)
        BR_Vars_Base.PPFD_ENERGY = PPFD
        MC_costs = BioReactMats_MC(BR_Vars_Base, MEMB_COSTS, MEMB_PENALTIES, MEMB_NAMES, ELEC_COSTS, ELEC_V0, ELEC_NAMES, N0_value, c_elect, lang, 0.2)
        CAPEX_values_for_PPFD = MC_costs['CAPEX_RAW'][config_index]
        OPEX_values_for_PPFD = MC_costs['OPEX_RAW'][config_index]
        CAPEX = CAPEX_values_for_PPFD
        OPEX = OPEX_values_for_PPFD
        CAPEX_values_PPFD.append(CAPEX)
        OPEX_values_PPFD.append(OPEX)
    
    CAPEX_PPFD_dict[config] = np.array(CAPEX_values_PPFD)
    OPEX_PPFD_dict[config] = np.array(OPEX_values_PPFD)

# Line plots
plot_line(PPFD_Range, CAPEX_PPFD_dict, configurations, 'CAPEX vs. PPFD Energy for Each Configuration', 'PPFD/ kW/cum', 'CAPEX/ M£', Cur_folder+'CAPEX_vs_PPFD.png')
plot_line(PPFD_Range, OPEX_PPFD_dict, configurations, 'OPEX vs. PPFD Energy for Each Configuration', 'PPFD/ kW/cum', 'OPEX/ M£/yr', Cur_folder+'OPEX_vs_PPFD.png',type='Log')

