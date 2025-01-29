import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import copy 

def conduct_sensitivity_single_source(CC_VARS_Base, prop_name, prop_range, lang, C_electric, N0, source, num_points=100):
    # Store original property value
    try:
        original_value = getattr(CC_VARS_Base, prop_name)
        original_ERR = CC_VARS_Base.ERR.copy()
        CC_VARS_Base.assign_err(prop_name, 0)
    except:
        original_value=None



    # Set properties from the source dictionary
    for key, value in source.items():
        if key != 'NAMES':
            setattr(CC_VARS_Base, key, value)
    
    # Initialize lists to store results
    opex_results = []
    capex_results = []
    
    # Vary the property within the specified range
    for prop_value in np.linspace(prop_range[0], prop_range[1], num_points):
        # Set the property value
        if original_value != None:
            setattr(CC_VARS_Base, prop_name, prop_value)
        else:
            N0 = prop_value

        
        # Perform Monte Carlo analysis
        COST_SET = CC_VARS_Base.get_COST_SET_MC(lang, N0)
        
        # Store OPEX and CAPEX results
        opex_results.append(COST_SET['OPEX_RAW'])
        capex_results.append(COST_SET['CAPEX_RAW'])
    
    # Reset the property value to its original value
    if original_value != None:
        setattr(CC_VARS_Base, prop_name, original_value)
        CC_VARS_Base.ERR = original_ERR
    # Return the results
    return opex_results, capex_results

def conduct_sensitivity_all_sources(CC_VARS_Base, prop_name, prop_range, lang, C_electric, N0, source_dict, num_points=100):
    opex_results_all = {}
    capex_results_all = {}
    
    n = len(source_dict[list(source_dict.keys())[0]])  # Get the length of lists
    for i in range(n):
        source_params = {key: value[i] for key, value in source_dict.items()}  # Extract i'th value of each entry
        source_name = source_params['NAMES']
        opex_results, capex_results = conduct_sensitivity_single_source(CC_VARS_Base, prop_name, prop_range, lang, C_electric, N0, source_params, num_points)
        opex_results_all[source_name] = opex_results
        capex_results_all[source_name] = capex_results
    
    return opex_results_all, capex_results_all, np.linspace(prop_range[0], prop_range[1], num_points)

import os
import pickle

def plot_results(prop_values, opex_results, capex_results, prop_name, save_folder):
    # Define the filename for saving the plots and the arguments
    base_filename = os.path.join(save_folder, 'Results_vs_{}.pkl'.format(prop_name.replace('/', '')))
    
    # Save input arguments to a .pkl file
    with open(base_filename, 'wb') as f:
        pickle.dump({
            'prop_values': prop_values,
            'opex_results': opex_results,
            'capex_results': capex_results,
            'prop_name': prop_name,
            'save_folder': save_folder
        }, f)

    clean_prop_name  =(prop_name
                      .replace('/', '')
                      .replace('$', '')
                      .replace('{', '')
                      .replace('}', '')
                      .replace('\\', ''))
    
    # Plot OPEX Results
    xlabel = prop_name
    plt.figure(figsize=(10, 5))  # Twice as wide as default
    for source_name, opex_data in opex_results.items():
        opex_data = (np.asarray(opex_data) / 1E+6).tolist()
        plt.plot(prop_values, np.mean(opex_data, axis=1), label='{} OPEX'.format(source_name))
        plt.fill_between(prop_values, np.percentile(opex_data, 25, axis=1), np.percentile(opex_data, 75, axis=1), alpha=0.2)
    
    # Set x-axis ticks at start and end only
    x_start, x_end = min(prop_values), max(prop_values)
    x_ticks = [x_start, x_end] if x_start != x_end else [x_start]
    plt.xticks(x_ticks, fontsize=20)
    
    # Set y-axis ticks at start and end only with integer spacing
    opex_min = np.min([(np.min([np.percentile(data,25) for data in opex_results[source]])) for source in opex_results])/1E+6
    opex_max = np.max([(np.max([np.percentile(data,75) for data in opex_results[source]])) for source in opex_results])/1E+6
    opex_min = min([opex_min,0])
    y_start, y_end = np.floor(opex_min), np.ceil(opex_max)
    y_ticks = [y_start, y_end] if y_start != y_end else [y_start]
    plt.yticks(y_ticks, fontsize=20)
    
    plt.xlabel(xlabel, fontsize=36)
    plt.ylabel('BASE OPEX/ M£/yr', fontsize=36)
    
    # Move the legend to the right half of the figure
    plt.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))  # Right half, centered vertically
    
    plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(right=0.5)  # Leave space on the right for the legend
    plt.savefig(os.path.join(save_folder, f'OPEX_vs_{clean_prop_name}.png'), dpi=600)
    plt.close()
    
    # Plot CAPEX Results
    plt.figure(figsize=(10, 5))  # Twice as wide as default
    for source_name, capex_data in capex_results.items():
        capex_data = (np.asarray(capex_data) / 1E+6).tolist()
        plt.plot(prop_values, np.mean(capex_data, axis=1), label='{} CAPEX'.format(source_name))
        plt.fill_between(prop_values, np.percentile(capex_data, 25, axis=1), np.percentile(capex_data, 75, axis=1), alpha=0.2)
    
    # Set x-axis ticks at start and end only
    plt.xticks(x_ticks, fontsize=20)
    
    # Set y-axis ticks at start and end only with integer spacing
    capex_min = np.min([(np.min([np.percentile(data,25) for data in capex_results[source]])) for source in capex_results])/1E+6
    capex_max = np.max([(np.max([np.percentile(data,75) for data in capex_results[source]])) for source in capex_results])/1E+6
    capex_min = min([capex_min,0])
    y_start, y_end = np.floor(capex_min), np.ceil(capex_max)
    y_ticks = [y_start, y_end] if y_start != y_end else [y_start]
    plt.yticks(y_ticks, fontsize=20)
    
    plt.xlabel(xlabel, fontsize=36)
    plt.ylabel('CAPEX/ M£', fontsize=36)
    
    # Move the legend to the right half of the figure
    plt.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))  # Right half, centered vertically
    
    plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(right=0.5)  # Leave space on the right for the legend
    plt.savefig(os.path.join(save_folder, f'CAPEX_vs_{clean_prop_name}.png'), dpi=600)
    plt.close()

def conduct_sensitivity_plant_size(CC_VARS_Base, plant_size_range, lang, C_electric, source_dict, num_points=100):
    
    opex_results, capex_results, prop_values = conduct_sensitivity_all_sources(CC_VARS_Base, 'N0', plant_size_range, lang, C_electric, N0, source_dict, num_points)
    
    return opex_results, capex_results, prop_values

def conduct_2d_sensitivity_analysis(CC_VARS_Base, prop_name1, prop_name2, prop_range1, prop_range2, lang, C_electric, N0, source_dict, num_points=100):
    opex_results_all = {}
    capex_results_all = {}
    n = len(source_dict[list(source_dict.keys())[0]])  # Get the length of lists
    for i in range(n):
        source_params = {key: value[i] for key, value in source_dict.items()}  # Extract i'th value of each entry
        opex_results = np.zeros((num_points, num_points))
        capex_results = np.zeros((num_points, num_points))
            # Set properties from the source dictionary
        for key, value in source_params.items():
            if key != 'NAMES':
                setattr(CC_VARS_Base, key, value)
        source_name = source_params['NAMES']
        for i, prop_value1 in enumerate(np.linspace(prop_range1[0], prop_range1[1], num_points)):
            for j, prop_value2 in enumerate(np.linspace(prop_range2[0], prop_range2[1], num_points)):
                setattr(CC_VARS_Base, prop_name1, prop_value1)
                setattr(CC_VARS_Base, prop_name2, prop_value2)
                
                COST_SET = CC_VARS_Base.get_COST_SET(lang, N0)
                opex_results[i, j] = np.mean(COST_SET['OPEX'])
                capex_results[i,j] = np.mean(COST_SET['CAPEX'])

        
        opex_results_all[source_name] = opex_results
        capex_results_all[source_name] = capex_results
    
    return opex_results_all, capex_results_all,  np.linspace(prop_range1[0], prop_range1[1], num_points), np.linspace(prop_range2[0], prop_range2[1], num_points)

def plot_2d_results(results, prop_values1, prop_values2, prop_name1, prop_name2,z_name,save_folder):
    for source_name, data in results.items():
        data = (np.asarray(data)/1E+6)
        plt.figure(figsize=(6, 5))
        #plt.rcParams.update({
        #    "text.usetex": True,
        #})
        plt.imshow(data, origin='lower', aspect='auto', extent=[prop_values2[0], prop_values2[-1], prop_values1[0], prop_values1[-1]], cmap='plasma', norm=matplotlib.colors.LogNorm(vmin=data.min(), vmax=data.max()))
        cbar = plt.colorbar(label=z_name, format='%.0e', pad=0.02)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(z_name, fontsize=18, weight='bold')
        plt.contour(prop_values2, prop_values1, data, levels=np.logspace(np.log10(data.min()), np.log10(data.max()), 10), colors='k', linewidths=1.5)
        plt.xlabel(prop_name2, fontsize=18, weight='bold')
        plt.ylabel(prop_name1, fontsize=18, weight='bold')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)


        prop_label1 = clean_latex_string(prop_name1)
        prop_label2 = clean_latex_string(prop_name2)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, z_name.replace('/','')+'_colormap_{}_{}_vs_{}.png'.format(source_name, prop_label1, prop_label2)), bbox_inches='tight', dpi=600)
        plt.close()

def clean_latex_string(latex_string):
    if latex_string.startswith(r'$'):
        processed_string = latex_string[2:-1].replace('\\', '')
        processed_string = processed_string.replace('/','')
    else:
        processed_string = latex_string

    return processed_string

# Example usage
import sys

sys.path.append('/home/hs280/TEA/Bin')
import CarbonCaptureSubSys as CCSS

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
CC_VARS_Base.ERR = 0.5 # Base Param err
CC_VARS_Base.p = .95 #Outlet purity 
CC_VARS_Base.c_elect = 0.28

# Assign errors to variables
ERR_array = [0, 0, 0, 0.5, 0.2]
ERR_names = ['E0_CO2', 'x0', 'p', 'C0', 'n']
CC_VARS_Base.assign_err(value=CC_VARS_Base.ERR)
CC_VARS_Base.assign_err(ERR_names, ERR_array)

# Design Variables
Source = {}
Source['x0'] = [480E-6, 0.15]
Source['NAMES'] = ["DAC", "Flue"]
Source['C0'] = [8.5724E+6, 8.5724E+6]
N0 = 1000

opex_results_all, capex_results_all, prop_values = conduct_sensitivity_all_sources(CC_VARS_Base, 'p', [0.05, 0.99], lang, c_elect,N0, Source)
plot_results(prop_values, opex_results_all, capex_results_all, 'Required Purity','/home/hs280/TEA/Test/CC_Tests')

opex_results_all, capex_results_all, prop_values = conduct_sensitivity_all_sources(CC_VARS_Base, 'nu_2nd', [0.05, 0.99], lang, c_elect,N0, Source)
plot_results(prop_values, opex_results_all, capex_results_all, r'$\eta_{2nd}$','/home/hs280/TEA/Test/CC_Tests')

opex_results_all, capex_results_all, prop_values = conduct_sensitivity_all_sources(CC_VARS_Base, 'nu_cap', [0.05, 0.99], lang, c_elect,N0, Source)
plot_results(prop_values, opex_results_all, capex_results_all, r'$\eta_{cap}$','/home/hs280/TEA/Test/CC_Tests')

opex_results_all, capex_results_all, prop_values = conduct_sensitivity_all_sources(CC_VARS_Base, 'N0', [1, 5000], lang, c_elect,N0, Source)
plot_results(prop_values, opex_results_all, capex_results_all,'Scale/ Tonnes/ yr','/home/hs280/TEA/Test/CC_Tests')

opex_results_all, capex_results_all, prop_values1, prop_values2 = conduct_2d_sensitivity_analysis(CC_VARS_Base, 'p', 'nu_2nd', [0.01, 0.95], [0.01, 0.95], lang, c_elect, N0, Source)
plot_2d_results(opex_results_all, prop_values1, prop_values2, 'Required Purity', r'$\eta_{2nd}$','BASE OPEX/ M£/yr','/home/hs280/TEA/Test/CC_Tests')
plot_2d_results(capex_results_all, prop_values1, prop_values2, 'Required Purity', r'$\eta_{2nd}$','CAPEX/ M£', '/home/hs280/TEA/Test/CC_Tests')
