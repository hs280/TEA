import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns


# Define the plot_line function
def plot_line(N0_range, data, labels, title, xlabel, ylabel, filename, type=None):
    custom_palette = sns.color_palette("tab10", n_colors=len(data))
    plt.figure(figsize=(6, 6))
    i = 0
    for label in labels:
        data_config = data[label]
        data_config = (np.asarray(data_config)/1E+6).tolist()
        plt.plot(N0_range, np.median(data_config,axis=1), label=label, color=custom_palette[i])
        plt.fill_between(N0_range,np.percentile(data_config, 25, axis=1), np.percentile(data_config, 75, axis=1), alpha=0.08, color=custom_palette[i])
        plt.plot(N0_range, np.percentile(data_config, 25, axis=1), color=custom_palette[i], linestyle='--', alpha=0.2)  # Add lower quartile line
        plt.plot(N0_range,np.percentile(data_config, 75, axis=1), color=custom_palette[i], linestyle='--', alpha=0.2)  # Add upper quartile line
        i+=1
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    if type =='Log':
        plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.show()

# Define the modified conduct_sensitivity_single_source function
def conduct_sensitivity_single_source(CC_VARS_Base, prop_name, prop_range, C_electric, N0, num_points=100):
    # Store original property value
    try:
        original_value = getattr(CC_VARS_Base, prop_name)
        original_ERR = CC_VARS_Base.ERR.copy()
        CC_VARS_Base.assign_err(prop_name, 0)
    except:
        original_value=None

    # Set properties from the source dictionary
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
        COST_SET = CC_VARS_Base.get_COST_SET_MC(N0, C_electric)
        
        # Store OPEX and CAPEX results
        opex_results.append(COST_SET['OPEX_RAW'])
        capex_results.append(COST_SET['CAPEX_RAW'])
    
    # Reset the property value to its original value
    if original_value != None:
        setattr(CC_VARS_Base, prop_name, original_value)
        CC_VARS_Base.ERR = original_ERR
    
    # Return the results
    return opex_results, capex_results, np.linspace(prop_range[0], prop_range[1], num_points)

import os

def plot_results(prop_values, opex_results, capex_results, prop_name, save_folder):
    plt.figure(figsize=(10, 5))
    plt.plot(prop_values, np.mean(opex_results, axis=1), label='OPEX')
    plt.fill_between(prop_values, np.percentile(opex_results, 25, axis=1), np.percentile(opex_results, 75, axis=1), alpha=0.2)
    plt.xlabel(prop_name)
    plt.ylabel('OPEX')
    plt.title('OPEX vs. {}'.format(prop_name))
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, 'OPEX_vs_{}.png'.format(prop_name)))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(prop_values, np.mean(capex_results, axis=1), label='{} CAPEX'.format(capex_results))
    plt.fill_between(prop_values, np.percentile(capex_results, 25, axis=1), np.percentile(capex_results, 75, axis=1), alpha=0.2)
    plt.xlabel(prop_name)
    plt.ylabel('CAPEX')
    plt.title('CAPEX vs. {}'.format(prop_name))
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, 'CAPEX_vs_{}.png'.format(prop_name)),bbox_inches='tight', dpi=300)
    plt.close()

def conduct_2d_sensitivity_analysis(CLASS, prop_name1, prop_name2, prop_range1, prop_range2, C_electric, N0, num_points=100):
    opex_results = np.zeros((num_points, num_points))
    original_value1 = getattr(CLASS, prop_name1)
    original_value2 = getattr(CLASS, prop_name2)

    for i, prop_value1 in enumerate(np.linspace(prop_range1[0], prop_range1[1], num_points)):
        for j, prop_value2 in enumerate(np.linspace(prop_range2[0], prop_range2[1], num_points)):
            setattr(CLASS, prop_name1, prop_value1)
            setattr(CLASS, prop_name2, prop_value2)
            
            _,tmp_OPEX,_ = CLASS.get_total_costs(N0,C_electric)
            opex_results[i, j] = tmp_OPEX.copy()

    setattr(CLASS, prop_name1,original_value1)
    setattr(CLASS, prop_name2,original_value2)
    
    
    
    return opex_results, np.linspace(prop_range1[0], prop_range1[1], num_points), np.linspace(prop_range2[0], prop_range2[1], num_points)

def plot_2d_results(opex_data, prop_values1, prop_values2, prop_name1, prop_name2, save_folder):
    plt.figure(figsize=(8, 6))
    plt.imshow(opex_data, origin='lower', aspect='auto', extent=[prop_values2[0], prop_values2[-1], prop_values1[0], prop_values1[-1]], cmap='plasma', norm=matplotlib.colors.LogNorm(vmin=opex_data.min(), vmax=opex_data.max()))
    cbar = plt.colorbar(label='OPEX', format='%.0e')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('OPEX', fontsize=12, weight='bold')
    plt.contour(prop_values2, prop_values1, opex_data, levels=np.logspace(np.log10(opex_data.min()), np.log10(opex_data.max()), 10), colors='k', linewidths=1.5)
    plt.xlabel(prop_name2, fontsize=14, weight='bold')
    plt.ylabel(prop_name1, fontsize=14, weight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(os.path.join(save_folder, 'OPEX_colormap_{}_vs_{}.png'.format(prop_name1, prop_name2)), bbox_inches='tight', dpi=300)
    plt.close()


# Example usage
import sys

sys.path.append('/home/hs280/TEA/Bin')
import BioSepSubSys as CCSS

SS_Vars_Base = CCSS.SeparationSubSys()

C_electric = 0.28

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
SS_Vars_Base.yield_ = [0.1,0.7,0.2]  # yield of product type list sum ==1
SS_Vars_Base.value = [25E+6,1.5E+6,7.5E+3]  # Value £/MMol product type list 
SS_Vars_Base.trials = 5000  # Num trials for MC
SS_Vars_Base.ERR = 0.2 # Base Param err

# Choices
N0 = 1000
SS_Vars_Base.assign_err(value=0.5)

opex_results_all, capex_results_all, prop_values = conduct_sensitivity_single_source(SS_Vars_Base, 'nu_2nd', [0.01, 0.2], C_electric,N0)
plot_results(prop_values, opex_results_all, capex_results_all, 'nu_2nd','/home/hs280/TEA/Test/BS_Tests')

opex_results_all, capex_results_all, prop_values = conduct_sensitivity_single_source(SS_Vars_Base, 'mass_fraction', [0.001, 0.01], C_electric,N0)
plot_results(prop_values, opex_results_all, capex_results_all, 'mass_fraction','/home/hs280/TEA/Test/BS_Tests')

opex_results_all, prop_values1, prop_values2 = conduct_2d_sensitivity_analysis(SS_Vars_Base, 'mass_fraction', 'nu_2nd', [0.01, 0.5], [0.001, 0.2], C_electric, N0,)
plot_2d_results(opex_results_all, prop_values1, prop_values2, 'mass_fraction', 'nu_2nd', '/home/hs280/TEA/Test/BS_Tests')

import numpy as np
import plotly.figure_factory as ff

def generate_ternary_contour(CLASS, C_electric, N0,save_path):
    # Generate lists of yield values for each product
    prod1_yields = np.linspace(0,1, 10)
    prod2_yields = np.linspace(0,1, 10)

    # Initialize arrays to store results
    capex = []
    opex = []
    revenue = []
    payback_time = []
    y1 = []
    y2 = []
    y3 = []

    # Iterate through all combinations of yield values
    for yield1 in prod1_yields:
        for yield2 in prod2_yields:
            # Ensure the sum of yields is equal to 1
            yield3 = max(1-(yield1 + yield2),0)
            total_yield = yield1 + yield2 + yield3
            yield_vals = [yield1 / total_yield, yield2 / total_yield, yield3 / total_yield]

            # Set the yield values for the current iteration
            setattr(CLASS, 'yield_', yield_vals)

            # Get total costs
            CAPEX, OPEX, REVENUE = CLASS.get_total_costs(N0, C_electric)

            # Store OPEX, CAPEX, and GROSS results
            capex.append(CAPEX)
            opex.append(OPEX)
            revenue.append(REVENUE)
            y1.append(yield_vals[0])
            y2.append(yield_vals[1])
            y3.append(yield_vals[2])

            # Calculate PAYBACK TIME
            payback_time.append(CAPEX / max(1e-9, REVENUE - OPEX))

    # Convert lists to numpy arrays
    capex = np.array(capex)
    opex = np.array(opex)
    revenue = np.array(revenue)
    payback_time = np.array(payback_time)

    plot_ternary(y1,y2,y3,capex,'capex',save_path)
    plot_ternary(y1,y2,y3,opex,'opex',save_path)
    plot_ternary(y1,y2,y3,revenue,'revenue',save_path)
    #plot_ternary(y1,y2,y3,np.log10(np.log10(payback_time)),'payback_time',save_path)

import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

def plot_ternary(y1, y2, y3, param, param_name, save_path):
    # Create ternary contour plot
    fig = ff.create_ternary_contour(
        np.array([y1, y2, y3]), param,
        pole_labels=['RiboFlavin', 'PHB', 'BIOMASS'],
        interp_mode='cartesian',
        ncontours=20,
        colorscale='Viridis',
        showscale=True,
        title='Ternary Contour Plot - ' + param_name
    )

    # Calculate colorbar ticks: only min and max
    param_min = np.min(param)
    param_max = np.max(param)

    # Ensure ticks are integers
    tickvals = [int(round(param_min)), int(round(param_max))]

    # Update layout for font size, tick positions, figure size, and colorbar
    fig.update_layout(
        ternary=dict(
            aaxis=dict(
                title=dict(font=dict(size=30)),  # Pole label size
                tickvals=[0.5, 1],  # Ticks at 0.5 and 1
                tickfont=dict(size=30)  # Tick font size
            ),
            baxis=dict(
                title=dict(font=dict(size=30)),  # Pole label size
                tickvals=[0.5, 1],  # Ticks at 0.5 and 1
                tickfont=dict(size=30)  # Tick font size
            ),
            caxis=dict(
                title=dict(font=dict(size=30)),  # Pole label size
                tickvals=[0.5, 1],  # Ticks at 0.5 and 1
                tickfont=dict(size=30)  # Tick font size
            )
        ),
        title=dict(font=dict(size=10)),  # Title font size
        height=600,  # 6 inches (height in pixels)
        width=600,  # 6 inches (width in pixels)
        coloraxis=dict(
            colorbar=dict(
                x=1.5,  # Move colorbar to the right
                tickvals=tickvals,  # Only min and max ticks
                ticktext=[str(val) for val in tickvals],  # Convert to strings for display
                tickfont=dict(size=50),  # Tick font size
                title=dict(text=param_name, font=dict(size=20))  # Title font size
            )
        )
    )

    # Save plot
    fig.write_image(save_path + '/prod_' + param_name + '.png')


# Example usage
generate_ternary_contour(SS_Vars_Base, C_electric, N0,'/home/hs280/TEA/Test/BS_Tests')

import numpy as np
import matplotlib.pyplot as plt 
import os



# Define Yield cases
Cases = {}
Cases['80/0/20'] = [0.80, 0, 0.20]
Cases['0/80/20'] = [0, 0.80, 0.20]
Cases['40/40/20'] = [0.40, 0.40, 0.20]
Cases['20/0/80'] = [0.2,0,0.8]
Cases['0/0/100'] = [0, 0, 1]

# Perform sensitivity analysis for each Yield case
opex_results_all = {}
capex_results_all = {}

N0_range = [100,1000] # Adjust range and step as needed

for case_name, yield_values in Cases.items():
    print(SS_Vars_Base.nu_2nd)
    opex_results_list = []
    capex_results_list = []
    SS_Vars_Base.yield_ = yield_values
    opex_results, capex_results, N0_vals = conduct_sensitivity_single_source(SS_Vars_Base, 'N0', N0_range, C_electric, N0,)


    opex_results_all[case_name] = opex_results
    capex_results_all[case_name] = capex_results

# Plot OPEX and CAPEX vs. Plant Scale for all Yield cases
plot_line(N0_vals, opex_results_all, Cases.keys(), 'OPEX', 'Plant Scale/ Tonnes/yr', 'OPEX/ M£/yr', '/home/hs280/TEA/Test/BS_Tests/OP_vs_N0.png')
plot_line(N0_vals, capex_results_all, Cases.keys(), 'CAPEX', 'Plant Scale/ Tonnes/yr', 'CAPEX/ M£', '/home/hs280/TEA/Test/BS_Tests/CA_vs_N0.png')