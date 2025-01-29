import numpy as np
import scipy.stats as stats


def lognormal_dist():
    # Step 1: Create truncated normal distribution with mu = 0, sigma = 0.5, truncated at +/- 3*sigma
    base_dist = stats.truncnorm(-3, 3, loc=0, scale=0.5)
    
    # Step 2: Generate samples from the truncated normal distribution
    def transform():
        # Generate a random value from the truncated normal distribution
        sample = base_dist.rvs()
        
        # Step 3: Apply the log-normal transformation (exponentiate the value)
        log_normal_value = np.exp(sample)
        
        # Step 4: Shift and scale the value to be between -1 and 1
        # We shift and scale the distribution to fit within [-1, 1]
        scaled_value = 2 * (log_normal_value-np.exp(-0.25))/(np.exp(1.5)-np.exp(-1.5))
        return scaled_value

    return transform

class BioReactorSubSys:
    def __init__(self):
        # Reactor Specific Volume
        self.v0 = None  
        # Reactor cost Const
        self.a = None   
        # Reactor cost pre-exp
        self.b = None   
        # Reactor Cost Exp
        self.n = None   
        # Membrane Area
        self.A_m = None  
        # Membrane Cost
        self.C_m = None  
        # Electrode area
        self.A_E = None  
        # Electrode Cost
        self.C_E = None  
        # Convective Heat Loss
        self.H_air = None  
        # Temperature Above Ambient
        self.del_T = None  
        # Bioreactor Height/m
        self.D = None  
        # Insolation
        self.Is = None  
        # Energy Usage kWh/tonne
        self.E0 = None  
        # Energy required for light #kW/m2
        self.PPFD_ENERGY = None
        # LED Cost
        self.LED_Cost = None
        # Penalty Factor
        self.penalty = None  
        # Num trials for MC
        self.trials = 1000  
        # Error value
        self.ERR = 0.2

        self.dist_dict = {
            # Reactor Specific Volume (truncated normal)
            "v0": lambda: stats.truncnorm(-1, 1, loc=0, scale=1/3).rvs(),  
            
            # Reactor cost constants (truncated normal)
            "a": lognormal_dist(),   
            "b": lognormal_dist(),   
            "n": lambda: stats.truncnorm(-1, 1, loc=0, scale=1/3).rvs(),  
            
            # Membrane properties (uniform for area and cost)
            "A_m": lambda: stats.uniform(-1, 2).rvs(),  # Uniform [-1, 1], range = 2
            "C_m": lambda: stats.uniform(-1, 2).rvs(),  # Uniform [-1, 1], range = 2
            
            # Electrode properties
            "A_E": lambda: stats.truncnorm(-1, 1, loc=0, scale=1/3).rvs(),  # Truncated normal
            "C_E": lambda: stats.truncnorm(-1, 1, loc=0, scale=1/3).rvs(),  # Truncated normal
            
            # Heat and temperature (truncated normal)
            "H_air": lambda: stats.truncnorm(-1, 1, loc=0, scale=1/3).rvs(),  
            "del_T": lambda: stats.truncnorm(-1, 1, loc=0, scale=1/3).rvs(),    
            
            # Bioreactor dimensions (uniform for height)
            "D": lambda: stats.uniform(-1, 2).rvs(),  # Uniform [-1, 1], range = 2
            
            # Environmental parameters (truncated normal for insolation)
            "Is": lambda: stats.truncnorm(-1, 1, loc=0, scale=1/3).rvs(),  
            
            # Energy usage
            "E0": lognormal_dist(),  # Truncated normal
            "PPFD_ENERGY": lambda: stats.truncnorm(-1, 1, loc=0, scale=1/3).rvs(),  
            
            # Costs (log-normal for LED costs, reflecting potential skew)
            "LED_Cost": lognormal_dist(),  
            
            # Penalty factor (truncated normal, small variability)
            "penalty": lambda: stats.truncnorm(-1, 1, loc=0, scale=1/3).rvs(),  
            
            # Monte Carlo trials and error (fixed values, no distributions)
            "trials": None,
            "ERR": None,
        }

    def get_COST_SET_MC(self, lang, CO2_conversion_rate, C_electric):
        """
        Calculate the cost with Monte Carlo simulation.
        Args:
            lang (float): Language variable.
            CO2_conversion_rate (float): CO2 conversion rate in tonnes per year.
            C_electric (float): Electrical cost.
        Returns:
            dict: Dictionary containing calculated costs.
        """
        BASE_OPEX = self.get_operating_cost(CO2_conversion_rate, C_electric)
        BASE_CAPEX = self.get_total_CAPEX(lang, CO2_conversion_rate)
        MAX_OPEX = BASE_OPEX
        MIN_OPEX = BASE_OPEX
        SUM_OPEX = BASE_OPEX
        MAX_CAPEX = BASE_CAPEX
        MIN_CAPEX = BASE_CAPEX
        SUM_CAPEX = BASE_CAPEX
        CAPEX_RAW = [BASE_CAPEX]
        OPEX_RAW = [BASE_OPEX]

        for i in range(self.trials):
            obj_p = self.perturb_obj()
            TEMP_OPEX = obj_p.get_operating_cost(CO2_conversion_rate, C_electric)
            TEMP_CAPEX = obj_p.get_total_CAPEX(lang, CO2_conversion_rate)

            MAX_CAPEX = max(MAX_CAPEX, TEMP_CAPEX)
            MIN_CAPEX = min(MIN_CAPEX, TEMP_CAPEX)
            SUM_CAPEX += TEMP_CAPEX
            MAX_OPEX = max(MAX_OPEX, TEMP_OPEX)
            MIN_OPEX = min(MIN_OPEX, TEMP_OPEX)
            SUM_OPEX += TEMP_OPEX
            OPEX_RAW.append(TEMP_OPEX)
            CAPEX_RAW.append(TEMP_CAPEX)

        COST_SET = {
            'CAPEX': [BASE_CAPEX, MAX_CAPEX, MIN_CAPEX, SUM_CAPEX / (self.trials + 1)],
            'OPEX': [BASE_OPEX, MAX_OPEX, MIN_OPEX, SUM_OPEX / (self.trials + 1)],
            'CAPEX_RAW': CAPEX_RAW,
            'OPEX_RAW': OPEX_RAW
        }
        return COST_SET

    def perturb_obj(self):
        """
        Perturb the object properties for Monte Carlo simulation.
        Returns:
            BioReactorSubSys: Object with perturbed properties.
        """
        obj_p = BioReactorSubSys()
        i = 0
        for prop in self.__dict__:
            if prop not in ['trials', 'ERR','dist_dict']:
                setattr(obj_p, prop, getattr(self, prop) * (1 + self.dist_dict[prop]()* self.ERR[i]))
        return obj_p

    def assign_err(self, property=None, value=0):
        """
        Assign error to a specific property.
        Args:
            property (str): Name of the property.
            value (float): Error value.
        """
        if property==None:
            self.ERR = [value] * (len(vars(self)) - 2)
        else:
            index = None
            props = vars(self).keys()
            if property in props:
                index = list(props).index(property)

            if index is not None:
                self.ERR[index] = value

    # Methods for Capital Cost Calculation
    def get_total_CAPEX(self, lang, CO2_conversion_rate):
        """
        Calculate the total capital cost.
        Args:
            lang (float): Language variable.
            CO2_conversion_rate (float): CO2 conversion rate in tonnes per year.
        Returns:
            float: Total capital cost.
        """
        LED_Cost = self.LED_Cost*(self.get_LED_Duty(CO2_conversion_rate)/(5*8.760))**(0.6)
        return lang * (self.get_total_reactor_cost(CO2_conversion_rate)+LED_Cost)

    def get_total_reactor_cost(self, CO2_conversion_rate):
        """
        Calculate the total reactor cost.
        Args:
            CO2_conversion_rate (float): CO2 conversion rate in tonnes per year.
        Returns:
            float: Total reactor cost.
        """
        glass_cost = self.get_reactor_chamber_cost(CO2_conversion_rate)
        Membrane_Cost = self.get_material_cost(self.A_m, CO2_conversion_rate, self.C_m)
        Electrode_Cost = self.get_material_cost(self.A_E, CO2_conversion_rate, self.C_E)
        total_cost = glass_cost + Membrane_Cost + Electrode_Cost
        return total_cost

    def get_reactor_chamber_cost(self, CO2_conversion_rate):
        """
        Calculate the reactor chamber cost.
        Args:
            CO2_conversion_rate (float): CO2 conversion rate in tonnes per year.
        Returns:
            float: Reactor chamber cost.
        """
        V = self.v0 * CO2_conversion_rate
        cost = self.a + self.b * V ** self.n
        return cost

    def get_material_cost(self, A_s, CO2_conversion_rate, C):
        """
        Calculate the material cost.
        Args:
            A_s (float): Surface area.
            CO2_conversion_rate (float): CO2 conversion rate in tonnes per year.
            C (float): Cost factor.
        Returns:
            float: Material cost.
        """
        Material_Cost = A_s * C * CO2_conversion_rate ** (0.6)
        return Material_Cost

    # Methods for OPEX Calculation
    def get_operating_cost(self, CO2_conversion_rate, C_electric):
        """
        Calculate the operating cost.
        Args:
            CO2_conversion_rate (float): CO2 conversion rate in tonnes per year.
            C_electric (float): Electrical cost.
        Returns:
            float: Operating cost.
        """
        heat_duty = self.get_heat_energy(CO2_conversion_rate)
        electrical_duty = self.get_electrical_duty(CO2_conversion_rate)
        LED_duty = self.get_LED_Duty(CO2_conversion_rate)
        total_duty = electrical_duty + heat_duty +LED_duty
        total_cost = C_electric * total_duty
        return total_cost

    def get_heat_energy(self, CO2_conversion_rate):
        """
        Calculate the heat energy.
        Args:
            CO2_conversion_rate (float): CO2 conversion rate in tonnes per year.
        Returns:
            float: Heat energy.
        """
        V = self.v0 * CO2_conversion_rate
        A_top = V / self.D
        A_sides = 4 * (V * CO2_conversion_rate) ** (0.5)
        heat_in = self.Is * A_top
        heat_out = self.H_air * self.del_T * (A_top + A_sides)
        heat_duty = max((heat_out - heat_in), 0)
        return heat_duty

    def get_electrical_duty(self, CO2_conversion_rate):
        """
        Calculate the electrical duty.
        Args:
            CO2_conversion_rate (float): CO2 conversion rate in tonnes per year.
        Returns:
            float: Electrical duty.
        """
        electrical_duty = CO2_conversion_rate * self.E0 * (1 / (1 - self.penalty))
        return electrical_duty
    
    def get_LED_Duty(self,CO2_conversion_rate):
         
        V_react = self.v0*CO2_conversion_rate
        A_react = V_react/self.D
        attenuation_factor = (0.45)**(np.min((A_react,self.D)))
        return 8760*CO2_conversion_rate*self.PPFD_ENERGY/attenuation_factor


