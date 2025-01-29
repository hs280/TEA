import numpy as np

class CarbonCaptureSubSys:
    def __init__(self):
        # Initialize properties with default values
        self.x0 = None              # Source conc
        self.p = None               # Reactor purity req.
        self.a = None               # Pump Pre-Exponential Const/£
        self.b = None               # Pump Pre-Exponential Factor/£
        self.n = None               # Pump Exponent
        self.E0_CO2 = None          # Ideal Gas const*T kWh/Tonne
        self.C0 = None              # 1 Tonne reference Plant cost
        self.Pressure = None        # Pumping Presure Req./kPa
        self.nu_pump = None         # Pump Efficiencey
        self.nu_conv = None         # Reactor Conversion
        self.nu_cap = None          # Capture Efficiencey
        self.nu_2nd = None          # 2nd Law efficiencey
        self.c_elect = None         # Cost of electricity £/kWh
        self.trials = 100           # Num trials for MC
        self.ERR = 1E-9             # Base Param err

    # Monte Carlo Style Cost Assessment
    def get_COST_SET_MC(self, lang, N):
        BASE_OPEX = self.get_operating_cost(N)
        BASE_CAPEX = self.get_total_CAPEX(lang, N)
        MAX_OPEX = BASE_OPEX
        MIN_OPEX = BASE_OPEX
        SUM_OPEX = BASE_OPEX
        MAX_CAPEX = BASE_CAPEX
        MIN_CAPEX = BASE_CAPEX
        SUM_CAPEX = BASE_CAPEX
        CAPEX_RAW = [BASE_CAPEX]
        OPEX_RAW = [BASE_OPEX]
        for _ in range(self.trials):
            obj_p = self.perturb_obj()
            TEMP_OPEX = obj_p.get_operating_cost(N)
            TEMP_CAPEX = obj_p.get_total_CAPEX(lang, N)
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
    
    #Non MC Costassement 
    def get_COST_SET(self, lang, N):
        BASE_OPEX = self.get_operating_cost(N)
        BASE_CAPEX = self.get_total_CAPEX(lang, N)
        COST_SET = {
            'CAPEX': [BASE_CAPEX],
            'OPEX': [BASE_OPEX],
        }
        return COST_SET
    

    def perturb_obj(self):
        """
        Perturb the object properties for Monte Carlo simulation.
        Returns:
            BioReactorSubSys: Object with perturbed properties.
        """
        obj_p = CarbonCaptureSubSys()
        i = 0
        for prop in self.__dict__:
            if prop not in ['trials', 'ERR']:
                setattr(obj_p, prop, getattr(self, prop) * (1 + (np.random.rand() - 0.5) * 2 * self.ERR[i]))
                i+=1
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
        elif type(property) is list:
            for i in range(len(property)):
                prop = property[i]
                val = value[i]
                self.assign_err(property=prop, value=val)
        else:
            index = None
            props = vars(self).keys()
            if property in props:
                index = list(props).index(property)

            if index is not None:
                self.ERR[index] = value

    # Capital Cost Methods
    def get_total_CAPEX(self, lang, N):
        pump_capital = lang * self.get_pump_capital(N)
        if self.p < self.x0:
            Separation_capital = 0
        else:
            Separation_capital = self.C0*max((N*self.get_energy()/8760E+3),0)**(0.55) #self.get_sep_cap(N)
        capital_cost = pump_capital + Separation_capital
        return capital_cost

    def get_pump_capital(self, N):
        Power = N * self.get_pump_energy() / (365 * 24)
        pump_capital = self.a + self.b * Power ** self.n
        pump_capital = max(pump_capital, 1000 * self.Pressure ** 0.6)
        return pump_capital

    def get_sep_cap(self, N):
        Separation_capital = self.C0 * N ** 0.6 * ((self.p - self.x0)) ** 0.6
        return Separation_capital

    # OPEX METHODS
    def get_operating_cost(self, N):
        E = self.get_total_energy()
        total_cost = N * self.c_elect * E
        return total_cost

    def get_total_energy(self):
        if self.x0 > self.p:
            E_CC = 0
        else:
            E_CC = self.get_energy()
        E_Pump = self.get_pump_energy()
        E = E_CC + E_Pump
        return E

    def get_pump_energy(self):
        E_min = self.get_min_pump_energy()
        E = E_min / (self.nu_pump * self.nu_conv)
        return E

    def get_min_pump_energy(self):
        mass_conc = 44 * self.p / (44 * self.p + 29 * (1 - self.p))
        mass_flow = 1000 / mass_conc
        density = 1.225 * (1.57 * self.p + (1 - self.p))
        V = mass_flow / density
        E_min = self.Pressure * V / (60 ** 2)
        return E_min

    def get_energy(self):
        E_min = self.get_min_energy()
        E = E_min / self.nu_2nd
        return E

    def get_min_energy(self):
        x2 = self.p
        N2 = self.x0 * self.nu_cap / (self.p)
        a2 = self.get_binary_mix(x2)

        N3 = 1 - N2
        x3 = self.x0 * (1 - self.nu_cap) / N3
        a3 = self.get_binary_mix(x3)

        E_min = self.E0_CO2 * (N2 * a2 + N3 * a3) / (N2 * x2)
        return E_min

    def get_binary_mix(self, x):
        a = self.get_fact(x, self.x0) + self.get_fact(1 - x, 1 - self.x0)
        return a

    def get_fact(self, x, x0):
        x = min(max(x,1E-9),1-1E-9)
        x0 = min(max(x0,1E-9),1-1E-9)
        alpha = x * np.log(x / x0)
        return alpha
