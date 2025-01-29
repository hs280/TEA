import numpy as np

class SeparationSubSys:
    def __init__(self):
        self.C0 = None  # ref.capital /unit kWh/yr
        self.Kp = None  # kg CO2/kg product
        self.E0 = None  # refrence energy in kWh/mol
        self.Mr_product = None  # Product Molecular Mass type list
        self.Mr_solvent = None  # Solvent Molecular Mass type list
        self.mass_fraction = None  # Mass fraction of product type list 
        self.p = None  # desired product purity type list 
        self.nu_cap = None  # capture efficiencey of product
        self.nu_2nd = None  # 2nd Law efficenciey
        self.yield_ = None  # yield of product type list sum ==1
        self.value = None  # Value /kg product type list 
        self.trials = 100  # Num trials for MC
        self.ERR = 0  # Base Param err

    def get_COST_SET_MC(self, N, C_electric):
        CAPEX_N, OPEX_N, REV_N = self.get_total_costs(N,C_electric)

        MIN_OPEX, MAX_OPEX, OPEX_RAW, SUM_OPEX = self.evaluate_params(OPEX_N, np.inf, 0, [], 0)
        MIN_CAPEX, MAX_CAPEX, CAPEX_RAW, SUM_CAPEX = self.evaluate_params(CAPEX_N, np.inf, 0, [], 0)
        MIN_GROSS, MAX_GROSS, GROSS_RAW, GROSS_SUM = self.evaluate_params(REV_N, np.inf, 0, [], 0)

        for i in range(self.trials):
            obj_p = self.perturb_obj()
            CAPEX_N, OPEX_N, REV_N = obj_p.get_total_costs(N,C_electric)
            MIN_OPEX, MAX_OPEX, OPEX_RAW, SUM_OPEX = self.evaluate_params(OPEX_N, MIN_OPEX, MAX_OPEX, OPEX_RAW, SUM_OPEX)
            MIN_CAPEX, MAX_CAPEX, CAPEX_RAW, SUM_CAPEX = self.evaluate_params(CAPEX_N, MIN_CAPEX, MAX_CAPEX, CAPEX_RAW, SUM_CAPEX)
            MIN_GROSS, MAX_GROSS, GROSS_RAW, GROSS_SUM = self.evaluate_params(REV_N, MIN_GROSS, MAX_GROSS, GROSS_RAW, GROSS_SUM)

        COST_SET = {
            'CAPEX': [CAPEX_RAW[0], MAX_CAPEX, MIN_CAPEX, SUM_CAPEX / (self.trials + 1)],
            'OPEX': [OPEX_RAW[0], MAX_OPEX, MIN_OPEX, SUM_OPEX / (self.trials + 1)],
            'GROSS': [GROSS_RAW[0], MAX_GROSS, MIN_GROSS, GROSS_SUM / (self.trials + 1)],
            'CAPEX_RAW': CAPEX_RAW,
            'OPEX_RAW': OPEX_RAW,
            'GROSS_RAW': GROSS_RAW
        }
        return COST_SET

    def evaluate_params(self, var, Min_old, Max_old, Ref, Sum):
        Temp = var
        Min = min(Min_old, Temp)
        Max = max(Max_old, Temp)
        Ref.append(Temp)
        Sum += Temp
        return Min, Max, Ref, Sum

    def perturb_obj(self):
        """
        Perturb the object properties for Monte Carlo simulation.
        Returns:
            BioReactorSubSys: Object with perturbed properties.
        """
        obj_p = SeparationSubSys()
        i = 0
        for prop in self.__dict__:
            if prop not in ['trials', 'ERR']:
                property = getattr(self, prop)
                if type(property) is list:
                    new_prop = [p*(1 + (np.random.rand() - 0.5) * 2 * self.ERR[i]) for p in property]
                else:
                    new_prop = property*(1 + (np.random.rand() - 0.5) * 2 * self.ERR[i])
                setattr(obj_p,prop, new_prop)
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


    def get_total_costs(self,N,C_electric):
        CAPEX=0
        REVENUE=0
        OPEX=0
        Feed = self.get_mol_flows(N)
        for i in range(len(self.Kp)):
            ENERGY,REV_n, OUTLET=self.get_subs_cost(Feed,i)
            CAPEX+=self.C0*(ENERGY/(8760E+3))**0.55
            OPEX+=ENERGY*C_electric
            REVENUE+=REV_n
            Feed = OUTLET
        
        return CAPEX, OPEX, REVENUE


    def get_mol_flows(self,N):
        mol_flow = []
        total_flow = N/self.mass_fraction
        for i in range(len(self.Kp)):
            mol_flow.append(total_flow*self.mass_fraction*self.yield_[i]/self.Kp[i]/self.Mr_product[i])
        mol_flow.append(total_flow*(1-self.mass_fraction)/self.Mr_solvent)

        return mol_flow
    
    def get_subs_cost(self,Feed,i):
        ENERGY, OUTLET = self.get_sub_sys_energy(Feed,i)
        REVENUE = (Feed[i]-OUTLET[i])*self.value[i]
        return ENERGY, REVENUE, OUTLET


    def get_sub_sys_energy(self,mf1,i):
        if self.p[i] ==0:
            mf3 = mf1.copy()
            mf3[i] = 0
            return 0, mf3
        else:
            F1 = sum(mf1)
            x1 = [mf/F1 for mf in mf1]   

            x_product = x1[i]
            x_remaining = x1.copy()
            x_remaining[i] = 0

            F2 = self.nu_cap[i]*F1*x_product/self.p[i]
            x2 = [x*(1-self.p[i]) for x in x_remaining]
            x2[i] = self.p[i]
            mf2 = [F2*x2i for x2i in x2]

            F3 = F1-F2
            mf3 = [mf1i-mf2i for mf1i, mf2i in zip(mf1,mf2)]
            x3 = [mf3i/F3 for mf3i in mf3]

            E1 = self.calculate_energy(mf1,x1)
            E2 = self.calculate_energy(mf2,x2)
            E3 = self.calculate_energy(mf3,x3)

            return E2+E3-E1, mf3


    def calculate_energy(self,mf,x):
        Energy =sum([self.E0 * mfi*np.log(max(xi,1E-9)) for mfi, xi in zip(mf,x)])/self.nu_2nd
        return Energy

