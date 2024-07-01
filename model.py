import torch
import re
import yaml

ACCEPTABLE_INIT = [
    "zeros",
    "ones",
    "random",
    "randn"
]

class SlidingMode(torch.nn.Module):
    def __init__(self, config_path, init="ones", mse_reduction="sum"):
        super().__init__()
        with open(config_path, 'r') as fp:
            config = yaml.safe_load(fp)

        self.system_string = config["system"]
        self.control_string = config["control"]
        self.desired_string = config["desired"]
        initial = config["initial"]
        self.param_names = config["params"]
        self.system_variable_names = initial.keys()

        if not(len(self.system_string.keys()) == len(self.desired_string.keys()) == len(initial.keys())):
            raise ValueError
        else:
            for sk, dk, ik in zip(self.system_string.keys(), self.desired_string.keys(), initial.keys()):
                if not('d' + dk == 'd' + ik == sk):
                    raise ValueError
        
        self.system_functions = [
            self.__parse_equation(el, system_variables=initial.keys(), control_variables=self.control_string.keys(), params=self.param_names)
            for el in self.system_string.values()
        ]

        self.control_functions = [
            self.__parse_equation(el, system_variables=initial.keys(), control_variables=self.control_string.keys(), params=self.param_names)
            for el in self.control_string.values()
        ]

        self.desired_functions = [
            self.__parse_equation(el, system_variables=initial.keys(), control_variables=self.control_string.keys(), params=self.param_names)
            for el in self.desired_string.values()
        ]

        self.params = self.set_params(len(self.param_names), init=init)

        self.initial_values = torch.tensor([el for el in initial.values()], dtype=torch.float32)

        self.criterion = torch.nn.MSELoss(reduction=mse_reduction)
    
    def set_params(self, n, init="ones"):
        if init == "zeros":
            params = torch.zeros(n, dtype=torch.float32)
        elif init == "ones":
            params = torch.ones(n, dtype=torch.float32)
        elif init == "random":
            params = torch.rand(n, dtype=torch.float32)
        elif init == "randn":
            params = torch.randn(n, dtype=torch.float32)
        elif isinstance(init, torch.Tensor):
            assert len(init) == n
            params = init
        else:
            raise NotImplementedError(f"Choose from {ACCEPTABLE_INIT}")
        
        return torch.nn.parameter.Parameter(params)

    def __parse_equation(self, equation_string, system_variables, control_variables, params):
        result = equation_string

        # 2. Change system variables
        for i, sv in enumerate(system_variables):
            result = re.sub(sv, f"sv[{i}]", result)
        
        # 3. Change control variables
        for i, cv in enumerate(control_variables):
            result = re.sub(cv, f"cv[{i}]", result)
        
        # 4. Change params
        for i, par in enumerate(params):
            result = re.sub(par, f"par[{i}]", result)
        def func(sv, cv, par, t):
            eq = eval(result)
            return eq

        return func
    
    def forward(self, time, delta, param_values=None):
        params = self.params if param_values is None else param_values

        time_steps = torch.arange(0.0, time + delta, delta) # time + delta so the time window would be [0..time] and not [0..time)

        # system_variables = torch.zeros((time_steps.shape[0], len(self.initial_values)))
        # system_variables[0,:] = self.initial_values.clone()
        system_variables = self.initial_values.clone()
        all_system_variables = [system_variables.clone().tolist()]

        desired_values = torch.tensor([[df(None, None, None, t) for df in self.desired_functions] for t in time_steps])

        total_loss = torch.tensor(0.0)

        for i, t in enumerate(time_steps[1:]):
            # calculate control variables
            control_variables = [cf(system_variables, None, params, t) for cf in self.control_functions]
            control_variables = torch.stack(control_variables)

            # calculate system variables
            dsystem_variables = [sf(system_variables, control_variables, params, t) for sf in self.system_functions]
            dsystem_variables = torch.stack(dsystem_variables)

            system_variables = system_variables + delta * dsystem_variables
        
            total_loss = total_loss + self.criterion(system_variables, desired_values[i+1]) # don't count initial values
            
            all_system_variables.append(system_variables.clone().tolist())
        
        all_system_variables = torch.tensor(all_system_variables)
        assert all_system_variables.shape == desired_values.shape

        return (
            all_system_variables,
            desired_values,
            total_loss
        )