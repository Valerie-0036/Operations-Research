from pyworkforce.queuing import ErlangC,MultiErlangC
interval = 60
transactions = [160, 165, 171, 175, 180]
aht=200/60
asa = 120 / 60
shrinkage=0.30 #to accommodate down time
erlang = ErlangC(transactions=transactions[3], interval=interval, asa=asa, aht=aht,shrinkage=shrinkage)
requirements = erlang.required_positions(service_level=0.8)

print(requirements)

param_grid = {"transactions": transactions, "aht": [aht], "interval": [interval], "asa": [asa], "shrinkage": [shrinkage]}
erlang = MultiErlangC(param_grid=param_grid, n_jobs=-1)

print(erlang.param_list)

service_level_scenarios = {"positions": [requirements["positions"]]}

requirements = erlang.service_level(arguments_grid=service_level_scenarios)

# 1 service level result per combination in the erlang.param_list and service_level_scenarios
print(requirements)
