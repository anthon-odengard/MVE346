from pyomo.environ import *
import pyomo.environ
import pandas as pd
import numpy as np
import gurobipy
import matplotlib.pyplot as plt

model = ConcreteModel()

# DATA
countries = ['DE', 'DK', 'SE']
loads = ['Load_DE', 'Load_DK', 'Load_SE']
techs = ['Wind', 'PV', 'Gas', 'Hydro', 'Battery']
eta = {'Wind': 1, 'PV': 1, 'Gas': 0.4, 'Hydro': 1,
       'Battery': 0.9}  # you could also formulate eta as a time series with the capacity factors for PV and wind
lifetime = {  # years
    "Wind": 25,
    "PV": 25,
    "Gas": 30,
    "Hydro": 80,
    "Battery": 10,
}
investment_cost = {
    "Wind": 1100000,
    "PV": 600000,
    "Gas": 550000,
    "Hydro": 0,
    "Battery": 150000,
}

running_cost = {  # €/MWh_elec
    "Wind": 0.1,
    "PV": 0.1,
    "Gas": 2,
    "Hydro": 0.1,
    "Battery": 0.1,
}

fuel_cost = {  # €/MWh_fuel
    "Wind": 0,
    "PV": 0,
    "Gas": 22,
    "Hydro": 0,
    "Battery": 0
}

discountrate = 0.05
batteryOn = True
CO2_cap = 0.1
input_data = pd.read_csv('TimeSeries.csv', header=[0], index_col=[0])
co2_tot_1 = sum([125243664.86937931, 8552556.240328334, 4978228.174867922])
total_cost_1 = 37238.0733983158  # M Euro


# TIME SERIES HANDLING
def demandData():
    demand = {}
    for n in model.nodes:
        for t in model.hours:
            demand[n, t] = float(input_data['Load_' + str(n)][t])
    return demand


def factor_data():
    prodFactor = {}
    for n in model.nodes:
        for g in model.gens:
            for t in model.hours:
                if g in ['Wind', 'PV']:
                    prodFactor[n, g, t] = float(input_data[str(g) + "_" + str(n)][t])
                else:
                    prodFactor[n, g, t] = 1
    return prodFactor


# SETS
model.nodes = Set(initialize=countries, doc='countries')
model.hours = Set(initialize=input_data.index, doc='hours')
model.gens = Set(initialize=techs, doc='techs')

# PARAMETERS
model.demand = Param(model.nodes, model.hours, initialize=demandData())
model.eta = Param(model.gens, initialize=eta, doc='Conversion efficiency')
model.prodFactor = Param(model.nodes, model.gens, model.hours, initialize=factor_data())
model.lifetime = Param(model.gens, initialize=lifetime, doc='Conversion efficiency')
model.investment_cost = Param(model.gens, initialize=investment_cost, doc='Investment cost per kW')
model.running_cost = Param(model.gens, initialize=running_cost, doc='Running cost per MWh_elec')
model.fuel_cost = Param(model.gens, initialize=fuel_cost, doc='fuel cost per MWh_fuel')


def annuity():
    a = {}
    for g in model.gens:
        a[g] = discountrate / (1 - 1 / (1 + discountrate) ** lifetime[g])
    return a


model.annuity = Param(model.gens, initialize=annuity())


def production_cost():
    prod_cost = {}
    for g in model.gens:
        prod_cost[g] = model.running_cost[g] + model.fuel_cost[g] / model.eta[g]
    return prod_cost


model.production_cost = Param(model.gens, initialize=production_cost())

# VARIABLES
capMaxdata = pd.read_csv('CapMax.csv', index_col=[0])


def capacity_max(model, n, g):
    capMax = {}
    if g in capMaxdata.columns:
        capMax[n, g] = float(capMaxdata[g][n]) * 1e3
        return 0.0, capMax[n, g]
    elif g == 'Battery' and not batteryOn:
        return 0.0, 0.0
    else:
        return 0.0, None


model.invested_capacity = Var(model.nodes, model.gens, bounds=capacity_max, doc='Generator cap')
model.production = Var(model.nodes, model.gens, model.hours, doc='Production cap')
model.battery_power_stored = Var(model.nodes, model.hours, doc="Battery storage")
model.in_power = Var(model.nodes, model.hours, doc="Battery power in")
model.transmission = Var(model.nodes, model.nodes, model.hours, doc="Transmission")
model.invested_transmission = Var(model.nodes, model.nodes, bounds=(0.0, None), doc="Transmission cap")


# CONSTRAINTS
# Production >= demand at all times
def prod_demand_rule(model, nodes, time):
    total_prod_node = 0
    transmission_in = 0
    transmission_out = 0
    for n in model.nodes:
        transmission_in = transmission_in + model.transmission[n, nodes, time]
        transmission_out = transmission_out + model.transmission[nodes, n, time]
    for g in model.gens:
        total_prod_node = total_prod_node + model.production[nodes, g, time]
    return total_prod_node - transmission_out / 0.98 >= model.demand[nodes, time] + model.in_power[nodes, time] - \
           transmission_in


model.prodDemand = Constraint(model.nodes, model.hours, rule=prod_demand_rule)


def self_transmission_rule(model, node, time):
    return model.transmission[node, node, time] == 0


model.self_transmission = Constraint(model.nodes, model.hours, rule=self_transmission_rule)


def positive_transmission_rule(model, nodes1, nodes2, time):
    return model.transmission[nodes1, nodes2, time] >= 0


model.positive_transmission_ = Constraint(model.nodes, model.nodes, model.hours, rule=positive_transmission_rule)


def invested_transmission_rule(model, nodes1, nodes2):
    return model.invested_transmission[nodes1, nodes2] == model.invested_transmission[nodes2, nodes1]


model.invested_transmission_constraint = Constraint(model.nodes, model.nodes, rule=invested_transmission_rule)


def transmission_constraint(model, nodes1, nodes2, time):
    return model.transmission[nodes1, nodes2, time] <= model.invested_transmission[nodes1, nodes2]


model.transmission_cap = Constraint(model.nodes, model.nodes, model.hours, rule=transmission_constraint)


def in_power_positive_rule(model, nodes, time):
    return model.in_power[nodes, time] >= 0


model.in_power_positive = Constraint(model.nodes, model.hours, rule=in_power_positive_rule)


def in_power_capacity_rule(model, nodes, time):
    return model.in_power[nodes, time] <= model.invested_capacity[nodes, "Battery"] / 12


model.in_power_capacity = Constraint(model.nodes, model.hours, rule=in_power_capacity_rule)


def battery_storage_rule(model, nodes, time):
    if time == 0:
        return model.battery_power_stored[nodes, 0] == model.in_power[nodes, 0] - model.production[nodes, "Battery", 0]
    elif time > 0:
        return model.battery_power_stored[nodes, time] == model.battery_power_stored[nodes, time - 1] + \
               model.in_power[nodes, time] - model.production[nodes, "Battery", time]


model.battery_storage = Constraint(model.nodes, model.hours, rule=battery_storage_rule)


def battery_storage_end_rule(model, nodes):
    return model.battery_power_stored[nodes, model.hours[-1]] == model.battery_power_stored[nodes, 0]


model.battery_storage_end = Constraint(model.nodes, rule=battery_storage_end_rule)


def battery_upper_storage_rule(model, nodes, hours):
    return model.battery_power_stored[nodes, hours] <= model.invested_capacity[nodes, "Battery"]


model.battery_upper_storage = Constraint(model.nodes, model.hours, rule=battery_upper_storage_rule)


def battery_lower_storage_rule(model, nodes, hours):
    return model.battery_power_stored[nodes, hours] >= 0


model.battery_lower_storage = Constraint(model.nodes, model.hours, rule=battery_lower_storage_rule)


# Production <= total installed capacity * production factor
def prod_capacity_rule(model, nodes, gens, time):
    return model.production[nodes, gens, time] <= model.invested_capacity[nodes, gens] * model.prodFactor[
        nodes, gens, time]


model.prodCapa = Constraint(model.nodes, model.gens, model.hours, rule=prod_capacity_rule)


def hydro_reserve_rule(model, nodes):
    total_hydro_reserve = 0
    for t in model.hours:
        total_hydro_reserve = total_hydro_reserve + input_data["Hydro_inflow"][t] - model.production[nodes, "Hydro", t]
    return (0, total_hydro_reserve, 33 * 1e12)


model.hydro_reserve = Constraint(model.nodes, rule=hydro_reserve_rule)


def positive_prod_rule(model, nodes, gens, time):
    return model.production[nodes, gens, time] >= 0


model.positve_production = Constraint(model.nodes, model.gens, model.hours, rule=positive_prod_rule)


def co2_cap_rule(model):
    co2_total = 0
    for n in model.nodes:
        for t in model.hours:
            co2_total = co2_total + model.production[n, "Gas", t] * 0.202 / model.eta["Gas"]
    return co2_total <= CO2_cap * co2_tot_1


model.co2_cap = Constraint(rule=co2_cap_rule)


# OBJECTIVE FUNCTION


def system_cost():  # Total cost
    total_investment_cost = 0
    total_running_cost = 0
    for n in model.nodes:
        for g in model.gens:
            total_investment_cost = total_investment_cost + model.investment_cost[g] * model.annuity[g] * \
                                    model.invested_capacity[n, g]  # € + €/MW * 1 * MW = €
            for t in model.hours:
                total_running_cost = total_running_cost + model.production_cost[g] * model.production[
                    n, g, t]  # € + €/MWh * MWh = €
    return total_investment_cost + total_running_cost


def transmission_cost():
    total_investment_cost = 0
    annuity_trans = discountrate / (1 - 1 / (1 + discountrate) ** 50)
    for n1 in model.nodes:
        for n2 in model.nodes:
            total_investment_cost = total_investment_cost + 2500000 * annuity_trans * \
                                    model.invested_transmission[n1, n2]  # € + €/MW * 1 * MW = €
    return total_investment_cost / 2


def objective_rule(model):
    return system_cost() + transmission_cost()


model.objective = Objective(rule=objective_rule, sense=minimize, doc='Objective function')

if __name__ == '__main__':
    from pyomo.opt import SolverFactory
    import pyomo.environ
    import pandas as pd

    opt = SolverFactory("gurobi_direct")
    opt.options["threads"] = 4
    print('Solving')
    results = opt.solve(model, tee=True)

    results.write()

    # Reading output - example
    capTot = {}
    for n in model.nodes:
        for g in model.gens:
            capTot[n, g] = model.invested_capacity[n, g].value / 1e3  # GW

    costTot = value(model.objective) / 1e6  # Million EUR

    co2_total = 0
    for n in model.nodes:
        for t in model.hours:
            co2_total = co2_total + model.production[n, "Gas", t].value * 0.202 / model.eta["Gas"]

    cap = {}
    for g in model.gens:
        cap.update({g: np.zeros(3)})
        i = 0
        for n in model.nodes:
            cap[g][i] = model.invested_capacity[n, g].value
            i = i + 1

    annual_production = {}
    for g in model.gens:
        annual_production.update({g: np.zeros(3)})
        i = 0
        for n in model.nodes:
            for t in model.hours:
                annual_production[g][i] = annual_production[g][i] + model.production[n, g, t].value
            i = i + 1

    DE_gens_prod = {}
    for g in model.gens:
        DE_gens_prod.update({g: np.zeros(168)})
        for t in range(0, 168):
            DE_gens_prod[g][t] = model.production["DE", g, t].value

    DE_demand = np.zeros(168)
    for t in range(0, 168):
        DE_demand[t] = float(input_data['Load_DE'][t])

    DE_in_power = np.zeros(168)
    for t in range(0, 168):
        DE_in_power[t] = model.in_power["DE", t].value

    DE_transmission_in = np.zeros(168)
    for t in range(0, 168):
        for n in model.nodes:
            DE_transmission_in[t] = DE_transmission_in[t] + model.transmission[n, "DE", t].value

    #trans_invest = {}
    #for n1 in model.nodes:
    #    for n2 in model.nodes:
    #        trans_invest[n1, n2] = model.invested_transmission[n1, n2].value

    print("Total cost: ")
    print(costTot)

    print("Total co2: ")
    print(co2_total)
    model.invested_transmission.pprint()

    # print("Invested capacity result: ")
    # model.invested_capacity.pprint()

    ####### Plot results #########
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    x = np.arange(len(model.nodes))

    b_gas = list(np.add(cap["Wind"], cap["PV"]))
    b_hydro = list(np.add(b_gas, cap["Gas"]))
    b_battery = list(np.add(b_hydro, cap["Hydro"]))

    ax1.bar(x, cap["Wind"], color='black', label="Wind")
    ax1.bar(x, cap["PV"], color='red', bottom=cap["Wind"], label="PV")
    ax1.bar(x, cap["Gas"], color='blue', bottom=b_gas, label="Gas")
    ax1.bar(x, cap["Hydro"], color='cyan', bottom=b_hydro, label="Hydro")
    ax1.bar(x, cap["Battery"], color='green', bottom=b_battery, label="Batteries")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model.nodes)
    ax1.set_ylabel("Invested capacity MW")
    ax1.set_title("Total invested capacity per country")
    ax1.legend(loc="best")

    b_gas = list(np.add(annual_production["Wind"], annual_production["PV"]))
    b_hydro = list(np.add(b_gas, annual_production["Gas"]))
    b_battery = list(np.add(b_hydro, annual_production["Hydro"]))

    ax2.bar(x, annual_production["Wind"], color='black', label="Wind")
    ax2.bar(x, annual_production["PV"], color='red', bottom=annual_production["Wind"], label="PV")
    ax2.bar(x, annual_production["Gas"], color='blue', bottom=b_gas, label="Gas")
    ax2.bar(x, annual_production["Hydro"], color='cyan', bottom=b_hydro, label="Hydro")
    ax2.bar(x, annual_production["Battery"], color='green', bottom=b_battery, label="Batteries")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model.nodes)
    ax2.set_ylabel("MWh")
    ax2.set_title("Total annual production")
    ax2.legend(loc="best")

    x3 = np.arange(0, 168)
    ax3.stackplot(x3, DE_gens_prod["Wind"], DE_gens_prod["PV"], DE_gens_prod["Gas"], DE_gens_prod["Battery"],
                  -DE_in_power, DE_transmission_in,
                  labels=["Wind", "PV", "Gas", "Battery", "In_power", "Transmission_in"])
    ax3.plot(x3, DE_demand, label="Demand")
    ax3.set_ylabel("MWh")
    ax3.set_xlabel("Hour")
    ax3.set_title("Domestic production in Germany")
    ax3.legend(loc="best")

    ax4.bar(x, [model.invested_transmission['DE', 'DK'].value, model.invested_transmission['DE', 'SE'].value, \
                model.invested_transmission['DK', 'SE'].value])
    ax4.set_xticks(x)
    ax4.set_xticklabels(["DE-DK", "DE-SE", "DK-SE"])
    ax4.set_ylabel("Invested capacity")
    ax4.set_title("Total transmisson investment")
    plt.show()