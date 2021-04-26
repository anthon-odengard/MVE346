
def cap():
    battery_level=0
    for n in model.nodes:
        for t in model.hours:
            battery_level = battery_level + model.production[n, "battery", t]
    return battery_level >= 0
'''
def battery_upper_production_rule(model, nodes, time):
    capa = model.invested_capacity[nodes, "Battery"]
    upper_bound = capa * model.eta["Battery"]
    return model.production[nodes, "Battery", time] <= upper_bound


model.battery_production_upper = Constraint(model.nodes, model.hours, rule=battery_upper_production_rule)


def battery_lower_production_rule(model, nodes, time):
    capa = model.invested_capacity[nodes, "Battery"]
    lower_bound = capa * -1
    return model.production[nodes, "Battery", time] >= lower_bound


model.battery_production_lower = Constraint(model.nodes, model.hours, rule=battery_lower_production_rule)


def battery_balance_rule(model, nodes):
    battery_storage = 0
    for t in model.hours:
        battery_storage = battery_storage - model.production[nodes, "Battery", t]
    return battery_storage == 0


model.battery_balance = Constraint(model.nodes, rule=battery_balance_rule)
'''


def battery_storage_rule(model, nodes, hours):
    if hours == 0:
        return model.battery_storage[nodes, hours] == - model.production[nodes, "Battery", hours]
    elif hours == model.hours[-1]:
        return model.battery_storage[nodes, hours] == model.battery_storage[nodes, 0]
    else:
        return model.battery_storage[nodes, hours] == model.battery_storage[nodes, hours-1] - model.production[nodes, "Battery", hours] * model.eta["Battery"]/2


model.battery_storage_constraint = Constraint(model.nodes, model.hours, rule=battery_storage_rule)


def battery_upper_storage_rule(model, nodes, hours):
    return model.battery_storage[nodes, hours] <= model.invested_capacity[nodes, "Battery"]


model.battery_upper_storage = Constraint(model.nodes, model.hours, rule=battery_upper_storage_rule)


def battery_lower_storage_rule(model, nodes, hours):
    return model.battery_storage[nodes, hours] >= 0


model.battery_lower_storage = Constraint(model.nodes, model.hours, rule=battery_lower_storage_rule)

