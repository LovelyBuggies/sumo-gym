class ElectricVehicles(object):
    def __init__(self, id, speed, indicator, capacity):
        self.id = id
        self.speed = speed
        self.indicator = indicator
        self.capacity = capacity


class ChargingStation(object):
    def __init__(self, location, indicator, charging_speed):
        self.location = location
        self.indicator = indicator
        self.charging_speed = charging_speed
