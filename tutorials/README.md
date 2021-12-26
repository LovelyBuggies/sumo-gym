# Tutorials


## `make-fmpenv-xml.py`

This file demonstrates the ability to pass in SUMO files that will ultimately be parsed to define a FMP environment. The SUMO files referenced here collectively define a simulation over a very simple road network in which all edges are bidirectional. There are 15 non-internal nodes (2 are demand nodes), one charging station, and two electric vehicles.

### Expected Format of SUMO Input Files
***
#### Demand Nodes

Each node/vertex is encoded as a `junction`. In order to define a demand node, add a generic parameter to the junction with the `key` value set to "destination". For example, in `simple.net.xml`, the junctions with ids `v1` and `v15` are demand nodes. As shown below, the destination of `v1` is `v5`.
```
<junction id="v1" type="priority" x="-18.12" y="77.63" incLanes="gneE15_0 -gneE1_0" intLanes=":v1_0_0 :v1_1_0" shape="-15.32,80.49 -16.15,74.14 -22.02,76.71">
        <request index="0" response="00" foes="00" cont="0"/>
        <request index="1" response="00" foes="00" cont="0"/>
        <param key="destination" value="v5"/>
</junction>
```
***
#### Charging Stations

Charging stations are not defined in the net.xml or rou.xml files, and are instead specified in an additional one. The FMP environment requires knowledge of each charging station's exact x-y coordinate location, but charging stations in SUMO define the location by the lane ID and a range within the lane w.r.t. the length. Therefore, to ease the process of parsing, when creating charging stations, an approximation of the location needs to be provided in a generic parameter with the `key` attribute set to "approx_loc" and the `value` attribute set to a space separated string with two floats. You can hover over the station in the GUI to get these estimates. 

For example, after loading `simple.sumocfg` in `SUMO GUI` with delay set to 80 ms, I determined that (32.36, 23.80) was a good estimate. Here is the content of `simple_charging_station_additional.xml`:

```
<additional>
    <chargingStation power="10000" chargeInTransit="0" chrgpower="200000" efficiency="1" endPos="25" id="cS_1" lane="-gneE18_0" startPos="20">
        <param key="approx_loc" value="32.36 23.80"/>
    </chargingStation>
</additional>
```
Also, as shown above, charging speeds can be specified through the `power` attribute (W/h). Always set `efficiency` to 1. 
Moreover, right now we assume the charging station to be on "Edge" when determining the actions of each vehicle since we treat the charging station as a "Vertex" to route the vehicle, while in sumo, the charging stations are set on "Lane". As a work around, we create two charging stations symmetrically on both lanes of the edge.

**NOTE: The decoder currently assumes that there will be at most one charging station per lane. **
***
#### Electric Vehicles

Electric vehicles are defined in the rou.xml file. The decoder currently assumes that there will only be one type of electric vehicle, which should be defined by a `vType` as shown below (from `simple.rou.xml`):

```
<vType id="EV" accel="3.0" decel="6.0" length="5.0" minGap="2.5" maxSpeed="50.0" sigma="0.5" vClass="evehicle" emissionClass="Energy" guiShape="evehicle">
        <param key="has.battery.device" value="true"/>
        <param key="maximumBatteryCapacity" value="2000"/>
        <param key="maximumPower" value="1000"/>
        <param key="vehicleMass" value="10000"/>
        <param key="frontSurfaceArea" value="5"/>
        <param key="airDragCoefficient" value="0.6"/>
        <param key="internalMomentOfInertia" value="0.01"/>
        <param key="radialDragCoefficient" value="0.5"/>
        <param key="rollDragCoefficient" value="0.01"/>
        <param key="constantPowerIntake" value="100"/>
        <param key="propulsionEfficiency" value="0.9"/>
        <param key="recuperationEfficiency" value="0.9"/>
        <param key="stoppingTreshold" value="0.1"/>
</vType>
```
The electric vehicles need to be defined by setting the `type` attribute to the id of the previously defined `vType`. Additionally all vehicles must have a `route` defined and the format **must match** the one shown below (i.e., there needs to be a `route` element within the `vehicle` element even though it is possible to define routes for vehicles in other ways). These routes should consist of only one edge, but more than one can be provided (they will just be ignored); this is because all vehicles must have a predefined starting point in the environment. 

Here are the vehicle definitions in `simple.rou.xml`:

```
<vehicle id="0" type="EV" depart="0" color="1,0,0">
    <param key="actualBatteryCapacity" value="2000"/>
    <route edges="-gneE15"/>
</vehicle>
<vehicle id="1" type="EV" depart="0" color="1,0,0">
    <param key="actualBatteryCapacity" value="2000"/>
     <route edges="-gneE19 -gneE18">
        <stop chargingStation="cS_1" until="50"/>
     </route>
</vehicle>
```


   