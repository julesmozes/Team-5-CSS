import osmap

map = osmap.Map()
map.build("Oost, Amsterdam, Netherlands")
map.buildTraffic(minSpatialWavelength=300, minTemporalWavelength=2, modulationDepth=5)

map.animate_costs(filename="trafficAnimation.gif")

