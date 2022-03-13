from time import sleep
from utilities import *

if __name__ == "__main__":
    sumoCmd = ["sumo-gui", "-c", "fixedtime.sumocfg"]
    for _ in range(1):
        tr.start(sumoCmd)
        step = 0
        print(int(tr.simulation.getTime()))
        while tr.simulation.getMinExpectedNumber() > 0:
            tr.simulationStep()
            # sleep(0.02)
            step += 1
        tr.close()