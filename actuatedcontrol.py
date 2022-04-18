from multi_agent_utilities import *

def pack_state_info(Tl):
    state = {}
    phase = tr.trafficlight.getRedYellowGreenState(Tl)
    slane_list = tr.trafficlight.getControlledLanes(Tl)
    phase_num = None
    for p in PHASE_INFO:
        if phase == p:
            phase_num = PHASE_INFO[p]
            break 
    state["phase"] = phase_num
    pressures = []
    intersection_pressure = 0
    for slane in slane_list:
        elane = tr.lane.getLinks(slane)[0][0]
        print(tr.lane.getLinks(slane))
        nslane = tr.lane.getLastStepVehicleNumber(slane)
        nelane = tr.lane.getLastStepVehicleNumber(elane)
        pressure = nslane - nelane
        tmNp = (slane + "->" + elane, pressure)
        intersection_pressure += pressure
        pressures.append(tmNp)
    state["intersection_pressure"] = abs(intersection_pressure)
    state["phase_pressure"] = get_pressure_for_phase(phase, Tl)
    state["pressures"] = pressures
    return state

def get_states():
    states = {}
    for Tl in tr.trafficlight.getIDList():
        state = pack_state_info(Tl)
        states[Tl] = state
    return states

def get_pressure_for_phase(phase, Tl):
    phase_list = list(phase.strip(" "))
    slane_list = tr.trafficlight.getControlledLanes(Tl)
    total_pressure = 0
    for i in range(len(phase_list)):
        if i % 3 != 0 and phase_list[i] == "G":
            slane = slane_list[i]
            elane = tr.lane.getLinks(slane)[0][0]
            nslane = tr.lane.getLastStepVehicleNumber(slane)
            nelane = tr.lane.getLastStepVehicleNumber(elane)
            pressure = nslane - nelane
            total_pressure += pressure
    return total_pressure

def run_simul():
    step = 0
    while tr.simulation.getMinExpectedNumber() > 0:
        tr.simulationStep()
        # network_state = get_states()
        # pp.pprint(network_state)
        # if step % 10 == 0 and step != 0:
        #     for Tl in network_state:
        #         maxpressure = float("-Inf")
        #         argmaxp = None
        #         for p in ACTION_PHASES:
        #             pressure = get_pressure_for_phase(p, Tl)
        #             if pressure > maxpressure:
        #                 maxpressure = pressure
        #                 argmaxp = p 
        #         tr.trafficlight.setRedYellowGreenState(Tl, argmaxp)
        # step += 1
    tr.close()

if __name__ == "__main__":
    sumoCmd = ["sumo-gui",
            "-c", "sumo\\actuated.sumocfg",
            "--no-step-log", "true",
            "-W", "true", 
            "--duration-log.disable"]
    tr.start(sumoCmd)
    run_simul()