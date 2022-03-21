from centralised_utilities import *

class DQN(nn.Module):
    def __init__(self, outputs = OUTPUTS):
        super().__init__()

        self.fc1 = nn.Linear(in_features = INPUTS, out_features = FC1)
        self.fc2 = nn.Linear(in_features = FC1, out_features = FC2)
        self.out = nn.Linear(in_features = FC2, out_features = outputs)

    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

class MemoryManager():
    def __init__(self, capacity = CAPACITY, batch_size = BATCH_SIZE):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = []
        self.push_count = 0
        self.exp = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            # Overwrite experiences if memory is full
            self.memory[self.push_count % self.capacity] = experience 
        self.push_count += 1

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def is_sampleable(self):
        return len(self.memory) >= self.batch_size
    
    def Exp(self):
        return self.exp
    
    def unzip_exps(self, exps):
        batch = self.exp(*zip(*exps))
        s = torch.stack(batch.state)
        a = torch.cat(batch.action)
        r = torch.cat(batch.reward)
        ns = torch.stack(batch.next_state)
        return (s,a,r,ns)

class ActionStrategy(): 
    # Epsilon Greedy
    def __init__(self, start = EPS_I, end = EPS_E, decay = EPS_DECAY):
        self.start = start
        self.end = end
        self.decay = decay
    
    def get_epsilon(self, experience_step): 
        # Exponential decay
        return self.end + (self.start - self.end)*(np.exp(-1*self.decay*experience_step))
    
class Agent():
    def __init__(self, strategy, num_actions = NUM_OF_ACTIONS, device = GPU):
        self.experience_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
        self.epsilon = None
    
    def select_action(self, state, policy_net, tl = CONTROLLED_SIGNAL, greedy_biased = False):
        if not greedy_biased:
            self.epsilon = self.strategy.get_epsilon(self.experience_step)
            self.experience_step += 1
            if self.epsilon > random.random():
                rand_action = random.randrange(self.num_actions)
                return torch.tensor([rand_action], device= self.device)
            else:
                with torch.no_grad():
                    return policy_net(state).unsqueeze(dim=0).argmax(dim=1).to(self.device)
        else:
            maxpressure = float("-Inf")
            argmaxp = None
            for p in PHASE_INFO:
                pressure = self.greedy_get_pressure_for_phase(p, tl)
                if pressure > maxpressure:
                    maxpressure = pressure
                    argmaxp = p
            # print(argmaxp)
            return torch.tensor([int(PHASE_INFO[argmaxp])], device= self.device)

    def select_central_action(self, state, policy_net, num_actions):
        self.epsilon = self.strategy.get_epsilon(self.experience_step)
        self.experience_step += 1
        if self.epsilon > random.random():
            rand_action = random.randrange(num_actions)
            return torch.tensor([rand_action], device= self.device)
        else:
            with torch.no_grad():
                return policy_net(state).unsqueeze(dim=0).argmax(dim=1).to(self.device)
                
    def greedy_get_pressure_for_phase(self, phase, Tl):
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

    @property
    def exploration_rate(self):
        return self.epsilon
    
    def get_qvalues(self, policy_net, states, actions):
        return policy_net(states).gather(dim = 1, index = actions.unsqueeze(-1))
    
    def get_maxqs_for_ns(self, target_net, next_states):
        return target_net(next_states).max(1)[0].detach()

    def get_bellman_targets(self, rewards, next_qvalues):
        b_targets = rewards + GAMMA*next_qvalues
        return b_targets.unsqueeze(-1)
    
class SumoTrafficState():
    def pack_state_info(self, Tl):
        state = {}
        phase = tr.trafficlight.getRedYellowGreenState(Tl)
        slane_list = tr.trafficlight.getControlledLanes(Tl)
        pressures = []
        intersection_pressure = 0
        for slane in slane_list:
            elane = tr.lane.getLinks(slane)[0][0]
            nslane = tr.lane.getLastStepVehicleNumber(slane)
            nelane = tr.lane.getLastStepVehicleNumber(elane)
            pressure = nslane - nelane
            tmNp = (slane + "->" + elane, pressure)
            intersection_pressure += pressure
            pressures.append(tmNp)
        state["waiting_times"], state["tot_waiting_time"] = self.get_total_waiting_time(self, Tl)
        state["pressures"] = pressures
        state["intersection_pressure"] = intersection_pressure
        state["phase"] = self.get_phase_num(self, phase)
        state["phase_pressure"] = self.get_pressure_for_phase(self, phase, Tl)
        return state

    def get_total_waiting_time(self, Tl):
        waiting_time_list = []
        total_waiting_time = 0
        slanes = tr.trafficlight.getControlledLanes(Tl)
        for slane in slanes:
            elane = tr.lane.getLinks(slane)[0][0]
            wt = tr.lane.getWaitingTime(slane)
            total_waiting_time += wt
            tmNwt = (slane + "->" + elane, wt)
            waiting_time_list.append(tmNwt)
        return (waiting_time_list, total_waiting_time)

    def get_phase_num(self, phase):
        phase_num = None
        for p in PHASE_INFO:
            if phase == p:
                phase_num = PHASE_INFO[p]
                break 
        return phase_num

    def get_pressure_for_phase(self, phase, Tl):
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
    
    @classmethod
    def get(self):
        states = []
        for Tl in tr.trafficlight.getIDList():
            state = self.pack_state_info(self, Tl)
            states.append((Tl, state))
        return states

class SumoManager():
    def __init__(self, sumocmd = SUMOCMD, active_signal = CONTROLLED_SIGNAL, tms = NUM_TMS, maxp = MAXP, maxwt = MAXWT, device = GPU):
        self.device = device
        self.sumocmd = sumocmd
        self.state = None
        self._step = 0
        self.done = False
        self.TL = active_signal
        self.tms = tms
        self.maxp = maxp
        self.maxwt = maxwt
        self.action_record = []
        self.setup()

    def setup(self):
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")
        self.start()
        self.close()

    def start(self):
        tr.start(self.sumocmd, label="master")
        self.state = self.parse_state_info()
        self.TLIds = tr.trafficlight.getIDList()
        self.tls_num = len(self.TLIds)
        self.joint_actions = list(product(
            list(ACTION_PHASES.keys()), 
            repeat = self.tls_num))
        self.num_actions = len(self.joint_actions)

    def close(self):
        tr.close()

    def reset(self):
        if not self.done:
            self.start()
            self._step = 0
            self.action_record = []
        else:
            self.close()
            self._step = 0
            self.action_record = []
            self.start()

    def step(self):
        self.done = False if tr.simulation.getMinExpectedNumber() > 0 else True
        tr.simulationStep()
        self._step += 1
        if GUI_ACTIVE:
            time.sleep(TIME_ELAPSE)
        if not self.done:
            self.state = self.parse_state_info()
        else:
            self.state = torch.zeros_like(self.state, device = self.device).float()

    def get_state(self, elapsed_state = False):
        if not elapsed_state:
            return self.state
        else:
            for _ in range(ACTION_DELAY): 
                self.step()
                if self.done: break
            return self.state
    
    def set_tl_state(self, action_index):
        joint_action = self.joint_actions[action_index]
        self.tl_record = {}
        for tl, phase in zip(sm.TLIds, joint_action): 
            tr.trafficlight.setRedYellowGreenState(tl, ACTION_PHASES[phase])
            self.tl_record[tl] = phase

    def take_action_get_reward(self, action):
        action_index = int(action.item())
        self.set_tl_state(action_index)
        self.action_record.append((self._step, self.tl_record))
        self.get_state(elapsed_state = True)
        reward = self.compute_reward()
        return torch.tensor([reward], device = self.device)

    def compute_reward(self):
        tot_reward = 0
        for tl in self.TLIds:
            n = self.TLIds.index(tl)
            p = self.state[(n)*self.tms:(n + 1)*self.tms]
            tot_reward += -1*(abs(sum(p)))
        # wt = self.state[self.tms:]
        # wt = self.maxp*torch.tanh((wt/(self.maxwt/2) - 1))
        return tot_reward
        
    def parse_state_info(self):
        state_info = SumoTrafficState.get()
        pNwt_cat = []
        for tl in state_info:
            p = tl[1]["pressures"]  
            # wt = state_info[self.TL]["waiting_times"]  
            for i in p:
                pNwt_cat.append(i[1])
        # for i in wt:
        #     pNwt_cat.append(i[1])
        return torch.tensor(pNwt_cat, device = self.device).float()

class PerfomanceMeter():
    def __init__(self):
        self.open = False
    
    def print_time(self):
        t = time.localtime()
        print(time.strftime("%H:%M:%S", t))

    def plot_returns(self, returns, durs, avg_reward, bell_loss, period):
        self.period = period
        if not self.open:
            self.open = True
        plt.clf()
        plt.title(f"{str(period)} period Moving Average of Episodic Values")
        fig, self.axes = plt.subplots(2, 2)
        mav1 = self.get_moving_avgs(returns)
        mav2 = self.get_moving_avgs(durs)
        mav3 = self.get_moving_avgs(avg_reward)
        mav4 = self.get_moving_avgs(bell_loss)
        self.axes[0, 0].plot(returns) 
        self.axes[0, 0].plot(mav1) 
        self.axes[0, 0].set_title("Return")
        self.axes[0, 1].plot(durs)
        self.axes[0, 1].plot(mav2) 
        self.axes[0, 1].set_title("Duration")
        self.axes[1, 0].plot(avg_reward)
        self.axes[1, 0].plot(mav3)
        self.axes[1, 0].set_title("Avg Reward")
        self.axes[1, 1].plot(bell_loss)
        self.axes[1, 1].plot(mav4)
        self.axes[1, 1].set_title("Avg Bellman Loss")
        plt.tight_layout()
        plt.pause(0.001)
    
    def get_moving_avgs(self, values):
        values = torch.tensor(values).float()
        if len(values) >= self.period:
            mov_avgs = values.unfold(0, self.period, 1).mean(1)
            mov_avgs = torch.cat((torch.zeros(self.period - 1), mov_avgs))
            return mov_avgs.detach().numpy()
        else:
            mov_avgs = torch.zeros(len(values))
            return mov_avgs.detach().numpy()
    
    def save(self, msg):
        plt.savefig("data\\" + msg + ".png")

    def write_record(self, name, list):
        list = str(list)
        file = open("logs\\" + name + ".txt", "w")
        file.write(list)
        file.close()

if __name__ == "__main__":

    sm = SumoManager()
    epsgreedy = ActionStrategy()
    agent = Agent(epsgreedy)
    memory = MemoryManager()
    pm = PerfomanceMeter()
    Exp = memory.Exp()
    policy_net = DQN(outputs = sm.num_actions).to(GPU)
    target_net = DQN(outputs = sm.num_actions).to(GPU)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=ALPHA)

    episode_returns = []
    episodic_losses = []
    avg_rewards = []
    avg_bellman_losses = []
    max_return = float("-Inf")
    min_loss = float("Inf")
    pm.print_time()

    for episode in range(NUM_EPISODES):
        print("EPISODE: ", episode)
        sm.reset()
        state = sm.get_state()
        return_val = 0
        tot_reward = 0
        loss_val = 0
        for agent_step in count():
            action = agent.select_central_action(state, policy_net, sm.num_actions)
            reward = sm.take_action_get_reward(action)
            if agent_step >= MAX_EPISODES:
                sm.set_done()
            tot_reward += reward.item()
            return_val += (GAMMA**(agent_step))*(reward.item())
            next_state = sm.get_state()
            memory.push(Exp(state, action, reward, next_state))
            state = next_state
            if memory.is_sampleable():
                exps = memory.sample()
                states, actions, rewards, next_states = memory.unzip_exps(exps)
                cur_qvalues = agent.get_qvalues(policy_net, states, actions)
                next_qvalues = agent.get_maxqs_for_ns(target_net, next_states)
                bellman_targets = agent.get_bellman_targets(rewards, next_qvalues)
                loss = F.smooth_l1_loss(cur_qvalues, bellman_targets)
                loss_val += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if sm.done:
                    if return_val > max_return:
                        pm.write_record(GRAPH_NAME + "_max_return", sm.action_record)
                        max_return = return_val
                    if agent_step < min_loss:
                        pm.write_record(GRAPH_NAME + "_min_loss", sm.action_record)
                        min_loss = agent_step
                        all_time_low = True
                    avg_reward = tot_reward/agent_step
                    avg_loss = loss_val/agent_step
                    print(f"RETURN for EPISODE {episode}:", return_val)
                    print(f"LOSS for EPISODE {episode}:", agent_step)
                    episode_returns.append(return_val)
                    episodic_losses.append(agent_step)
                    avg_rewards.append(avg_reward)
                    avg_bellman_losses.append(avg_loss)
                    if GRAPH_SHOW:
                        pm.plot_returns(episode_returns, episodic_losses, avg_rewards, avg_bellman_losses, MAV_COUNT)             
                    break 
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    pm.print_time()
    if not GRAPH_SHOW:
        pm.plot_returns(episode_returns, episodic_losses, avg_rewards, avg_bellman_losses, MAV_COUNT)
    pm.write_record(GRAPH_NAME + "_returns", str(episode_returns))
    pm.write_record(GRAPH_NAME + "_losses", str(episodic_losses))
    pm.save(GRAPH_NAME)
    sm.close()