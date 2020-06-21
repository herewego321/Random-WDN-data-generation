import random
import numpy as np
import os
import wntr
import pandas as pd
from tqdm import tqdm
import time


class ModelGenerator:
    def __init__(self, wn, ratio, base=0.002, deviation=None, roughness=100):
        self.wn_tr = wn
        self.base = base
        self.ratio = ratio
        self.deviation = deviation
        self.roughness = roughness

    def random_generator(self, base, deviation=None, method=None):
        if method == 'gauss':
            value = random.gauss(base, deviation)
        elif method == 'uniform':
            value = base + random.uniform(-1 * base, 5 * base)
        elif method == 'pipe':
            value = random.uniform(base, 3 * base)
        else:
            print("sorry, we don't have such method!")

        return abs(value)

    def generate_demand(self):
        if os.path.exists('../data/node base demand.npy'):
            base_demand_value = np.load('../data/node base demand.npy', allow_pickle=True).item()

        else:
            base_demand_value = {}
            for junction in self.wn_tr.junction_name_list:
                base_demand_value[junction] = self.random_generator(base=self.base, method='uniform')

            np.save('../data/node base demand.npy', base_demand_value, allow_pickle=True)

        return base_demand_value

    def generate_roughness(self):

        if os.path.exists('../data/pipe roughness.npy'):
            pipe_roughness = np.load('../data/pipe roughness.npy', allow_pickle=True).item()

        else:
            pipe_roughness = {}
            for pipe in self.wn_tr.pipe_name_list:
                pipe_roughness[pipe] = self.random_generator(base=self.roughness, method='pipe')

            np.save('../data/pipe roughness.npy', pipe_roughness, allow_pickle=True)

        return pipe_roughness

    def generate_model(self):
        base_demand_value = self.generate_demand()
        pipe_roughness = self.generate_roughness()
        wn = self.wn_tr
        for key, value in base_demand_value.items():
            base_demand_value[key] = value * self.ratio

        base_demand_value['leak_pos'] = 0

        for junction in wn.junction_name_list:
            juu = wn.get_node(junction)
            juu.demand_timeseries_list[0].base_value = base_demand_value[junction]

        for pipe_name in wn.pipe_name_list:
            pipe = wn.get_link(pipe_name)
            pipe.roughness = pipe_roughness[pipe_name]

        return wn, base_demand_value


class DataGenerator:
    def __init__(self, wn, demand, ratio, path, nonleak_data=False, failure_pipe='198', leak_size=0.1, type='pressure',
                 last_time=50, noise = 0.01):
        self.wn = wn
        self.failure_pipe = failure_pipe
        self.type = type
        self.last_time = last_time
        self.demand = demand
        self.leak_size = leak_size
        self.ratio = ratio
        self.path = path
        self.nonleak_data = nonleak_data
        self.noise = noise

    def base_demand(self, base, devation, method):
        if method == 'gauss':
            demand = random.gauss(base, devation)
        elif method == 'uniform':
            demand = random.uniform(0.008, 0.012)
        else:
            print("sorry, we don't have such method!")

        return abs(demand)

    def node_result(self, result, index, leak):
        resu = []
        fina_r = []
        for i in result:
            pp = result[i].node[index]
            resu.append(pp)
        if leak is False:
            final = pd.concat(resu)
        if leak is True:
            final = pd.concat(resu).drop(['leak_pos'], axis=1)
        return final.values

    def hydraulic_model(self, wn, leak_position, leak, demand, failure=None, leak_size=0.1):
        # This part shoud input a modified wn model. Which means the base demand at each node is well difined
        # and the leak position is well d'efined.
        resu = {}
        wn.reset_initial_values()

        for junction_name, junction in wn.junctions():
            junction.minimum_pressure = 5.0
            junction.nominal_pressure = 30.0

        print('leak_size for this scenario is %.3f, leak is %s, failure pipe is %s' % (leak_size, leak, failure))
        for t in tqdm(range(self.last_time)):
            wn.options.time.duration = t*wn.options.time.report_timestep
            for junction in wn.junction_name_list:
                ba_va = demand[junction]
                juu = wn.get_node(junction)
                juu.demand_timeseries_list[0].base_value = self.base_demand(ba_va, devation=self.noise, method='gauss')

            if leak is True:
                junction_leak = wn.get_node(leak_position)
                if junction_leak.demand_timeseries_list[0].base_value != 0:
                    junction_leak.demand_timeseries_list[0].base_value = 0
                #             print('check the demand at leaking is %f'%junction_leak.demand_timeseries_list[0].base_value)
                junction_leak.remove_leak(wn)

                r = leak_size * (t / self.last_time) ** (0.5)
                #             leak_size_r = random.uniform(1.0*leak_size, 1.5*leak_size)
                leak_size_r = leak_size
                junction_leak.add_leak(wn, area=3.1415926 * (leak_size_r / 2) ** 2, start_time=t)
                sim = wntr.sim.WNTRSimulator(wn, mode='PDD')
                result = sim.run_sim()
                resu[t] = result
            else:
                sim = wntr.sim.WNTRSimulator(wn, mode='PDD')
                result = sim.run_sim()
                resu[t] = result
            final_pressure = self.node_result(resu, 'pressure', leak)

        return final_pressure

    # this part we want to modify the base_demand model
    def stochastic_model(self, wn_tr, leak, base_demand_r, failure_pipe=None, head_file_name=None):
        wn = wn_tr
        training_data_pressure = []
        if leak is False:
            last_time = time.time()
            #         print('leak is False!')
            t = 1
            for iterations in range(1):
                if t % 10 == 0:
                    print('this is the %d time in %d' % (t, 100))
                t += 1
                head = self.hydraulic_model(wn, leak_position=0, leak=False, demand=base_demand_r)
                training_data_pressure.append([head, '0'])
                np.save(head_file_name, training_data_pressure)
                wn = wn_tr
            print('non leak loop took {} seconds'.format(time.time() - last_time))
        if leak is True:
            t = 1
            for failure in [failure_pipe]:
                print('this is the %d pipe in %d pipes' % (t, len(wn.pipe_name_list)))
                last_time = time.time()
                t += 1
                wn = wntr.morph.split_pipe(wn, failure, failure + '_B', 'leak_pos')
                for iterations in range(1):
                    leak_size = self.leak_size
                    pressure = self.hydraulic_model(wn, leak=leak, demand=base_demand_r,
                                                    leak_position='leak_pos', failure=failure,
                                                    leak_size=leak_size)
                    training_data_pressure.append([pressure, failure])

                    np.save(head_file_name, training_data_pressure)

                print('loop took {} seconds'.format(time.time() - last_time))

    def compute(self):
        rate = round(self.ratio, 2)

        # head_file_name = '../data/training_data_pressure_198_pipe_{}_leak'.format(rate)

        if self.nonleak_data:
            head_file_name = '../data/nonleak/training_data_pressure_none_leak_{}'.format(rate)
            self.stochastic_model(self.wn, leak=False, base_demand_r=self.demand, failure_pipe=self.failure_pipe,
                                  head_file_name=head_file_name)
        else:
            head_file_name = os.path.join(self.path, 'training_data_pressure_{}'.format(rate))
            self.stochastic_model(self.wn, leak=True, base_demand_r=self.demand, failure_pipe=self.failure_pipe,
                                  head_file_name=head_file_name)


class DataPreprocess:

    def __init__(self, wn, ratio_range, pipe):
        self.ratio = ratio_range
        self.wn = wn
        self.pipe = pipe

    def get_data(self):
        wn = self.wn
        leak_pressure = []
        non_leak_pressure = []
        for rate in self.ratio:
            rate = round(rate, 2)
            leak = pd.DataFrame(
                np.load('../data/pipe_{}/training_data_pressure_{}.npy'.format(self.pipe, rate), allow_pickle=True)[0][0],
                columns=wn.node_name_list)
            non_leak = pd.DataFrame(
                np.load('../data/nonleak/training_data_pressure_none_leak_{}.npy'.format(rate), allow_pickle=True)[0][0],
                columns=wn.node_name_list)
            leak_pressure.append(leak)
            non_leak_pressure.append(non_leak)

        leaking_data = pd.concat(leak_pressure)
        non_leaking_data = pd.concat(non_leak_pressure)
        return leaking_data, non_leaking_data
