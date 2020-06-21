from ModelGeneratorClass import ModelGenerator
from ModelGeneratorClass import DataGenerator
from ModelGeneratorClass import DataPreprocess
import numpy as np
import wntr
import os
from multiprocessing import Pool
import SendEmail
import multiprocessing


# for link in wn_tr.link_name_list:
#     link = '198'
#     path = '../data/pipe_{}'.format(link)
#     if os.path.exists(path):
#         continue
#     else:
#         os.mkdir(path)


def generate_data(link):
    path = '../data/pipe_{}'.format(link)
    #    os.mkdir(path)
    wn_tr = wntr.network.WaterNetworkModel('../data/Fairfield_modified_2.inp')
    wn_tr.options.time.hydraulic_timestep = 1
    wn_tr.options.time.report_timestep = 1
    junction_1 = wn_tr.get_node('1')
    junction_1.base_head = 300
    for ratio in np.arange(0.3, 1.4, 0.1):
        model = ModelGenerator(wn=wn_tr, ratio=ratio)
        path = '../data/pipe_{}'.format(link)
        wn, demand = model.generate_model()
        Data = DataGenerator(wn=wn, path=path, demand=demand, ratio=ratio, failure_pipe=link, last_time=200)
        tt = Data.compute()


# if __name__ == '__main__':
#    pool = Pool(os.cpu_count()-1)
#    wn_tr = wntr.network.WaterNetworkModel('../data/Fairfield_modified_2.inp')
#    results = pool.map(generate_data, wn_tr.link_name_list)
#    SendEmail.send_email('computation done!\n\nThe computaion is finished!')

##Used to generate non leaking data


def nonleak(ratio):
    path = '../data/pipe_{}'.format('198')
    wn_tr = wntr.network.WaterNetworkModel('../data/Fairfield_modified_2.inp')
    wn_tr.options.time.hydraulic_timestep = 1
    wn_tr.options.time.report_timestep = 1
    junction_1 = wn_tr.get_node('1')
    junction_1.base_head = 300
    model = ModelGenerator(wn=wn_tr, ratio=round(ratio, 2))
    wn, demand = model.generate_model()
    Data = DataGenerator(wn=wn, path=path, demand=demand, ratio=ratio, failure_pipe='198', nonleak_data=True,
                         last_time=200)
    tt = Data.compute()


if __name__ == '__main__':
    pool = Pool(os.cpu_count() - 1)
    wn_tr = wntr.network.WaterNetworkModel('../data/Fairfield_modified_2.inp')
    results = pool.map(nonleak, np.arange(0.3, 1.4, 0.1))
