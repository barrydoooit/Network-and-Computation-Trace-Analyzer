import math

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline, interp1d

# matplotlib.style.use("ggplot")
from PacketProcessor import PacketProcessor, DATA_ROOT
from utils.figure_configs import linestyles

roblox_data_root = 'network/roblox_data/'

if __name__ == '__main__':
    fig, ax = plt.subplots(dpi=80)
    fig.subplots_adjust(bottom=0.2)
    plt.grid()
    mcpc_java_throughput = pd.read_csv(
        '../../data/network/minecraft_data/Java_E5_2U4R_20211201/PC/processed_data/throughput/connect_TCP_throughput.csv')
    mcpc_cpp_throughput = pd.read_csv(
        '../../data/network/minecraft_data/CPP_Version_20211202/PC/processed_data/throughput/connect_UDP_throughput.csv')
    mcpe_cpp_throughput = pd.read_csv(
        '../../data/network/minecraft_data/CPP_Version_20211202/PE/processed_data/throughput/connect_UDP_throughput.csv')

    mcpc_rtt_raknet = pd.read_csv(
        '../../data/network/minecraft_data/CPP_Version_20211202/PC/processed_data/rtt/connect_rtt.csv')
    mcpc_rtt_raknet = mcpc_rtt_raknet[mcpc_rtt_raknet.is_pingpong == 0]
    mcpe_rtt_raknet = pd.read_csv(
        '../../data/network/minecraft_data/CPP_Version_20211202/PE/processed_data/rtt/connect_rtt.csv')
    mcpe_rtt_raknet = mcpc_rtt_raknet[mcpc_rtt_raknet.is_pingpong == 0]
    mcpc_rtt_tcp = pd.read_csv(
        '../../data/network/minecraft_data/Java_E5_2U4R_20211201/PC/processed_data/rtt/connect_rtt.csv')
    style_idx = 0

    def parse_rtt(rtt, interval=0.5):
        time = interval
        rtt_sum = 0
        count = 0
        df = pd.DataFrame(columns=['time', 'rtt'])
        for _, row in rtt.iterrows():
            while time < row.time:
                res = 0
                if count == 0:
                    if len(df.index) > 0:
                        res =  df.at[len(df.index)-1, 'rtt']
                else:
                    res = rtt_sum/count
                df.loc[len(df.index)] = [time, res]
                time += interval
                rtt_sum = 0
                count = 0
            rtt_sum += row.sample_rtt
            count += 1
        return df


    mcpc_rtt_raknet = parse_rtt(mcpc_rtt_raknet)
    mcpe_rtt_raknet = parse_rtt(mcpe_rtt_raknet)
    mcpc_rtt_tcp = parse_rtt(mcpc_rtt_tcp)
    mcpe_rtt_raknet.to_csv('../../cache/temp.csv')

    def parse_throughput(rtt, interval=0.5):
        time = interval
        egress_sum = 0
        df = pd.DataFrame(columns=['time', 'egress'])
        for _, row in rtt.iterrows():
            while time < row.time:

                df.loc[len(df.index)] = [time, egress_sum]
                time += interval
                egress_sum = 0
            egress_sum += row.egress
        return df


    mcpc_java_throughput = parse_throughput(mcpc_java_throughput)
    mcpc_cpp_throughput = parse_throughput(mcpc_cpp_throughput)
    mcpe_cpp_throughput = parse_throughput(mcpe_cpp_throughput)

    def index_a():
        alpha = 0.002
        beta = -1
        mcpc_java = pd.concat([mcpc_java_throughput.set_index('time'),
                               mcpc_rtt_tcp.set_index('time')], axis=1, join='inner')
        mcpc_java['idx'] = alpha*mcpc_java.egress + beta * mcpc_java.rtt
        mcpc_cpp = pd.concat([mcpc_cpp_throughput.set_index('time'),
                               mcpc_rtt_raknet.set_index('time')], axis=1, join='inner')
        mcpc_cpp['idx'] = alpha * mcpc_cpp.egress + beta * mcpc_cpp.rtt
        mcpe_cpp = pd.concat([mcpe_cpp_throughput.set_index('time'),
                               mcpe_rtt_raknet.set_index('time')], axis=1, join='inner')
        mcpe_cpp['idx'] = alpha * mcpe_cpp.egress + beta * mcpe_cpp.rtt

        print(mcpe_cpp.head())
        mcpc_java.plot( y='idx', ax=ax, label='Minecraft Java PC')
        mcpc_cpp.plot(y='idx', ax=ax, label='Minecraft C++ PC')
        mcpe_cpp.plot(y='idx', ax=ax, label='Minecraft C++ Mobile')
        plt.grid()
        plt.ylabel('Index A = 0.002*throughput - rtt')
        plt.savefig('../../figs/section4/fig17.pdf')
        plt.show()
    index_a()