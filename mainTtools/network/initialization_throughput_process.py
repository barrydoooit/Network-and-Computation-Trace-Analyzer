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
    rblx = PacketProcessor(server_ip='183.47.97.149',
                           server_port=17658,
                           client_ip='192.168.65.174',
                           client_port=62930,
                           packets='connect',
                           protocol='RakNet',
                           RAW_DATA_ROOT=roblox_data_root + 'raw_data/',
                           PROCESSED_DATA_ROOT=roblox_data_root + 'processed_data/')
    rblx.parse_throughput(suffix='_UDP')
    rblx_throughput_udp = pd.read_csv(
        '../../data/network/roblox_data/processed_data/throughput/connect_UDP_throughput.csv')
    vrc_throughput_udp = pd.read_csv(
        '../../data/network/vrchat_data/20211202_01\processed_data/throughput/connect_UDP_throughput.csv')
    vrc_throughput_tcp = pd.read_csv(
        '../../data/network/vrchat_data/20211202_01\processed_data/throughput/connect_TCP_throughput.csv')

    style_idx = 0
    def throughput_cdf():

        def plot_cdf(y, label):
            global style_idx
            count, bins_count = np.histogram(y, bins=2000)
            pdf = count / sum(count)
            cdf = np.cumsum(pdf)
            ax.plot(bins_count[1:], cdf, label=label, lw=1.5, ls=list(linestyles.items())[style_idx][1])
            style_idx = (style_idx + 1) % len(list(linestyles.items()))
            # y = cdf
            # x = bins_count[1:]
            # X_Y_Spline = make_interp_spline(x, y)
            # X_ = np.linspace(x.min(), x.max(), 10000)
            # Y_ = X_Y_Spline(X_)
            # ax.plot(X_, Y_, label=label, ls='-.')

        plot_cdf(rblx_throughput_udp.egress.values*10/1024, 'Roblox RakNet Flow')
        plot_cdf(vrc_throughput_udp.egress.values*10/1024, 'VRChat UDP Flow')
        plot_cdf(vrc_throughput_tcp.egress.values*10/1024, 'VRChat TCP Flow')
        plot_cdf(mcpc_java_throughput.egress.values*10/1024, 'Minecraft (Java) TCP Flow')
        plot_cdf(mcpc_cpp_throughput.egress.values*10/1024, 'Minecraft (C++) RakNet Flow')
        plt.xlim(-20, 750)
        plt.ylabel('CDF')
        plt.xlabel('Throughput (KBytes/s)')
        plt.savefig('../../figs/section4/fig4.pdf')
        plt.legend()
        plt.show()
        return


    def throughput_bar():
        data = [rblx_throughput_udp.egress.values,
                vrc_throughput_udp.egress.values,
                vrc_throughput_tcp.egress.values,
                mcpc_java_throughput.egress.values,
                mcpc_cpp_throughput.egress.values]
        labels = ['Roblox RakNet Flow',
                  'VRChat UDP Flow',
                  'VRChat TCP Flow',
                  'Minecraft (Java) TCP Flow',
                  'Minecraft (C++) RakNet Flow']
        percentiles = (50, 75, 90, 95, 100,)
        y = pd.DataFrame(
            columns=["0-%d%%" % percentiles[0]] + ["%d%%-%d%%" % (percentiles[i], percentiles[i + 1]) for i in
                                                   range(len(percentiles) - 1)], index=labels)
        for d, l in zip(data, labels):
            d = d / 0.1 / 1024
            percentile_diff = np.insert(np.diff([np.percentile(np.log2(d+1), p) for p in percentiles]), 0,
                                        np.percentile(np.log2(d+1), percentiles[0]))
            y.loc[l] = percentile_diff

        y.plot(kind="bar", stacked=True, colormap="Blues_r", ax=ax, zorder=3, edgecolor='black', linewidth=1)
        ax.grid(zorder=0)
        ax.set_xticklabels(labels=labels, rotation=15, minor=False, fontsize=8)
        ax.set_ylabel("Egress Throughput [Log2] (KBytes/s)", fontsize=12)
        plt.savefig('../../figs/section4/fig5.pdf')
        plt.show()
        return


    def throughput_timeline(group=5):
        style_idx = 0

        data = [rblx_throughput_udp,
                vrc_throughput_udp,
                vrc_throughput_tcp,
                mcpc_java_throughput,
                mcpc_cpp_throughput]
        labels = ['Roblox RakNet Flow',
                  'VRChat UDP Flow',
                  'VRChat TCP Flow',
                  'Minecraft (Java) TCP Flow',
                  'Minecraft (C++) RakNet Flow']

        for d, l in zip(data, labels):

            egress = d.egress.values / (0.1*group) / 1024
            print(l, ': ', np.average(egress))
            times = [d.time.values[i*group] for i in range(math.ceil(len(egress)/group))]
            throughput_sum = np.log2(np.add.reduceat(egress, np.arange(0, len(egress), group))+1)
            ax.plot(times, throughput_sum, lw=2, label=l, ls=list(linestyles.items())[style_idx][1])
            style_idx = (style_idx + 1) % len(list(linestyles.items()))
        ax.set_xlabel('time (s)')
        ax.set_ylabel("Egress Throughput [Log2] (KBytes/s)", fontsize=12)
        plt.legend()
        plt.savefig('../../figs/section4/fig6.pdf')
        plt.show()
        return



    # throughput_cdf()
    # throughput_bar()
    throughput_timeline()
