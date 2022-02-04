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
        '../../data/network/minecraft_data/Java_E5_2U4R_20211201/PC/processed_data/throughput/stand_still_TCP_throughput.csv')
    mcpc_cpp_throughput = pd.read_csv(
        '../../data/network/minecraft_data/CPP_Version_20211202/PC/processed_data/throughput/stand_still_UDP_throughput.csv')
    rblx = PacketProcessor(server_ip='183.47.97.149',
                           server_port=17658,
                           client_ip='192.168.65.174',
                           client_port=62930,
                           packets='stand_still_with_creatures',
                           protocol='RakNet',
                           RAW_DATA_ROOT=roblox_data_root + 'raw_data/',
                           PROCESSED_DATA_ROOT=roblox_data_root + 'processed_data/')
    rblx.parse_throughput(suffix='_UDP')
    rblx_throughput_udp = pd.read_csv(
        '../../data/network/roblox_data/processed_data/throughput/stand_still_with_creatures_UDP_throughput.csv')
    vrc_throughput_udp = pd.read_csv(
        '../../data/network/vrchat_data/20211202_01\processed_data/throughput/stand_still_UDP_throughput.csv')
    vrc_throughput_tcp = pd.read_csv(
        '../../data/network/vrchat_data/20211202_01\processed_data/throughput/stand_still_TCP_throughput.csv')

    style_idx = 0


    def throughput_cdf(data=None,
                       labels=None):

        if labels is None:
            labels = ['Roblox UDP Flow',
                      'VRChat UDP Flow',
                      'VRChat TCP Flow',
                      'Minecraft (Java) TCP Flow',
                      'Minecraft (C++) UDP Flow']
        if data is None:
            data = [rblx_throughput_udp.egress.values * 10 / 1024,
                    vrc_throughput_udp.egress.values * 10 / 1024,
                    vrc_throughput_tcp.egress.values * 10 / 1024,
                    mcpc_java_throughput.egress.values * 10 / 1024,
                    mcpc_cpp_throughput.egress.values * 10 / 1024]

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

        for d, l in zip(data, labels):
            plot_cdf(d, l)

        plt.ylabel('CDF')
        plt.xlabel('Throughput (KBytes/s)')
        plt.xlim(-10)
        plt.legend()
        plt.savefig('../../figs/section5/fig4.pdf')
        plt.show()
        return


    def throughput_bar(data=None,
                       labels=None):

        if labels is None:
            labels = ['Roblox UDP Flow',
                      'VRChat UDP Flow',
                      'VRChat TCP Flow',
                      'Minecraft (Java) TCP Flow',
                      'Minecraft (C++) UDP Flow']
        if data is None:
            data = [rblx_throughput_udp.egress.values * 10 / 1024,
                    vrc_throughput_udp.egress.values * 10 / 1024,
                    vrc_throughput_tcp.egress.values * 10 / 1024,
                    mcpc_java_throughput.egress.values * 10 / 1024,
                    mcpc_cpp_throughput.egress.values * 10 / 1024]
        percentiles = (50, 75, 90, 95, 100,)
        y = pd.DataFrame(
            columns=["0-%d%%" % percentiles[0]] + ["%d%%-%d%%" % (percentiles[i], percentiles[i + 1]) for i in
                                                   range(len(percentiles) - 1)], index=labels)
        for d, l in zip(data, labels):
            percentile_diff = np.insert(np.diff([np.percentile(d, p) for p in percentiles]), 0,
                                        np.percentile(d, percentiles[0]))
            y.loc[l] = percentile_diff
        y.plot(kind="bar", stacked=True, colormap="Blues_r", ax=ax, zorder=3, edgecolor='black', linewidth=1)
        ax.grid(zorder=0)
        ax.set_xticklabels(labels=labels, rotation=15, minor=False, fontsize=8)
        ax.set_ylabel("Egress Throughput (KBytes/s)", fontsize=12)
        plt.savefig('../../figs/section5/fig5.pdf')
        plt.show()
        return


    def throughput_timeline(group=5, data=None,
                            labels=None, fig_path='../../figs/section5/fig6.pdf'):
        if labels is None:
            labels = ['Roblox UDP Flow',
                      'VRChat UDP Flow',
                      'VRChat TCP Flow',
                      'Minecraft (Java) TCP Flow',
                      'Minecraft (C++) UDP Flow']
        if data is None:
            data = [rblx_throughput_udp,
                    vrc_throughput_udp,
                    vrc_throughput_tcp,
                    mcpc_java_throughput,
                    mcpc_cpp_throughput]
        style_idx = 0

        for d, l in zip(data, labels):
            times = [d.time.values[i * group] for i in range(math.ceil(len(d.egress.values) / group))]
            throughput_sum = np.add.reduceat(d.egress.values, np.arange(0, len(d.egress.values), group)) / (
                        0.1 * group) / 1024
            ax.plot(times, throughput_sum, lw=2, label=l, ls=list(linestyles.items())[style_idx][1])
            style_idx = (style_idx + 1) % len(list(linestyles.items()))
        ax.set_xlabel('time (s)')
        ax.set_ylabel("Egress Throughput (KBytes/s)", fontsize=12)
        plt.legend()
        plt.savefig(fig_path)
        plt.show()
        return


    def effect():
        root = 'network/minecraft_data/Java_Pan_2U4R_20220121/PC/'
        short_fancy = PacketProcessor(server_ip='123.57.84.206',
                                      server_port=25565,
                                      client_ip='192.168.65.66',
                                      client_port=59406,
                                      packets='move_chunk_load_8_fancy',
                                      protocol='TCP',
                                      RAW_DATA_ROOT=root + 'raw_data/',
                                      PROCESSED_DATA_ROOT=root + 'processed_data/')
        short_fancy_t = short_fancy.parse_throughput(suffix='_TCP')
        double_fancy = PacketProcessor(server_ip='123.57.84.206',
                                       server_port=25565,
                                       client_ip='192.168.65.66',
                                       client_port=59406,
                                       packets='move_chunk_load_16_fancy',
                                       protocol='TCP',
                                       RAW_DATA_ROOT=root + 'raw_data/',
                                       PROCESSED_DATA_ROOT=root + 'processed_data/')
        double_fancy_t = double_fancy.parse_throughput(suffix='_TCP')
        double_fast = PacketProcessor(server_ip='123.57.84.206',
                                      server_port=25565,
                                      client_ip='192.168.65.66',
                                      client_port=62755,
                                      packets='move_chunk_load_16_fast',
                                      protocol='TCP',
                                      RAW_DATA_ROOT=root + 'raw_data/',
                                      PROCESSED_DATA_ROOT=root + 'processed_data/')
        double_fast_t = double_fast.parse_throughput(suffix='_TCP')
        short_fast = PacketProcessor(server_ip='123.57.84.206',
                                     server_port=25565,
                                     client_ip='192.168.65.66',
                                     client_port=62755,
                                     packets='move_chunk_load_8_fast',
                                     protocol='TCP',
                                     RAW_DATA_ROOT=root + 'raw_data/',
                                     PROCESSED_DATA_ROOT=root + 'processed_data/')
        short_fast_t = short_fast.parse_throughput(suffix='_TCP')
        throughput_timeline(5, [short_fancy_t,
                                double_fancy_t,
                                short_fast_t, double_fast_t],
                            ['Short Range (Fancy)', 'Double Range (Fancy)', 'Short Range (Fast)', 'Double Range (Fast)'],
                            '../../figs/section5/fig12.pdf')


    # throughput_cdf()
    # throughput_bar()
    # throughput_timeline()
    effect()
