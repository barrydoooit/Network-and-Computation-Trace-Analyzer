import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline, interp1d

# matplotlib.style.use("ggplot")
from PacketProcessor import PacketProcessor, DATA_ROOT

minecraft_pe_data_root = 'network/minecraft_data/CPP_Version_20211202/PE/'
minecraft_pc_data_root = 'network/minecraft_data/CPP_Version_20211202/PC/'
if __name__ == '__main__':
    mcpe = PacketProcessor(server_ip='52.166.239.43',
                         server_port=19132,
                         client_ip='192.168.67.64',
                         client_port=62990,
                         packets='connect',
                         protocol='RakNet',
                         RAW_DATA_ROOT=minecraft_pe_data_root + 'raw_data/',
                         PROCESSED_DATA_ROOT=minecraft_pe_data_root + 'processed_data/')
    #mcpe.to_dataframe(cover=True)
    mcpe.parse_sample_rtt(cover=False)

    mcpc = PacketProcessor(server_ip='52.166.239.43',
                           server_port=19132,
                           client_ip='192.168.67.64',
                           client_port=53169,
                           packets='connect',
                           protocol='RakNet',
                           RAW_DATA_ROOT=minecraft_pc_data_root + 'raw_data/',
                           PROCESSED_DATA_ROOT=minecraft_pc_data_root + 'processed_data/')
    #mcpc.to_dataframe(cover=True)
    mcpc.parse_sample_rtt(cover=False)

    # vr_tcp = PacketProcessor(server_ip='13.225.100.227',
    #                          server_port=443,
    #                          client_ip='192.168.67.64',
    #                          client_port=54735)
    fig, ax = plt.subplots(dpi=80)
    fig.subplots_adjust(bottom=0.2)
    mcpc_rtt_raknet = pd.read_csv(
        '../../data/network/minecraft_data/CPP_Version_20211202/PC/processed_data/rtt/connect_rtt.csv')
    mcpc_rtt_raknet = mcpc_rtt_raknet[mcpc_rtt_raknet.is_pingpong == 0]
    mcpc_rtt_tcp = pd.read_csv(
        '../../data/network/minecraft_data/Java_E5_2U4R_20211201/PC/processed_data/rtt/connect_rtt.csv')
    def delay_cdf():
        return
    def delay_timeline():
        times = mcpc_rtt_raknet.time.values
        # xnew = np.linspace(times.min(), times.max(), 100)
        # spl = make_interp_spline(times, mcpc_rtt_raknet.sample_rtt.values, k=1)
        # value_np_smooth = spl(xnew)
        # plt.plot(xnew, value_np_smooth)

        mcpc_rtt_raknet.plot(x='time', y='sample_rtt', ax=ax, label="Minecraft RakNet")
        mcpc_rtt_tcp.plot(x='time', y='sample_rtt', ax=ax, label="Minecraft TCP")
        ax.legend()
        ax.set_xlabel("time (s)")
        ax.set_ylabel("RTT (ms)")
        ax.grid()
        plt.savefig('../../figs/section4/fig3.pdf')
        plt.show()

    delay_timeline()