# created by Yukuan DING, modified by Yanting LIU, Lai WEI
# filter the pcapng traces using scapy
# filter the RTT log given different status
import scapy.all as scapy
import pandas as pd
from enum import Enum
import numpy as np
import os.path
import matplotlib.pyplot as plt


class Role(Enum):
    SERVER = 0
    CLIENT = 1


class PacketProcessor(object):
    def __init__(self, server_ip='222.187.232.216', server_port=11701,
                 client_ip='192.168.65.174', client_port=64798, packets='short_range_move',
                 protocol='TCP', RAW_DATA_ROOT = 'raw_data/', PROCESSED_DATA_ROOT = 'processed_data/'):
        self.RAW_DATA_ROOT = RAW_DATA_ROOT
        self.PROCESSED_DATA_ROOT = PROCESSED_DATA_ROOT
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_ip = client_ip
        self.client_port = client_port
        self.packets_path = packets
        self.protocol = protocol



    def filter(self):
        file_path = self.RAW_DATA_ROOT + 'filtered_caps/' + self.packets_path + '.pcap'
        IPs = [self.client_ip, self.server_ip]
        ports = [self.client_port, self.server_port]
        if os.path.isfile(file_path):
            filtered_packets = scapy.rdpcap(file_path)
        else:
            self.packets = scapy.rdpcap(self.RAW_DATA_ROOT + 'caps/' + self.packets_path + '.pcapng')
            filtered_packets = [pkt for pkt in self.packets if
                                "IP" in pkt and (pkt['IP'].src in IPs and pkt['IP'].dst in IPs) and
                                self.protocol in pkt and (pkt[self.protocol].sport in ports and pkt[self.protocol].dport in ports)]
            scapy.wrpcap(file_path, filtered_packets)
        return filtered_packets

    def to_dataframe(self, packets=None, log_level=False):
        file_path = self.RAW_DATA_ROOT + 'caps_csv/' + self.packets_path + '.csv'
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
        else:
            if packets is None:
                packets = self.filter()
            start_time = packets[0].time if len(packets) > 0 else 0
            print(start_time)
            if self.protocol == 'TCP':
                df = pd.DataFrame(columns=["time", "src", "dst", "seq", "ack", "len", "payload", "role"])
                start_seq = packets[0]['TCP'].seq - 1 if len(packets) > 0 else 0
                start_ack = packets[0]['TCP'].ack - 1 if len(packets) > 0 else 0
                start_side = Role.SERVER if packets[0]['TCP'].sport == self.server_port else Role.CLIENT
                for pkt in packets:
                    role = Role.SERVER if pkt['TCP'].sport == self.server_port else Role.CLIENT
                    if log_level:
                        print([float(pkt.time) - start_time, pkt['IP'].src, pkt['IP'].dst,
                               pkt['TCP'].seq - (start_seq if role == start_side else start_ack),
                               pkt['TCP'].ack - (start_seq if role != start_side else start_ack),
                               len(pkt['TCP'].payload), str(pkt['TCP'].payload), role])
                    df.loc[len(df.index)] = [float(pkt.time) - start_time, pkt['IP'].src, pkt['IP'].dst,
                                             pkt['TCP'].seq - (start_seq if role == start_side else start_ack),
                                             pkt['TCP'].ack - (start_seq if role != start_side else start_ack),
                                             len(pkt['TCP'].payload), pkt['TCP'].payload, role]
            elif self.protocol == 'UDP':
                df = pd.DataFrame(columns=["time", "src", "dst", "len", "role"])
                for pkt in packets:
                    role = Role.SERVER if pkt['UDP'].sport == self.server_port else Role.CLIENT
                    df.loc[len(df.index)] = [pkt.time - start_time, pkt['IP'].src, pkt['IP'].dst, len(pkt['UDP'].payload), role]
            df.to_csv(file_path)
        if log_level:
            print(df.info())
        return df

    def parse_sample_rtt(self, filtered_packets=None):
        file_path = self.PROCESSED_DATA_ROOT + 'rtt/' + self.packets_path + '_rtt.csv'
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
        else:
            if filtered_packets is None:
                filtered_packets = self.to_dataframe()
            df = pd.DataFrame(columns=["time", "estimated_rtt", "sample_rtt"])
            for idx, row in filtered_packets.iterrows():
                if row.role == Role.CLIENT and row.len > 0:
                    origin_pkt = row
                    for n_idx, n_row in filtered_packets.iloc[idx + 1: idx + 201, :].iterrows():
                        if n_row.role == Role.SERVER:
                            df.loc[len(df.index)] = [n_row.time, 0, (n_row.time - origin_pkt.time) * 1000]
                            break
            df.to_csv(file_path)
        return df

    def parse_throughput(self, filtered_packets=None):#UNTESTED
        if filtered_packets is None:
            filtered_packets = self.to_dataframe()
        df = pd.DataFrame(columns=["time", "ingress", "egress"])
        time = 1
        ingress = 0
        egress = 0
        for _, row in filtered_packets.iterrows():
            while time < row.time:
                df.loc[len(df.index)] = [time, ingress, egress]
                time += 1
                ingress = 0
                egress = 0
            if row.role == Role.CLIENT:
                ingress += row.len
            else:
                egress += row.len
        return df

    def parse_packet_len_count(self, filtered_packets=None):
        file_path_ingress = self.PROCESSED_DATA_ROOT + 'pkt_len_distribution/' + self.packets_path + '_pktld_ingress.csv'
        file_path_egress = self.PROCESSED_DATA_ROOT + 'pkt_len_distribution/' + self.packets_path + '_pktld_egress.csv'
        if os.path.isfile(file_path_ingress) and os.path.isfile(file_path_egress):
            df_ingress = pd.read_csv(file_path_ingress, index_col=0)
            df_egress = pd.read_csv(file_path_egress, index_col=0)
        else:
            if filtered_packets is None:
                filtered_packets = self.to_dataframe()
            egress_pkts = filtered_packets[filtered_packets.role == 'Role.CLIENT']
            ingress_pkts = filtered_packets[filtered_packets.role == 'Role.SERVER']
            max_pkt_size_egress = egress_pkts.len.max()
            min_pkt_size_egress = egress_pkts[egress_pkts.len > 0].len.min()
            max_pkt_size_ingress = ingress_pkts.len.max()
            min_pkt_size_ingress = ingress_pkts[ingress_pkts.len > 0].len.min()
            print(max_pkt_size_egress)
            idx_to_pkt_size_egress = np.zeros(max_pkt_size_egress + 1, dtype=int)
            column_list_egress = ["time"] + [str(i) for i in range(min_pkt_size_egress, max_pkt_size_egress + 1)]
            print("length is %d" % len(column_list_egress))
            df_egress = pd.DataFrame(columns=column_list_egress)
            packet_count_egress = np.zeros(max_pkt_size_egress + 1, dtype=int)
            time_egress = 1
            for _, row in egress_pkts.iterrows():
                while time_egress < row.time:
                    new = [time_egress] + packet_count_egress[min_pkt_size_egress:].tolist()
                    df_egress.loc[df_egress.shape[0]] = new
                    packet_count_egress = np.zeros(max_pkt_size_egress + 1, dtype=int)
                    time_egress += 1
                pkt_size = row.len
                if idx_to_pkt_size_egress[pkt_size] == 0:
                    idx_to_pkt_size_egress[pkt_size] = 1
                packet_count_egress[pkt_size] += 1
            df_egress.loc[df_egress.shape[0]] = [time_egress] + packet_count_egress[min_pkt_size_egress:].tolist()
            df_egress = df_egress.drop(columns=[str(i) for i in range(min_pkt_size_egress, max_pkt_size_egress + 1) if idx_to_pkt_size_egress[i] == 0])
            df_egress.to_csv(file_path_egress)

            idx_to_pkt_size_ingress = np.zeros(max_pkt_size_ingress + 1, dtype=int)
            column_list_ingress = ["time"] + [str(i) for i in range(min_pkt_size_ingress, max_pkt_size_ingress + 1)]
            df_ingress = pd.DataFrame(columns=column_list_ingress)
            packet_count_ingress = np.zeros(max_pkt_size_ingress + 1, dtype=int)
            time_ingress = 1
            for _, row in ingress_pkts.iterrows():
                while time_ingress < row.time:
                    new = [time_ingress] + packet_count_ingress[min_pkt_size_ingress:].tolist()
                    df_ingress.loc[df_ingress.shape[0]] = new
                    packet_count_ingress = np.zeros(max_pkt_size_ingress + 1, dtype=int)
                    time_ingress += 1
                pkt_size = row.len
                if idx_to_pkt_size_ingress[pkt_size] == 0:
                    idx_to_pkt_size_ingress[pkt_size] = 1
                packet_count_ingress[pkt_size] += 1
            df_ingress.loc[df_ingress.shape[0]] = [time_ingress] + packet_count_ingress[min_pkt_size_ingress:].tolist()
            df_ingress = df_ingress.drop(columns=[str(i) for i in range(min_pkt_size_ingress, max_pkt_size_ingress + 1) if idx_to_pkt_size_ingress[i] == 0])
            df_ingress.to_csv(file_path_ingress)
        return df_egress, df_ingress

    def get_top_n_freq_packets(self, n=10):
        df_egress, df_ingress = self.parse_packet_len_count()
        pkt_cnt_sum_egress = df_egress.drop(columns=['time']).sum(axis=0).to_frame().nlargest(n, 0)
        pkt_cnt_sum_ingress = df_ingress.drop(columns=['time']).sum(axis=0).to_frame().nlargest(n, 0)
        return pkt_cnt_sum_egress, pkt_cnt_sum_ingress

    def get_pkt_len_division_list(self, division_threshold=[0, 50, 100, 1000, 10000]):
        df_egress, df_ingress = self.parse_packet_len_count()
        df_grouped_egress = pd.DataFrame()
        df_grouped_ingress = pd.DataFrame()
        for i in range(1, len(division_threshold)):
            df = df_egress.drop(columns=['time'])[[j for j in df_egress.drop(columns=['time']).columns.values
                            if (int(j) <= division_threshold[i] and int(j) >division_threshold[i-1])]]
            df_grouped_egress["%d-%d"%(division_threshold[i-1]+1, division_threshold[i])] = df.sum(axis=1)
        for i in range(1, len(division_threshold)):
            df = df_ingress.drop(columns=['time'])[[j for j in df_ingress.drop(columns=['time']).columns.values
                            if (int(j) <= division_threshold[i] and int(j) >division_threshold[i-1])]]
            df_grouped_ingress["%d-%d"%(division_threshold[i-1]+1, division_threshold[i])] = df.sum(axis=1)
        return df_grouped_egress, df_grouped_ingress

    """def test_multiple_violin_plot(self, roblox, mc):
        X = ['Connecting', 'Waiting']
        X_axis = np.arange(len(X))
        vp1 = plt.violinplot(roblox, X_axis - 0.2, 0.2)
        vp2 = plt.violinplot(mc, X_axis + 0.2, 0.2)
        plt.xticks(X_axis, X)
        plt.xlabel("Scenarios")
        plt.ylabel("Thoughput")
        plt.title("Throughput of MC and Roblox in Different Scenarios")
        #plt.legend()
        plt.show()"""

#import seaborn as sns
class PacketPlotter(object):
    def __init__(self, processors=None):
        if processors is None:
            processors = []
        self.processors = processors

    def plt_pkt_len_breakup(self):
        df_egress, df_ingress = self.processors[0].get_pkt_len_division_list()
        df_egress.plot(kind='bar', stacked=True)
        plt.title('Roblox '+self.processors[0].packets_path+' Packet Length Distribution - egress')
        plt.xlabel("Time (s)")
        plt.ylabel("Packet Count")
        plt.show()

        plt.figure(500)
        df_ingress.plot(kind='bar', stacked=True)
        plt.title('Roblox '+self.processors[0].packets_path+' Packet Length Distribution - ingress')
        plt.xlabel("Time (s)")
        plt.ylabel("Packet Count")
        plt.show()
if __name__ == "__main__":
    MC_PATH = '../data/network/vrchat_data/20211202_01'
    pp = PacketProcessor(packets='connect',RAW_DATA_ROOT=MC_PATH+'raw_data/', PROCESSED_DATA_ROOT=MC_PATH+'processed_data/')
    pp.parse_sample_rtt()
    """filtered_df = None
    status_list = ['connect','fast_chunk_load','fast_chunk_reload',
              'short_range_move','stan_still_with_creatures',
              'stand_still']
    cache_root = 'rtt'
    pcapng_root = 'caps'
    for status in status_list:
        cache_path = os.path.join(cache_root,status+'_rtt.csv')
        pcapng_path = os.path.join(pcapng_root,status+'.pcapng')
        if os.path.isfile(cache_path):
            rtts = pd.read_csv(cache_path)
        else:
            pp = PacketProcessor(server_ip='222.187.232.216', server_port=11701,
                 client_ip='192.168.65.174', client_port=64798, packets=pcapng_path,
                 protocol='TCP')
            filtered_df = pp.to_dataframe(pp.filter(), log_level=False)
            filtered_df.to_csv(cache_path)
            rtts = pp.parse_sample_rtt(filtered_df)
            rtts.to_csv(cache_path)
        plt.title("sample RTT when editing world")
        plt.plot(rtts["time"], rtts["sample_rtt"])
        plt.xlabel("time(s)")
        plt.ylabel("Sample RTT (ms)")
        plt.grid()
        plt.savefig(status+".png")
        plt.close()"""
    # plt.show()
    # pp.test_violin_plot()
