# created by Yukuan DING, modified by Yanting LIU, Lai WEI
# filter the pcapng traces using scapy
# filter the RTT log given different status
import scapy.all as scapy
import pandas as pd
from enum import Enum
import numpy as np
import os.path
import matplotlib.pyplot as plt
from scapy.packet import Raw
from scapy.utils import hexdump


class Role(Enum):
    SERVER = 0
    CLIENT = 1


class Protocol(Enum):
    TCP = 0
    UDP = 1
    RakNet = 2


DATA_ROOT = '../../data/'


class PacketProcessor(object):
    def __init__(self, server_ip, server_port,
                 client_ip, client_port, packets='short_range_move',
                 protocol='TCP', RAW_DATA_ROOT='raw_data/', PROCESSED_DATA_ROOT='processed_data/'):
        self.RAW_DATA_ROOT = DATA_ROOT + RAW_DATA_ROOT
        self.PROCESSED_DATA_ROOT = DATA_ROOT + PROCESSED_DATA_ROOT
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_ip = client_ip
        self.client_port = client_port
        self.packets_path = packets
        self.protocol = protocol
        """new_dirs = [RAW_DATA_ROOT+'filtered_caps', RAW_DATA_ROOT+'caps_csv',
                    PROCESSED_DATA_ROOT+'throughput', PROCESSED_DATA_ROOT+'rtt']
        for d in new_dirs:
            if not os.path.exists(d):
                os.mkdir(d)"""

    def filter(self, cover=False, protocol=''):
        file_path = self.RAW_DATA_ROOT + 'filtered_caps/' + self.packets_path + protocol + '.pcap'
        IPs = [self.client_ip, self.server_ip]
        ports = [self.client_port, self.server_port]
        if os.path.isfile(file_path) and not cover:
            filtered_packets = scapy.rdpcap(file_path)
        else:
            self.packets = scapy.rdpcap(self.RAW_DATA_ROOT + 'caps/' + self.packets_path + '.pcapng')
            filtered_packets = [pkt for pkt in self.packets if
                                "IP" in pkt and (pkt['IP'].src in IPs and pkt['IP'].dst in IPs) and
                                self.protocol in pkt and (
                                        pkt[self.protocol].sport in ports and pkt[self.protocol].dport in ports)]
            scapy.wrpcap(file_path, filtered_packets)
        return filtered_packets

    def to_dataframe(self, packets=None, log_level=False, cover=False,  protocol=''):
        file_path = self.RAW_DATA_ROOT + 'caps_csv/' + self.packets_path + protocol +  '.csv'
        if os.path.isfile(file_path) and not cover:
            df = pd.read_csv(file_path, index_col=0)
        else:
            if packets is None:
                packets = self.filter()
            start_time = packets[0].time if len(packets) > 0 else 0
            if self.protocol == 'TCP':
                df = pd.DataFrame(columns=["time", "src", "dst", "seq", "ack", "len", "payload", "role"])
                start_seq = packets[0]['TCP'].seq - 1 if len(packets) > 0 else 0
                start_ack = packets[0]['TCP'].ack - 1 if len(packets) > 0 else 0
                start_side = Role.SERVER.value if packets[0]['TCP'].sport == self.server_port else Role.CLIENT.value
                for pkt in packets:
                    role = Role.SERVER.value if pkt['TCP'].sport == self.server_port else Role.CLIENT.value
                    if log_level:
                        print([float(pkt.time) - start_time, pkt['IP'].src, pkt['IP'].dst,
                               pkt['TCP'].seq - (start_seq if role == start_side else start_ack),
                               pkt['TCP'].ack - (start_seq if role != start_side else start_ack),
                               len(pkt), str(pkt['TCP'].payload), role])
                    df.loc[len(df.index)] = [float(pkt.time - start_time), pkt['IP'].src, pkt['IP'].dst,
                                             pkt['TCP'].seq - (start_seq if role == start_side else start_ack),
                                             pkt['TCP'].ack - (start_seq if role != start_side else start_ack),
                                             len(pkt['TCP'].payload), pkt['TCP'].payload, role]
            elif self.protocol == 'UDP' or self.protocol == 'RakNet':
                df = pd.DataFrame(columns=["time", "src", "dst", "len", "role"])
                for pkt in packets:
                    role = Role.SERVER.value if pkt['UDP'].sport == self.server_port else Role.CLIENT.value
                    df.loc[len(df.index)] = [pkt.time - start_time, pkt['IP'].src, pkt['IP'].dst, len(pkt), role]
                if self.protocol == 'RakNet':
                    df['seq'] = -1
                    df['ack'] = -1
                    idx = 0
                    for pkt in packets:
                        if pkt['UDP'].sport == self.server_port:
                            if idx+1 == 1291:
                                print(hexdump(pkt[Raw].load, dump=True))
                        if True:
                            seq_arr = hexdump(pkt['UDP'].load, dump=True).split()[1: 5]
                            seq_arr.reverse()
                            if seq_arr[-1] == '84' or seq_arr[-1] == '8C':
                                seq = int(''.join(seq_arr[:-1]), 16)
                                df.loc[idx, 'seq'] = seq
                            else:
                                ack_arr_raw = hexdump(pkt['UDP'].load, dump=True).split()[1:]
                                ack_arr = []
                                for ack in ack_arr_raw:
                                    if len(ack) == 2:
                                        ack_arr.append(ack)

                                if ack_arr[0] == 'C0':

                                    record_count = int(ack_arr[2], 16)
                                    ack_df_val = []
                                    ack_idx = 3
                                    for record_idx in range(record_count):
                                        start_ack_arr = ack_arr[ack_idx + 1: ack_idx + 4]
                                        start_ack_arr.reverse()
                                        start_ack = int(''.join(start_ack_arr), 16)
                                        ack_df_val.append(str(start_ack))

                                        if ack_arr[ack_idx] == '00':
                                            end_ack_arr = ack_arr[ack_idx + 4: ack_idx + 7]
                                            end_ack_arr.reverse()
                                            end_ack = int(''.join(end_ack_arr), 16)
                                            ack_df_val.append(str(end_ack))
                                            ack_idx += 7
                                        else:
                                            ack_idx += 4
                                        ack_df_val.append("#")
                                    df.loc[idx, 'ack'] = (' '.join(ack_df_val))[:-1]
                        # elif Raw in pkt and pkt['UDP'].sport == self.server_port:
                        #     seq_arr = hexdump(pkt.lastlayer().load, dump=True).split()
                        #     print(seq_arr)
                        #     print()
                        idx += 1
            df.to_csv(file_path)
        if log_level:
            print(df.info())
        return df

    def parse_sample_rtt(self, filtered_packets=None, cover=False, suffix=''):
        file_path = self.PROCESSED_DATA_ROOT + 'rtt/' + self.packets_path + suffix + '_rtt.csv'
        if os.path.isfile(file_path) and not cover:
            df = pd.read_csv(file_path)
        else:
            if filtered_packets is None:
                filtered_packets = self.to_dataframe()
            if self.protocol == 'TCP':
                df = pd.DataFrame(columns=["time", "estimated_rtt", "sample_rtt"])
                for idx, row in filtered_packets.iterrows():
                    if row.role == Role.CLIENT.value and row.len > 0:
                        origin_pkt = row
                        for n_idx, n_row in filtered_packets.iloc[idx + 1: idx + 201, :].iterrows():
                            if n_row.role == Role.SERVER.value and n_row.ack == origin_pkt.len + origin_pkt.seq:
                                df.loc[len(df.index)] = [n_row.time, 0, (n_row.time - origin_pkt.time) * 1000]
                                break
            elif self.protocol == 'RakNet':
                df = pd.DataFrame(columns=["time", "sample_rtt", "is_pingpong"])
                pkt_server_idx = 0
                pkt_client_idx = 0
                for _, pkt in filtered_packets.iterrows():
                    if pkt.role == Role.SERVER.value:
                        if pkt.ack != '-1':
                            max_ack = 0
                            ack_arr = [record.split() for record in pkt.ack.split("#")]
                            for ack_range in ack_arr:
                                ack = int(ack_range[0])
                                if len(ack_range) == 2:
                                     ack = int(ack_range[1])
                                if ack > max_ack:
                                    max_ack = ack
                            source_pkt_time = -1.0
                            source_pkts = filtered_packets[(filtered_packets.role==Role.CLIENT.value) & (filtered_packets.seq != '-1')].reset_index(drop=True)
                            pkt_count = source_pkts.shape[0]
                            while pkt_server_idx < pkt_count:
                                if source_pkts.at[pkt_server_idx, 'seq'] == max_ack:
                                    source_pkt_time = source_pkts.at[pkt_server_idx, 'time']
                                    pkt_server_idx += 1
                                    break
                                pkt_server_idx += 1
                            if source_pkt_time > 0:
                                df.loc[len(df.index)] = [pkt.time, 1000 * (pkt.time - source_pkt_time), 0]
                    else:
                        if pkt.ack != '-1':
                            max_ack = 0
                            ack_arr = [record.split() for record in pkt.ack.split("#")]
                            for ack_range in ack_arr:
                                ack = int(ack_range[0])
                                if len(ack_range) == 2:
                                    ack = int(ack_range[1])
                                if ack > max_ack:
                                    max_ack = ack
                            source_pkt_time = -1.0
                            source_pkts = filtered_packets[
                                (filtered_packets.role == Role.SERVER.value) & (filtered_packets.seq != '-1')].reset_index(drop=True)
                            pkt_count = source_pkts.shape[0]
                            while pkt_client_idx < pkt_count:
                                if source_pkts.at[pkt_client_idx, 'seq'] > max_ack:
                                    source_pkt_time = source_pkts.at[pkt_client_idx, 'time']
                                    pkt_client_idx += 1
                                    break
                                pkt_client_idx += 1
                            if source_pkt_time > 0:
                                df.loc[len(df.index)] = [pkt.time,  1000* (source_pkt_time - pkt.time), 1]
            df.to_csv(file_path)
        return df

    def parse_throughput(self, interval=0.1, filtered_packets=None, suffix=''):
        file_path = self.PROCESSED_DATA_ROOT + 'throughput/' + self.packets_path + suffix + '_throughput.csv'
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
        else:
            if filtered_packets is None:
                filtered_packets = self.to_dataframe()
            df = pd.DataFrame(columns=["time", "ingress", "egress"])
            time = interval
            ingress = 0
            egress = 0
            for _, row in filtered_packets.iterrows():
                while time < row.time:
                    df.loc[len(df.index)] = [time, ingress, egress]
                    time += interval
                    ingress = 0
                    egress = 0
                if row.role == Role.CLIENT.value:
                    ingress += row.len
                else:
                    egress += row.len
            df.to_csv(file_path)
        return df

    def clear_cache(self):
        file_paths = [self.RAW_DATA_ROOT + 'caps_csv/' + self.packets_path + '.csv',
                      self.RAW_DATA_ROOT + 'filtered_caps/' + self.packets_path + '.pcap',
                      self.PROCESSED_DATA_ROOT + 'throughput/' + self.packets_path + '_UDP_throughput.csv',
                      self.PROCESSED_DATA_ROOT + 'throughput/' + self.packets_path + '_TCP_throughput.csv']
        for p in file_paths:
            if os.path.isfile(p):
                os.remove(p)

    def parse_packet_len_count(self, filtered_packets=None):
        file_path_ingress = self.PROCESSED_DATA_ROOT + 'pkt_len_distribution/' + self.packets_path + '_pktld_ingress.csv'
        file_path_egress = self.PROCESSED_DATA_ROOT + 'pkt_len_distribution/' + self.packets_path + '_pktld_egress.csv'
        if os.path.isfile(file_path_ingress) and os.path.isfile(file_path_egress):
            df_ingress = pd.read_csv(file_path_ingress, index_col=0)
            df_egress = pd.read_csv(file_path_egress, index_col=0)
        else:
            if filtered_packets is None:
                filtered_packets = self.to_dataframe()
            egress_pkts = filtered_packets[filtered_packets.role == Role.CLIENT.value]
            ingress_pkts = filtered_packets[filtered_packets.role == Role.SERVER.value]
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
            df_egress = df_egress.drop(columns=[str(i) for i in range(min_pkt_size_egress, max_pkt_size_egress + 1) if
                                                idx_to_pkt_size_egress[i] == 0])
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
            df_ingress = df_ingress.drop(
                columns=[str(i) for i in range(min_pkt_size_ingress, max_pkt_size_ingress + 1) if
                         idx_to_pkt_size_ingress[i] == 0])
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
                                                   if (int(j) <= division_threshold[i] and int(j) > division_threshold[
                    i - 1])]]
            df_grouped_egress["%d-%d" % (division_threshold[i - 1] + 1, division_threshold[i])] = df.sum(axis=1)
        for i in range(1, len(division_threshold)):
            df = df_ingress.drop(columns=['time'])[[j for j in df_ingress.drop(columns=['time']).columns.values
                                                    if (int(j) <= division_threshold[i] and int(j) > division_threshold[
                    i - 1])]]
            df_grouped_ingress["%d-%d" % (division_threshold[i - 1] + 1, division_threshold[i])] = df.sum(axis=1)
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


# import seaborn as sns
class PacketPlotter(object):
    def __init__(self, processors=None):
        if processors is None:
            processors = []
        self.processors = processors

    def plt_pkt_len_breakup(self):
        df_egress, df_ingress = self.processors[0].get_pkt_len_division_list()
        df_egress.plot(kind='bar', stacked=True)
        plt.title('Roblox ' + self.processors[0].packets_path + ' Packet Length Distribution - egress')
        plt.xlabel("Time (s)")
        plt.ylabel("Packet Count")
        plt.show()

        plt.figure(500)
        df_ingress.plot(kind='bar', stacked=True)
        plt.title('Roblox ' + self.processors[0].packets_path + ' Packet Length Distribution - ingress')
        plt.xlabel("Time (s)")
        plt.ylabel("Packet Count")
        plt.show()


if __name__ == "__main__":
    MC_PATH = 'roblox_data/'
    pp = PacketProcessor(RAW_DATA_ROOT=MC_PATH + 'raw_data/', PROCESSED_DATA_ROOT=MC_PATH + 'processed_data/')
    pp.parse_sample_rtt()
