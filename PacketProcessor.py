import scapy.all as scapy
import pandas as pd
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt


class Role(Enum):
    SERVER = 0
    CLIENT = 1


class PacketProcessor(object):
    def __init__(self, server_ip='222.187.232.216', server_port=11701,
                 client_ip='192.168.65.174', client_port=64798, packets='caps/fast_chunk_load.pcapng',
                 protocol='TCP'):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_ip = client_ip
        self.client_port = client_port
        self.packets = scapy.rdpcap(packets)
        self.protocol = protocol

    def filter(self):
        IPs = [self.client_ip, self.server_ip]
        ports = [self.client_port, self.server_port]
        return [pkt for pkt in self.packets if
                "IP" in pkt and (pkt['IP'].src in IPs and pkt['IP'].dst in IPs) and
                "TCP" in pkt and (pkt["TCP"].sport in ports and pkt["TCP"].dport in ports)]

    def to_dataframe(self, packets=None, log_level=False):
        df = pd.DataFrame(columns=["time", "src", "dst", "seq", "ack", "len", "payload", "role"])
        start_time = packets[0].time if len(packets) > 0 else 0
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
        if log_level:
            print(df.info())
        return df

    def estimate_rtt(self, filtered_packets):
        df = pd.DataFrame(columns=["time", "estimated_rtt", "sample_rtt"])
        for idx, row in filtered_packets.iterrows():
            if row.role == Role.CLIENT and row.len > 0:
                origin_pkt = row
                for n_idx, n_row in filtered_packets.iloc[idx+1: idx+201, :].iterrows():
                    if n_row.role == Role.SERVER and n_row.ack == origin_pkt.len+origin_pkt.seq:
                        df.loc[len(df.index)] = [n_row.time, 0, (n_row.time-origin_pkt.time)*1000]
                        break
        return df

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


if __name__ == "__main__":
    pp = PacketProcessor()
    filtered_df = pp.to_dataframe(pp.filter(), log_level=False)
    rtts = pp.estimate_rtt(filtered_df)
    plt.title("sample RTT when editing world")
    plt.plot(rtts["time"], rtts["sample_rtt"])
    plt.xlabel("time(s)")
    plt.ylabel("Sample RTT (ms)")
    plt.grid()
    plt.show()
    # pp.test_violin_plot()
