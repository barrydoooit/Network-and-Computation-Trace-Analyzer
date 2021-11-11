import scapy.all as scapy
import pandas as pd


class PacketProcessor(object):
    def __init__(self, server_ip='222.187.232.216', server_port=11701,
                 client_ip='192.168.65.174', client_port=64798, packets='caps/short_range_move.pcapng',
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
        return (pkt for pkt in self.packets if
                        "IP" in pkt and (pkt['IP'].src in IPs and pkt['IP'].dst in IPs) and
                        "TCP" in pkt and (pkt["TCP"].sport in ports and pkt["TCP"].dport in ports))

    def to_dataframe(self, packets=None):
        df = pd.DataFrame(columns=["time", "src", "dst", "packet_length"])
        if len(packets)>0:
            start_time = packets[0].time
        for pkt in packets:
            df.iloc[-1]=[float(pkt.time)-start_time, pkt['IP'].src, pkt['IP'].dst, pkt['TCP'].len]


if __name__ == "__main__":
    print("hello metaverse")
