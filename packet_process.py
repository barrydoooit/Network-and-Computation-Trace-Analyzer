from PacketProcessor import PacketProcessor

if __name__ == "__main__":
    ROBLOX_PATH = 'roblox_data/'
    pp = PacketProcessor(server_ip='183.47.97.149', server_port=17658,
                 client_ip='192.168.65.174', client_port=62930, packets='connect',
                 protocol='UDP', RAW_DATA_ROOT=ROBLOX_PATH+'raw_data/', PROCESSED_DATA_ROOT=ROBLOX_PATH+'processed_data/')
    pp.get_top_n_freq_packets().to_csv("test.csv")