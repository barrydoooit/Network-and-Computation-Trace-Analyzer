from PacketProcessor import PacketProcessor, PacketPlotter



def sample_udp_process(sip, cip, spt, cpt, pkt_path, ROOT_PATH = 'minecraft_data/E5_2U4R_20211201/PC/'):

    pp = PacketProcessor(server_ip=sip, server_port=spt,
                         client_ip=cip, client_port=cpt, packets=pkt_path, protocol='UDP',
                         RAW_DATA_ROOT=ROOT_PATH + 'raw_data/', PROCESSED_DATA_ROOT=ROOT_PATH + 'processed_data/')
    pp.clear_cache()
    pp.parse_throughput(suffix='_UDP')

def sample_tcp_process(sip, cip, spt, cpt, pkt_path, ROOT_PATH, need_rtt=False):

    pp = PacketProcessor(server_ip=sip, server_port=spt,
                         client_ip=cip, client_port=cpt, packets=pkt_path,
                         RAW_DATA_ROOT=ROOT_PATH + 'raw_data/', PROCESSED_DATA_ROOT=ROOT_PATH + 'processed_data/')
    pp.clear_cache()
    if need_rtt:
        pp.parse_sample_rtt()
    pp.parse_throughput(suffix='_TCP')

def minecraft_1203():

    sip = "52.166.239.43"
    cip = "192.168.67.64"
    spt = 19132
    cpt = 62990
    sample_udp_process(sip, cip, spt, cpt, 'connect', ROOT_PATH='minecraft_data/CPP_Version_20211202/PE/')
    sip = "52.166.239.97"
    cip = "192.168.67.64"
    spt = 19510
    cpt = 63159
    for n in ['short_range_move', 'stand_still', 'battle']:
        sample_udp_process(sip, cip, spt, cpt, n, ROOT_PATH='minecraft_data/CPP_Version_20211202/PE/')

def vrchat_1203():
    sip = "13.225.100.227"
    cip = "192.168.67.64"
    spt = 443
    cpt = 54372
    sample_tcp_process(sip, cip, spt, cpt, 'connect', ROOT_PATH='vrchat_data/20211202_01/')
    cpt = 54837
    for n in ['short_range_move', 'stand_still', 'item_moving', 'drawing']:
        sample_tcp_process(sip, cip, spt, cpt, n, ROOT_PATH='vrchat_data/20211202_01/')
    sip = "172.65.221.66"
    cip = "192.168.67.64"
    spt = 5055
    cpt = 55032
    sample_udp_process(sip, cip, spt, cpt, 'connect', ROOT_PATH='vrchat_data/20211202_01/')
    sip = "172.65.237.193"
    cip = "192.168.67.64"
    spt = 5056
    cpt = 55033
    for n in ['short_range_move', 'stand_still', 'item_moving', 'drawing']:
        sample_udp_process(sip, cip, spt, cpt, n, ROOT_PATH='vrchat_data/20211202_01/')
if __name__ == "__main__":
    sample_tcp_process('124.132.136.201', '192.168.67.64', 11501, 49873, 'connect', ROOT_PATH='minecraft_data/Java_E5_2U4R_20211201/PC/', need_rtt=True)
    sample_tcp_process('124.132.136.201', '192.168.67.64', 11501, 53136, 'battle', ROOT_PATH='minecraft_data/Java_E5_2U4R_20211201/PC/', need_rtt=True)
    sample_tcp_process('124.132.136.201', '192.168.67.64', 11501, 53136, 'resource_loading', ROOT_PATH='minecraft_data/Java_E5_2U4R_20211201/PC/', need_rtt=True)
    sample_tcp_process('124.132.136.201', '192.168.67.64', 11501, 49934, 'short_range_move', ROOT_PATH='minecraft_data/Java_E5_2U4R_20211201/PC/', need_rtt=True)
    sample_tcp_process('124.132.136.201', '192.168.67.64', 11501, 49934, 'stand_still', ROOT_PATH='minecraft_data/Java_E5_2U4R_20211201/PC/', need_rtt=True)
    sample_tcp_process('124.132.136.201', '192.168.67.64', 11501, 53136, 'world_edit', ROOT_PATH='minecraft_data/Java_E5_2U4R_20211201/PC/', need_rtt=True)
