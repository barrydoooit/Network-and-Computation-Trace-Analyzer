import scapy.all as scapy
import csv
import matplotlib.pyplot as plt
import numpy as np

filename = 'Output/output_mc_short_range_move.csv'
packets = scapy.rdpcap('raw_data/caps/short_range_move.pcapng')
source = '222.187.232.216'
host = '192.168.65.174'
out_record = "Record/record_short_range_move.csv"


def _open():
    cnt = 1
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Packet_Number', 'Time', 'Src', 'Dst', 'Length'])
        for packet in packets:
            if cnt == 1:
                initial = int(packet.time)
            if 'IP' in packet and (packet['IP'].src == source or packet['IP'].dst == source):
                print(packet['IP'].info)
                writer.writerow([cnt, float(packet.time) - initial, packet['IP'].src, packet['IP'].dst, packet['IP'].len])
            cnt += 1


"""_open()
cnt = 1
X = []
Y = []
Delay = []
T = []
B = [0, 0, 0, 0, 0, 0]
with open(out_record, 'w') as recordFile:
    csv_writer = csv.writer(recordFile)
    csv_writer.writerow(["On Time/s", "End Time/s", "Delay/ms", "Length/bytes", "Throughput(KBytes/s)"])
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        flag = False
        length = 0
        on_time = 0
        for packet in csv_reader:
            if len(packet) == 0:
                continue
            if cnt == 1:
                cnt += 1
                continue
            cnt += 1
            if packet[2] == source and packet[3] == host:
                if not flag:
                    flag = True
                    on_time = float(packet[1])
                length += int(packet[4])
            else:
                flag = False
                if length != 0:
                    if length < 200:
                        B[0] += 1
                    elif length < 500:
                        B[1] += 1
                    elif length < 1000:
                        B[2] += 1
                    elif length < 10000:
                        B[3] += 1
                    elif length < 50000:
                        B[4] += 1
                    else:
                        B[5] += 1
                    X.append(on_time)
                    Y.append(length)
                    Delay.append((float(packet[1]) - on_time) * 1000)
                    T.append(min(length/((float(packet[1]) - on_time) * 1024 * 1024), 120))
                    csv_writer.writerow([on_time, float(packet[1]),
                                         (float(packet[1]) - on_time) * 1000, length,
                                         min(length/((float(packet[1]) - on_time) * 1024 * 1024), 120)])
                length = 0

Delay[0] = 10
"""
def length_pic(X, Y):
    plt.plot(X, Y, color = 'r')
    plt.xlabel('start_time / (s)')
    plt.ylabel('packet_length / (bytes)')
    plt.show()
    plt.cla()


def length_frequency(B):
    plt.bar(range(len(B)), B, tick_label = ['0-200','200-500','500-1k','1k-10k','10k-50k','others'], color = 'cyan')
    plt.xlabel('packet_length / (bytes)')
    plt.ylabel('Frequency')
    for a,b in zip(range(len(B)), B):
        plt.text(a,b,'%d'%b,ha='center',va='bottom',fontsize=10);
    plt.show()


def packet_delay(X, Delay):
    plt.plot(X, Delay, color='r')
    plt.xlabel('start_time / (s)')
    plt.ylabel('packet_delay / (ms)')
    plt.show()
    plt.cla()


def packet_throughput(X, Throughput):
    plt.plot(X, Throughput, color='blue')
    plt.xlabel('start_time / (s)')
    plt.ylabel('Throughput / (MB/s)')
    plt.show()
    plt.cla()


"""length_pic(X, Y)
length_frequency(B)
packet_delay(X, Delay)
packet_throughput(X, T)"""

if __name__ == "__main__":
    print(packets[0]["IP"])