#created by Lai WEI, at 13:46 2021/11/29
#compute the throughput and rtt
#and add it to the trace

import os.path
import pandas as pd
import numpy as np

def compute_and_add_throughput(input_path,output_path):
    dfi = pd.read_csv(input_path)
    ths = []
    last_time = 0
    for index,row in dfi.iterrows():
        cur_time = row["time"]
        span = cur_time - last_time
        last_time = cur_time
        packet_length = row["len"]
        if span == 0:
            th = 0
        else:
            th = packet_length/span
        ths.append(th/1000000)
    dfo = pd.DataFrame({"Length/bytes":dfi["len"],"Throughput(KBytes/s)":ths})
    dfo.to_csv(output_path,index=False)
    return

def compute_and_add_rtt(input_path,output_path):
    dfi = pd.read_csv(input_path)
    rtts = []
    last_time = 0
    for index,row in dfi.iterrows():
        cur_time = row["time"]
        span = cur_time - last_time
        last_time = cur_time
        rtts.append(span*1000)
    dfo = pd.DataFrame({"Length/bytes":dfi["len"],"RTT":rtts})
    dfo.to_csv(output_path,index=False)
    return

if __name__=="__main__":
    input_root = "cache/th_mc_pc_vs_phone"
    output_root = "cache/th_mc_pc_vs_phone"
    input_name = "Phone_raw.csv"
    output_name = "Phone.csv"
    input_path = os.path.join(input_root,input_name)
    output_path = os.path.join(output_root,output_name)
    compute_and_add_throughput(input_path,output_path)
    #output_root = "cache/rtt_roblox_vs_mc"
    #output_path = os.path.join(output_root, output_name)
    #compute_and_add_rtt(input_path, output_path)