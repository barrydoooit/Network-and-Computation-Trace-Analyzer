
import pandas as pd
import numpy as np

path = "cache/th_roblox_vs_mc_vs_vrchat"
vrpath = path+"/VRChat_raw.csv"
vrpatho = path+"/VRChat.csv"
df = pd.read_csv(vrpath)
lens = df["len"]
ths = []
last_time = 0
for index, row in df.iterrows():
    len = row["len"]
    cur_time = row["time"]
    span = cur_time - last_time
    if span == 0:
        th = 0
    else:
        th = len / span
    ths.append(th/100)

dfo = pd.DataFrame({"Length/bytes":lens,"Throughput(KBytes/s)":ths})
dfo.to_csv(vrpatho,index=False)