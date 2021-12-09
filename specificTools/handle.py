import csv
import pandas as pd

def get_throughput_from_record(record):
    df = pd.read_csv(record)
    trps = df[["On Time/s", "Throughput(KBytes/s)"]]
    time = 1
    throughputs = []
    basetime = float(trps.iloc[0]["On Time/s"])
    sum = 0
    count = 0
    for i in range(1, len(trps.index)):
        row = trps.iloc[i]
        if float(row["On Time/s"]) - basetime < time:
            sum += row["Throughput(KBytes/s)"]
            count += 1
        else:
            print(count)
            if count == 0:
                print(throughputs)
                if len(throughputs) > 0:
                    throughputs.append(throughputs[-1])
                else:
                    throughputs.append(0)
            else:
                throughputs.append(sum / count)
            sum = row["Throughput(KBytes/s)"]
            count = 1
            time += 1
    return throughputs


delay_half = 0
length_half = 0
throughput_half = 0
cnt_half = 171
with open('Record/record4.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    cnt = 1
    delay_sum = 0
    length_sum = 0
    throughput_sum = 0
    for packet_row in csv_reader:
        if len(packet_row) == 0:
            continue
        if cnt == 1:
            cnt += 1
            continue
        delay_sum += float(packet_row[2])
        length_sum += float(packet_row[3])
        throughput_sum += float(packet_row[4])
        if cnt == cnt_half:
            print("loading Delay: ",delay_sum/cnt)
            print("loading Length: ", length_sum/cnt)
            print("loading Throughput: ", throughput_sum/cnt)
            delay_half = delay_sum
            length_half = length_sum
            throughput_half = throughput_sum
        cnt += 1

print("Normal Delay:", (delay_sum - delay_half)/(cnt - cnt_half))
print("Normal Length:", (length_sum - length_half)/(cnt - cnt_half))
print("Normal Throughput:", (throughput_sum - throughput_half)/(cnt - cnt_half))

print("Total Delay: ", delay_sum/cnt)
print("Total Length: ", length_sum/cnt)
print("Total Throughput: ", throughput_sum/cnt)
