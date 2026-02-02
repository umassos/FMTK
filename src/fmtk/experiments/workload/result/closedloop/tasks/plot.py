import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob

def get_throughput(file_name):
    df = pd.read_csv(file_name)
    df["sec"] = df["Timestamp"].astype(int)
    reqs_per_sec = df.groupby("sec").size()
    avg_throughput = reqs_per_sec.mean()
    return avg_throughput
    
def get_latency(file_name):
    df = pd.read_csv(file_name)
    df["TotalTime(ms)"] = df["TotalTime(ms)"].astype(float)
    avg_latency = df["TotalTime(ms)"].mean()
    return avg_latency

# folder=A100/sharing_withoutbatching
#iterate over all clients 1,2,4,8,10 and csv files inside it
# get average throughput and latency for each client

folder_path = "A100/normal_workers" # Replace with the actual path to your folder
client_throughputs_n = {}
client_latencies_n = {}
for client_count in [2,4,8,6,10]:
    client_folder = os.path.join(folder_path, str(client_count))
    if not os.path.exists(client_folder):
        print(f"Folder {client_folder} does not exist, skipping.")
        continue
    csv_files = glob.glob(os.path.join(client_folder, "*.csv"))
    throughputs = []
    latencies = []
    for file_path in csv_files:
        throughput = get_throughput(file_path)
        latency = get_latency(file_path)
        throughputs.append(throughput)
        latencies.append(latency)
    avg_throughput = np.sum(throughputs)
    avg_latency = np.mean(latencies)

    #store the avg throughput and latency in dictionary for plotting
    client_throughputs_n[client_count] = avg_throughput
    client_latencies_n[client_count] = avg_latency

folder_path = "A100/sharing_withoutbatching" # Replace with the actual path to your folder
client_throughputs_s = {}
client_latencies_s = {}
for client_count in [2,4,6,8,10]:
    client_folder = os.path.join(folder_path, str(client_count))
    if not os.path.exists(client_folder):
        print(f"Folder {client_folder} does not exist, skipping.")
        continue
    csv_files = glob.glob(os.path.join(client_folder, "*.csv"))
    throughputs = []
    latencies = []
    for file_path in csv_files:
        throughput = get_throughput(file_path)
        latency = get_latency(file_path)
        throughputs.append(throughput)
        latencies.append(latency)
    avg_throughput = np.sum(throughputs)
    avg_latency = np.mean(latencies)

    #store the avg throughput and latency in dictionary for plotting
    client_throughputs_s[client_count] = avg_throughput
    client_latencies_s[client_count] = avg_latency

client_throughputs_sb = {}
client_latencies_sb = {}
# get for A100/sharing_withbatching
folder_path = "A100/sharing_withbatching" # Replace with the actual path to your folder
for client_count in [2,4,6,8,10]:   
    client_folder = os.path.join(folder_path, str(client_count))
    if not os.path.exists(client_folder):
        print(f"Folder {client_folder} does not exist, skipping.")
        continue
    csv_files = glob.glob(os.path.join(client_folder, "*.csv"))
    throughputs = []
    latencies = []
    for file_path in csv_files:
        throughput = get_throughput(file_path)
        latency = get_latency(file_path)
        throughputs.append(throughput)
        latencies.append(latency)
    avg_throughput = np.sum(throughputs)
    avg_latency = np.mean(latencies)

    #store the avg throughput and latency in dictionary for plotting
    client_throughputs_sb[client_count] = avg_throughput
    client_latencies_sb[client_count] = avg_latency

# bar plot side by side for sharing_withbatching and sharing_withoutbatching with y axis as throughput and x axis as client count
x = list(client_throughputs_s.keys())
y1 = list(client_throughputs_s.values())
y2 = list(client_throughputs_sb.values())
y3=list(client_throughputs_n.values())
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
bars1= ax.bar(np.array(x) - width/2, y3, width, label='Normal Workers')
bars2 = ax.bar(np.array(x) + width/2, y1, width, label='Sharing without batching')
bars3 = ax.bar(np.array(x) + 3*width/2, y2, width, label='Sharing with batching')           
ax.set_xlabel('#Tasks')
ax.set_ylabel('Throughput (requests/sec)')
ax.set_title('Throughput vs Number of Clients')
ax.set_xticks(x)
ax.legend()
# add value on top of each bar
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)
plt.savefig('throughput_vs_clients.png')
plt.clf()   

# bar plot side by side for sharing_withbatching and sharing_withoutbatching with y axis as latency and x axis as client count
x = list(client_latencies_s.keys())     
y1 = list(client_latencies_s.values())
y2 = list(client_latencies_sb.values())
y3=list(client_latencies_n.values())
width = 0.35  # the width of the bars           
fig, ax = plt.subplots()
bars1= ax.bar(np.array(x) - width/2, y3, width, label='Normal Workers')
bars2 = ax.bar(np.array(x) + width/2, y1, width,
                label='Sharing without batching')   
bars3 = ax.bar(np.array(x) + 3*width/2, y2, width,
                label='Sharing with batching')

ax.set_xlabel('#Tasks')
ax.set_ylabel('Latency (ms)')
ax.set_title('Latency vs Number of Clients')
ax.set_xticks(x)
ax.legend()
# add value on top of each bar
add_value_labels(bars1)
add_value_labels(bars2)       
add_value_labels(bars3)  
plt.savefig('latency_vs_clients.png')
plt.clf()       
# plt.show()    
# print the dictionaries        
print("Client Throughputs Without Batching:", client_throughputs_s)
print("Client Latencies Without Batching:", client_latencies_s)
print("Client Throughputs With Batching:", client_throughputs_sb)
print("Client Latencies With Batching:", client_latencies_sb)
print("Client Throughputs Normal Workers:", client_throughputs_n)
print("Client Latencies Normal Workers:", client_latencies_n)

memory={
    "Normal": {2:2923.90,4:5847.8,6:8771.70,8:11695.6,10:14619.50},
    "Sharing without batching": {2:1463.00,4:1464.06,6:1514.92,8:1615.58,10:1716.25},
    "Sharing with batching": {2:1463.00,4:1464.06,6:1514.92,8:1615.58,10:1716.25},
}

#plot memory vs number of tasks 
# with memory data for normal, sharing without batching and sharing with batching
# in memory dictionary
x = list(memory["Normal"].keys())
y1 = memory["Normal"].values()
y2 = memory["Sharing without batching"].values()
y3 = memory["Sharing with batching"].values()
width = 0.25  # the width of the bars
fig, ax = plt.subplots()
bars1= ax.bar(np.array(x) - width, y1, width, label='Normal Workers')
bars2 = ax.bar(np.array(x), y2, width, label='Sharing without batching')
bars3 = ax.bar(np.array(x) + width, y3, width, label='Sharing with batching')
ax.set_xlabel('#Tasks')
ax.set_ylabel('Memory (MB)')
ax.set_title('Memory vs Number of Clients')
ax.set_xticks(x)
ax.legend()
# add value on top of each bar
add_value_labels(bars1)
add_value_labels(bars2)     
add_value_labels(bars3)
plt.savefig('memory_vs_clients.png')
plt.clf()
# plt.show()    