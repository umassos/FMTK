import pandas as pd

def get_throughput(folder_name,task_name):
    df = pd.read_csv(f'result/closedloop/{folder_name}/{task_name}.csv')
    df["sec"] = df["Timestamp"].astype(int)
    reqs_per_sec = df.groupby("sec").size()
    avg_throughput = reqs_per_sec.mean()

    # print("Requests per second (per bucket):")
    # print(reqs_per_sec)
    # print("\nAverage throughput:", avg_throughput, "req/s")
    return avg_throughput
def get_latency(folder_name,task_name):
    df = pd.read_csv(f'result/closedloop/{folder_name}/{task_name}.csv')
    df["TotalTime(ms)"] = df["TotalTime(ms)"].astype(float)
    avg_latency = df["TotalTime(ms)"].mean()

    return avg_latency

print("Throughput (req/s):")
print("Normal")
folder_name='3_client/normal'
task_name='hr'
print(get_throughput(folder_name,task_name))

task_name='diasbp'
print(get_throughput(folder_name,task_name))

task_name='sysbp'
print(get_throughput(folder_name,task_name))

print("Sharing without batching")
folder_name='3_client/sharing_withoutbatching'
task_name='hr'
print(get_throughput(folder_name,task_name))

task_name='diasbp'
print(get_throughput(folder_name,task_name))

task_name='sysbp'
print(get_throughput(folder_name,task_name))

print("Sharing with batching")
folder_name='3_client/sharing_withbatching'
task_name='hr'
print(get_throughput(folder_name,task_name))

task_name='diasbp'
print(get_throughput(folder_name,task_name))

task_name='sysbp'
print(get_throughput(folder_name,task_name))

print("Normal workers")
folder_name='3_client/normal_workers'
task_name='hr'
print(get_throughput(folder_name,task_name))

task_name='diasbp'
print(get_throughput(folder_name,task_name))

task_name='sysbp'
print(get_throughput(folder_name,task_name))

print("Normal servers")
folder_name='3_client/normal_servers'
task_name='hr'
print(get_throughput(folder_name,task_name))

task_name='diasbp'
print(get_throughput(folder_name,task_name))

task_name='sysbp'
print(get_throughput(folder_name,task_name))


print("Latency (ms):")
print("Normal")
folder_name='3_client/normal'
task_name='hr'
print(get_latency(folder_name,task_name))

task_name='diasbp'
print(get_latency(folder_name,task_name))

task_name='sysbp'
print(get_latency(folder_name,task_name))

print("Sharing without batching")
folder_name='3_client/sharing_withoutbatching'
task_name='hr'
print(get_latency(folder_name,task_name))

task_name='diasbp'
print(get_latency(folder_name,task_name))

task_name='sysbp'
print(get_latency(folder_name,task_name))

print("Sharing with batching")
folder_name='3_client/sharing_withbatching'
task_name='hr'
print(get_latency(folder_name,task_name))

task_name='diasbp'
print(get_latency(folder_name,task_name))

task_name='sysbp'
print(get_latency(folder_name,task_name))

print("Normal workers")
folder_name='3_client/normal_workers'
task_name='hr'
print(get_latency(folder_name,task_name))

task_name='diasbp'
print(get_latency(folder_name,task_name))

task_name='sysbp'
print(get_latency(folder_name,task_name))

print("Normal servers")
folder_name='3_client/normal_servers'
task_name='hr'
print(get_latency(folder_name,task_name))

task_name='diasbp'
print(get_latency(folder_name,task_name))

task_name='sysbp'
print(get_latency(folder_name,task_name))