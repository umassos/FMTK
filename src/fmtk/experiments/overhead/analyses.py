import json 
from statistics import mean
def analyse_result(json_path,metric):
    with open(json_path, "r") as f:
        data = json.load(f)
        recs = data.get("records", [])
        result={}
        for r in recs:
            if r['section'] not in result:
                result[r['section']]=[r[metric]]
            else:
                result[r['section']].append(r[metric])
        final_result={}
        for f,v in result.items():
            if metric=='gpu_alloc_peak':
                final_result[f]=mean(v)/10**6
            elif metric=='gpu_energy_mJ':
                final_result[f]=mean(v)/1000
            else:
                final_result[f]=mean(v)
    return final_result

baseline_data=analyse_result('./logs/moment_ecg_baseline.json','wall_time_sec')
sdk_data=analyse_result('./logs/moment_ecg_sdk.json','wall_time_sec')
print(baseline_data)
print(sdk_data)

baseline_data=analyse_result('./logs/moment_ecg_baseline.json','gpu_alloc_peak')
sdk_data=analyse_result('./logs/moment_ecg_sdk.json','gpu_alloc_peak')
print(baseline_data)
print(sdk_data)

baseline_data=analyse_result('./logs/moment_ecg_baseline.json','gpu_energy_mJ')
sdk_data=analyse_result('./logs/moment_ecg_sdk.json','gpu_energy_mJ')
print(baseline_data)
print(sdk_data)


baseline_data=analyse_result('./logs/moment_etth1_baseline.json','wall_time_sec')
sdk_data=analyse_result('./logs/moment_etth1_sdk.json','wall_time_sec')
print(baseline_data)
print(sdk_data)

baseline_data=analyse_result('./logs/moment_etth1_baseline.json','gpu_alloc_peak')
sdk_data=analyse_result('./logs/moment_etth1_sdk.json','gpu_alloc_peak')
print(baseline_data)
print(sdk_data)

baseline_data=analyse_result('./logs/moment_etth1_baseline.json','gpu_energy_mJ')
sdk_data=analyse_result('./logs/moment_etth1_sdk.json','gpu_energy_mJ')
print(baseline_data)
print(sdk_data)