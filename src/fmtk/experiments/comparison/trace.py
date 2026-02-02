from collections import Counter
import json
import logging
from itertools import groupby
import numpy as np
import pickle
from typing import List, Tuple, Any

import numpy as np


class Request:
    def __init__(self, req_id, task, dataloader , req_time):
        self.req_id = req_id
        self.task = task 
        self.dataloader = dataloader
        self.req_time = req_time

    
    def __repr__(self):
        return f"req_id={self.req_id}, " \
               f"task={self.task}, dataloader={self.dataloader}, "  \
               f"req_time={self.req_time}"


def generate_requests(num_tasks, alpha, req_rate, cv, duration,
                      tasks, # (base_dir, adapter_dir)
                      seed=42)-> List[Request]:
    np.random.seed(seed)

    tot_req = int(req_rate * duration)

    # generate adapter id
    probs = np.random.power(alpha, tot_req)
    ind = (probs * num_tasks).astype(int)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / req_rate
    # intervals = np.random.exponential(1.0 / req_rate, tot_req)
    intervals = np.random.gamma(shape, scale, tot_req)
    for i in range(tot_req):
        tic += intervals[i]
        requests.append(Request(i, tasks[ind[i]][0], tasks[ind[i]][1],tic))
    return requests


