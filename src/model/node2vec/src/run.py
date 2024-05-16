import os
import subprocess
from time import sleep
from common_util.data_loader.local_graph.local_graph import kLocalGraph, load_local_graph
graphs = [kLocalGraph.ms_academic_cs, kLocalGraph.amazon_electronics_computers, kLocalGraph.amazon_electronics_photo]

for graph in graphs:
    graph_size = load_local_graph(graph)
    walk_length = 80
    window_size = 10
    for p in [0.25,0.50,1,2,4]:
        for q in [0.25,0.50,1,2,4]:
            cmd = f"python main.py \
                --input {graph.value} \
                --output emb/barbell.emb \
                --num-walks 10 \
                --walk-length {walk_length} \
                --window-size {window_size} \
                --p {p}\
                --q {q}\
                --dimensions 128 "
            cmd = cmd.split()
            subprocess.run(cmd)