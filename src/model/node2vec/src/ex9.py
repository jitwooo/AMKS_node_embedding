import sys
package_path = '/home/zjh/graph_embedding/src'
if package_path not in sys.path:
    sys.path.append(package_path)

'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from model.local_graph import load_local_graph_by_name
from experiment_util.nodes_classify import load_node_label, kLabelType, knn_evaluate
from experiment_util.cross_graph import load_node_name, emb_evaluate
from model.local_graph import kLocalGraph, load_local_graph

def parse_args():
    '''
	Parses the node2vec arguments.
	'''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='cora',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph(index):
    '''
	Reads the input network in networkx.
	'''
    if args.weighted:
        G = nx.read_edgelist(args.input[index], nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = load_local_graph_by_name(args.input[index])
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        tag = G.graph_tag
        G = G.to_undirected()
        G.graph_tag = tag

    return G


def learn_embeddings(walks):
    '''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1,
                     workers=args.workers, epochs=args.iter)

    return model.wv


def main(args):
    '''
	Pipeline for representational learning for all nodes in a graph.
	'''
    # KDD-ICDM-1 ; SIGIR-CIKM ; SIGMOD-ICDE-1
    graph_name_one = kLocalGraph.sigir
    graph_name_two = kLocalGraph.cikm

    nx_graph_one = load_local_graph(graph_name_one)
    graph_one_node_name = load_node_name(nx_graph_one)
    for edge in nx_graph_one.edges():
        nx_graph_one[edge[0]][edge[1]]['weight'] = 1

    nx_graph_two = load_local_graph(graph_name_two)
    graph_two_node_name = load_node_name(nx_graph_two)
    for edge in nx_graph_two.edges():
        nx_graph_two[edge[0]][edge[1]]['weight'] = 1

    common_node_name = set([k for k in graph_one_node_name if graph_one_node_name[k] in nx_graph_one]).intersection(
        set([k for k in graph_two_node_name if graph_two_node_name[k] in nx_graph_two]))
    graph_one_common_id = [graph_one_node_name[name] for name in sorted(common_node_name)]
    graph_two_common_id = [graph_two_node_name[name] for name in sorted(common_node_name)]

    k_list = [20, 40]  # Top-k



    graph_one = node2vec.Graph(nx_graph_one, args.directed, args.p, args.q)
    graph_two = node2vec.Graph(nx_graph_two, args.directed, args.p, args.q)

    graph_one.preprocess_transition_probs()
    walks = graph_one.simulate_walks(args.num_walks, args.walk_length)
    wv = learn_embeddings(walks)

    filter_list = list(sorted(nx_graph_one.nodes))
    f2 = [str(ele) for ele in filter_list]
    embedding_vec1 = np.array(wv[f2])

    print(len(graph_one_common_id), len(embedding_vec1))
    graph_one_embed_vec = embedding_vec1[graph_one_common_id]

    graph_two.preprocess_transition_probs()
    walks = graph_two.simulate_walks(args.num_walks, args.walk_length)
    wv = learn_embeddings(walks)

    filter_list = list(sorted(nx_graph_two.nodes))
    f2 = [str(ele) for ele in filter_list]
    embedding_vec2 = np.array(wv[f2])

    graph_two_embed_vec = embedding_vec2[graph_two_common_id]

    ans = emb_evaluate(graph_one_embed_vec, graph_two_embed_vec, k_list)
    top_20_score = ans[k_list[0]]
    top_40_score = ans[k_list[1]]

    # labels = load_node_label(nx_G, kLabelType.eigencentrality)
    with open("ex9.txt", 'a+') as f:
        f.write("#" * 20 + f"{graph_name_one}-{graph_name_two}, top20: {top_20_score}, top40: {top_40_score}")
        f.write("\n")




if __name__ == "__main__":
    args = parse_args()
    main(args)
