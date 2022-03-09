import numpy as np


def calculate_dissim(graphs, method="density", norm=None, normalize=True):
    """
    Calculate the dissimilarity matrix using the input kernel. 
    """
    glob = False
    node = False
    edge = False

    if method == "density":
        glob = True
        num_nodes = graphs.shape[1]
        num_nodes_possible = num_nodes ** 2 - num_nodes

        metric = np.zeros(len(graphs))
        for i in range(len(graphs)):
            num_edges = np.count_nonzero(graphs[i])
            metric[i] = num_edges / num_nodes_possible
    
    elif method == "avgedgeweight":
        glob = True
        metric = np.zeros(len(graphs))
        for i in range(len(graphs)):
            num_edges = np.count_nonzero(graphs[i])
            metric[i] = np.sum(graphs[i]) / num_edges
    
    elif method == "avgadjmatrix":
        glob = True
        metric = np.zeros(len(graphs))
        for i in range(len(graphs)):
            metric[i] = np.average(graphs[i])   

    elif method == "degree":
        node = True
        metric = np.zeros((graphs.shape[0], graphs.shape[1]))
        for i, graph in enumerate(graphs):
            for j, row in enumerate(graph):
                metric[i, j] = np.count_nonzero(row)
    
    elif method == "strength":
        node = True
        metric = np.zeros((graphs.shape[0], graphs.shape[1]))
        for i, graph in enumerate(graphs):
            for j, row in enumerate(graph):
                metric[i, j] = np.sum(row)

    elif method == "edgeweight":
        edge = True
        metric = graphs
    
    else:
        print("Not a valid kernel name.")
    
    dissim_matrix = np.zeros((len(graphs), len(graphs)))
    for i, metric1 in enumerate(metric):
        for j, metric2 in enumerate(metric):
            if glob and norm == None:
                diff = np.abs(metric1 - metric2)
            elif (node or edge) and norm == "l1":
                diff = np.linalg.norm(metric1 - metric2, ord=1)
            elif (node or edge) and norm == "l2":
                diff = np.linalg.norm(metric1 - metric2, ord=2)
            else:
                print("L1, L2 norms only apply to node or edge-wise kernels.")
            
            dissim_matrix[i, j] = diff

    if normalize:
        dissim_matrix = dissim_matrix / np.max(dissim_matrix)
    
    return dissim_matrix