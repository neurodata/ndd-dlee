import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from graspologic.embed import ClassicalMDS


def calculate_dissim(graphs, method="density", norm=None, normalize=True):
    """ Calculate the dissimilarity matrix using the input kernel. """
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


def cluster_dissim(dissim_matrix: np.ndarray, labels: list, method="agg"):
    """
    Cluster dissimilarity matrix using Agglomerative, K-means, or GMM. 
    """
    if method == "agg":
        # Agglomerative clustering
        agg = AgglomerativeClustering(n_clusters=4, affinity='precomputed', linkage='average', \
            compute_distances=True).fit(dissim_matrix, y=labels)
        pred = AgglomerativeClustering(n_clusters=4, affinity='precomputed', linkage='average', \
            compute_distances=True).fit_predict(dissim_matrix, y=labels)

        # construct linkage matrix
        counts = np.zeros(agg.children_.shape[0])
        n_samples = len(agg.labels_)

        for i, merge in enumerate(agg.children_):
            temp_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    temp_count += 1 
                else:
                    temp_count += counts[child_idx - n_samples]
            counts[i] = temp_count

        linkage_matrix = np.column_stack([agg.children_, agg.distances_, counts]).astype(float)

        return linkage_matrix, pred
    
    elif method in ["gmm", "kmeans"]:
        # Classical MDS
        cmds = ClassicalMDS(n_components=2, dissimilarity="precomputed")
        cmds_embedding = cmds.fit_transform(dissim_matrix)

        # cluster using GMM or K-means
        if method == "gmm":
            clustering = GaussianMixture(n_components=4, n_init=25).fit_predict(cmds_embedding, y=labels)
        elif method == "kmeans":
            clustering = KMeans(n_clusters=4, n_init=25).fit_predict(cmds_embedding, y=labels)
        else:
            print("Not a valid kernel name.")

        return cmds_embedding, clustering
    
    else:
        print("Not a valid kernel name.")