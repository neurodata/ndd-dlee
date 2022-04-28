import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from graspologic.embed import ClassicalMDS
from graspologic.utils import pass_to_ranks, to_laplacian
import pandas as pd
from graspologic.utils import remap_labels
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp


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
        for i, graph in enumerate(graphs):
            num_edges = np.count_nonzero(graph)
            metric[i] = num_edges / num_nodes_possible
    
    elif method == "avgedgeweight":
        glob = True
        metric = np.zeros(len(graphs))
        for i, graph in enumerate(graphs):
            #graph = pass_to_ranks(graph)
            num_edges = np.count_nonzero(graph)
            metric[i] = np.sum(graph) / num_edges
    
    elif method == "avgadjmatrix":
        glob = True
        metric = np.zeros(len(graphs))
        for i, graph in enumerate(graphs):
            graph = pass_to_ranks(graph)
            metric[i] = np.average(graph)   

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
    """
    for i in range(len(metric)):
        for j in range(i, len(metric)):
            if glob and norm == None:
                diff = np.abs(metric[i] - metric[j])
            elif (node or edge) and norm == "l1":
                diff = np.linalg.norm(metric[i] - metric[j], ord=1)
            elif (node or edge) and norm == "l2":
                diff = np.linalg.norm(metric[i] - metric[j], ord=2)
            else:
                print("L1, L2 norms only apply to node or edge-wise kernels.")
            
            dissim_matrix[j, i] = diff
    """
    for i, metric1 in enumerate(metric):
        for j, metric2 in enumerate(metric):
            if glob and norm == None:
                diff = np.abs(metric1- metric2)
            elif (node or edge) and norm == "l1":
                diff = np.linalg.norm(metric1 - metric2, ord=1)
            elif (node or edge) and norm == "l2":
                diff = np.linalg.norm(metric1 - metric2, ord=2)
            else:
                print("L1, L2 norms only apply to node or edge-wise kernels.")
            
            dissim_matrix[j, i] = diff

    if normalize:
        dissim_matrix = dissim_matrix / np.max(dissim_matrix)
    
    return dissim_matrix


def calculate_dissim_unmatched(graphs, method="degree", normalize=True):
    """ Calculate the dissimilarity matrix using the input kernel. """
    if method == "degree":
        metric = np.zeros((graphs.shape[0], graphs.shape[1]))
        for i, graph in enumerate(graphs):
            for j, node in enumerate(graph):
                metric[i, j] = np.count_nonzero(node)
    
    elif method == "strength":
        metric = np.zeros((graphs.shape[0], graphs.shape[1]))
        for i, graph in enumerate(graphs):
            for j, node in enumerate(graph):
                metric[i, j] = np.sum(node)

    elif method == "edgeweight":
        metric = []
        for graph in graphs:
            metric.append(np.ravel(np.nonzero(graph)))
            
    else:
        print("Not a valid kernel name.")
    
    dissim_matrix = np.zeros((len(graphs), len(graphs)))
    for i, metric1 in enumerate(metric):
        for j, metric2 in enumerate(metric):
            diff, _ = ks_2samp(np.array(metric1), np.array(metric2), mode='asymp')
            dissim_matrix[i, j] = diff
            
    if normalize:
        dissim_matrix = dissim_matrix / np.max(dissim_matrix)
    
    return dissim_matrix


def laplacian_dissim(graphs, transform: str=None, metric: str='l2', smooth_eigvals: bool=False, \
    normalize=True):
    if transform == 'pass-to-ranks':
        for i, graph in enumerate(graphs):
            graph = pass_to_ranks(graph)
            graphs[i] = graph
    elif transform == 'binarize':
        graphs[graphs != 0] = 1
    elif transform == None:
        graphs = graphs
    else:
        print('Supported transformations are "pass-to-ranks", "binarize", or None.')
    
    eigs = []
    for i, graph in enumerate(graphs):
        # calculate laplacian
        lap = to_laplacian(graph, 'I-DAD')

        # find and sort eigenvalues
        w = np.linalg.eigvals(lap)
        w = np.sort(w)

        if smooth_eigvals:
            kde = KernelDensity(kernel='gaussian', bandwidth=0.015).fit(w.reshape(-1, 1))
            xs = np.linspace(0, 2, 2000)
            xs = xs[:, np.newaxis]
            log_dens = kde.score_samples(xs)
            eigs.append(np.exp(log_dens))
        else:
            eigs.append(w)

    dissim_matrix = np.zeros((len(graphs), len(graphs)))
    for i, eig1 in enumerate(eigs):
        for j, eig2 in enumerate(eigs):
            if metric == 'cosine':
                diff = cosine(eig1, eig2)
            elif metric == 'l1':
                diff = np.linalg.norm(eig1 - eig2, ord=1)
            elif metric == 'l2':
                diff = np.linalg.norm(eig1 - eig2, ord=2)
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
            clustering = GaussianMixture(n_components=2, n_init=25).fit_predict(cmds_embedding, y=labels)
        elif method == "kmeans":
            clustering = KMeans(n_clusters=2, n_init=25).fit_predict(cmds_embedding, y=labels)
        else:
            print("Not a valid kernel name.")

        return cmds_embedding, clustering
    
    else:
        print("Not a valid kernel name.")

def _compare_preds(y_true, y_pred):
    """ Compare true and predicted labels and return list of 'Correct' or 'Incorrect'. """
    preds = []
    for i, pred in enumerate(y_pred):
        if pred == y_true[i]:
            preds.append("Correct")
        else:
            preds.append("Incorrect")
    
    return preds

def construct_df(embedding, labels, y_true, y_pred):
    """ Construct dataframe for plotting and remapped labels. Only for 'gmm' or 'kmeans'. """
    # x,y coordinates
    plot_df = pd.DataFrame(embedding, columns=["Dimension 1", "Dimension 2"])

    # predicted labels
    mapper_inv = {}
    for i, label in enumerate(set(labels)):
        mapper_inv[i] = label
    y_pred = remap_labels(y_true=y_true, y_pred=y_pred)
    pred_str = np.array([mapper_inv[l] for l in list(y_pred)])
    plot_df["Order"] = pred_str

    # determine if predictions are accurate for each sample
    preds = _compare_preds(y_true, y_pred)
    plot_df["Predictions"] = preds

    return plot_df, y_pred

def plot_clustering(
    labels: list,
    algorithm: str = "agg",
    dissim_matrix: np.ndarray = None,
    linkage_matrix: np.ndarray = None,
    data: pd.DataFrame = None,
    **kwargs,
):
    sns.set_context("talk", font_scale=0.85)
    palette = dict(zip(set(labels), sns.color_palette("colorblind", len(set(labels)))))
    colors = np.array([palette[l] for l in labels])

    if algorithm == "agg" and dissim_matrix is not None and linkage_matrix is not None:
        clustergrid = sns.clustermap(dissim_matrix, row_linkage=linkage_matrix, col_linkage=linkage_matrix, row_colors=colors, \
            col_colors=colors, cmap="RdBu_r", center=0, xticklabels=False, yticklabels=False, **kwargs)

        clustergrid.figure.set_facecolor('w')
        clustergrid.ax_cbar.set_title("Dissimilarity") # add color bar
        col_ax = clustergrid.ax_col_colors # add row and column colors
        patches = [Patch(color=v, label=k) for k,v in palette.items()] # add legend
        clustergrid.figure.legend(handles=patches, bbox_to_anchor = (1.23, 1), title='Order')

        return clustergrid
        
    elif algorithm in ["gmm", "kmeans"] and data is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='w')
        markers = {"Incorrect": "X", "Correct": "o"}
        sns.scatterplot(x="Dimension 1", y="Dimension 2", hue=labels, style="Predictions", data=data, \
            palette=palette, markers=markers, ax=ax, **kwargs)
        plt.legend(bbox_to_anchor = (1.05, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('CMDS Dimension 1')
        ax.set_ylabel('CMDS Dimension 2')
        ax.tick_params(left=False, bottom=False)

        return ax
    
    else:
        print("If algorithm is 'agg', dissim_matrix and linkage_matrix must be given. If algorithm is 'gmm' or 'kmeans', \
            data must be given.")
    