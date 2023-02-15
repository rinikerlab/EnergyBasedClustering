from sklearn.decomposition import PCA
from scipy.spatial import KDTree as KDTreeSP
from scipy.spatial.distance import pdist
from scipy.linalg import qr, svd

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as snp

from Hierarchical import *


class EBC:
    def __init__(self, 
                 n_clusters=None, 
                 temperature=1, 
                 percentile=1, 
                 proto_radius=None, 
                 knn=8, 
                 knn_distance=None, 
                 gamma=1.25,
                 n_samples=500, 
                 pca_steps=8, 
                 pca_threshold=0.8, 
                 pca_components=None, 
                 verbose=True, 
                 mode='knn', 
                 mode_energy='min', 
                 use_sparse=True, 
                 boxsize=None):
        self._n_clusters = n_clusters
        self._temperature = temperature
        self._percentile = percentile
        self._proto_radius = proto_radius
        self._knn = knn 
        self._knn_distance = knn_distance
        self._gamma = gamma
        self._n_samples = n_samples    
        self._pca_steps = pca_steps
        self._pca_threshold = pca_threshold
        self._pca_components = pca_components
        self._verbose = verbose
        self._mode = mode        
        self._mode_energy = mode_energy
        self._use_sparse = use_sparse
        self._boxsize = boxsize
        self._visualisation_ready = False
        self._pi = None
        
    def _from_array(self, states, energies, stride=1, start=0, end=None):        
        states, energies = states[start:end][::stride], energies.squeeze()[start:end][::stride]
        states, states_pca = self._prepare_states(states)
        self._states, self.states_raw, self.energies = states_pca, states, energies
        
    def _prepare_states(self, states):
        self._num_states = states.shape[0]
        states = states.reshape((self._num_states, -1))        
        if self._pca_components is None:            
            self._pca_components = self._guess_pca_dimensionality(states)
        if self._pca_components >= states.shape[-1]:                
            return states, states
        pca = PCA(n_components=self._pca_components)
        states_pca = pca.fit_transform(states)      
        if self._verbose:
            variance_ratio = np.round(np.sum(pca.explained_variance_ratio_) * 100, 1)
            print(f'Using {self._pca_components} components, explaining {variance_ratio}% of variance') 
        return states, states_pca 
        
    def _guess_pca_dimensionality(self, states):
        explained_variance_ratios = {}    
        sample_states = states[np.random.choice(len(states), self._n_samples)]
        # Check explained variance percentage for a couple of guesses.
        num_components = np.ceil((np.logspace(-2, -4e-2, self._pca_steps) * (states.shape[-1]))).astype(np.int32)
        num_components = np.unique(num_components)
        num_components = num_components[num_components > 1]
        for num_dims in num_components:
            if num_dims < self._n_samples:
                pca = PCA(n_components=num_dims)
                pca.fit_transform(sample_states )
                if np.sum(pca.explained_variance_ratio_) >= self._pca_threshold:
                    return num_dims
        # If no guess is successfull, use full dimensionality.
        return states.shape[-1]
    
    def _set_proto_radius(self):
        if self._proto_radius is None:
            distances_samples = pdist(self._states[np.random.choice(self._states.shape[0], self._n_samples)])   
            if self._verbose:
                print('Proto radius set to: ', np.percentile(distances_samples, self._percentile))
            self._proto_radius = np.percentile(distances_samples, self._percentile)
    
    # Take neighbour sets sorted by their potential energy and assign the sample 
    # with the lowest potential as the cluster center.
    # Returns the neighbourhoods as dictionaries indexed by the cluster center.
    def _prepare_proto_clusters(self):
        tree = KDTreeSP(self._states, boxsize=self._boxsize)
        proto_clusters = {}
        unassigned_centers, assigned_centers = set(list(range(len(self._states)))), set()
        nodes = np.array(list(unassigned_centers))[np.argsort(self._energies)]
        for node in nodes:
            if node in unassigned_centers:     
                neighbours = tree.query_ball_point(self._states[node:node+1], r=self._proto_radius)[0]
                neighbours = np.array(list(set(neighbours) - assigned_centers))
                selected_center = neighbours[np.argmin(self._energies[neighbours])]
                neighbours = set(neighbours) 
                proto_clusters[selected_center] = neighbours
                assigned_centers |= neighbours
                unassigned_centers -= neighbours
        self._num_proto_clusters = len(proto_clusters)
        self._proto_centers = np.array(list(proto_clusters.keys()))
        self._proto_coords = self._states[self._proto_centers]
        self._tree = KDTreeSP(self._proto_coords, boxsize=self._boxsize)
        if self._verbose:
            print('Using {} proto clusters.'.format(self._num_proto_clusters))
        return proto_clusters 
    
    def _prepare_energies(self):
        if self._mode_energy == 'min_mode':
            self._proto_energies = self._energies[self._proto_centers]
        else:
            mean_energy = lambda key: np.mean(self._energies[list(self._proto_clusters[key])])
            self._proto_energies = np.array([mean_energy(key) for key in self._proto_clusters])
            
    def _distance_kernel(self, distances):
        r0 = 2 * self._proto_radius
        alpha = 2.145966026289347 / (self._knn_distance - r0) # 2.62826088488
        return np.exp(-np.square(np.maximum(distances - r0, 0) * alpha))
    
    def _build_diffusion_matrix_knn(self):       
        indices_a = self._tree.query(self._proto_coords, self._knn + 1)[1][:, 1:].flatten()
        indices_b = np.tile(np.arange(self._num_proto_clusters)[:, None], self._knn).flatten() 
        dEs = self._proto_energies[indices_b] - self._proto_energies[indices_a]
        metropolis_weights = np.exp(np.minimum(np.divide(dEs, self.temperature), 0))
        diffusion_matrix = np.zeros((self._num_proto_clusters, self._num_proto_clusters), dtype=np.float64)
        diffusion_matrix[indices_a, indices_b] = metropolis_weights / (self._knn)
        diffusion_matrix[np.diag_indices(n=self._num_proto_clusters)] = 0
        diffusion_matrix[np.diag_indices(n=self._num_proto_clusters)] = 1 - np.sum(diffusion_matrix, axis=0)
        diffusion_matrix = diffusion_matrix.T
        return snp.csr_matrix(diffusion_matrix), diffusion_matrix
    
    def _build_diffusion_matrix_knnr(self):
        if self._knn_distance is None:
            self._knn_distance = np.amax(self._tree.query(self._proto_coords, k=2)[0]) * self._gamma
        distance_matrix_sp = self._tree.sparse_distance_matrix(self._tree, max_distance=self._knn_distance, output_type='coo_matrix')
        distance_matrix_sp.eliminate_zeros()
        knn = 1 / distance_matrix_sp.sign().sum(axis=1).A1
        indices_a, indices_b, distances = *distance_matrix_sp.nonzero(), distance_matrix_sp.data
        distance_weights = self._distance_kernel(distances)
        selection_probabilities = knn[indices_a]
        dEs = self._proto_energies[indices_b] - self._proto_energies[indices_a]
        metropolis_weights = np.exp(np.minimum(np.divide(dEs, self.temperature), 0)) 
        diffusion_matrix_sp = metropolis_weights * distance_weights * selection_probabilities
        diffusion_matrix_sp = snp.coo_matrix((diffusion_matrix_sp, (indices_a, indices_b)))
        diffusion_matrix_sp.setdiag(1 - np.sum(diffusion_matrix_sp, axis=0).A1)
        diffusion_matrix_sp = diffusion_matrix_sp.transpose()
        return diffusion_matrix_sp, diffusion_matrix_sp.todense().A
    
    def _build_diffusion_matrix(self):
        if self._mode == 'knn':
            diffusion_matrix_sp, diffusion_matrix = self._build_diffusion_matrix_knn()
        elif self._mode == 'knnr':
            diffusion_matrix_sp, diffusion_matrix = self._build_diffusion_matrix_knnr()
        return diffusion_matrix_sp, diffusion_matrix
    
    # QR rank from : https://academic.oup.com/imaiai/article/8/1/181/5045955?login=true
    # Implementation from SKLearn: https://github.com/scikit-learn/scikit-learn/blob/baf0ea25d/sklearn/cluster/_spectral.py#L25
    def _extract_labels(self, vectors):    
        k = vectors.shape[1]
        _, _, piv = qr(vectors.T, pivoting=True)
        ut, _, v = svd(vectors[piv[:k], :].T)
        vectors = np.abs(np.dot(vectors, np.dot(ut, v.conj())))
        return vectors.argmax(axis=1)
    
    def _eigen(self):
        w, v = np.linalg.eig(self._diffusion_matrix)
        argsort = np.argsort(-w)
        self.w, self.v = w[argsort].real, v[:, argsort].real
        self._check_gap()
        return self.v[:, :self._n_clusters]

    def _eigen_sparse(self):
        w, v = snp.linalg.eigs(self._diffusion_matrix_sp, k=self._n_clusters, which='LM')
        argsort = np.argsort(-w)
        self.w, self.v = w[argsort].real, v[:, argsort].real
        self._check_gap()
        return self.v

    def _extract_clusters(self, n_clusters=None):
        if n_clusters is not None:
            self._n_clusters = n_clusters
        proto_labels = self._extract_labels(self._eigen_sparse()) if self._use_sparse else self._extract_labels(self._eigen())
        state_labels = np.zeros((len(self._states)), dtype=np.int64)
        for idk, (core_key, cluster_set) in enumerate(self._proto_clusters.items()):
            state_labels[list(cluster_set)] = proto_labels[idk]
        return proto_labels, state_labels
    
    def extract_clusters(self, n_clusters=None):
        self._proto_labels, self._state_labels = self._extract_clusters(n_clusters=n_clusters)
        self._cluster_ids = list(set(self._proto_labels))

    def fit(self, states, energies):
        self._from_array(states, energies)
        self._set_proto_radius()
        self._proto_clusters = self._prepare_proto_clusters()    
        self._prepare_energies()
        self._diffusion_matrix_sp, self._diffusion_matrix = self._build_diffusion_matrix()
        if self._n_clusters is not None:
            self.extract_clusters()
            
    def fit_transform(self, states, energies):
        self.fit(states, energies)
        return self.state_labels
            
    def _check_gap(self):
        if self.w.size > 1:
            gap = self.w[0] - self.w[1]
            if gap < 1e-5:
                print(f'The gap between the two largest eigenvalues is {gap}.')
                print('Your graph might consist of multiple disconnected components which can result in unexpected behaviour.')
                print('Consider increasing the number of knn and/or the temperature.')
        if self._verbose:
            print(f'Using the eigenvector corresponding to an eigenvalue of {self.w[0]}.')
        
    @property
    def pi(self):
        if self._pi is None:
            pi = snp.linalg.eigs(self._diffusion_matrix_sp.T, k=1, which='LM')[1].real.squeeze()
            self._pi = pi / np.sum(pi)            
            #pi = self.v[:, 0].squeeze()
            #self._pi = pi / np.sum(pi)
        return self._pi
    
    @property
    def labels(self):
        return self._state_labels.copy()
    
    @property
    def state_labels(self):
        return self._state_labels.copy()
    
    @property
    def proto_labels(self):
        return self._proto_labels.copy()
    
    @property
    def distance_matrix(self):
        return self._distance_matrix.copy()
    
    @property
    def diffusion_matrix(self):
        return self._diffusion_matrix.copy()
    
    @property
    def diffusion_matrix_sp(self):
        return self._diffusion_matrix_sp.copy()
    
    @property
    def temperature(self):
        return self._temperature
    
    @temperature.setter
    def temperature(self, temperature):
        if temperature <= 0:
            temperature = np.nextafter(0, 1, dtype=np.float32)
            print(f'Temperature cannot be <= 0, set to {temperature}.')
        self._temperature = temperature
        
    @property
    def energies(self):
        return self._energies
    
    @energies.setter
    def energies(self, energies):
        self._energies = energies

    def select(self, ids=None):
        if ids is None:
            return self._proto_centers
        return self._proto_centers[ids]
    
    def _prepare_coordinates_2D(self):
        if self._states.shape[-1] <= 2:
            self._states_2D = self._states.copy()
            self._proto_2D = self._proto_coords.copy()
        else:
            self._pca_2d = PCA(n_components=2)
            self._states_2D = self._pca_2d.fit_transform(self._states)    
            self._proto_2D = self._pca_2d.transform(self._proto_coords)
        self._visualisation_ready = True    
        
    def get_cluster_members(self, cluster_key, only_proto=True):
        if only_proto:
            return np.where(self._proto_labels == cluster_key)
        return np.where(self._state_labels == cluster_key)

    def show(self, savename=None, show_clusters=True, s=10, s_big=250, alpha_proto=0.8, colormap=None):
        if not self._visualisation_ready:
            self._prepare_coordinates_2D()  
        self._prepare_colormap(colormap=colormap)
        fig = plt.figure(0, figsize=(20, 20), dpi=200, facecolor='white')
        plt.scatter(self._states_2D[:, 0], self._states_2D[:, 1], c=-self.energies, s=s, cmap=plt.cm.inferno, zorder=-1, alpha=1.0) 
        if show_clusters:
            for idk, cluster_key in enumerate(self._cluster_ids):
                coords = self._proto_2D[self.get_cluster_members(cluster_key)]
                plt.scatter(coords[:, 0], coords[:, 1], color=self.cluster_colormap[idk], s=s_big, alpha=alpha_proto)
        return self._plot_clean_up(plt, savename=savename)
        
    def _prepare_colormap(self, colormap=None):
        if colormap is None:
            if self._n_clusters <= 10:
                colormap = plt.cm.tab10.colors
            elif self._n_clusters <= 20:
                colormap = plt.cm.tab20.colors
            else:
                colormap = np.array(plt.cm.cividis.colors)[np.linspace(0, 255, self._n_clusters, dtype=np.int32)]
        else:
            colormap = np.array(colormap.colors)[np.linspace(0, 255, self._n_clusters, dtype=np.int32)]
        self.colormap = colormap
        self.cluster_colormap = {cluster_id: color for cluster_id, color in enumerate(colormap)}        
        
    def _plot_clean_up(self, plt, savename=None):
        ax = plt.gca()
        ax.grid(False)
        plt.axis('off')
        if savename is not None:
            plt.savefig(savename, bbox_inches='tight')
        return plt.gcf() 
        
    def plot_graph(self, threshold=5e-3, tau=10, scaling=5, show_labels=False, save=False, savename=None, dpi=200, colormap=None):
        if not self._visualisation_ready:
            self._prepare_coordinates_2D()
        coords = self._states_2D[self._proto_centers]
        flow = np.linalg.matrix_power(self.diffusion_matrix, tau)
        flow[flow < threshold] = 0
        g = nx.from_numpy_matrix(flow)
        g.remove_edges_from(nx.selfloop_edges(g))
        population = self.pi
        for edge in g.edges:
            receiver, sender = edge
            weight = population[sender] * flow[sender, receiver] + population[receiver] * flow[receiver, sender]
            #weight = flow[sender, receiver] + flow[receiver, sender]
            g[edge[0]][edge[1]]['weight'] = weight
        max_cluster = np.amax(list(self._proto_clusters.keys()))
        fig = plt.figure(0, figsize=(20, 16), dpi=dpi, facecolor='white')
        #plt.scatter(coords[:, 0], coords[:, 1], c=population * 100, zorder=2, cmap=plt.cm.plasma, edgecolor='black', s=250)
        plt.scatter(self._states_2D[:, 0], self._states_2D[:, 1], c=-self.energies, s=0.5, cmap=plt.cm.plasma, zorder=-10, alpha=0.6)
        plt.scatter(self._proto_2D[:, 0], self._proto_2D[:, 1], c=self.pi * 100, s=125, cmap=plt.cm.plasma, zorder=100, alpha=0.8)
        plt.colorbar(label='Population [\%]')
        plt.box(False)
        ax = plt.gca()    
        pos = {key: coords[key] for key in g.nodes}
        weights = [g[u][v]['weight'] for u,v in g.edges]
        weights = np.array(weights)
        weights /= np.amax(weights) 
        weights *= scaling
        nx.draw_networkx_edges(g, pos, ax=ax, width=weights)
        if show_labels:
            self._prepare_colormap(colormap=colormap)
            labeldict = {idk: label for idk, label in enumerate(self.proto_labels)}
            node_colors = [self.cluster_colormap[key] for key in self.proto_labels]
            nx.draw_networkx_nodes(g, pos=pos, ax=ax, node_color=node_colors)
            nx.draw_networkx_labels(g, pos=pos, labels=labeldict, ax=ax)
        ax.set_facecolor('white')    
        if save:
            plt.savefig(savename, bbox_inches='tight')
            
    def write_clusters(self, rdkit_mol):
        from Utilities import write_xyz
        conformations = np.array([x.GetPositions() for x in rdkit_mol.GetConformers()])
        symbols = [e.GetSymbol() for e in rdkit_mol.GetAtoms()]
        for idk, cluster_key in enumerate(self._cluster_ids):
            indices = self.get_cluster_members(cluster_key)[0]
            index_max_pop = indices[np.argmax(self.pi[indices])]
            trajectory_index = self.select(index_max_pop)
            write_xyz(conformations[trajectory_index], symbols, file_name=f'cluster_{cluster_key}.xyz')
            
    def hierarchical(self, n_steps=250, use_colormap=True, figsize=(20, 12), dpi=100, ymin=None, ymax=None, fontsize=18, labelpad=20):
        self._prepare_colormap()
        fe, e_min, e_max = get_fes(self.pi)
        fe_cutoffs = np.logspace(np.log10(e_max) , np.log10(e_min), n_steps) + 0.1
        g = nx.from_scipy_sparse_array(self._diffusion_matrix_sp)
        cc_dict, components_depth, final_merge = get_components(g, fe_cutoffs, fe)
        edges, components_dict, fe_dict, components_cluster = get_edges(cc_dict, fe_cutoffs, fe, self.cluster_colormap, self.proto_labels)    
        graph, n_neighbours, root_node, leafs = build_graph(edges, components_dict)
        levels, targets, children = prepare_tree_structure(graph, n_neighbours, fe_dict, root_node, leafs, )
        positions = prepare_positions(graph, root_node, children, leafs, fe_dict)
        return plot_tree(positions, components_cluster, targets, e_min, final_merge, use_colormap=use_colormap, figsize=figsize, dpi=dpi, ymin=ymin, ymax=ymax, fontsize=fontsize, labelpad=labelpad)
        
    