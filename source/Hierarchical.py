import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def get_fes(pi):
    fe = -np.log(pi)
    fe[np.where(np.isnan(fe))] = -np.log(np.maximum(1e-15, np.amin(pi)))
    e_min, e_max = np.amin(fe), np.amax(fe)
    return fe, e_min, e_max

def build_graph(edges, components_dict):
    graph = nx.DiGraph()
    for edge in edges.values():
        graph.add_edge(edge[1], edge[0])
    graph.remove_edges_from(nx.selfloop_edges(graph))
    n_neighbours = {node: graph.in_degree(node) for node in graph}
    root_node = list(components_dict.keys())[0]
    leafs = [node for node in graph.nodes if n_neighbours[node] == 0]
    return graph, n_neighbours, root_node, leafs

def get_components(g, fe_cutoffs, fe):    
    cc_dict, final_merge = {}, fe_cutoffs[0]
    for fe_cutoff in fe_cutoffs:
        g.remove_nodes_from(np.where(fe > fe_cutoff)[0])
        connected_components = nx.connected_components(g)
        connected_components = [cc for cc in connected_components]
        components_depth = [np.amin(fe[list(cc)]) for cc in connected_components]
        cc_dict[fe_cutoff] = [x for _, x in sorted(zip(components_depth, connected_components))]
        if len(connected_components) > 1 and final_merge == fe_cutoffs[0]:
            final_merge = fe_cutoff
    return cc_dict, components_depth, final_merge

def get_edges(cc_dict, fe_cutoffs, fe, colormap, proto_labels):
    edges, components_dict, fe_dict, components_cluster = {}, {}, {}, {}
    for idx in range(len(fe_cutoffs) - 1):
        fe_cutoff_0, fe_cutoff_1 = fe_cutoffs[idx], fe_cutoffs[idx + 1]
        for component_a in cc_dict[fe_cutoff_0]:
            hash_comp_a = hash(tuple(component_a))
            fe_dict[hash_comp_a] = fe_cutoff_0
            components_dict[hash_comp_a] = (fe_cutoff_0, component_a)
            comp_a = list(component_a)
            for component_b in cc_dict[fe_cutoff_1]:
                comp_b = list(component_b)
                hash_comp_b = hash(tuple(component_b))
                if not component_b.isdisjoint(component_a):
                    components_dict[hash_comp_b] = (fe_cutoff_1, component_b)
                    fe_dict[hash_comp_b] = fe_cutoff_1
                    edges[(hash_comp_a, hash_comp_b)] = (hash_comp_a, hash_comp_b, fe_cutoff_0, fe_cutoff_1)
                components_cluster[hash_comp_a] = colormap[proto_labels[comp_a[np.argmin(fe[comp_a])]]]
            components_cluster[hash_comp_b] = colormap[proto_labels[comp_b[np.argmin(fe[comp_b])]]]
    return edges, components_dict, fe_dict, components_cluster

def prepare_tree_structure(graph, n_neighbours, fe_dict, root_node, leafs, ):
    levels, targets, children = {}, {}, {}
    for node in leafs:
        merge_node = node
        while merge_node != root_node:
            descendants = {descendant: fe_dict[descendant] for descendant in nx.descendants(graph, node) if n_neighbours[descendant] > 0}
            merge_node = root_node
            if len(descendants) > 1:
                merge_node = min(descendants, key=descendants.get)                
            levels[node] = fe_dict[merge_node]
            targets[node] = merge_node
            if merge_node not in children:
                children[merge_node] = set()
            children[merge_node].update([node])
            node = merge_node
    levels = dict(sorted(levels.items(), key=lambda item: item[1], reverse=False))
    return levels, targets, children

def prepare_positions(graph, root_node, children, leafs, fe_dict):
    x_positions, current_x = {}, 0
    dfs_order = np.flip([key for key in nx.dfs_preorder_nodes(graph.reverse(), source=root_node)]) 
    for node in dfs_order:
        if node in leafs:
            x_positions[node] = current_x
            current_x += 0.1
    for node in dfs_order:
        if node not in leafs and node in children:
            child_pos = [x_positions[child] for child in children[node]]
            x_positions[node] = np.mean(child_pos)
    return {node: (x_pos, fe_dict[node]) for node, x_pos in x_positions.items()}

def plot_tree(positions, components_cluster, targets, e_min, final_merge, width=4, dpi=200, figsize=(12, 7), use_colormap=True, ymin=None, ymax=None, fontsize=18, labelpad=20):
    plt.figure(0, figsize=figsize, dpi=dpi)
    for node, (x0, y0) in positions.items():
        if node in targets:
            x1, y1 = positions[targets[node]]
            color = 'black'
            if use_colormap:
                try:
                    color = components_cluster[node]
                except:
                    pass
            linecoll = plt.vlines(x0, y0, y1, linewidths=width, color=color)#'black')
            linecoll = plt.hlines(y1, x0, x1, linewidths=width, color=color)#'black')
            linecoll.set_capstyle('round')
        else:
            plt.vlines(x0, y0, y0)
    ax = plt.gca()
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)    
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks(np.arange(np.ceil(e_min), final_merge * 1.2, 1))
    if ymin is None:
        ymin = np.floor(e_min)
    if ymax is None:
        ymax = final_merge * 1.2
    ax.set_ylim(ymin, ymax) # 
    ax.set_ylabel('Free Energy [a.u.]', fontsize=fontsize, labelpad=labelpad)
    plt.yticks(fontsize=fontsize)
    return plt.gcf()


        
    


