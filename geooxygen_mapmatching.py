# This code is based on the algorithm in Mustiere and Devogele 2007 
# "Matching Networks with Different Levels of Detail", DOI: 10.1007/s10707-007-0040-1

# The inputs to the core process, NetMatcher, should be two road networks: 
# Net1 at coarse scale and Net2 at fine scale. 

# The overall workflow as decribed in the paper:
#                Networks preparation (1)
#                        |
#                   ------------
#                   |          |
#     Nodes pre-matching (2)   Arcs pre-matching (3)
#            |                        |
#            |  ----------------------|
#            |  |                     |
#               v                     |
# Nodes matching (4)  --------->  Arcs matching (5)
#            |                        | 
#            --------------------------
#                         |
#                     Global evaluation (6)

import os
import geopandas as gpd
import networkx as nx
from pathlib import Path
from scipy.spatial.distance import cdist, euclidean
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, LineString
import configparser
from enum import Enum

NodeStatus = Enum('NodeStatus', ['COMPLETE', 'INCOMPLETE', 'IMPOSSIBLE'])
MatchingStatus = Enum('MatchingStatus', ['UNMATCHED', 'CERTAIN', 'UNCERTAIN'])

D_MAX = 100 # possible max distance in displacement, default as Geoxygen
D_RES = 30 # tolorence, default as Geoxygen
ID_COLNAME = 'arc_id'
LENGTH_COLNAME = 'length'


def load_input(data_root: Path, 
               coarse_shp_fn: str,
               fine_shp_fn: str):
    """
        Load the geometry of road networks at coarse and fine as shapefile. 

        :param data_root: path to folder with the shapefiles.
        :param coarse_shp_fn: shapefile name of the coarse road network
        :param fine_shp_fn: shapefile name of the fine road network
    """
    coarse_gdf = gpd.read_file(data_root / coarse_shp_fn) # Net1
    fine_gdf = gpd.read_file(data_root / fine_shp_fn) # Net2

    raw_columns_coarse = coarse_gdf.columns
    raw_columns_fine = fine_gdf.columns

    coarse_gdf['length'] = coarse_gdf.geometry.length
    fine_gdf['length'] = fine_gdf.geometry.length

    coarse_gdf['arc_id'] = coarse_gdf.index
    fine_gdf['arc_id'] = fine_gdf.index

    coarse_gdf['s'] = coarse_gdf.geometry.apply(lambda x: x.coords[0])
    coarse_gdf['e'] = coarse_gdf.geometry.apply(lambda x: x.coords[-1])
    fine_gdf['s'] = fine_gdf.geometry.apply(lambda x: x.coords[0])
    fine_gdf['e'] = fine_gdf.geometry.apply(lambda x: x.coords[-1])

    return coarse_gdf, fine_gdf, raw_columns_coarse, raw_columns_fine


def gdf_to_nx(gdf: gpd.GeoDataFrame, approach='primal', 
              length: str='SHAPE_Leng', indexcol: str='arc_id', startcol: str='s', endcol: str='e', directed=False):
    """
        convert a road network in geopandas GeoDataFrame to networkx MultiGraph. 

        :param gdf: a GeoDataFrame
        :param approach: string indicating data model. 
            'primal' means treat intersection as nodes and street segments as edges
        :param length: the column that indicates road segment length;
        :param indexcol: the column used as the unique id for road arcs, 
            as the same pair of endpoints may have multiple arcs connecting; served as key for multigraph
        :param startcol: the column with the start endpoint, as a tuple of coordinates 
        :param endcol: the column with the end endpoint, as a tuple of coordinates
        :param directed: if the road network is directed. The default is False. 
            The directed version is not considered in the current implementation.

        :return a multigraph net
    """

    try:
        length_loc = gdf.columns.get_loc(length)
    except:
        raise Exception("Column for length does not exist.")
    try:
        id_loc = gdf.columns.get_loc(indexcol)
    except:
        raise Exception("Column for FID does not exist.")
    try:
        s_loc = gdf.columns.get_loc(startcol)
    except:
        raise Exception("Column for start node s does not exist.")
    try:
        e_loc = gdf.columns.get_loc(endcol)
    except:
        raise Exception("Column for end node e does not exist.")
    
    net = nx.MultiGraph()

    for row in gdf.itertuples(index=False):
        # use arc_id twice for fast access
        net.add_edge(row[s_loc], row[e_loc], key=row[id_loc], length=row[length_loc], arc_id=row[id_loc])
    return net


def semi_Hausdorff(arc2, arc1) -> float:
    """
        defined as max_arc2(min_arc1(dist(p2, p1))), p2 belongs to arc2, p1 belongs to arc1

        :param arc2: an array with coordinates as items
        :param arc1: an array with coordinates as items
        :return float. representing the semi Hausdorff distance
    """    
    return np.max(np.min(cdist(arc2, arc1), axis=1))


def prematching_arcs(coarse_gdf: gpd.GeoDataFrame, fine_gdf: gpd.GeoDataFrame) -> dict:
    """
        Querying with semi-Hausdorff is inefficient, therefore we make it into two steps:
        1. Using Euclidean distance to find the candidate arcs1 for each arc2, so we can use spatial index
        2. Only apply semi-Hausdorff calculation for the candidate arcs1
        
        :param coarse_gdf: Geopandas GeoDataFrame represents the road network at smaller scale
        :param fine_gdf: Geopandas GeoDataFrame represents the road network at larger scale
        :return a dictionary, key is the unique id in coarse_gdf assigned by ID_COLNAME, 
                value is a list unique ids in fine_gdf with the same ID_COLNAME
    """
    # step 1
    coarse_gdf['buffer'] = coarse_gdf.buffer(D_MAX)
    coarse_gdf['line'] = coarse_gdf['geometry'].copy()
    
    coarse_gdf.geometry = coarse_gdf['buffer']
    fine_candidates = coarse_gdf.sjoin(fine_gdf, how='inner', predicate='intersects', 
                                       lsuffix='coarse', rsuffix='fine')
    # switch the geometry back to polyline
    coarse_gdf.geometry = coarse_gdf['line']
    
    # step 2
    id_loc = fine_gdf.columns.get_loc(ID_COLNAME)

    arc_matching_coarse2fine = {}

    coarse_fine_groups = fine_candidates.groupby(f"{ID_COLNAME}_coarse")

    # TODO: This part can be accelerated by parallel computing
    for coarse_fid, candidate_series in tqdm(coarse_fine_groups, total=coarse_fine_groups.ngroups):
        arc1 = coarse_gdf[coarse_gdf[ID_COLNAME] == coarse_fid]['line']

        # arc1_cand = fine_gdf[fine_gdf[FID_COLNAME].isin(candidate_series[f"{FID_COLNAME}_fine"])]
        arc2_cand = fine_gdf.iloc[candidate_series['index_fine']] # should be faster than the line above

        id_hausdist = {row[id_loc]: semi_Hausdorff(arc1.geometry.values[0].coords, row.geometry.coords) 
                        for row in arc2_cand.itertuples(index=False)} 

        a2_closest_id = min(id_hausdist, key=id_hausdist.get)
        a2_closest_dist = id_hausdist[a2_closest_id]
        a2_threshold = min(D_MAX, a2_closest_dist + D_RES)
        a2_candidate_fids = [k for k, v in id_hausdist.items() if v < a2_threshold]

        arc_matching_coarse2fine[coarse_fid] = a2_candidate_fids
        
    return arc_matching_coarse2fine


def pre_match_nodes(node1_tar: tuple, arc_matching_coarse2fine: dict, coarse_net: nx.MultiGraph, fine_net: nx.MultiGraph, fine_gdf: gpd.GeoDataFrame) -> list:
    """
        Find candidate matched node2 for node1 to determine if each n
        
        :param node1_tar: the targeted node1 in the coarse network that should be investigated
        :param arc_matching_coarse2fine: the pre-matched arc1-arc2s pairs
        :param coarse_net: networkx graph represents the topology road network at smaller scale
        :param fine_net: networkx graph represents the topology road network at larger scale
        :param fine_gdf: Geopandas GeoDataFrame represents the road network at larger scale
        :return: (node1_tar_arcs,matched_results, correspond_arc2), 
            node1_tar_arcs: arc ids connecting to the targeted node1
            matched_results: a list with tuple items (node1_tar, node2, NodeStatus)
            correspond_arc2: for visualization purpose only

    """
    matched_results = []

    # For each node1_tar, find corresponding node2s (as start and end points of an arc2) 
    #   by checking pre-matched arc1 and arc2
    # For each corresponding node2, check connected arc2 and see if arc2s match arc1s connected to node1_tar

    node1_tar_arcs = [edge[2] for edge in coarse_net.edges(node1_tar, data='arc_id')]

    # Possible solution 1: do not differenciate arcs and endpoints
    # Collect all possile arc2
    correspond_arc2 = set()
    for arc1_id in node1_tar_arcs:
        if arc1_id in arc_matching_coarse2fine:
            correspond_arc2.update(set(arc_matching_coarse2fine[arc1_id]))

    correspond_node2 = set()
    for arc2_id in correspond_arc2:
        arc2_s = fine_gdf.loc[arc2_id, 's']
        arc2_e = fine_gdf.loc[arc2_id, 'e']
        correspond_node2.add(arc2_s)
        correspond_node2.add(arc2_e)

    for cand_node2 in correspond_node2:
        cand_node2_arcs = set([edge[2] for edge in fine_net.edges(cand_node2, data='arc_id')])
        
        matched_num = 0
        matched_arc2 = set()
        processed_arc1 = set()
        for arc1_id in node1_tar_arcs:
            # every arc to node1 has to have a DIFFERENT matched arc to node2

            # here is to avoid M arc1 to match 1 arc2
            # TODO: no direction or clockwise issues has been considered
            matched_arc2_candidates = set(arc_matching_coarse2fine[arc1_id]).intersection(cand_node2_arcs)
            processed_arc1.add(arc1_id)        
            if len(matched_arc2_candidates) > 0:
                matched_arc2 = matched_arc2.union(matched_arc2_candidates)
                if len(matched_arc2) >= len(processed_arc1):
                    matched_num += 1

        if matched_num == len(node1_tar_arcs):
            matched_results.append((node1_tar, cand_node2, NodeStatus.COMPLETE.value))
            # node2_statuses[cand_node2] = NodeStatus.COMPLETE.value
        elif matched_num > 0:
            matched_results.append((node1_tar, cand_node2, NodeStatus.INCOMPLETE.value))
            # if node2_statuses[cand_node2] != NodeStatus.COMPLETE.value:
            #     node2_statuses[cand_node2] = NodeStatus.INCOMPLETE.value
        else:
            pass
            # No need to record impossible situation
            # matched_results.append((node1_tar, cand_node2, NodeStatus.IMPOSSIBLE))
    return node1_tar_arcs, matched_results, correspond_arc2


def filter_pre_matched_nodes(node1_tar: tuple, node1_tar_arcs: list, matched_results: list, fine_net: nx.MultiGraph, arc_matching_coarse2fine: dict):
    """
        further filter matched node2s based on workflow Fig.8, which has 2 steps
            Step 1: Search for 1-1 matching links
            Step 2: Search for 1-N matching links
        to determine node1-node2 matching status

        :param node1_tar: the targeted node1 in the coarse network that should be investigated
        :param node1_tar_arcs: arc ids in the coarse road network connecting to the targeted node1, 
            first item of the output of FUNCTION pre_match_nodes 
        :param matched_results: matched_results: a list with tuple items (node1_tar, node2, NodeStatus),
            second item of the output of FUNCTION pre_match_nodes 
        :param fine_net: for 1-N matching, the fine road network (Net2) is needed to find subgraph2
        :param arc_matching_coarse2fine: result of pre-matching arcs
        :return node_matching_final: a list of matched results, whose items are tuple of 
            (node1_tar, node2, MatchingStatus)
    """

    node_matching_final = []

    # Step 1: 1-1
    complete_matching = [_ for _ in matched_results if _[2] == NodeStatus.COMPLETE.value]
    incomplete_matching = [_ for _ in matched_results if _[2] == NodeStatus.INCOMPLETE.value]

    complete_matching_num = len(complete_matching)
    incomplete_matching_num = len(incomplete_matching)

    if len(matched_results) == 0:
        # the same order as matched_results
        node_matching_final.append((node1_tar, None, MatchingStatus.UNMATCHED.value)) 
    elif complete_matching_num == 0: # no complete match but several incomplete match
        if incomplete_matching_num == 1:
            node_matching_final.append((node1_tar, 
                                        [_ for _ in matched_results if _[2] == NodeStatus.INCOMPLETE.value][0][1],
                                        MatchingStatus.UNCERTAIN.value))
        else: # incomplete_matching_num > 1
            # Step 2: 1-N 
            incomplete_node2 = [_[1] for _ in incomplete_matching]
            subnet2 = fine_net.subgraph(incomplete_node2)
            comps_subnet2 = list(nx.connected_components(subnet2))

            comps_status = {}
            # treat each comp as a node
            for comp2_idx, comp2 in enumerate(comps_subnet2):
                matched_num = 0
                matched_arc2 = set()
                processed_arc1 = set()

                cand_comp2_arcs = set([edge[2] for edge in fine_net.subgraph(comp2).edges(data='arc_id')])
                
                for arc1_id in node1_tar_arcs:
                    # here is to avoid M arc1 to match 1 arc2 in comp2
                    # TODO: no direction or clockwise issues has been considered
                    matched_arc_comp2_candidates = set(arc_matching_coarse2fine[arc1_id]).intersection(cand_comp2_arcs)
                    processed_arc1.add(arc1_id)        
                    if len(matched_arc_comp2_candidates) > 0:
                        matched_arc2 = matched_arc2.union(matched_arc_comp2_candidates)
                        if len(matched_arc2) >= len(processed_arc1):
                            matched_num += 1
                
                if matched_num == len(node1_tar_arcs):
                    # matched_results.append((node1_tar, cand_node2, NodeStatus.COMPLETE.value))
                    comps_status[comp2_idx] = NodeStatus.COMPLETE.value
                elif matched_num > 0:
                    # matched_results.append((node1_tar, cand_node2, NodeStatus.INCOMPLETE.value))
                    comps_status[comp2_idx] = NodeStatus.INCOMPLETE.value
                else:
                    pass
            
            complete_matched_group = [comp_idx for comp_idx, label in comps_status.items() 
                                    if label == NodeStatus.COMPLETE.value]
            incomplete_matched_group = [comp_idx for comp_idx, label in comps_status.items() 
                                        if label == NodeStatus.INCOMPLETE.value]

            # no candidate group in Net2 or all subgraphs are impossible
            if len(comps_status) == 0: 
                node_matching_final.append((node1_tar, None, MatchingStatus.UNMATCHED.value))
            # if only one complete matched subgraph, marked as certain
            elif len(complete_matched_group) == 1:
                for node2 in comps_subnet2[complete_matched_group[0]]:
                    node_matching_final.append((node1_tar, node2, MatchingStatus.CERTAIN.value))
            # if several complete matched subgraph (rare case), choose the closest
            # DEFINITION: distance between node1 and subgraph2 as the shortest distance between node1
            #   and node2s in subgraph2
            elif len(complete_matched_group) > 1:
                min_dists = [min([euclidean(node1_tar, node2) for node2 in comps_subnet2[comp_idx]]) 
                                for comp_idx in complete_matched_group]
                matched_comp_idx = np.argmin(np.array(min_dists)) #position in min_dists
                matched_comp = comps_subnet2[complete_matched_group[matched_comp_idx]]
                for node2 in matched_comp:
                    node_matching_final.append((node1_tar, node2, MatchingStatus.UNCERTAIN.value))
            # no complete matched
            else:
                if len(incomplete_matched_group) == 1:
                    for node2 in comps_subnet2[incomplete_matched_group[0]]:
                        # TODO: filter the incomplete matched comp
                        node_matching_final.append((node1_tar, node2, MatchingStatus.UNCERTAIN.value))
                else:
                    for comp_idx in incomplete_matched_group:
                        for node2 in comps_subnet2[comp_idx]:
                            node_matching_final.append((node1_tar, node2, MatchingStatus.UNCERTAIN.value))

    elif complete_matching_num == 1: # only one complete match
        node_matching_final.append((node1_tar, 
                                    [_ for _ in matched_results if _[2] == NodeStatus.COMPLETE.value][0][1],
                                    MatchingStatus.CERTAIN.value))
    elif complete_matching_num > 1:
        candidate_idx = [(idx, euclidean(rec[0], rec[1])) for idx, rec in enumerate(matched_results) 
                            if rec[2] == NodeStatus.COMPLETE.value]
        closest_node2_candidate_idx = candidate_idx[np.argmin(np.array(candidate_idx)[:, 1])][0]

        closest_node2 = matched_results[closest_node2_candidate_idx][1]
        node_matching_final.append((node1_tar,                       
                                    closest_node2,
                                    MatchingStatus.UNCERTAIN.value))
    return node_matching_final


def match_arcs(arc_matching_coarse2fine: dict, coarse_gdf: gpd.GeoDataFrame,
               fine_gdf: gpd.GeoDataFrame, fine_net: nx.multigraph,
               node1_matched_node2: dict, ) -> dict:
    """
        based on algorithm described as Fig. 10: 
        For arc1 whose both endpoints (node1s) should have matched node2s
        Among subgraph formed by the node2s, find a path that has the cloest distance to arc1

        :param arc_matching_coarse2fine: result of pre-matching arcs
        :param coarse_gdf: GeoDataFrame that represents the geometry of road network at coarse scale
        :param fine_gdf: GeoDataFrame that represents the geometry of road network at fine scale
        :param fine_net: Multigraph that represents the topology of road network at fine scale
        :param node1_matched_node2: matched coarse-fine node pairs
        :return matched_arc1_arc2_id_pair: matched coarse arc to fine arc pairs as dictionary
    """

    # store id pairs as each pair of start-end points may have several arcs
    matched_arc1_arc2_id_pair = {} 

    # As arc1 must have prematched arc2arr in arc_matching_coarse2fine, we just need to filter arc2arr further
    for arc1_id in arc_matching_coarse2fine:
        arc1s = coarse_gdf.loc[arc1_id, 's']
        arc1e = coarse_gdf.loc[arc1_id, 'e']

        matched_node2_arc1s = set([_[0] for _ in node1_matched_node2[arc1s]])
        matched_node2_arc1e = set([_[0] for _ in node1_matched_node2[arc1e]])

        matched_arc2_ids = []
        matched_node2arr = set()

        if (arc1s in node1_matched_node2 and arc1e in node1_matched_node2):
            for arc2_id in arc_matching_coarse2fine[arc1_id]:
                arc2s = fine_gdf.loc[arc2_id, 's']
                arc2e = fine_gdf.loc[arc2_id, 'e']

                if ((arc2s in matched_node2_arc1s and arc2e in matched_node2_arc1e) 
                    or (arc2s in matched_node2_arc1e and arc2e in matched_node2_arc1s)):
                    matched_arc2_ids.append(arc2_id)
                    matched_node2arr.add(arc2s)
                    matched_node2arr.add(arc2e)
            
        if len(matched_arc2_ids) > 0:
            if len(matched_arc2_ids) == 1:
                # to match with path (as a set of arcs), here to convert it as a list of tuples
                matched_arc1_arc2_id_pair[arc1_id] = [tuple(matched_arc2_ids)]
            # with multiple matching, we need to find the cloest path (a set of arcs)
            else:
                matched_node2arr = list(matched_node2arr)
                subnet2 = fine_net.subgraph(matched_node2arr).copy()

                remove_edges = []
                matched_arc2_ids = set(matched_arc2_ids)

                for edge in subnet2.edges(data=True):
                    if edge[2]['arc_id'] not in matched_arc2_ids:
                        remove_edges.append(edge)

                if len(remove_edges) > 0:
                    for e in remove_edges:
                        subnet2.remove_edge(e[0], e[1], key=e[2]['arc_id'])
                
                all_paths = set()
                subnet2_nodes = list(subnet2.nodes())
                all_node_pairs = [(a, b) for idx, a in enumerate(subnet2_nodes) for b in subnet2_nodes[idx + 1:]]

                for pair in all_node_pairs:
                    for path in nx.all_simple_edge_paths(subnet2, source=pair[0], target=pair[1]):
                        all_paths.add(tuple([_[2] for _ in path]))

                path_dist2arc1 = []
                length_diff2arc1 = []

                arc1_length = coarse_gdf.loc[arc1_id, 'length']
                arc1_coords = coarse_gdf.loc[arc1_id, 'geometry'].coords

                for p in all_paths:
                    arc2 = []
                    arc2_length = 0
                    for arc2_id in p:
                        arc2.extend(fine_gdf.loc[arc2_id, 'geometry'].coords[:])
                        arc2_length += fine_gdf.loc[arc2_id, LENGTH_COLNAME]
                    # there is a risk that several paths share the same smallest distance
                    path_dist2arc1.append(semi_Hausdorff(arc1_coords, arc2))
                    length_diff2arc1.append(abs(arc1_length - arc2_length))

                path_dist2arc1 = np.array(path_dist2arc1)
                length_diff2arc1 = np.array(length_diff2arc1)

                path_cand_idx = np.where(path_dist2arc1 == path_dist2arc1.min())

                # consider the length difference between arc1 and path2arr as an additional creteria
                # because of length preservation, the path2 candidate with smallest length difference should be selected
                length_diff2arc1 = length_diff2arc1[path_cand_idx]
                length_diff_cand_idx = np.where(length_diff2arc1 == length_diff2arc1.min())

                if len(length_diff_cand_idx) > 1:
                    selected_length_idx = np.random.choice(length_diff_cand_idx[0], 1)
                else:
                    selected_length_idx = length_diff_cand_idx[0]

                selected_path_idx = path_cand_idx[0][selected_length_idx]

                all_paths = list(all_paths)
                selected_path = all_paths[selected_path_idx[0]]

                if arc1_id not in matched_arc1_arc2_id_pair:
                    matched_arc1_arc2_id_pair[arc1_id] = [selected_path]
                else:
                    matched_arc1_arc2_id_pair[arc1_id].append(selected_path)
    
    return matched_arc1_arc2_id_pair 


def output_files(out_folder: str, matched_arc1_arc2_id_pair: dict, node_matching_result: list,
                 coarse_gdf: gpd.GeoDataFrame, fine_gdf: gpd.GeoDataFrame,
                 raw_columns_coarse: list, raw_columns_fine: list,
                 coarse_file_prefix: str, fine_file_prefix: str):
    """
        output the results as files

        :param out_folder: path to the output folder
        :param matched_arc1_arc2_id_pair: a dictionary as the result of function match_arcs
        :param coarse_gdf: GeoDataFrame that represents the geometry of road network at coarse scale
        :param fine_gdf: GeoDataFrame that represents the geometry of road network at fine scale
        :param raw_columns_coarse: list stores original column names of coarse_gdf
        :param raw_columns_fine: list stores original column names of fine_gdf
        :param coarse_file_prefix: file name's prefix
        :param fine_file_prefix: file name's prefix
        :return None
    """
    out_folder = Path(out_folder)

    out_coarse_columns = list(raw_columns_coarse) + [LENGTH_COLNAME, ID_COLNAME]
    out_fine_columns = list(raw_columns_fine) + [LENGTH_COLNAME, ID_COLNAME]

    # output the raw shapefiles with additional columns created by this algorithm
    coarse_gdf[out_coarse_columns].to_file(out_folder / f'{coarse_file_prefix}_coarse.shp')
    fine_gdf[out_fine_columns].to_file(out_folder / f'{fine_file_prefix}_fine.shp')

    flatten_matched_arc1_arc2_id_pair = []

    for arc1_id, path2s in matched_arc1_arc2_id_pair.items():
        for path2 in path2s:
            for arc2_id in path2:
                flatten_matched_arc1_arc2_id_pair.append((arc1_id, arc2_id))

    out_df_arc1_arc2 = pd.DataFrame.from_records(flatten_matched_arc1_arc2_id_pair, columns=['arc1', 'arc2'])

    out_df_arc1_arc2 = out_df_arc1_arc2.merge(coarse_gdf['geometry'], 
                       left_on='arc1', right_index=True, how='left',
                       suffixes=('_match', '_coarse')).merge(fine_gdf['geometry'],
                                                                        left_on='arc2', right_index=True, how='left', 
                                                                        suffixes=('', '_fine'))
    out_df_arc1_arc2.columns = ['arc1', 'arc2', 'arc1_geometry', 'arc2_geometry']

    # this line shapefile will by displayed as arrows from corase road segment to their matched fine road segments
    out_gdf_arc1_arc2 = gpd.GeoDataFrame(out_df_arc1_arc2, geometry='arc1_geometry')
    out_gdf_arc1_arc2['match_vec'] = out_gdf_arc1_arc2.apply(lambda x: LineString([x['arc1_geometry'].centroid, 
                                                                                x['arc2_geometry'].centroid]), axis=1)
    out_gdf_arc1_arc2.set_geometry('match_vec', inplace=True)
    out_gdf_arc1_arc2.set_crs(coarse_gdf.crs, inplace=True)
    out_gdf_arc1_arc2 = out_gdf_arc1_arc2.drop(['arc1_geometry', 'arc2_geometry'], axis=1)    
    
    out_gdf_arc1_arc2.to_file(out_folder / f'arc_{coarse_file_prefix}_{fine_file_prefix}_matched.shp', driver='ESRI Shapefile')


    out_df_node_match = pd.DataFrame.from_records(node_matching_result, 
                                          columns=['node1', 'node2', 'matching_status_val'])
    out_df_node_match['matching_status'] = MatchingStatus.UNMATCHED.name
    out_df_node_match.loc[out_df_node_match['matching_status_val'] 
                    == MatchingStatus.CERTAIN.value, 'matching_status'] = MatchingStatus.CERTAIN.name
    out_df_node_match.loc[out_df_node_match['matching_status_val']
                    == MatchingStatus.UNCERTAIN.value, 'matching_status'] = MatchingStatus.UNCERTAIN.name
    out_df_node_match['geometry'] = out_df_node_match[~out_df_node_match['node2'].isna()].apply(lambda x: 
                                                                                            LineString([x['node1'], x['node2']]), axis=1) 
    
    out_gdf_node1_match_status = gpd.GeoDataFrame(out_df_node_match[['node1', 'matching_status']])
    out_gdf_node1_match_status.columns = ['node1', 'matchstatus']
    out_gdf_node1_match_status['node1pt'] = out_gdf_node1_match_status['node1'].apply(lambda x: Point(x))
    out_gdf_node1_match_status.set_geometry('node1pt', inplace=True)
    out_gdf_node1_match_status.set_crs(coarse_gdf.crs, inplace=True)
    out_gdf_node1_match_status = out_gdf_node1_match_status.drop('node1', axis=1)
    out_gdf_node1_match_status[['matchstatus', 'node1pt']].to_file(out_folder / f'{coarse_file_prefix}_node_match_status.shp', 
                                                    driver='ESRI Shapefile')
    
    out_gdf_node_match = gpd.GeoDataFrame(out_df_node_match[~out_df_node_match['node2'].isna()],
                                      geometry='geometry')
    out_gdf_node_match = out_gdf_node_match.set_crs(coarse_gdf.crs, inplace=True)
    out_gdf_node_match.drop(['node1', 'node2'], axis = 1, inplace=True)
    out_gdf_node_match.to_file(out_folder / f'{coarse_file_prefix}_node_match.shp', 
                                                    driver='ESRI Shapefile')



def geoxygen_road_match(data_root: str, coarse_shp_fn: str,
               fine_shp_fn: str, out_folder: str,
               coarse_file_prefix: str = 'coarse', fine_file_prefix: str = 'fine',
               d_max: float =100, d_res: float = 30):
    """
        The main entrance function to be called 

        :param data_root: path to folder with the shapefiles.
        :param coarse_shp_fn: shapefile name of the coarse road network
        :param fine_shp_fn: shapefile name of the fine road network
        :param out_folder: path to folder to save output files
        :param d_max: possible max distance in displacement, default as Geoxygen
        :param d_res: tolorence, default as Geoxygen
    """
    D_MAX = d_max
    D_RES = d_res    

    data_root = Path(data_root)

    coarse_gdf, fine_gdf, raw_columns_coarse, raw_columns_fine = load_input(data_root, coarse_shp_fn, fine_shp_fn) 
    
    coarse_net = gdf_to_nx(coarse_gdf, length='length', indexcol='arc_id')
    fine_net = gdf_to_nx(fine_gdf, length='length', indexcol='arc_id')

    ## Main step 1: find pre-matched arc pairs
    arc_matching_coarse2fine = prematching_arcs(coarse_gdf, fine_gdf)

    # build a inverse lookup table for the coarse-fine arc pairs
    arc_matching_fine2coarse = {}
    for coarse_arc_id, fine_arc_ids in arc_matching_coarse2fine.items():
        for fine_arc_id in fine_arc_ids:
            if fine_arc_id in arc_matching_fine2coarse:
                arc_matching_fine2coarse[fine_arc_id].append(coarse_arc_id)
            else:
                arc_matching_fine2coarse[fine_arc_id] = [coarse_arc_id]


    # node matching rules:
    # Node2 in Net2 (fine scale network) is complete if in the prematching arcs, 
    #   all arcs connected to Node1 correspond to some arcs to Node2
    # Node 2 in Net2 is incomplete if at least one arc connected to Node 1 correspond to one arc to Node2
    # Node 2 is impossible if no corresponds in prematching arcs

    node1_linked = set()

    for arc1_id in arc_matching_coarse2fine:
        node1s = coarse_gdf.loc[arc1_id, 's']
        node1e = coarse_gdf.loc[arc1_id, 'e']
        node1_linked.add(node1s)
        node1_linked.add(node1e)

    node1_linked = list(node1_linked)

    node2_linked = set()

    for arc2_id in arc_matching_fine2coarse:
        node2s = fine_gdf.loc[arc2_id, 's']
        node2e = fine_gdf.loc[arc2_id, 'e']
        node2_linked.add(node2s)
        node2_linked.add(node2e)

    node2_linked = list(node2_linked)

    node2_statuses = {n: NodeStatus.IMPOSSIBLE.value for n in node2_linked}
    
    ## Main step 2: find matched node pairs with filtering
    node_matching_result = []

    for idx, node1_tar in enumerate(node1_linked):
        node1_tar_arcs, matched_results,correspond_arc2 = pre_match_nodes(node1_tar, 
                                                                        arc_matching_coarse2fine, 
                                                                        coarse_net, 
                                                                        fine_net, fine_gdf)
        node_matching_final = filter_pre_matched_nodes(node1_tar, node1_tar_arcs, matched_results, fine_net, arc_matching_coarse2fine)
        node_matching_result.extend(node_matching_final)

    ## Main step 3: match arc pairs based on pre-matched arcs and matched nodes
    node1_matched_node2 = {}

    for r in node_matching_result:
        if r[2] != MatchingStatus.UNMATCHED:
            if r[0] in node1_matched_node2:
                node1_matched_node2[r[0]].append(r[1:]) 
            else:
                node1_matched_node2[r[0]] = [r[1:]]

    # print(len(node1_matched_node2))

    matched_arc1_arc2_id_pair = match_arcs(arc_matching_coarse2fine,
                                           coarse_gdf, fine_gdf, fine_net, node1_matched_node2)   
       
    # output results
    output_files(out_folder, matched_arc1_arc2_id_pair, node_matching_result,
                 coarse_gdf, fine_gdf,
                 raw_columns_coarse, raw_columns_fine,
                 coarse_file_prefix, fine_file_prefix)



def __main__(self):
    pass
    # config = configparser.ConfigParser()
    # config['datainput'] = {}
    # config['datainput']['data_folder'] = r''
    # config['datainput']['coarse_shp_fn'] = r''
    # config['datainput']['fine_shp_fn'] = r''
    # config['dataoutput'] = {}
    # config['dataoutput']['out_folder']

