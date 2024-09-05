import csv
import time
import numpy as np
import networkx as nx
import random
from collections import defaultdict
import copy
from multiprocessing import Pool
import matplotlib.pyplot as plt
import math
from networkx.algorithms import approximation as approx
from tqdm import tqdm
from itertools import permutations
from itertools import combinations
import pandas as pd
from geopy.distance import geodesic
from tqdm import tqdm
from scipy.stats import f_oneway, wilcoxon, kruskal, ttest_rel
from scipy.stats import kruskal, mannwhitneyu


class Resilience:
    def __init__(self, attr_name, indexing=False):
        self.name = attr_name
        self.G = nx.DiGraph()  # graph model
        # geospatial data
        self.node_coordinates = {}  # (lat, lon)
        # flow-weighted model
        # self.flow_matrix = None
        self.od_flow = {}
        self.node_flow = {}
        self.indexing = indexing
        self.record_matrix = None
        if self.indexing:
            self.node2index = None
            self.index2node = None
        self._matrix_header = None  # load from adjacency matrix
        self._edge_dict = None
        self._relocation_edge_dict = None
        self._relocation_edge_weight = None
        self._restoration_edge_dict = None
        self._restoration_node_weight = None
        self.distance_matrix = None
        self.df = None
        self._relocation_weight_dict = None
        self.trip_index = None
        self.flow_matrix = None

        # capacity-weighted model based on trips/routes (GTFS data)
        #         self.network = Network(network_name=graph_name)  # network-route-trip structure
        # self.routes = {}  # system structure
        # self.stops = {}  # NOTE: standalone stop repository
        # self.route_edge = defaultdict(list)
        # self.edge_route = None
        # self.trip_param = defaultdict(dict)
        # self.node_param = defaultdict(dict)
        # self.edge_capacity = {}
        # self.node_capacity = {}
        # multi-processing
        self.core_num = 5

    #     def load_adjacency_matrix(self, file_path, contain_header=True):
    #         node_list = []
    #         with open(file_path, 'r') as csv_file:
    #             csv_reader = csv.reader(csv_file, delimiter=',')
    #             matrix = []
    #             for row in csv_reader:
    #                 matrix.append(row)
    #         if contain_header:
    #             for i in range(len(matrix[0])):
    #                 if matrix[0][i] != matrix[i][0]:
    #                     print('error: adjacency matrix has asymmetric headers')
    #             # delete headers
    #             for row in matrix:
    #                 del row[0]
    #             header = matrix.pop(0)
    #             # use (network_name, node_name) represents node
    #             self._matrix_header = header
    #             node_list = [(self.name, node) for node in header]
    #         self.G.add_nodes_from(node_list)
    #         if self.indexing:
    #             self.node2index = {node: index for index, node in enumerate(node_list)}
    #             self.index2node = {index: node for index, node in enumerate(node_list)}
    #         for idx, x in enumerate(node_list):
    #             for idy, y in enumerate(node_list):
    #                 if int(matrix[idx][idy]) > 0:
    #                     self.G.add_edge(x, y)
    #         print('\nnetwork created:',
    #               f'name = {self.name}, '
    #               f'number of nodes = {self.G.number_of_nodes()}, '
    #               f'number of edges = {self.G.number_of_edges()}')

    #     def load_ori_data(self, filepath):

    #         trips = pd.read_csv(filepath)
    #         trips['age_group'] = trips['age'].map(self.age_range)

    #         trips['income_group'] = trips['income'].map(self.income_range)
    #         # 合并数据集，将站台名称替换为 code reference

    #         merged_df = trips.merge(station_index, left_on='origin', right_on='station', how='left')

    #         merged_df = merged_df.rename(columns={'index': 'origin_index'})

    #         merged_df_des = merged_df.merge(station_index, left_on='destination', right_on='station', how='left')

    #         merged_df_des = merged_df_des.rename(columns={'index': 'destination_index'})

    #         trip_index = merged_df_des[['time','sex','age','industry','income', 'income_group', 'origin','destination','origin_index','destination_index','age_group']]

    #         self.trip_index = trip_index

    #         else:

    #             xlsx_file = openpyxl.load_workbook(filepath)
    #             data_dict = {}

    #             for sheet_name in xlsx_file.sheetnames:
    #                 sheet = xlsx_file[sheet_name]
    #                 sheet_data = {}
    #                 for row in sheet.iter_rows(values_only=True):
    #                     sheet_data[row[0]] = row[1]  # 假设第一列是键，后面的列是值
    #                 data_dict[sheet_name] = sheet_data

    #             self.trip_index = data_dict
    #         return self.trip_index

    def load_data(self, filepath):
        trips = pd.read_csv(filepath)
        self.trip_index = trips

    def age_range(self, x):

        if 0 < x <= 14:
            return 0  # children
        elif 14 < x <= 24:
            return 1  # senior students / young adults
        elif 24 < x <= 34:
            return 2  # young working-age group
        elif 34 < x <= 44:
            return 3  #
        elif 44 < x <= 54:
            return 4  #
        elif 54 < x <= 64:
            return 5  #
        elif 64 < x:
            return 6  # retired / elderly

        else:
            print('age filter error:', x)

    def map_income_com(self, income):
        if income < 7 or income == 97.0:
            return "0"
        elif 7 <= income <= 12:
            return "1"
        elif 13 <= income <= 19:
            return "2"
        else:
            print('income filter error:', income)

    def get_node_list(self):
        return list(self.G.nodes)

    def get_edge_list(self):
        return list(self.G.edges)

    def get_edge_dict(self, update=True):
        if update:
            edge_dict = defaultdict(list)
            for edge in self.G.edges():
                x, y = edge[0], edge[1]
                edge_dict[x].append(y)
            self._edge_dict = edge_dict

    #         return self._edge_dict

    def get_adjacency_matrix(self, file_path):

        df = pd.read_csv(file_path, index_col=0)
        suffix = ' Station'
        df.columns = df.columns + suffix
        df.index = df.index + suffix
        df = df.rename(columns={'Admiralty Station': 'Admiralty station'})
        df = df.rename(index={'Admiralty Station': 'Admiralty station'})

        df.index = range(len(df))
        df.columns = range(len(df))

        self.df = df

        adjacency_df = pd.DataFrame(columns=['Source', 'Target'])

        for node1 in df.index:
            for node2 in df.columns:
                if df.loc[node1, node2] == 1:
                    self.G.add_edge(node1, node2)
                    new_row = pd.Series({'Source': node1, 'Target': node2})
                    adjacency_df = adjacency_df.append(new_row, ignore_index=True)

        print('\nnetwork created:'
              f'number of nodes = {self.G.number_of_nodes()}, '
              f'number of edges = {self.G.number_of_edges()}')

    #         return adjacency_df

    def get_station_index(self, filepath):

        station_index = pd.read_csv(filepath)

        self.station_index = station_index

    #         return self.station_index

    def o_d_matrix(self):

        record_matrix = pd.DataFrame(
            [[0 for _ in range(len(self.station_index))] for _ in range(len(self.station_index))])

        record_matrix.columns = range(len(self.df.columns))

        record_matrix.index = range(len(self.df.columns))

        for record in self.trip_index.values:
            ori = record[-3]
            des = record[-2]

            record_matrix[ori][des] += 1

        self.flow_matrix = record_matrix

        for idx in range(len(record_matrix)):
            for idy in range(len(record_matrix[idx])):
                if idx != idy:
                    self.od_flow[idx, idy] = record_matrix[idx][idy]

    #         if self.trip_index is dict:
    #             for key, value in self.trip_index.items():

    #                 record_matrix = pd.DataFrame([[0 for _ in range(len(self.station_index))] for _ in range(len(self.station_index))])

    #                 record_matrix.columns = range(len(self.df.columns))

    #                 record_matrix.index = range(len(self.df.columns))

    #                 for record in value:

    #                     ori = record[-3]
    #                     des = record[-2]

    #                     record_matrix[ori][des]+=1

    #                 self.flow_matrix[key] = record_matrix

    #         return self.flow_matrix

    def load_gps_coordinates(self, file_path, contain_header=True,
                             node_col=0, lat_col=1, lon_col=2):
        # # table: Node, Lat, Lon
        coordinates = []
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                coordinates.append(row)
        # print(coordinates)
        if contain_header:
            del coordinates[0]
        for row in coordinates:
            node, lat, lon = row[node_col], eval(row[lat_col]), eval(row[lon_col])
            self.node_coordinates[(self.name, node)] = (lat, lon)

    def reachable_nodes(self, origin, edge_dict=None):
        current_node = origin
        visited = []
        path = [current_node]
        if edge_dict is None:
            edge_dict = self.get_edge_dict()
        while path:
            current_node = path.pop(0)
            visited.append(current_node)
            destinations = edge_dict[current_node]
            for next_node in destinations:
                if next_node not in visited and next_node not in path:
                    path.append(next_node)
        return visited

    def get_node_degree(self):
        return dict(nx.degree(self.G))

    def get_node_betweenness_centrality(self):
        return dict(nx.betweenness_centrality(self.G))

    def get_local_node_connectivity(self):
        od_pairs = permutations(self.get_node_list(), 2)
        return {od: approx.local_node_connectivity(G=self.G, source=od[0], target=od[1]) for od in od_pairs}

    def get_node_flow(self):
        if not self.od_flow:
            return None
        self.node_flow = {}
        for u in self.get_node_list():
            self.node_flow[u] = 0
        for u in self.get_node_list():
            # for v in self.reachable_nodes(u):
            for v in self.get_node_list():
                if v != u:
                    self.node_flow[u] += self.od_flow[u, v]

    #         return self.node_flow

    def _relocation_dijsktra_weighted(self, initial, end):
        # shortest paths is a dict of nodes
        # whose value is a tuple of (previous node, weight)
        edge_dict = self._relocation_edge_dict
        edge_weight = self._relocation_edge_weight
        shortest_paths = {initial: (None, 0)}
        current_node = initial
        visited = set()
        while current_node != end:
            visited.add(current_node)
            destinations = edge_dict[current_node]
            weight_to_current_node = shortest_paths[current_node][1]
            for next_node in destinations:
                weight = edge_weight[(current_node, next_node)] + weight_to_current_node
                # weight = 1 + weight_to_current_node
                if next_node not in shortest_paths:
                    shortest_paths[next_node] = (current_node, weight)
                else:
                    current_shortest_weight = shortest_paths[next_node][1]
                    if current_shortest_weight > weight:
                        shortest_paths[next_node] = (current_node, weight)
            next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
            if not next_destinations:
                return math.inf
            # next node is the destination with the lowest weight
            current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
        return shortest_paths[end][1]

    def _haversine_distance_between_nodes(self, node_x, node_y):
        lo1 = self.node_coordinates[(self.name, str(node_x))]
        lo2 = self.node_coordinates[(self.name, str(node_y))]
        return haversine(lo1, lo2)

    def unweighted_relocation(self, d_max):
        """
        :return: dict of node relocation
        """

        def df(d, d_max):
            will = 1 - d / d_max
            if 0 <= will <= 1:
                return will
            else:
                return 0

        od_pairs = permutations(self.get_node_list(), 2)
        distance_matrix = {od: self._haversine_distance_between_nodes(od[0], od[1]) for od in od_pairs}
        relocation_matrix = {od: df(dst, d_max) for od, dst in distance_matrix.items()}
        relocation_potential = {}
        for node in self.get_node_list():
            relocation_potential[node] = np.sum(
                [relocation_matrix[node, other] for other in self.get_node_list() if node != other])
        return relocation_potential

    #     def haversine_distance_matrix(self, d_max):

    #         #return matrix 84*84, with the haversine distance
    #         # if distance > d_max, value is set to be 0, else set the original distance

    #         distance_matrix = pd.DataFrame([[0 for _ in range(len(self.station_index))] for _ in range(len(self.station_index))])

    #         distance_matrix.columns = range(len(self.df.columns))

    #         record_matrix.index = range(len(self.df.columns))

    #         od_pairs = permutations(self.get_node_list(), 2)

    #         distance_dic = {od: self._haversine_distance_between_nodes(od[0], od[1]) for od in od_pairs}

    #         for key, value in distance_dic.items():
    #             ori = key[0]
    #             des = key[1]

    #             if value > d_max:

    #                 distance_matrix[ori][des] = 0

    #             else:

    #                 distance_matrix[ori][des] = value

    #         self.distance_matrix = distance_matrix

    #         return self.distance_matrix

    def reachable_matrix(self, target):
        # return matrix 84*84, with 0 or 1, 0 represents two station is not reachable
        temp_G = copy.deepcopy(self.G)

        temp_G.remove_node(target)

        edge_dict = defaultdict(list)

        for edge in temp_G.edges():
            x, y = edge[0], edge[1]
            edge_dict[x].append(y)

        reachable_matrix = pd.DataFrame(
            [[0 for _ in range(len(self.station_index))] for _ in range(len(self.station_index))])

        reachable_matrix.columns = range(len(self.df.columns))

        reachable_matrix.index = range(len(self.df.columns))

        for node in self.get_node_list():
            if node != target:
                reached_nodes = self.reachable_nodes(node, edge_dict)

                for nd in reached_nodes:
                    reachable_matrix[node][nd] = 1

        return reachable_matrix

    def get_travel_distance(self, mean_value=False):
        spl = dict(nx.all_pairs_shortest_path_length(self.G))
        travel_distance_distribution = {}
        for od_pair, flow in self.od_flow.items():
            trip_len = spl[od_pair[0]][od_pair[1]]

            if trip_len in travel_distance_distribution.keys():
                travel_distance_distribution[trip_len] += flow
            else:
                travel_distance_distribution[trip_len] = flow
        # print(travel_distance_distribution)
        if mean_value:
            total_trip_len, total_flow = 0.0, 0.0
            for trip_len, flow in travel_distance_distribution.items():
                total_trip_len += trip_len * flow
                total_flow += flow
            mean_trip_len = total_trip_len / total_flow
            return round(mean_trip_len, 3)
        else:
            return travel_distance_distribution

    def flow_weighted_recovery(self, d_max):
        def df(d, d_max):
            will = 1 - d / d_max
            if 0 <= will <= 1:
                return will
            else:
                return 0

        self._relocation_edge_dict = defaultdict(set)  # edge_dict[current_node]=next_nodes`
        self._relocation_weight_dict = defaultdict(list)
        self._relocation_edge_weight = {}

        max_flows = self.flow_matrix.sum().sum()

        for x in self.get_node_list():
            for y in self.get_node_list():
                if y != x:
                    #                     if y in self._edge_dict[x]:  # connected by operational metro line
                    #                         self._relocation_edge_dict[x].add(y)
                    #                         self._relocation_edge_weight[x, y] = 0

                    #                     else:
                    dst = self._haversine_distance_between_nodes(x, y)  # walking distance
                    if dst <= d_max:  # introduce a relocation edge if no farther than 1600 metres
                        self._relocation_edge_dict[x].add(y)
                        self._relocation_edge_dict[y].add(x)
                        #                     self._relocation_weight_dict[x].append(dst)
                        self._relocation_edge_weight[x, y] = dst
                        self._relocation_edge_weight[y, x] = dst

        node_recovery = {}

        for node in self.get_node_list():

            #             weight_sum = sum(self._relocation_weight_dict[node])

            neighbor_ls = self._relocation_edge_dict[node]

            record_matrix = pd.DataFrame()

            wills = []

            reachable_matrix = self.reachable_matrix(node)

            for neighbor in neighbor_ls:
                temp_df = reachable_matrix.loc[[neighbor]]

                record_matrix = pd.concat([record_matrix, temp_df], axis=0)

                wills.append(df(self._relocation_edge_weight[node, neighbor], d_max))

            wills_df = pd.DataFrame([wills])

            #             print('wills_df of station', node, 'is', wills_df)

            #             print('record_matrix  of station', node, 'is', record_matrix)

            result_df = record_matrix.mul(wills_df.values.T, axis=1)

            def max_non_zero(x):
                non_zero_values = list(filter(lambda i: i != 0, x))
                return max(non_zero_values) if non_zero_values else 0

            # 针对每列取非零最小值，如果全部为零则该列为0

            max_non_zero_values = result_df.apply(max_non_zero, axis=0)

            # 构造1行84列的DataFrame
            #             output_df = pd.DataFrame([min_non_zero_values])

            #             print('Reachable Destination Allocation is', pd.DataFrame(min_non_zero_values))

            #             reachable_index = pd.DataFrame(record_matrix.apply(lambda row: 1 if any(row != 0) else 0, axis=1), columns= ['Result'])

            o_flow = self.flow_matrix.loc[node].tolist()

            d_flow = self.flow_matrix[node].tolist()

            od_flow = o_flow + d_flow

            overall_flow = sum(od_flow)

            reachable_flow = sum([a * b for a, b in zip(od_flow, max_non_zero_values)])

            if overall_flow != 0:

                node_recovery[node] = reachable_flow / overall_flow

            else:
                node_recovery[node] = 0

            self.node_recovery = node_recovery

        return self.node_recovery

    #     def flow_weighted_recovery(self, d_max):
    #         def df(d, d_max):
    #             will = 1 - d / d_max
    #             if 0 <= will <= 1:
    #                 return will
    #             else:
    #                 return 0

    #         self._relocation_edge_dict = defaultdict(set)  # edge_dict[current_node]=next_nodes`
    #         self._relocation_weight_dict = defaultdict(list)
    #         self._relocation_edge_weight = {}

    #         max_flows = self.flow_matrix.sum().sum()

    #         for x in self.get_node_list():
    #             for y in self.get_node_list():
    #                 if y!= x:
    # #                     if y in self._edge_dict[x]:  # connected by operational metro line
    # #                         self._relocation_edge_dict[x].add(y)
    # #                         self._relocation_edge_weight[x, y] = 0

    # #                     else:
    #                         dst = self._haversine_distance_between_nodes(x, y)  # walking distance
    #                         if dst <= d_max:  # introduce a relocation edge if no farther than 1600 metres
    #                             self._relocation_edge_dict[x].add(y)
    #                             self._relocation_edge_dict[y].add(x)
    # #                     self._relocation_weight_dict[x].append(dst)
    #                             self._relocation_edge_weight[x, y] = dst
    #                             self._relocation_edge_weight[y, x] = dst

    #         node_recovery = {}

    #         for node in self.get_node_list():

    #             o_flow = self.flow_matrix.loc[node].tolist()

    #             d_flow = self.flow_matrix[node].tolist()

    #             od_flow = o_flow + d_flow

    #             overall_flow = sum(od_flow)

    # #             weight_sum = sum(self._relocation_weight_dict[node])

    #             neighbor_ls = self._relocation_edge_dict[node]

    #             reachable_matrix = self.reachable_matrix(node)

    #             record_matrix = []

    #             lik = []

    #             wills = []

    #             for neighbor in neighbor_ls:

    #                 lik.append(self._relocation_edge_weight[node, neighbor])

    #                 temp_ls = reachable_matrix[neighbor].tolist()

    #                 reachable_flow = sum([a*b for a, b in zip(od_flow, temp_ls)])

    #                 record_matrix.append(reachable_flow)

    # #                 record_matrix.append(pd.concat([record_matrix, temp_df], axis=1))

    #                 wills.append(df(self._relocation_edge_weight[node, neighbor], d_max))

    #             liks = sum(lik)

    #             cr = [liks/i for i in lik]

    #             cns = sum(cr)

    #             cn = [i/cns for i in cr]

    #             print('will list is', wills)

    #             print('closeness list is', cn)

    #             print('will* closeness list is ', [wills[i] * cn[i] for i in range(len(cr))])

    #             relocation = sum([wills[i] * cn[i]*record_matrix[i] for i in range(len(cr))])

    # #             reachable_index = pd.DataFrame(record_matrix.apply(lambda row: 1 if any(row != 0) else 0, axis=1), columns= ['Result'])

    # #             reachable_flow = sum([a*b for a, b in zip(od_flow, reachable_index['Result'].tolist())])

    #             if overall_flow!=0:

    #                 station_relocation = relocation/(overall_flow)
    #                 node_recovery[node] = station_relocation

    #             else:
    #                 node_recovery[node] = 0

    #             self.node_recovery = node_recovery

    #         return self.node_recovery

    def get_relocation_edge_dict(self):

        return self._relocation_edge_dict

    def get_relocation_edge_weight(self):

        return self._relocation_edge_weight


def haversine(p1, p2):  # (decimal）

    dst = geodesic(p1, p2).kilometers
    return dst

Age_dic = {}
trip_dic = {}

for i in tqdm(range(7)):
    tmp = Resilience(f'age_{i}')
    tmp.load_data(f'./group_data/CSV/age_{i}.csv')
    tmp.load_gps_coordinates('GPS MTR Stations 2011_with_name_new.csv', contain_header=False)
    tmp.get_station_index('OD matrix station index.csv')
    tmp.get_adjacency_matrix('HK metro Space L 2011.csv')
    tmp.o_d_matrix()
    tmp.get_edge_dict()

    trip_distance = tmp.get_travel_distance()

    trip_dic[f'age_{i}'] = trip_distance

    if i == 0:

        Age_dic[f'age_{i}'] = tmp.flow_weighted_recovery(1.28)

    elif i == 6:
        Age_dic[f'age_{i}'] = tmp.flow_weighted_recovery(1.12)

    else:

        Age_dic[f'age_{i}'] = tmp.flow_weighted_recovery(1.6)

