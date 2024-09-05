from OD_trip_request_for_TCS2011 import *
import datetime
from numpy import nan
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm


def export_csv(list, path_to_file, ntf=True):
    with open(path_to_file, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file, delimiter=',')
        for x in list:
            csv_writer.writerow(x)
    if ntf:
        print('file:', path_to_file, 'created')


def save_pet(pet, filename='temporary file'):
    with open(filename, 'w') as f:
        f.write(json.dumps(str(pet)))


def load_pet(filename):
    with open(filename) as f:
        pet = json.loads(f.read())
    return eval(pet)


def load_csv(path_to_file="trips.csv"):
    '''
    Note: Intermediate stations discarded
    :param path_to_file: string
    :return: list
    '''
    with open(path_to_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        table = []
        for row in csv_reader:
            attr = eval(row[0])
            attr.extend([row[1], row[-1]])  # Note: Intermediate stations discarded
            table.append(attr)
    return table


def trip_filter(trip, time=False, sex=False, age=False, industry=False, income=False):
    """
    to divide trips list into different groups based on the specified attributes and attribute value
    :param trip: dataframe
    :param time:
    :param sex:
    :param age:
    :param industry:
    :param income:
    :return: dict of dataframe: groups[key] = group
    """

    def time_range(x):
        for t in range(24):
            if datetime.time(t, 0) <= x <= datetime.time(t, 59):
                # return t
                if t in [7, 8, 9]:
                    return 1  # AM peak
                elif t in [17, 18, 19]:
                    return 2  # PM peak
                else:
                    return 0  # non-peak
        print('time filter error:', x)

    def age_range(x):
        """
           age  : i-index
         2<=x<8 : 0
         8<=x<13: 1
        13<=x<18: 2
        18<=x<23: 3
        ...
        for the groups that have i > 0, calculate the age range with i-index:
        5(i-1)+8 <= age < 5i+8
        """
        # if 1 < x < 100:
        #     i = (x - 8) / 5.0 + 1
        #     if i < 1:
        #         return 0
        #     else:
        #         return int(math.floor(i))
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

    attr_rubric = {'time': '0 to 23, representing snapshots from 00:00-00:59 to 23:00-23:59', 'sex': '1:Male, 2:Female',
                   'age': '5(i-1)+8 <= age < 5i+8 (i>0), details in def age_range',
                   'industry': '1 to 20, details in Page 142 @ OM(TCS2011)_Final.pdf',
                   'income': '1 to 19 (97 for no income), details in Page 144 @ OM(TCS2011)_Final.pdf'}
    attr_list = ['time', 'sex', 'age', 'industry', 'income']
    attr_dict = {'time': time, 'sex': sex, 'age': age, 'industry': industry, 'income': income}
    attr_size = {'time': 3, 'sex': 2, 'age': 7, 'industry': 20, 'income': 20}
    attr_map = {'time': trip['time'].map(time_range), 'sex': trip['sex'], 'age': trip['age'].map(age_range),
                'industry': trip['industry'], 'income': trip['income']}
    number_of_groups = 1
    grpby = []
    num_of_attr_used = 0
    for attr in attr_list:
        if attr_dict[attr]:
            print(f'attribute: {attr:<8}  size: {attr_size[attr]:<8}  rubric: {attr_rubric[attr]}')
            grpby.append(attr_map[attr])
            number_of_groups = number_of_groups * attr_size[attr]
            num_of_attr_used += 1
    if num_of_attr_used > 0:
        grped_trip = trip.groupby(grpby)
        grps = {}
        for key, values in grped_trip:
            grp = grped_trip.get_group(key)
            if type(key) is tuple:
                key = tuple([int(key_item) for key_item in key])
            else:
                key = int(key)
            grps[key] = grp
            # print(gp)
        print('number_of_groups=', number_of_groups)
    else:
        grps = {}
        time_filter = [i for i in range(24) if i not in [1, 2, 3, 4, 5]]
        print(time_filter)
        trip['time'] = trip['time'].map(time_range)
        grps[0] = trip.loc[trip['time'].isin(time_filter)]
    return grps


def load_(path_to_file='OD matrix station index.csv'):
    si = pd.read_csv(path_to_file)
    return si


def to_od_matrix(grps, si, output_folder='matrices', export=False, benchmark=False):
    """
    to convert groups of trips to matrices
    :param grps: dataframe
    :param si: dataframe
    :param output_folder
    :param export: bool: whether export the matrices or not
    :param benchmark
    :return:
    """
    file_list = []
    if export:
        print("csv_export activated...")
    for key, grp in grps.items():
        matrix = [[0 for mx in range(84)] for my in range(84)]
        # print(gp)
        for index, row in grp.iterrows():
            try:
                o = si.loc[si['station'] == row['origin']].index[0]
                d = si.loc[si['station'] == row['destination']].index[0]
            except:
                print('od stations indexing error')
            # print(f"o={o,row['origin']},d={d,row['destination']}")
            else:
                matrix[o][d] += 1
        print(f'total flow for {key} = {np.sum(matrix)}')
        if export:
            if benchmark:
                path_file = f'benchmark_matrices/{str(key)}.csv'
            else:
                path_file = f'{output_folder}/{str(key)}.csv'
            file_list.append(str(str(key) + '.csv'))
            export_csv(list=matrix, path_to_file=path_file, ntf=False)
    if export:
        if benchmark:
            save_pet(file_list, 'benchmark_matrices/file_list')
        else:
            save_pet(file_list, f'{output_folder}/file_list')


def time_range(x):
    for t in range(24):
        if datetime.time(t, 0) <= x <= datetime.time(t, 59):
            # return t
            if t in [7, 8, 9]:
                return 1  # AM peak
            elif t in [17, 18, 19]:
                return 2  # PM peak
            else:
                return 0  # non-peak
    print('time filter error:', x)


def age_range(x):
    # if 1 < x < 100:
    #     i = (x - 8) / 5.0 + 1
    #     if i < 1:
    #         return 0
    #     else:
    #         return int(math.floor(i))
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

trips = pd.read_csv('trips_df.csv')
trips['age_group'] = trips['age'].map(age_range)

df = pd.read_csv('HK metro Space L 2011.csv', index_col=0)

station_index = pd.read_csv('OD matrix station index.csv')

merged_df = trips.merge(station_index, left_on='origin', right_on='station', how='left')

merged_df = merged_df.rename(columns={'index': 'origin_index'})

merged_df_des = merged_df.merge(station_index, left_on='destination', right_on='station', how='left')

merged_df_des = merged_df_des.rename(columns={'index': 'destination_index'})

trip_index = merged_df_des[['time','sex','age','industry','income', 'origin','destination','origin_index','destination_index','age_group']]

attr_group_dic = {'time': [[] for i in range(3)], 'sex': [[] for i in range(2)],
                'age': [[] for i in range(7)], 'industry': [[] for i in range(20)],
                'income': [[] for i in range(20)], 'income_group':[[] for i in range(3)]
                 }

trip_index['income_group'] = trip_index['income'].apply(lambda x: map_income_com(x))
desired_position = 5  # 指定新列的位置
trip_index.insert(desired_position, 'income_group', trip_index.pop('income_group'))

attr_group_dic['age']= [[] for i in range(7)]

for index, trip in trip_index.iterrows():
    #     print(f'working with{index} trip information')

    trip_agegroup = trip[-1]
    trip_sex = trip[1]
    trip_income_group = trip[5]
    trip_income = trip[4]

    attr_group_dic['age'][int(trip_agegroup)].append(list(trip))
    attr_group_dic['sex'][int(trip_sex - 1)].append(list(trip))
    attr_group_dic['income_group'][int(trip_income_group)].append(list(trip))
    #     attr_group_dic['time'][int(trip_timegroup-1)].append(list(trip))
    if trip_income == 97:
        attr_group_dic['income'][-1].append(list(trip))
    else:
        attr_group_dic['income'][int(trip_income - 1)].append(list(trip))

suffix = ' Station'
df.columns = df.columns + suffix
df.index = df.index + suffix
df = df.rename(columns={'Admiralty Station': 'Admiralty station'} )
df = df.rename(index={'Admiralty Station': 'Admiralty station'} )


df.index = range(len(df))
df.columns = range(len(df))


def o_d_matrix(data):
    record_matrix = pd.DataFrame([[0 for _ in range(len(station_index))] for _ in range(len(station_index))])

    record_matrix.columns = range(len(df.columns))

    record_matrix.index = range(len(df.columns))

    for record in data:
        ori = record[-3]
        des = record[-2]

        record_matrix[ori][des] += 1

    #     record_matrix.to_csv(filepath, header= False, index= False)

    return record_matrix

station_index_dic = dict(zip(station_index['station'], station_index['index']))

G = nx.DiGraph()

# 遍历CSV文件中的行和列，并将有连接的节点添加到图中
for node1 in df.index:
    for node2 in df.columns:
        if df.loc[node1, node2] == 1:
            G.add_edge(node1, node2)


def weight_betweenness(data):
    node_list, queue = list(G.nodes), list(G.nodes)
    flow_matrix = o_d_matrix(data)
    node_flow_centrality = {}
    total_flow = flow_matrix.values.sum()
    sp = {}

    while queue:
        v = queue.pop(0)
        od_flow_v = 0
        for e1 in node_list:
            for e2 in node_list:

                od_flow = flow_matrix.loc[e1, e2]
                if e1 != e2 and v not in [e1, e2] and od_flow > 0:
                    if (e1, e2) not in sp.keys():
                        try:
                            paths = list(nx.all_shortest_paths(G, e1, e2))
                            sp[e1, e2] = paths
                        except:
                            paths = None
                            sp[e1, e2] = paths
                    else:
                        paths = sp[e1, e2]
                    if paths:
                        nosp = len(paths)
                        for path in paths:
                            if v in path:
                                od_flow_v += od_flow / nosp
        if total_flow == 0:
            node_flow_centrality[v] = 0
        else:
            node_flow_centrality[v] = od_flow_v / total_flow
    return node_flow_centrality


for age in tqdm(attr_group_dic['age']):
    attr_BC_new['age'].append(weight_betweenness(age))

for sex in tqdm(attr_group_dic['sex']):
    attr_BC_new['sex'].append(weight_betweenness(sex))

for income in tqdm(attr_group_dic['income']):
    attr_BC_new['income'].append(weight_betweenness(income))

for industry in tqdm(attr_group_dic['industry']):
    attr_BC_new['industry'].append(weight_betweenness(industry))

for income_group in tqdm(attr_group_dic['income_group']):
    attr_BC_new['income_group'].append(weight_betweenness(income_group))


def calculate_gini(data):
    #     data = np.array(list(data.values()))
    data = data.values()

    # 先将数据排序
    data_sorted = np.sort(list(data))

    # 计算累积相对频率
    total = sum(data_sorted)

    # 计算各类别频数累积占比
    cumulative_sum = np.cumsum(data_sorted)
    cumulative_percentage = cumulative_sum / total

    # 计算GINI指数
    gini = 1 - 2 * np.sum(cumulative_percentage) / len(data_sorted) + 1 / len(data_sorted)

    return gini

attr_BC_gini_new = {'sex': [[] for i in range(2)],
                'age': [[] for i in range(7)],
                'industry': [[] for i in range(20)],
                'income': [[] for i in range(20)],
                    'income_group': [[] for i in range(3)]
                          }

def calculate_gini_group(attrs):
    attrs_gini = []

    for attr in attrs:
        gini = calculate_gini(attr)

        attrs_gini.append(gini)
    return attrs_gini

attr_BC_gini_new['sex'] = calculate_gini_group(attr_BC_new['sex'])
attr_BC_gini_new['age'] = calculate_gini_group(attr_BC_new['age'])
attr_BC_gini_new['industry'] = calculate_gini_group(attr_BC_new['industry'])
attr_BC_gini_new['income'] = calculate_gini_group(attr_BC_new['income'])
attr_BC_gini_new['income_group'] = calculate_gini_group(attr_BC_new['income_group'])


def time_to_hours(time):
    # 将时间字符串转换为datetime对象

    time_datetime = datetime.strptime(time, '%H:%M:%S')  # 根据实际格式进行修改

    # 计算时间相对于午夜的小时数
    hours = time_datetime.hour + time_datetime.minute / 60 + time_datetime.second / 3600

    return hours


def group_data_by_time_interval(data, time_interval_minutes, time_filter):
    # 将时间间隔转换为小时单位
    time_interval_hours = time_interval_minutes / 60

    # 创建一个字典，用于存储按时间分组的数据
    grouped_data = defaultdict(list)

    for item in data:

        # 获取数据行中的时间值
        time_value = item[0]

        # 将时间字符串转换为小时表示
        time_hours = time_to_hours(time_value)

        # 计算数据应该属于哪个时间间隔
        time_group = int(time_hours / time_interval_hours)

        if time_group not in time_filter:
            # 如果时间间隔不存在于字典中，创建一个空列表
            #             if time_group not in grouped_data:
            #                 grouped_data[time_group] = []

            # 将数据添加到对应时间间隔的列表中
            grouped_data[time_group].append(item)

    # 将字典键按时间顺序排序
    sorted_grouped_data = dict(sorted(grouped_data.items()))

    return sorted_grouped_data


def whole_data_by_time_gini(attr, time_interval_minutes, time_filter):
    group_item = group_data_by_time_interval(attr.values, time_interval_minutes, time_filter)

    record_time = {}
    #     print(group_item.values)
    for key, lst in group_item.items():
        #         print(lst)
        record_time[key] = calculate_gini(weight_betweenness(lst))

    return record_time

whole_gini_1h = whole_data_by_time_gini(trip_index, 60, [i for i in range(6)])

age_gini_1h = group_data_by_time_gini(attr_group_dic['age'], 60,[i for i in range(6)])
sex_gini_1h = group_data_by_time_gini(attr_group_dic['sex'], 60,[i for i in range(6)])
income_gini_1h = group_data_by_time_gini(attr_group_dic['income'], 60,[i for i in range(6)])
industry_gini_1h = group_data_by_time_gini(attr_group_dic['industry'], 60,[i for i in range(6)])
income_group_gini_1h = group_data_by_time_gini(attr_group_dic['income_group'], 60,[i for i in range(6)])