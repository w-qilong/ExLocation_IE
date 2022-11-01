import json
import jieba
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import  Counter

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'

# 定义保存所有转换交通事故记录的图
all_sentence_rdf=[]
all_sentence_code_stakenumber=[]


f = open(r'output/output_model_with_dist_region.json', 'r', encoding='utf-8')
json_data = json.load(f)
for data in json_data:
    ner = data['ner']
    sentence = data['sentence']

    relationship = [
        ['event', 'has_time', 'time'],
        ['event', 'has', 'location'],
        ['expressway', 'has_direction', 'direction'],
        ['direction', 'has_origin', 'city'],
        ['direction', 'has_end', 'city'],
        ['expressway', 'has_section', 'roadsection'],
        ['roadsection', 'cover_way', 'expressway'],
        ['roadsection', 'located_in', 'city'],
        ['expressway', 'has_subsection', 'expressway'],
        ['expressway', 'has_code', 'expresswaycode'],

        ['expressway', 'has', 'stakenumber'],
        ['expressway', 'has', 'servicearea'],
        ['expressway', 'has', 'interchange'],
        ['expressway', 'has', 'station'],
        ['expressway', 'has', 'hub'],
        ['expressway', 'has', 'tunnel'],
        ['expressway', 'has', 'ramp'],
        ['expressway', 'has', 'flyover'],

        ['roadsection', 'cover_entity', 'stakenumber'],
        ['roadsection', 'cover_entity', 'servicearea'],
        ['roadsection', 'cover_entity', 'interchange'],
        ['roadsection', 'cover_entity', 'station'],
        ['roadsection', 'cover_entity', 'hub'],
        ['roadsection', 'cover_entity', 'tunnel'],
        ['roadsection', 'cover_entity', 'ramp'],
        ['roadsection', 'cover_entity', 'flyover'],
    ]

    entity_list = []
    nested_entity_list = []  # [['raodsection',[1,2,3],'成南段']，['expressway',[1,2],'成南']]
    rdf_list = []

    # 存储当前句子中对应的expresswaycode和stakenumber
    code_with_stakenumber=[]

    # 获得一个句子中的所有实体
    for item in ner:
        item_type = item['type']
        item_index = item['index']
        item_text = [sentence[i] for i in item['index']]
        if [item_type, item_index] not in entity_list:
            entity_list.append([item_type, item_index, ''.join(item_text)])

        if item_type=='expresswaycode' or item_type=='stakenumber':
            code_with_stakenumber.append([item_type,''.join(item_text)])
    all_sentence_code_stakenumber.append(code_with_stakenumber)


    # 查找出嵌套实体
    # item:['type',[1,2,3],'成南段']

    # 要建立实体之间的关系，需要分为以下几步
    # 1. 找出实体中的嵌套实体，嵌套实体和被嵌套实体的关系直接赋值为cover,嵌套实体主要包含在路段、方向、立交、枢纽描述中
    # 2. 如何判断实体是否是嵌套实体，如果该实体index不连续，则为不连续实体，不连续实体的index可能相互包含，需要先排除它们，防止与嵌套实体的判断混淆
    # 2* 如果嵌套实体类型为路段，则嵌套实体与被嵌套实体的关系为cover，如果被嵌套实体类型为高速，则嵌套实体与被嵌套实体的关系为cover，如果被嵌套实体数量
    # 2* 大于等于2，则说明被嵌套实体为路段的两个端点。
    # 3. 其余实体，需要细化方向，确定方向描述的起点和终点
    # 4. 对于道路的子路段的描述，需要依据先后关系进行判断，需要保存实体的index进行计算

    # 将所有实体分为连续实体和不连续实体
    discontinuous_entity_list = []
    continuous_entity_list = []

    for item in entity_list:  # item:['type',[1,2,3],'成南段']
        tag = True
        for i, index in enumerate(item[1]):  # 判断实体是否是连续实体，如果不连续，直接跳过
            if i < len(item[1]) - 1:
                if item[1][i + 1] != index + 1:
                    tag = False
        if tag:
            continuous_entity_list.append(item)
        else:
            discontinuous_entity_list.append(item)
    # print('discontinuous_entity_list',discontinuous_entity_list)
    # print('continuous_entity_list',continuous_entity_list)

    # 对于连续实体部分，检测其中的嵌套实体与之对应的被嵌套实体
    # 存储被嵌套实体，下面建立被嵌套实体与嵌套实体之间的关系，建立flat实体时，不用再考虑他们
    be_nested_entity_list = []
    for item in continuous_entity_list:  # item:['type',[1,2,3],'成南段']
        for item_second in continuous_entity_list:
            if item_second[1][0] >= item[1][0] and item_second[1][-1] <= item[1][-1] and item != item_second:
                nested_entity_list.append([item, item_second])
                be_nested_entity_list.append(item_second)
    # print('nested_entity_list', nested_entity_list)

    # 判断各嵌套实体之间的关系
    for item in nested_entity_list:
        subject = item[0]  # 嵌套实体
        object = item[1]  # 被嵌套实体
        # 针对嵌套实体，不同情况下对应不同实体关系
        if subject[0] == 'roadsection' and object[0] == 'expressway':
            relation = 'cover_way'
        else:
            relation = 'cover_entity'
        rdf_list.append([subject, relation, object])  # 将嵌套实体与被嵌套实体及其对应的实体关系cover加入三元组
    # 判断句子中其它实体之间的关系类型
    # flat实体和discontinue实体之间的关系
    other_entity_list = [i for i in entity_list if i not in be_nested_entity_list]
    print(other_entity_list)
    for item in other_entity_list:
        for item_second in other_entity_list:
            if item != item_second:
                for relation in relationship:
                    if (item[0] == relation[0] and item_second[0] == relation[2]):
                        relation_with_entities = relation[1]
                        # 对于两个实体之间的关系，根据不同关系类型，对不同实体进行不同的处理
                        if relation_with_entities == 'has_subsection' and item[2] != item_second[2]:
                            if item[1][-1] < item_second[1][1]:
                                rdf_list.append([item, relation_with_entities, item_second])
                        else:
                            rdf_list.append([item, relation_with_entities, item_second])
    print()
    print(''.join(sentence))
    # print(rdf_list)

    # 方向细化
    for rdf in rdf_list:
        if rdf[1] == 'has_direction':
            direction = rdf[2][2]
            direction_cut = list(jieba.cut(direction))
            if len(direction_cut) > 2:
                start_city = direction_cut[0]
                end_city=direction_cut[2]
                start_city_list = ['city', rdf[2][1][direction.find(start_city):direction.find(
                    start_city) + len(start_city)], start_city]
                end_city_list = ['city', rdf[2][1][direction.find(end_city):direction.find(
                    end_city) + len(end_city)], end_city]
                rdf_list.append([rdf[2], 'has_origin', start_city_list])
                rdf_list.append([rdf[2], 'has_end', end_city_list])


    # 为高速补充全称，尤其是嵌套实体中的高速
    for index,rdf in enumerate(rdf_list) :
        for index1,item in enumerate(rdf) :
            if item[0]=='expressway' and not item[2].endswith('高速'):
                rdf_list[index][index1][2]+='高速'

    # print(rdf_list)
    all_sentence_rdf.append(rdf_list)
    # print(rdf_list)



    # 绘制实体关系图
    nodes=[]
    for rdf in rdf_list:
        nodes.append(rdf[0][2])
        nodes.append(rdf[2][2])
    # 去除重复实体
    nodes=list(set(nodes))
    G = nx.Graph()
    for node in nodes:
        G.add_node(node, desc=node)
    for rdf in rdf_list:
        G.add_edge(rdf[0][2],rdf[2][2],name=rdf[1])
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    node_labels = nx.get_node_attributes(G, 'desc')
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(G, 'name')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

