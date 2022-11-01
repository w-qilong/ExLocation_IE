import logging
import pickle
import time


def get_logger(dataset):
    # 定义log文件名称
    pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    # 创建logger
    logger = logging.getLogger()
    # 设置日志等级
    logger.setLevel(logging.INFO)
    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')
    # 定义输出日志对象
    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 创建流文件，用于输出日志到屏幕
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    # 将两者添加到logger中
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)


def decode(outputs, entities, length):
    # outputs:模型输出字与字之间的关系矩阵
    # entities：真实的实体
    # length：句子长度
    ent_r, ent_p, ent_c = 0, 0, 0  # 真实实体数量，预测实体数量，预测正确的实体数量
    decode_entities = []
    for index, (instance, ent_set, l) in enumerate(zip(outputs, entities, length)):
        forward_dict = {}
        head_dict = {}
        ht_type_dict = {}
        for i in range(l):
            for j in range(i + 1, l):
                if instance[i, j] == 1:
                    if i not in forward_dict:
                        forward_dict[i] = [j]
                    else:
                        forward_dict[i].append(j)
        for i in range(l):
            for j in range(i, l):
                if instance[j, i] > 1:
                    ht_type_dict[(i, j)] = instance[j, i]
                    if i not in head_dict:
                        head_dict[i] = {j}
                    else:
                        head_dict[i].add(j)

        predicts = []

        def find_entity(key, entity, tails):
            entity.append(key)
            if key not in forward_dict:
                if key in tails:
                    predicts.append(entity.copy())
                entity.pop()
                return
            else:
                if key in tails:
                    predicts.append(entity.copy())
            for k in forward_dict[key]:
                find_entity(k, entity, tails)
            entity.pop()

        for head in head_dict:
            find_entity(head, [], head_dict[head])

        predicts = set(
            [convert_index_to_text(x, ht_type_dict[(x[0], x[-1])]) for x in predicts])  # 预测出的实体，形式为1-2-3-#-type
        decode_entities.append([convert_text_to_index(x) for x in predicts])  # 将predicts中的字符序列转转为index 序列与实体类型序列
        ent_r += len(ent_set)  # 真实实体数量
        ent_p += len(predicts)  # 预测实体数量
        for x in predicts:
            if x in ent_set:
                ent_c += 1
    return ent_c, ent_p, ent_r, decode_entities  # 正确实体数量，预测实体，真实实体，预测实体解码


def cal_f1(c, p, r):
    '''

    :param c: 预测正确的数量
    :param p: 预测出的数量
    :param r: 真实的实体数量
    :return:
    '''
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r


def cal_entitiy_f1(input):
    '''
    calculate prf for each entity type
    :param input:
    :return:
    '''
    # input为包含验证集多个batch中真实实体和预测的实体集合的列表【true，predict】

    # 对于每一类实体，计算prf
    dict_tmp = []
    d = {'ent_c': 0, 'ent_p': 0, 'ent_r': 0}
    for i in range(15):
        dict_tmp.append(d.copy())

    prf_dict = dict(zip(list(range(15)), dict_tmp))

    for batch in input:

        true, predict = batch[0], batch[1]
        # 预测值和真实值的格式不一致，统一为一种格式，对于一个batch [ [([0,1,2,2],1), ()], ]
        true2predict = []
        for sentence in true:
            sentence_entity_list = []
            for entity in sentence:
                sentence_entity_list.append(convert_text_to_index(entity))
            true2predict.append(sentence_entity_list)
        # 计算各句子中不同实体的识别准确率
        for index, sentence in enumerate(true2predict):
            predict_sentence = predict[index]
            for item in sentence:
                if item in predict_sentence:
                    prf_dict[item[1]]['ent_c'] += 1
        # 统计预测结果和真实结果中包含的各类型实体的数量
        for sentence in true2predict:
            for entity in sentence:
                prf_dict[entity[1]]['ent_r'] += 1
        for sentence in predict:
            for entity in sentence:
                prf_dict[entity[1]]['ent_p'] += 1

    return prf_dict


def cal_nested_discontinuous_f1(input):
    '''
    calculate prf for nest and discontinuous entity type
    :param input:
    :return:
    '''
    # input为包含验证集多个batch中真实实体和预测的实体集合的列表【true,predict】

    # 对于每一类实体，计算prf
    # dict_tmp = []
    # d = {'ent_c': 0, 'ent_p': 0, 'ent_r': 0}
    # for i in range(15):
    #     dict_tmp.append(d.copy())
    #
    # prf_dict = dict(zip(list(range(15)), dict_tmp))
    #
    # for batch in input:
    #
    #     true, predict = batch[0], batch[1]
    #     # 预测值和真实值的格式不一致，统一为一种格式，对于一个batch [[([0,1,2,2],1), ()], ]
    #     true2predict = []
    #     for sentence in true:
    #         sentence_entity_list = []
    #         for entity in sentence:
    #             sentence_entity_list.append(convert_text_to_index(entity))
    #         true2predict.append(sentence_entity_list)
    #
    #     # 计算各句子中不同实体的识别准确率
    #     for index, sentence in enumerate(true2predict):
    #         predict_sentence = predict[index]
    #         for item in sentence:
    #             if item in predict_sentence:
    #                 prf_dict[item[1]]['ent_c'] += 1
    #     # 统计预测结果和真实结果中包含的各类型实体的数量
    #     for sentence in true2predict:
    #         for entity in sentence:
    #             prf_dict[entity[1]]['ent_r'] += 1
    #     for sentence in predict:
    #         for entity in sentence:
    #             prf_dict[entity[1]]['ent_p'] += 1

    prf_dict = {'nested': {'ent_c': 0, 'ent_p': 0, 'ent_r': 0}, 'discontinuous': {'ent_c': 0, 'ent_p': 0, 'ent_r': 0}}

    for batch in input:
        true, predict = batch[0], batch[1]
        # 预测值和真实值的格式不一致，统一为一种格式，对于一个batch [[([0,1,2,3],1), ()], ]
        true2predict = []
        for sentence in true:
            sentence_entity_list = []
            for entity in sentence:
                sentence_entity_list.append(convert_text_to_index(entity))
            true2predict.append(sentence_entity_list)

        # find nested and discontinuous entity in each sentence
        for index, sentence in enumerate(true2predict):
            predict_sentence = predict[index]
            # 将所有实体分为连续实体和不连续实体
            discontinuous_entity_list = []
            continuous_entity_list = []
            for item in sentence:
                tag = True
                for i, index1 in enumerate(item[0]):
                    if i < len(item[0]) - 1:
                        if item[0][i + 1] != index1 + 1:
                            tag = False
                if tag:
                    continuous_entity_list.append(item)
                else:
                    discontinuous_entity_list.append(item)
            prf_dict['discontinuous']['ent_r'] += len(discontinuous_entity_list)

            # get the nested entity for real sentence
            be_nested_entity_list = []
            for item in continuous_entity_list:
                for item_second in continuous_entity_list:
                    if item_second[0][0] >= item[0][0] and item_second[0][-1] <= item[0][-1] and item != item_second:
                        be_nested_entity_list.append(item_second)
            prf_dict['nested']['ent_r'] += len(be_nested_entity_list)

            # get the according predict result for real sentence
            for item in discontinuous_entity_list:
                if item in predict_sentence:
                    prf_dict['discontinuous']['ent_c'] += 1
            for item in be_nested_entity_list:
                if item in predict_sentence:
                    prf_dict['nested']['ent_c'] += 1

            # get the discontinuous entity for predict sentence
            discontinuous_entity_list_predict = []
            continuous_entity_list_predict = []
            for item in predict_sentence:
                tag = True
                for i, index2 in enumerate(item[0]):
                    if i < len(item[0]) - 1:
                        if item[0][i + 1] != index2 + 1:
                            tag = False
                if not tag:
                    discontinuous_entity_list_predict.append(item)
                else:
                    continuous_entity_list_predict.append(item)
            prf_dict['discontinuous']['ent_p'] += len(discontinuous_entity_list_predict)

            # get the nested entity for predict sentence
            be_nested_entity_list_predict = []
            for item in continuous_entity_list_predict:
                for item_second in continuous_entity_list_predict:
                    if item_second[0][0] >= item[0][0] and item_second[0][-1] <= item[0][-1] and item != item_second:
                        be_nested_entity_list_predict.append(item_second)
            prf_dict['nested']['ent_p'] += len(be_nested_entity_list_predict)

    return prf_dict
