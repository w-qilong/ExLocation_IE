import argparse
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
import config
import data_loader
import utils
from model import Model
from torch.utils.tensorboard import SummaryWriter
import datetime
from loss_funcitons import focal_loss
from sklearn.metrics import confusion_matrix


class Trainer(object):
    def __init__(self, model, writer):

        self.model = model

        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
        # 使用focal loss作为损失函数
        # self.criterion = focal_loss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []

        for i, data_batch in enumerate(data_loader):
            data_batch = [data.cuda() for data in data_batch[:-1]]

            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

            outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)

            grid_mask2d = grid_mask2d.clone()
            pre = outputs[grid_mask2d]
            labels = grid_labels[grid_mask2d]
            loss = self.criterion(pre, labels)
            writer.add_scalar('Train %s batch loss' % epoch, loss, i)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            outputs = torch.argmax(outputs, -1)
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)

            label_result.append(grid_labels)
            pred_result.append(outputs)

            self.scheduler.step()

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(),
                                                      pred_result.cpu().numpy(),
                                                      average="macro")

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        # 记录误差值
        writer.add_scalar('Train epoch mean Loss', np.mean(loss_list), epoch)
        writer.add_scalar('Train F1 score', f1, epoch)
        logger.info("\n{}".format(table))
        return f1

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        pred_result = []
        label_result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        true_and_predict_entities = []  # 用于记录下每个batch中预测出的实体及句子中真实的实体

        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)  # 得到各label矩阵
                ent_c, ent_p, ent_r, _ = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                #####################
                # 记录batch中真实实体类型与预测出的实体类型，用于计算各类型实体的识别效果
                true_and_predict_entities.append([entity_text, _])
                #####################

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels)
                pred_result.append(outputs)

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(),
                                                      pred_result.cpu().numpy(),
                                                      average="macro")  # 预测label对应的p，r，f
        cm = confusion_matrix(label_result.cpu().numpy(), pred_result.cpu().numpy())
        logger.info('confusion matrix:')
        logger.info('\n{}'.format(cm))

        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)  # 实体对应的p，r，f

        title = "EVAL" if not is_test else "TEST"
        logger.info('{} Label F1 {}'.format(title, f1_score(label_result.cpu().numpy(),
                                                            pred_result.cpu().numpy(),
                                                            average=None)))
        writer.add_scalar('eval entity F1 score', e_f1, epoch)

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        logger.info("\n{}".format(table))

        ##########################calculate each entity type prf################################
        entity_predict = utils.cal_entitiy_f1(true_and_predict_entities)
        table_entity = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])

        for key, value in entity_predict.items():
            entity_name = config.vocab.id_to_label(key)
            correct_num = value['ent_c']
            predict_num = value['ent_p']
            real_num = value['ent_r']

            f_eneity, p_entity, r_entity = utils.cal_f1(correct_num, predict_num, real_num)
            table_entity.add_row([entity_name] + ["{:3.4f}".format(x) for x in [f_eneity, p_entity, r_entity]])
        logger.info("\n{}".format(table_entity))
        ########################################################################################

        ###########################calculate nested and discontinuous prf################################
        entity_predict1 = utils.cal_nested_discontinuous_f1(true_and_predict_entities)
        table_nested_discontinuous_entity = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        for key, value in entity_predict1.items():
            correct_num = value['ent_c']
            predict_num = value['ent_p']
            real_num = value['ent_r']
            f_eneity, p_entity, r_entity = utils.cal_f1(correct_num, predict_num, real_num)
            table_nested_discontinuous_entity.add_row(
                [key] + ["{:3.4f}".format(x) for x in [f_eneity, p_entity, r_entity]])
        logger.info("\n{}".format(table_nested_discontinuous_entity))
        ##############################################################################

        return e_f1

    def predict(self, epoch, data_loader, data):
        self.model.eval()

        pred_result = []
        label_result = []

        result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        i = 0
        with torch.no_grad():
            for data_batch in data_loader:
                sentence_batch = data[i:i + config.batch_size]
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text,
                                                                    length.cpu().numpy())

                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence["sentence"]
                    instance = {"sentence": sentence, "entity": []}
                    for ent in ent_list:
                        instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                                                   "type": config.vocab.id_to_label(ent[1])})
                    result.append(instance)

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels)
                pred_result.append(outputs)
                i += config.batch_size

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(),
                                                      pred_result.cpu().numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "TEST"
        logger.info('{} Label F1 {}'.format("TEST", f1_score(label_result.cpu().numpy(),
                                                             pred_result.cpu().numpy(),
                                                             average=None)))
        # self.writer.add_scalar('test label F1 score', f1, epoch)
        # self.writer.add_scalar('test entity F1 score', e_f1, epoch)
        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        logger.info("\n{}".format(table))

        with open(config.predict_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        return e_f1

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/traffic.json')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--predict_path', type=str)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    config = config.Config(args)  # 定义超参数

    logger = utils.get_logger(config.dataset)
    logger.info(config)  # 输出args参数到文件和屏幕
    config.logger = logger  # 将logger添加到config中

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    logger.info("Loading Data")

    # 加载转换为模型可接受的数据格式，包括句子对应的bert id，mask以及句子中字之间的相对位置dist矩阵，句子对应的NNW和THW矩阵
    # datasets包含训练集、验证集和测试集，ori_data为原始句子
    datasets, ori_data = data_loader.load_data_bert(config)

    # train_loader, dev_loader, test_loader = (
    #     DataLoader(dataset=dataset,
    #                batch_size=config.batch_size,
    #                collate_fn=data_loader.collate_fn,
    #                shuffle=i == 0,
    #                num_workers=4,
    #                drop_last=i == 0)
    #     for i, dataset in enumerate(datasets)
    # )

    # updates_total = len(datasets[0]) // config.batch_size * config.epochs  # 所有epoch一共包含多少个batch
    #
    # # 初始化模型
    # logger.info("Building Model")
    # model = Model(config)
    # model = model.cuda()
    # trainer = Trainer(model)
    #
    # best_f1 = 0
    # best_test_f1 = 0
    # for i in range(config.epochs):
    #     logger.info("Epoch: {}".format(i))
    #     trainer.train(i, train_loader)
    #     f1 = trainer.eval(i, dev_loader)
    #     test_f1 = trainer.eval(i, test_loader, is_test=True)
    #     if f1 > best_f1:
    #         best_f1 = f1
    #         best_test_f1 = test_f1
    #         trainer.save(config.save_path)
    # logger.info("Best DEV F1: {:3.4f}".format(best_f1))
    # logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
    # trainer.load(config.save_path)
    # trainer.predict("Final", test_loader, ori_data[-1])

    #####################################################################################
    # 合并数据集，然后重新按照交叉验证的方式进行分割
    cat_datasets = torch.utils.data.ConcatDataset(datasets)
    # 定义测试集，查看模型识别效果
    test_dataloader = DataLoader(dataset=datasets[-1],
                                 batch_size=config.batch_size,
                                 collate_fn=data_loader.collate_fn,
                                 shuffle=True,
                                 num_workers=4)

    # print(len(test_dataloader.dataset))
    print('all_dataset length:{}'.format(len(cat_datasets)))

    K = 10
    leng = len(cat_datasets)
    every_k_len = leng // K
    print('each k-th length：%s' % every_k_len)
    eval_f1_scores = []

    for ki in range(K):
        logger.info("第{}折交叉验证".format(ki))
        # 划分验证集
        val_dataset = torch.utils.data.Subset(cat_datasets, np.arange(every_k_len * ki, every_k_len * (ki + 1)))
        print(every_k_len * ki, every_k_len * (ki + 1))
        # 划分训练集
        train_dataset = torch.utils.data.ConcatDataset(
            [torch.utils.data.Subset(cat_datasets, np.arange(0, every_k_len * ki)),
             torch.utils.data.Subset(cat_datasets, np.arange(every_k_len * (ki + 1), leng))])
        # print(len(train_dataset))

        updates_total = len(train_dataset) // config.batch_size * config.epochs  # 所有epoch一共包含多少个batch

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=config.batch_size,
                                      collate_fn=data_loader.collate_fn,
                                      shuffle=True,
                                      num_workers=4)

        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=config.batch_size,
                                    collate_fn=data_loader.collate_fn,
                                    shuffle=True,
                                    num_workers=4)

        logger.info("Building Model")
        model = Model(config)
        model = model.cuda()

        # 定义tensorboard文件路径
        comment = f'epoch={10}_bert=bert-base'
        writer = SummaryWriter(log_dir="tensorboard/" + '十折-%s验证CrossEntropy' % ki, comment=comment)
        # 初始化trainer
        trainer = Trainer(model, writer)

        best_f1 = 0
        for i in range(config.epochs):
            logger.info("Epoch: {}".format(i))
            trainer.train(i, train_dataloader)
            eval_f1 = trainer.eval(i, val_dataloader)

            if eval_f1 >= best_f1:
                best_f1 = eval_f1
                trainer.save('model_path/' + 'CrossEntropy-%s.pt' % ki)
            logger.info("Best DEV F1: {:3.4f}".format(best_f1))
        eval_f1_scores.append(best_f1)
        writer.add_scalar('CrossEntropy k-th 折 best f1', best_f1, ki)
        logger.info("{}折交叉验证的平均f1为：{}".format(K, np.mean(eval_f1_scores)))
        # trainer.load('model_path/' + 'CrossEntropy-%s.pt' % ki)
        # trainer.predict("Final", test_dataloader, ori_data[-1])
    #############################################
