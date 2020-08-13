# -*- coding: utf-8 -*-
'''
@Author: your name
@Date: 2020-06-18 21:15:35
@LastEditTime: 2020-06-30 14:20:08
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /bookClassification/src/DL/models/roberta.py
'''
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
from __init__ import *


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        model_config = RobertaConfig.from_pretrained(config.bert_path, num_labels=config.num_classes)
        self.roberta = RobertaForSequenceClassification.from_pretrained(config.bert_path, config=model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 杈撳叆鐨勫彞瀛�        mask = x[1]  # 瀵筽adding閮ㄥ垎杩涜�mask锛屽拰鍙ュ瓙涓�釜size锛宲adding閮ㄥ垎鐢�琛ㄧず锛屽�锛歔1, 1, 1, 1, 0, 0]
        token_type_ids = x[2]
        _, pooled = self.roberta(context, attention_mask=mask, token_type_ids=token_type_ids)
        out = self.fc(pooled)
        return out