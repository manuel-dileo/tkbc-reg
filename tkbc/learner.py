# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
from typing import Dict
import logging
import torch
from torch import optim

from datasets import TemporalDataset
from optimizers import TKBCOptimizer, IKBCOptimizer
from models import ComplEx, TComplEx, TNTComplEx, RTComplEx
from regularizers import N3, SmoothRegularizer, ExpDecayRegularizer, Np, Lp, Norm

parser = argparse.ArgumentParser(
    description="Temporal ComplEx"
)
parser.add_argument(
    '--dataset', type=str,
    help="Dataset name"
)
models = [
    'ComplEx', 'TComplEx', 'TNTComplEx', 'RTComplEx'
]
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)

rnns = ['GRU','RNN','LSTM', 'LinRNN']

parser.add_argument(
    '--rnn', choices=rnns, default='GRU',
    help="Model in {}".format(models)
)

parser.add_argument(
    '--rnn_size', default=10,
    help="RNN output size"
)

parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=100, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)

parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)

parser.add_argument(
    '--emb_reg_type', default='N3', type=str,
    help='Type of emb regularizer [N3, L2]'
)

parser.add_argument(
    '--time_reg_w', default=0., type=float,
    help="Timestamp regularizer strength"
)

parser.add_argument(
    '--time_reg', default='smooth', type=str,
    help='Type of time regularizer [smooth, expdecay]'
)

parser.add_argument(
    '--time_norm', default="Lp", type=str,
    help="Timestamp regularizer norm"
)

parser.add_argument(
    '--p_norm', default=2, type=int,
    help="p value for Lp/Np norm"
)

parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)


args = parser.parse_args()

dataset = TemporalDataset(args.dataset)

sizes = dataset.get_shape()
model = {
    'ComplEx': ComplEx(sizes, args.rank),
    'TComplEx': TComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
    'RTComplEx': RTComplEx(sizes, args.rank, no_time_emb=args.no_time_emb, rnnmodel=args.rnn, rnn_size=int(args.rnn_size)),
    'TNTComplEx': TNTComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
}[args.model]
model = model.cuda()

opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)

emb_reg = {
    'N3': N3(args.emb_reg),
    'L2': L2(args.emb_reg)
}[args.emb_reg_type]

norm = {
    'Lp': Lp(args.p_norm),
    'Np': Np(args.p_norm)
}[args.time_norm]

time_reg = {
    'smooth': SmoothRegularizer(args.time_reg_w, norm),
    'expdecay': ExpDecayRegularizer(args.time_reg_w, norm)
}[args.time_reg]

for epoch in range(args.max_epochs):
    examples = torch.from_numpy(
        dataset.get_train().astype('int64')
    )

    model.train()
    if dataset.has_intervals():
        optimizer = IKBCOptimizer(
            model, emb_reg, time_reg, opt, dataset,
            batch_size=args.batch_size
        )
        optimizer.epoch(examples)

    else:
        optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, opt,
            batch_size=args.batch_size
        )
        optimizer.epoch(examples)


    def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
        """
        aggregate metrics for missing lhs and rhs
        :param mrrs: d
        :param hits:
        :return:
        """
        m = (mrrs['lhs'] + mrrs['rhs']) / 2.
        h = (hits['lhs'] + hits['rhs']) / 2.
        return {'MRR': m, 'hits@[1,3,10]': h}

    if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
        if dataset.has_intervals():
            valid, test, train = [
                dataset.eval(model, split, -1 if split != 'train' else 50000)
                for split in ['valid', 'test', 'train']
            ]
            print("valid: ", valid)
            print("test: ", test)
            print("train: ", train)

        else:
            valid, test, train = [
                avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                for split in ['valid', 'test', 'train']
            ]
            print("valid: ", "MRR: ", valid['MRR'], "hits@[1,3,10]", valid['hits@[1,3,10]'])
            print("test: ", "MRR: ", test['MRR'], "hits@[1,3,10]", test['hits@[1,3,10]'])
            print("train: ", "MRR: ", train['MRR'], "hits@[1,3,10]", train['hits@[1,3,10]'])

