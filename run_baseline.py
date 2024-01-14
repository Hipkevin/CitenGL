import torch
import torch.nn.functional as F

import random
import numpy as np
import os
import argparse

from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomLinkSplit
from torchmetrics.functional import retrieval_recall, retrieval_reciprocal_rank

from util.tokenizers import tokenizer_opts, BertTokenizer
from util.textEncoderTool import get_basic_data, get_graph_data
from util.graph_model import GNNGL, TEGL, DFU, TEGL_LM

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


@torch.no_grad()
def evaluate(net, embedding, querys, relation_dict, dataset, name2id, k_list=10):
    net.eval()

    r = []
    for src, seg in zip(querys[0], querys[1]):
        r.append(model.rank(embedding, src.unsqueeze(0), seg))

    preds = torch.sigmoid(torch.stack(r, dim=0).squeeze()).to(device)
    # preds = torch.sigmoid(query @ embedding.t()).to(device)

    targets = []
    for item in dataset:
        cited_list = item[1]

        t = torch.zeros(len(relation_dict)).bool().to(device)
        for cited, _ in cited_list:
            if cited in relation_dict:
                t[name2id[cited]] = True

        targets.append(t)

    if type(k_list) == int:
        k = 10
        recall = 0
        MRR = 0
        for p, t in zip(preds, targets):
            recall += retrieval_recall(p, t, k=k)
            MRR += retrieval_reciprocal_rank(p, t)

        MRR /= len(preds)
        recall /= len(preds)

        return recall.item(), MRR.item()

    else:

        for k in k_list:
            recall = 0
            for p, t in zip(preds, targets):
                recall += retrieval_recall(p, t, k)

            recall /= len(preds)
            print(f'Recall@{k}:', recall.item())

        MRR = 0
        for p, t in zip(preds, targets):
            MRR += retrieval_reciprocal_rank(p, t)

        MRR /= len(preds)
        print('MRR:', MRR.item())


@torch.no_grad()
def evaluate1(embedding, query, relation_dict, dataset, name2id, k_list=10):
    preds = torch.sigmoid(query @ embedding.t())

    targets = []
    for item in dataset:
        cited_list = item[1]

        t = torch.zeros(len(relation_dict)).bool().to(device)
        for cited, _ in cited_list:
            if cited in relation_dict:
                t[name2id[cited]] = True

        targets.append(t)

    for k in k_list:
        recall = 0
        for p, t in zip(preds, targets):
            recall += retrieval_recall(p, t, k)

        recall /= len(preds)
        print(f'Recall@{k}:', recall.item())

    MRR = 0
    for p, t in zip(preds, targets):
        MRR += retrieval_reciprocal_rank(p, t)

    MRR /= len(preds)
    print('MRR:', MRR.item())


if __name__ == '__main__':
    del_ref_type = None
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    tokenizer_opts(parser)
    args = parser.parse_args([])
    args.seq_length = 200
    # args.vocab_path = 'sci_vocab.txt'
    args.vocab_path = 'LM/bert-tiny/vocab.txt'
    args.tokenizer = BertTokenizer(args)

    contexts, papers, wv_dict = get_basic_data('peerread')
    data, homo_data, \
    test_set, test_emb, val_set, val_emb, \
    relation_dict, name2id = get_graph_data(contexts, papers, wv_dict, args,
                                            text_type='title_abstract',
                                            val_rate=0.05, test_rate=0.1, del_ref_type=None)

    edge_types = [('paper', 'background', 'paper'),
                  ('paper', 'method', 'paper'),
                  ('paper', 'result', 'paper')]
    if del_ref_type:
        ref_dict = {'background': 0, 'method': 1, 'result': 2}
        for e_t in edge_types:
            if e_t[1] not in del_ref_type:
                edge_type = [e_t, ('paper', 'combine', 'paper')]

    transform = RandomLinkSplit(num_val=0, num_test=0, neg_sampling_ratio=2.0,
                                edge_types=edge_types)
    homo_data, _, _ = transform(homo_data)

    data = data.to(device)
    homo_data = homo_data.to(device)
    test_src, test_seg = test_emb[0].to(device), test_emb[1].to(device)
    val_src, val_seg = val_emb[0].to(device), val_emb[1].to(device)

    print('Training...')

    args.vocab_size = len(args.tokenizer.vocab)
    args.emb_size = 128
    args.gnn_hidden_size = 300
    args.text_hidden_size = 300
    args.class_num = len(edge_types)

    args.GNN_type = 'SAGE'

    epochs = 50
    batch_size = 256
    model = TEGL_LM(args, 'LM/bert-tiny', padding_len=200).to(device)
    # model = TEGL(args).to(device)
    model.apply(weight_init)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)

    train_loader = NeighborLoader(data,
                                  num_neighbors=[30] * 2,
                                  batch_size=batch_size,
                                  input_nodes=('paper'))

    for epoch in range(epochs):
        model.train()

        for batch_id, batch in enumerate(train_loader):
            homo_batch = batch.to_homogeneous()

            if len(homo_batch.edge_type) == 0:
                continue

            homo_batch, _, _ = transform(homo_batch)

            emb = model(homo_batch.x, homo_batch.seg, homo_batch.edge_index)

            pos_edge_index = homo_batch.edge_label_index.t()[homo_batch.edge_label == 1]
            neg_edge_index = homo_batch.edge_label_index.t()[homo_batch.edge_label == 0]
            edge_index = torch.cat([pos_edge_index.t(), neg_edge_index.t()], dim=-1)

            ranking_logits = model.rank(emb[edge_index[0]], homo_batch.x[edge_index[1]], homo_batch.seg[edge_index[1]])

            #     ranking_logits = (emb[edge_index[0]] * homo_data.x[edge_index[1]]).sum(dim=-1)
            #     motivation_logits = model.motivation(emb[edge_index[0]], homo_data.x[edge_index[1]])

            ranking_labels = homo_batch.edge_label

            ranking_loss = F.binary_cross_entropy_with_logits(ranking_logits.squeeze(), ranking_labels)

            optimizer.zero_grad()
            loss = ranking_loss
            loss.backward()
            optimizer.step()

            if batch_id % 10 == 0:
                all_emb = model(homo_data.x, homo_data.seg, homo_data.edge_index)
                # recall, mrr = evaluate(model, all_emb, val_emb, relation_dict, val_set, name2id, k_list=10)
                # print(f"Epoch: {epoch+1:03d}, batch: {batch_id:03d}, Loss: {loss:.4f}, rank: {ranking_loss:.4f}, motivate: {motivation_loss:.4f} | Recall@10: {recall:.4f}, MRR: {mrr:.4f}")

                print(f"Epoch: {epoch + 1:03d}, batch: {batch_id:03d}, Loss: {loss:.4f}, rank: {ranking_loss:.4f}")

        print('Testing...')
        all_emb = model(homo_data.x, homo_data.seg, homo_data.edge_index)
        evaluate(model, all_emb, (test_src, test_seg), relation_dict, test_set, name2id,
                 k_list=[10, 20, 100, 200, 500, 1000, 2000])