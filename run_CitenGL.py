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

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, RGCNConv, HypergraphConv

import torch.nn as nn
from torch_geometric.nn import HeteroLinear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
import math

GNN_TYPE_DICT = {'GCN': GCNConv,
                 'GAT': GATConv,
                 'SAGE': SAGEConv}

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
        r.append(model.rank(embedding, src, seg))

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
                recall += retrieval_recall(p, t, k=k)

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
            recall += retrieval_recall(p, t, k=k)

        recall /= len(preds)
        print(f'Recall@{k}:', recall.item())

    MRR = 0
    for p, t in zip(preds, targets):
        MRR += retrieval_reciprocal_rank(p, t)

    MRR /= len(preds)
    print('MRR:', MRR.item())


def edge2hyper(item):
    hyper_node = []
    hyper_edge = []
    node_set = set()
    hyper_count = 0

    edges = item.edge_index.T.cpu().tolist()
    for node in edges:
        if node[1] not in node_set:
            node_set.add(node[1])

            hyper_node.append(node[1])
            hyper_node.append(node[0])
            hyper_edge.append(hyper_count)
            hyper_edge.append(hyper_count)

            hyper_count += 1

        else:
            hyper_node.append(node[0])
            hyper_edge.append(hyper_count - 1)

    return torch.LongTensor([hyper_node, hyper_edge]).cuda()


class LiteratureEmbedding(torch.nn.Module):
    def __init__(self, args):
        super(LiteratureEmbedding, self).__init__()

        self.seq_length = args.seq_length
        self.emb_size = args.emb_size

        self.src_embedding = torch.nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.emb_size)
        self.seg_embedding = torch.nn.Embedding(num_embeddings=3, embedding_dim=args.emb_size)

        self.w = torch.nn.Parameter(torch.ones(args.seq_length), requires_grad=True)

    def forward(self, src, seg):
        src = self.src_embedding(src)
        seg = self.seg_embedding(seg)

        f = (F.normalize(self.w * src.view(-1, self.emb_size, self.seq_length)) +
             F.normalize(seg.view(-1, self.emb_size, self.seq_length))).sum(-1)

        return f


class HHGConv(MessagePassing):
    def __init__(self, dim_size, num_edge_heads, num_node_heads):
        super(HHGConv, self).__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.dim_size = dim_size
        self.num_edge_heads = num_edge_heads
        self.num_node_heads = num_node_heads

        self.Q1 = nn.Linear(self.dim_size, self.dim_size, bias=False)
        self.K1 = nn.Linear(self.dim_size, self.dim_size, bias=False)
        self.V1 = nn.Linear(self.dim_size, self.dim_size, bias=False)

        self.edge_linear = nn.Linear(self.dim_size, self.dim_size)

        self.head_tail_linear = nn.Linear(self.dim_size, self.dim_size)

        self.to_head_tail_linear = nn.Linear(self.dim_size, self.dim_size)

        self.Q2 = nn.Linear(self.dim_size, self.dim_size, bias=False)
        self.K2 = nn.Linear(self.dim_size, self.dim_size, bias=False)
        self.V2 = nn.Linear(self.dim_size, self.dim_size, bias=False)

        self.u1 = nn.Linear(self.dim_size, self.dim_size)
        self.u2 = nn.Linear(self.dim_size, self.dim_size)

        self.norm = GraphNorm(self.dim_size)

    def forward(self, x, edge_attr, edge_in_out_indexs, batch):
        # x [num_nodes, dim_size] edge_attr [num_edges, dim_size] edge_in_out_indexs [2, num_nodeedges] edge_in_out_head_tail [num_nodeedges]
        hyperedges = self.edge_updater(edge_in_out_indexs.flip([0]), x=x, edge_attr=edge_attr)
        # hyperedges [num_edges, dim_size]
        edge_attr_out = self.edge_linear(edge_attr)
        hyperedges = hyperedges + edge_attr_out
        out = self.propagate(edge_in_out_indexs, x=x, hyperedges=hyperedges, batch=batch)
        return out

    def edge_update(self, edge_index=None, x_j=None, edge_attr_i=None):
        m = self.head_tail_linear(x_j)
        # m, edge_attr_i [num_nodeedges, dim_size]
        query = self.Q1(edge_attr_i)
        key = self.K1(m)
        value = self.V1(m)

        query = query.reshape(-1, self.num_edge_heads, self.dim_size // self.num_edge_heads)
        key = key.reshape(-1, self.num_edge_heads, self.dim_size // self.num_edge_heads)
        value = value.reshape(-1, self.num_edge_heads, self.dim_size // self.num_edge_heads)
        # query, key, value [num_nodeedges, num_edge_heads, head_size]
        attn = (query * key).sum(dim=-1)
        attn = attn / math.sqrt(self.dim_size // self.num_edge_heads)
        # attn [num_nodeedges, num_edge_heads]
        attn_score = softmax(attn, edge_index[1])
        attn_score = attn_score.unsqueeze(-1)
        # attn_score [num_nodeedges, num_edge_heads, 1]
        out = value * attn_score
        # out [num_nodeedges, num_edge_heads, head_size]
        # out = scatter_add(out, edge_index[1], 0)
        # out [num_edges, num_edge_heads, head_size]
        out = out.reshape(-1, self.dim_size)

        return out

    def message(self, edge_index=None, x_i=None, hyperedges_j=None):
        m = self.to_head_tail_linear(hyperedges_j)
        # m, x_i [num_nodeedges, dim_size]
        query = self.Q2(x_i)
        key = self.K2(m)
        value = self.V2(m)

        query = query.reshape(-1, self.num_node_heads, self.dim_size // self.num_node_heads)
        key = key.reshape(-1, self.num_node_heads, self.dim_size // self.num_node_heads)
        value = value.reshape(-1, self.num_node_heads, self.dim_size // self.num_node_heads)
        # query, key, value [num_nodeedges, num_node_heads, head_size]
        attn = (query * key).sum(dim=-1)
        # attn [num_nodeedges, num_node_heads]
        attn = attn / math.sqrt(self.dim_size // self.num_node_heads)
        attn_score = softmax(attn, edge_index[1])
        attn_score = attn_score.unsqueeze(-1)
        # attn_score [num_nodeedges, num_node_heads, 1]
        out = value * attn_score
        # out [num_nodeedges, num_node_heads, head_size]

        return out

    def update(self, inputs, x=None, batch=None):
        inputs = inputs.reshape(-1, self.dim_size)
        # x, inputs [num_nodes, dim_size]
        inputs = self.u2(inputs)
        x = self.u1(x)
        out = inputs + x
        out = self.norm(out, batch)
        out = F.elu(out)
        return out


class HCitenGL(torch.nn.Module):
    def __init__(self, args):
        super(HCitenGL, self).__init__()

        self.literature_encoder = LiteratureEmbedding(args)
        self.edge_embedding = nn.Embedding(args.class_num + 1, args.emb_size, padding_idx=args.class_num)

        self.conv1 = HHGConv(args.emb_size, 8, 8)
        self.conv2 = HHGConv(args.emb_size, 8, 8)

        self.r_conv1 = RGCNConv(args.emb_size, args.gnn_hidden_size, num_relations=args.class_num)
        self.r_conv2 = RGCNConv(args.gnn_hidden_size, args.emb_size, num_relations=args.class_num)

        self.motivation_mlp = torch.nn.Linear(args.emb_size, args.class_num + 1)
        self.ranking_mlp = torch.nn.Linear(args.emb_size, 1)

        self.query_mlp = torch.nn.Sequential(
            torch.nn.Linear(args.emb_size, args.text_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.text_hidden_size, args.emb_size))

        self.linear = torch.nn.Linear(args.emb_size, args.emb_size)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, src, seg, edge_index, edge_type, batch):
        f = self.literature_encoder(src, seg)
        edge_attr = self.edge_embedding(edge_type)

        x = self.query_mlp(f)

        r_x = self.r_conv1(x, edge_index, edge_type)
        r_x = r_x.relu()
        r_x = self.r_conv2(r_x, edge_index, edge_type)
        r_x = r_x.relu()

        x = self.conv1(x, edge_attr, edge_index, batch)
        x = x.relu()
        x = self.conv2(x, edge_attr, edge_index, batch)
        x = x.relu()

        return self.dropout(self.linear(x + r_x))

    def rank(self, emb, src, seg):
        query = self.literature_encoder(src, seg)

        query = self.query_mlp(query)
        logits = self.dropout(self.ranking_mlp(emb * query + emb + query))
        return logits

    def motivation(self, emb, src, seg):
        query = self.literature_encoder(src, seg)

        query = self.query_mlp(query)
        logits = self.dropout(self.motivation_mlp(emb * query + emb + query))
        return logits


if __name__ == '__main__':
    del_ref_type = None
    # del_ref_type = ['result', 'method']  # B
    # del_ref_type = ['background', 'result']  # M
    # del_ref_type = ['background', 'method']  # R

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    tokenizer_opts(parser)
    args = parser.parse_args([])
    args.seq_length = 200
    args.vocab_path = 'sci_vocab.txt'
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

    epochs = 50
    batch_size = 256
    model = HCitenGL(args).to(device)
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
            # hyper_index = edge2hyper(homo_batch)

            if len(homo_batch.edge_type) == 0:
                continue

            homo_batch, _, _ = transform(homo_batch)

            emb = model(homo_batch.x, homo_batch.seg, homo_batch.edge_index, homo_batch.edge_type, homo_batch.batch)

            pos_edge_index = homo_batch.edge_label_index.t()[homo_batch.edge_label == 1]
            neg_edge_index = homo_batch.edge_label_index.t()[homo_batch.edge_label == 0]
            edge_index = torch.cat([pos_edge_index.t(), neg_edge_index.t()], dim=-1)

            ranking_logits = model.rank(emb[edge_index[0]], homo_batch.x[edge_index[1]], homo_batch.seg[edge_index[1]])
            motivation_logits = model.motivation(emb[edge_index[0]], homo_batch.x[edge_index[1]],
                                                 homo_batch.seg[edge_index[1]])

            #     ranking_logits = (emb[edge_index[0]] * homo_data.x[edge_index[1]]).sum(dim=-1)
            #     motivation_logits = model.motivation(emb[edge_index[0]], homo_data.x[edge_index[1]])

            ranking_labels = homo_batch.edge_label
            motivation_labels = torch.cat([homo_batch.edge_type, 3 + ranking_labels[ranking_labels == 0].long()],
                                          dim=-1)

            ranking_loss = F.binary_cross_entropy_with_logits(ranking_logits.squeeze(), ranking_labels)
            motivation_loss = F.cross_entropy(motivation_logits, motivation_labels)

            optimizer.zero_grad()
            loss = 0.8 * ranking_loss + 0.2 * motivation_loss
            loss.backward()
            optimizer.step()

            if batch_id % 10 == 0:
                # all_emb = model(homo_data.x, homo_data.seg, homo_data.edge_index, homo_data.edge_type)
                # recall, mrr = evaluate(model, all_emb, val_emb, relation_dict, val_set, name2id, k_list=10)
                # print(f"Epoch: {epoch+1:03d}, batch: {batch_id:03d}, Loss: {loss:.4f}, rank: {ranking_loss:.4f}, motivate: {motivation_loss:.4f} | Recall@10: {recall:.4f}, MRR: {mrr:.4f}")

                print(
                    f"Epoch: {epoch + 1:03d}, batch: {batch_id:03d}, Loss: {loss:.4f}, rank: {ranking_loss:.4f}, motivate: {motivation_loss:.4f}")

    print('Testing...')
    # all_hyper = edge2hyper(homo_data)
    all_emb = model(homo_data.x, homo_data.seg, homo_data.edge_index, homo_data.edge_type, None)
    evaluate(model, all_emb, (test_src, test_seg), relation_dict, test_set, name2id,
             k_list=[10, 20, 100, 200, 500, 1000, 2000])