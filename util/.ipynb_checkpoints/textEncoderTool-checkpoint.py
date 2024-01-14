import json
import torch
import numpy as np
import pickle as pkl

from torch_geometric.data import HeteroData
from .constants import *


def text_embedding(content: list, wv_dict: dict):
    vec = np.zeros((1, 128))
    for w in content:
        vec += wv_dict.get(w, np.zeros((1, 128)))

    return vec

def get_relation(papers, contexts):
    relation_dict = {}

    for paper_id, _ in papers.items():

        p_ref = []
        for k, v in contexts.items():
            if paper_id in k and paper_id != v['refid']:
                p_ref.append((v['refid'], v['ref_type']))
        
        # if p_ref != []:
        relation_dict[paper_id] = list(set(p_ref))

    return relation_dict

def sentence_tokenize(args, paper, text_type):
    if "_" not in text_type:  # Sentence classification.
        text_a = paper[text_type]
        src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
        seg = [1] * len(src)
    else:  # Sentence-pair classification.
        text_a, text_b = paper['title'], paper['abstract']
        src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
        src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
        src = src_a + src_b
        seg = [1] * len(src_a) + [2] * len(src_b)

    if len(src) > args.seq_length:
        src = src[: args.seq_length]
        seg = seg[: args.seq_length]
    PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
    while len(src) < args.seq_length:
        src.append(PAD_ID)
        seg.append(0)
        
    return src, seg

def get_basic_data(dataset_name):
    print('Data Loading...')
    contexts = json.load(open(f"paperRec_data/{dataset_name}/ref_contexts.json"))
    papers = json.load(open(f"paperRec_data/{dataset_name}/papers.json"))

    # print('wv Loading...')
    # wv_dict = pkl.load(open(f'paperRec_data/{dataset_name}/{dataset_name}_CBOW_128.pkl', 'rb'))

    return contexts, papers, None


def get_graph_data(contexts, papers, wv_dict, args,
                   text_type='title_abstract',
                   val_rate=0.05, test_rate=0.15, del_ref_type=None):
    print('Graph Constructing...')
    relation_dict = get_relation(papers, contexts)

    func_map = {'title': lambda paper: paper['title'].lower().split(' '),
                'abstract': lambda paper: paper['abstract'].lower().split(' '),
                'title_abstract':
                    lambda paper: paper['title'].lower().split(' ') + paper['abstract'].lower().split(' ')}

    val_size = int(len(papers) * val_rate)
    test_size = int(len(papers) * test_rate)

    print(f'train size: {len(papers) - val_size - test_size}, '
          f'test size: {test_size}, '
          f'val size: {val_size}')

    index_list = np.array(range(len(papers)))
    val_test_index = np.random.choice(index_list, size=val_size+test_size, replace=False)

    relation_list = list(relation_dict.items())
    val_test_set = []
    for idx in val_test_index:
        val_test_set.append(relation_list[idx])
        key = relation_list[idx][0]
        relation_dict.pop(key)

    val_set = val_test_set[0: val_size]
    test_set = val_test_set[val_size:]

    def dataset2tokens(dataset):
        src, seg = [], []
        for idx_, item in enumerate(dataset):
            name = item[0]
            src_, seg_ = sentence_tokenize(args, papers[name], text_type)
            src.append(src_)
            seg.append(seg_)

        return torch.LongTensor(src), torch.LongTensor(seg)

    test_emb = dataset2tokens(test_set)
    val_emb = dataset2tokens(val_set)

    name2id = {name: idx for idx, name in enumerate(relation_dict.keys())}

    background_adj = []
    method_adj = []
    result_adj = []
    
    src, seg = [], []
    adj = [background_adj, method_adj, result_adj]
    for name, relation in relation_dict.items():
        
        src_, seg_ = sentence_tokenize(args, papers[name], text_type)
        src.append(src_)
        seg.append(seg_)

        for r in relation:

            r, r_type = r
            if r in relation_dict:
                citing = name2id[name]
                ref = name2id[r]

                adj[r_type].append([ref, citing])

    data = HeteroData()
    data['paper'].x = torch.LongTensor(src)
    data['paper'].seg = torch.LongTensor(seg)

    ref_dict = {'background': 0, 'method': 1, 'result': 2}
    if del_ref_type:

        for ref_type in ref_dict.keys():
            combine = []
            if ref_type not in del_ref_type:
                data['paper', ref_type, 'paper'].edge_index = \
                    torch.tensor(adj[ref_dict[ref_type]], dtype=torch.long).t()

            else:
                combine += adj[ref_dict[ref_type]]

            data['paper', 'combine', 'paper'].edge_index = \
                torch.tensor(combine, dtype=torch.long).t()

    else:

        for ref_type in ref_dict.keys():
            data['paper', ref_type, 'paper'].edge_index = \
                torch.tensor(adj[ref_dict[ref_type]], dtype=torch.long).t()

    homo_data = data.to_homogeneous()
    data.ref_type = homo_data.edge_type

    return data, homo_data, test_set, test_emb, val_set, val_emb, relation_dict, name2id