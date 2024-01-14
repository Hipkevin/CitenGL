# !pip install whoosh
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import MultifieldParser
from whoosh import scoring, qparser

import torch
import torch.nn.functional as F

import random
import numpy as np
import os

from torch_geometric.transforms import RandomLinkSplit
from torchmetrics.functional import retrieval_recall, retrieval_reciprocal_rank

from util.dataTool import get_basic_data, get_graph_data
from util.graph_model import ReMoGNN

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


def evaluate(preds, relation_dict, dataset, name2id, k_list=10):
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


if __name__ == '__main__':
    contexts, papers, wv_dict = get_basic_data('refseer')
    data, homo_data, \
    test_set, test_emb, val_set, val_emb, \
    relation_dict, name2id = get_graph_data(contexts, papers, wv_dict,
                                            text_type='title_abstract',
                                            val_rate=0.05, test_rate=0.1, del_ref_type=None)

    schema = Schema(title=TEXT(stored=True), path=ID(stored=True), abstract=TEXT)
    ix = create_in("indexdir_refseer", schema)
    writer = ix.writer()

    for idx, item in relation_dict.items():
        title = papers[idx]['title']
        abstract = papers[idx]['abstract']

        writer.add_document(title=title, abstract=abstract, path=idx)
    writer.commit()

    searcher = ix.searcher(weighting=scoring.BM25F)
    query_parser = MultifieldParser(['title', 'abstract'],
                                    schema, group=qparser.OrGroup)

    from tqdm import tqdm

    preds = []
    for idx, item in tqdm(test_set):
        title = papers[idx]['title']
        abstract = papers[idx]['abstract']

        try:
            title_key_terms = ' '.join([
                t for t, _ in searcher.key_terms_from_text('title', title, numterms=1)])
        except:
            title += ' model'
            title_key_terms = ' '.join([
                t for t, _ in searcher.key_terms_from_text('title', title, numterms=1)])

        try:
            abstract_key_terms = ' '.join([
                t for t, _ in searcher.key_terms_from_text('abstract', abstract)])
        except:
            abstract += ' model'
            abstract_key_terms = ' '.join([
                t for t, _ in searcher.key_terms_from_text('abstract', abstract)])

        query = query_parser.parse(title_key_terms)
        results = searcher.search(query, limit=len(relation_dict), optimize=False, scored=True)

        pred = torch.zeros(len(relation_dict))
        for res in results:
            pred[name2id[res['path']]] = res.score

        preds.append(pred)

    preds = torch.stack(preds, dim=-1)

    evaluate(preds.t().cuda(), relation_dict, test_set, name2id, k_list=[10, 20, 100, 200, 500, 1000, 2000])