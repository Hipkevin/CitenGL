import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, RGCNConv, HypergraphConv

from transformers import AutoTokenizer, AutoModel
import torch

GNN_TYPE_DICT = {'GCN': GCNConv,
                 'GAT': GATConv,
                 'SAGE': SAGEConv}


class ReMoGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, class_num=3):
        super(ReMoGNN, self).__init__()

        self.conv1 = RGCNConv(in_channels, 768, num_relations=class_num)
        self.conv2 = RGCNConv(768, out_channels, num_relations=class_num)

        self.motivation_mlp = torch.nn.Linear(128, class_num+1)
        self.ranking_mlp = torch.nn.Linear(128, 1)
        
        self.query_mlp = torch.nn.Sequential(
            torch.nn.Linear(128, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, 128))
        
        self.linear = torch.nn.Linear(128, 128)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index, edge_type):
        x = self.query_mlp(x)
        x = self.conv1(x, edge_index, edge_type)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_type)
        x = x.relu()
        return self.dropout(self.linear(x))

    def rank(self, emb, query):
        query = self.query_mlp(query)
        logits = self.ranking_mlp(emb * query + emb + query)
        return logits

    def motivation(self, emb, query):
        query = self.query_mlp(query)
        logits = self.motivation_mlp(emb * query + emb + query)
        return logits

class HomoGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, GNN_type, class_num=3):
        super(HomoGNN, self).__init__()

        Conv = GNN_TYPE_DICT[GNN_type]

        self.conv1 = Conv(in_channels, 256)
        self.conv2 = Conv(256, out_channels)
        
        self.linear = torch.nn.Linear(128, 128)
        self.dropout = torch.nn.Dropout(0.25)

        self.motivation_mlp = torch.nn.Linear(128, class_num + 1)
        self.ranking_mlp = torch.nn.Linear(128, 1)
        
        self.query_linear1 = torch.nn.Linear(128, 256)
        self.query_linear2 = torch.nn.Linear(256, 128)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        return self.dropout(self.linear(x))

    def rank(self, emb, query):
        logits = self.ranking_mlp(emb * query + emb + query)
        return logits

    def motivation(self, emb, query):
        logits = self.motivation_mlp(emb * query + emb + query)
        return logits
    
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


class NormalLiteratureEmbedding(torch.nn.Module):
    def __init__(self, args):
        super(NormalLiteratureEmbedding, self).__init__()

        self.seq_length = args.seq_length
        self.emb_size = args.emb_size

        self.src_embedding = torch.nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.emb_size)

        self.w = torch.nn.Parameter(torch.ones(args.seq_length), requires_grad=True)

    def forward(self, src):
        src = self.src_embedding(src)

        f = (F.normalize(self.w * src.view(-1, self.emb_size, self.seq_length))).sum(-1)

        return f
    
class HCitenGL(torch.nn.Module):
    def __init__(self, args):
        super(HCitenGL, self).__init__()
        
        self.literature_encoder = LiteratureEmbedding(args)
        
        self.r_conv = RGCNConv(args.gnn_hidden_size, args.gnn_hidden_size, num_relations=args.class_num)
        self.h_conv = HypergraphConv(args.gnn_hidden_size, args.gnn_hidden_size)
        
        self.h_conv1 = HypergraphConv(args.emb_size, args.gnn_hidden_size)
        self.h_conv2 = HypergraphConv(args.gnn_hidden_size, args.emb_size)
        
        self.r_conv1 = RGCNConv(args.emb_size, args.gnn_hidden_size, num_relations=args.class_num)
        self.r_conv2 = RGCNConv(args.gnn_hidden_size, args.emb_size, num_relations=args.class_num)

        self.motivation_mlp = torch.nn.Linear(args.emb_size, args.class_num+1)
        self.ranking_mlp = torch.nn.Linear(args.emb_size, 1)
        
        self.query_mlp = torch.nn.Sequential(
            torch.nn.Linear(args.emb_size, args.text_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.text_hidden_size, args.emb_size))
        
        self.linear = torch.nn.Linear(args.emb_size, args.emb_size)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, src, seg, edge_index, hyper_index, edge_type):
        f = self.literature_encoder(src, seg)
        
        x = self.query_mlp(f)
        x = self.h_conv1(x, hyper_index)
        x = x.relu()
        x = self.r_conv(x, edge_index, edge_type)
        x = x.relu()
        x = self.h_conv2(x, hyper_index)
        x = x.relu()
        return self.dropout(self.linear(x))

    def rank(self, emb, src, seg):
        query = self.literature_encoder(src, seg)
        
        query = self.query_mlp(query)
        logits = self.ranking_mlp(emb * query + emb + query)
        return logits

    def motivation(self, emb, src, seg):
        query = self.literature_encoder(src, seg)
        
        query = self.query_mlp(query)
        logits = self.motivation_mlp(emb * query + emb + query)
        return logits
        
    
class RGCNCitenGL(torch.nn.Module):
    def __init__(self, args):
        super(RGCNCitenGL, self).__init__()
        
        self.literature_encoder = LiteratureEmbedding(args)
        
        self.conv1 = RGCNConv(args.emb_size, args.gnn_hidden_size, num_relations=args.class_num)
        self.conv2 = RGCNConv(args.gnn_hidden_size, args.emb_size, num_relations=args.class_num)

        self.motivation_mlp = torch.nn.Linear(args.emb_size, args.class_num+1)
        self.ranking_mlp = torch.nn.Linear(args.emb_size, 1)
        
        self.query_mlp = torch.nn.Sequential(
            torch.nn.Linear(args.emb_size, args.text_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.text_hidden_size, args.emb_size))
        
        self.linear = torch.nn.Linear(args.emb_size, args.emb_size)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, src, seg, edge_index, edge_type):
        f = self.literature_encoder(src, seg)
        
        x = self.query_mlp(f)
        x = self.conv1(x, edge_index, edge_type)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_type)
        x = x.relu()
        return self.dropout(self.linear(x))

    def rank(self, emb, src, seg):
        query = self.literature_encoder(src, seg)
        
        query = self.query_mlp(query)
        logits = self.ranking_mlp(emb * query + emb + query)
        return logits

    def motivation(self, emb, src, seg):
        query = self.literature_encoder(src, seg)
        
        query = self.query_mlp(query)
        logits = self.motivation_mlp(emb * query + emb + query)
        return logits
    
class GNNGL(torch.nn.Module):
    def __init__(self, args):
        super(GNNGL, self).__init__()
        
        self.literature_encoder = LiteratureEmbedding(args)
        
        Conv = GNN_TYPE_DICT[args.GNN_type]

        self.conv1 = Conv(args.emb_size, args.gnn_hidden_size)
        self.conv2 = Conv(args.gnn_hidden_size, args.emb_size)

        self.motivation_mlp = torch.nn.Linear(args.emb_size, args.class_num+1)
        self.ranking_mlp = torch.nn.Linear(args.emb_size, 1)
        
        self.query_mlp = torch.nn.Sequential(
            torch.nn.Linear(args.emb_size, args.text_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.text_hidden_size, args.emb_size))
        
        self.linear = torch.nn.Linear(args.emb_size, args.emb_size)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, src, seg, edge_index):
        f = self.literature_encoder(src, seg)
        
        x = self.query_mlp(f)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        return self.dropout(self.linear(x))

    def rank(self, emb, src, seg):
        query = self.literature_encoder(src, seg)
        
        query = self.query_mlp(query)
        logits = self.ranking_mlp(emb * query + emb + query)
        return logits

    def motivation(self, emb, src, seg):
        query = self.literature_encoder(src, seg)
        
        query = self.query_mlp(query)
        logits = self.motivation_mlp(emb * query + emb + query)
        return logits

class TEGL(torch.nn.Module):
    def __init__(self, args):
        super(TEGL, self).__init__()
        
        self.literature_encoder = LiteratureEmbedding(args)

        self.motivation_mlp = torch.nn.Linear(args.emb_size, args.class_num+1)
        self.ranking_mlp = torch.nn.Linear(args.emb_size, 1)
        
    def forward(self, src, seg, edge_index):
        f = self.literature_encoder(src, seg)
        return f

    def rank(self, emb, src, seg):
        query = self.literature_encoder(src, seg)
        
        logits = self.ranking_mlp(emb * query)
        return logits

    def motivation(self, emb, src, seg):
        query = self.literature_encoder(src, seg)
        
        logits = self.motivation_mlp(emb * query)
        return logits
    
class TEGL_LM(torch.nn.Module):
    def __init__(self, args, model_name_or_path, padding_len=256, embedding_size=768):
        super(TEGL_LM, self).__init__()
        
        # self.motivation_mlp = torch.nn.Linear(args.emb_size, args.class_num+1)
        self.ranking_mlp = torch.nn.Linear(args.emb_size, 1)
        self.mlp = torch.nn.Linear(128, args.emb_size)

        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                               cache_dir=model_name_or_path)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                       cache_dir=model_name_or_path)

        self.padding_len = padding_len
        self.embedding_size = embedding_size
        
        for param in self.model.parameters():
            param.requires_grad = False
        
    def forward(self, src, seg, edge_index):
        f = self.mlp(self.model(src).pooler_output)
        return f

    def rank(self, emb, src, seg):
        query = self.mlp(self.model(src).pooler_output)
        
        logits = self.ranking_mlp(emb * query)
        return logits

#     def motivation(self, emb, src, seg):
#         query = self.model(src).pooler_output
        
#         logits = self.motivation_mlp(emb * query)
#         return logits

class DFU(torch.nn.Module):
    def __init__(self, args):
        super(DFU, self).__init__()
        
        self.literature_encoder = LiteratureEmbedding(args)

        self.motivation_mlp = torch.nn.Linear(args.emb_size, args.class_num+1)
        self.ranking_mlp = torch.nn.Linear(args.emb_size, 1)
        
        self.query_mlp = torch.nn.Sequential(
            torch.nn.Linear(args.emb_size, args.text_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.text_hidden_size, args.emb_size))
        
        self.linear = torch.nn.Linear(args.emb_size, args.emb_size)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, src, seg, edge_index):
        f = self.literature_encoder(src, seg)
        
        x = self.query_mlp(f)
        return self.dropout(self.linear(x))

    def rank(self, emb, src, seg):
        query = self.literature_encoder(src, seg)
        
        query = self.query_mlp(query)
        logits = self.ranking_mlp(emb * query + emb + query)
        return logits

    def motivation(self, emb, src, seg):
        query = self.literature_encoder(src, seg)
        
        query = self.query_mlp(query)
        logits = self.motivation_mlp(emb * query + emb + query)
        return logits


class NormalDFU(torch.nn.Module):
    def __init__(self, args):
        super(NormalDFU, self).__init__()

        self.literature_encoder = LiteratureEmbedding(args)

        self.motivation_mlp = torch.nn.Linear(args.emb_size, args.class_num + 1)
        self.ranking_mlp = torch.nn.Linear(args.emb_size, 1)

        self.linear = torch.nn.Linear(args.emb_size, args.emb_size)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, src, seg, edge_index):
        f = self.literature_encoder(src, seg)

        return self.dropout(self.linear(f))

    def rank(self, emb, src, seg):
        query = self.literature_encoder(src, seg)

        logits = self.ranking_mlp(emb + query)
        return logits

    def motivation(self, emb, src, seg):
        query = self.literature_encoder(src, seg)

        logits = self.motivation_mlp(emb + query)
        return logits