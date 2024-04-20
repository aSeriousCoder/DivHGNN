import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from copy import deepcopy
import math

from config.Config import *
from model.Utils import kl
from model.Utils import attention

from model.Utils import reparametrize, attention


class MyRGATConv(nn.Module):
    def __init__(self, mods):  
        super().__init__()
        self._mods = mods

    def forward(self, g, inputs, mask=None):
        outputs = {nty : [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}

            for i, (stype, etype, dtype) in enumerate(sorted(g.canonical_etypes)):
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                elif stype not in src_inputs or dtype not in dst_inputs:
                    continue
                else:
                    if hparams['pruning']:
                        if mask[i] == 0:
                            continue
                    # Relational Conv With Shared Conv Model
                    dstdata = self._mods[etype](rel_graph, (src_inputs[stype], dst_inputs[dtype]))
                    outputs[dtype].append(dstdata)
        rsts = {}
        attn_map = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                feat = torch.cat(alist, dim=1)
                feat, attn = attention(feat, feat, feat)
                attn_map[nty] = attn.mean(dim=0)
                rsts[nty] = feat
            else:
                if g.number_of_dst_nodes(nty) != 0:
                    rsts[nty] = torch.zeros([g.number_of_dst_nodes(nty), 1, hparams['gnn_hidden_features']]).to(hparams['device'])
        
        # Sequence -> Vector
        for nty in rsts:
            rsts[nty] = rsts[nty].mean(dim=1)

        # concat last repr
        for nty in rsts:
            rsts[nty] = torch.cat([dst_inputs[nty], rsts[nty]], dim=1)
        
        return rsts, attn_map


class StochasticMultiLayerRGCN(nn.Module):
    def __init__(self, feat_dims, base_canonical_etypes, node_emb_meta, pruning_masks=None):
        super().__init__()
        # Conv Layers
        acc_feat_dim = [feat_dims[0]]
        for i in range(1, len(feat_dims)):
            acc_feat_dim.append(feat_dims[i] + acc_feat_dim[i-1])
        
        self.base_canonical_etypes = base_canonical_etypes
        self.convs = nn.ModuleList()
        for i in range(len(hparams['train_sampler_param'])):
            conv = MyRGATConv(
                mods=nn.ModuleDict({
                    rel[1]: dglnn.GATConv(
                        in_feats=acc_feat_dim[i], 
                        out_feats=feat_dims[i+1], 
                        num_heads=hparams['gnn_attention_head'], 
                        allow_zero_in_degree=True
                    ) for rel in base_canonical_etypes
                }),
            )
            self.convs.append(conv)
        
        if hparams['pruning']:
            if pruning_masks != None:
                self.pruning_masks = pruning_masks
            else:
                self.pruning_masks = torch.ones([len(hparams['train_sampler_param']), len(base_canonical_etypes)])
            self.history_pruning = {}
            self.acc_utility = []
    
    def collecting_metapath_utility(self):
        metapath_utility = []
        for i in range(len(self.convs)):
            metapath_utility.append([])
            for rel_name in self.convs[i]._mods:
                rel_utility = 0
                for _, param in self.convs[i]._mods[rel_name].named_parameters():
                    if hparams['pruning_order'] == 1:
                        rel_utility += param.abs().sum().detach()
                    elif hparams['pruning_order'] == 2:
                        if param.grad == None:
                            continue
                        else:
                            rel_utility += param.grad.abs().sum().detach()
                metapath_utility[i].append(rel_utility)
        metapath_utility = torch.Tensor(metapath_utility)
        self.acc_utility.append(metapath_utility)
    
    def pruning_metapath(self, epoch):
        pruning_count = 0
        epoch_metapath_utility = sum(self.acc_utility)
        for i in range(epoch_metapath_utility.shape[0]):
            for ep in sorted(hparams['pruning_thres'].keys(), reverse=True):
                if epoch >= ep:
                    pruning_thre = hparams['pruning_thres'][ep]
                    break
            pruning_threshold = epoch_metapath_utility[i][epoch_metapath_utility[i] != 0].mean() * pruning_thre
            for j in range(epoch_metapath_utility.shape[1]):
                if self.pruning_masks[i][j] == 0:
                    continue
                if epoch_metapath_utility[i][j] < pruning_threshold:
                    print('Layer{}_Metapath{}_{} is pruned ! '.format(i, j, self.base_canonical_etypes[j][1]))
                    self.history_pruning['Epoch{}_Pruning{}'.format(epoch, pruning_count)] = 'Layer{}_Metapath{}_{}'.format(i, j, self.base_canonical_etypes[j][1])
                    self.pruning_masks[i][j] = 0
                    pruning_count += 1

    def forward(self, blocks, x):
        attn_maps = []
        for i in range(len(blocks)):
            if hparams['pruning']:
                x, attn_map = self.convs[i](blocks[i], x, self.pruning_masks[i])
                attn_maps.append(attn_map)
            else:
                x, attn_map = self.convs[i](blocks[i], x)
                attn_maps.append(attn_map)
        return x, attn_maps


class ScorePredictor(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        if hparams['cross_scorer']:
            self.cross = nn.Sequential(
                nn.Linear(2 * hparams['gnn_out_features'], 2 * hparams['gnn_out_features']),
                nn.Linear(2 * hparams['gnn_out_features'], 2 * hparams['gnn_out_features']),
            )
    
    def get_representation(self, nodes):
        if hparams['sample'] <= 0:
            representation = nodes.data['GNN_Emb'][:, :hparams['gnn_out_features']]
            return {'Representation': representation}
        else:
            representation = []
            for i in range(hparams['sample']):
                sub_representation = reparametrize(nodes.data['GNN_Emb'][:, :hparams['gnn_out_features']], nodes.data['GNN_Emb'][:, hparams['gnn_out_features']:])
                representation.append(sub_representation)
            return {'Representation': torch.cat(representation, dim=-1)}

    def msgfunc_score_neg(self, edges):
        src_emb = edges.src['Representation']
        dst_emb = edges.dst['Representation']
        if hparams['cross_scorer']: 
            src_emb = src_emb.to(self.hparams['device'])
            dst_emb = src_emb.to(self.hparams['device'])
            if hparams['sample'] <= 0:
                crossed_emb = self.cross(torch.cat([src_emb, dst_emb], 1))
                crossed_src_emb = crossed_emb[:, :hparams['gnn_out_features']]
                crossed_dst_emb = crossed_emb[:, hparams['gnn_out_features']:]
                score = (crossed_src_emb * crossed_dst_emb).sum(dim=1).detach().cpu()
            else:
                scores = []
                for i in range(hparams['sample']):
                    sub_src_emb = src_emb[i*hparams['gnn_out_features'] : (i+1)*hparams['gnn_out_features']]
                    sub_dst_emb = dst_emb[i*hparams['gnn_out_features'] : (i+1)*hparams['gnn_out_features']]
                    crossed_emb = self.cross(torch.cat([sub_src_emb, sub_dst_emb], 1))
                    crossed_src_emb = crossed_emb[:, :hparams['gnn_out_features']]
                    crossed_dst_emb = crossed_emb[:, hparams['gnn_out_features']:]
                    score = (crossed_src_emb * crossed_dst_emb).sum(dim=1).detach().cpu()
                    scores.append(score)
                score = sum(scores) / len(scores)
        else:
            if hparams['sample'] <= 0:
                score = (src_emb * dst_emb).sum(dim=1)
            else:
                score = (src_emb * dst_emb).sum(dim=1) / hparams['sample']
        return {'neg_score': score, 'neg_news_representation': src_emb}

    def msgfunc_score_pos(self, edges):
        src_emb = edges.src['Representation']
        dst_emb = edges.dst['Representation']
        if hparams['cross_scorer']: 
            src_emb = src_emb.to(self.hparams['device'])
            dst_emb = src_emb.to(self.hparams['device'])
            if hparams['sample'] <= 0:
                crossed_emb = self.cross(torch.cat([src_emb, dst_emb], 1))
                crossed_src_emb = crossed_emb[:, :hparams['gnn_out_features']]
                crossed_dst_emb = crossed_emb[:, hparams['gnn_out_features']:]
                score = (crossed_src_emb * crossed_dst_emb).sum(dim=1).detach().cpu()
            else:
                scores = []
                for i in range(hparams['sample']):
                    sub_src_emb = src_emb[i*hparams['gnn_out_features'] : (i+1)*hparams['gnn_out_features']]
                    sub_dst_emb = dst_emb[i*hparams['gnn_out_features'] : (i+1)*hparams['gnn_out_features']]
                    crossed_emb = self.cross(torch.cat([sub_src_emb, sub_dst_emb], 1))
                    crossed_src_emb = crossed_emb[:, :hparams['gnn_out_features']]
                    crossed_dst_emb = crossed_emb[:, hparams['gnn_out_features']:]
                    score = (crossed_src_emb * crossed_dst_emb).sum(dim=1).detach().cpu()
                    scores.append(score)
                score = sum(scores) / len(scores)
        else:
            if hparams['sample'] <= 0:
                score = (src_emb * dst_emb).sum(dim=1)
            else:
                score = (src_emb * dst_emb).sum(dim=1) / hparams['sample']
        return {'pos_score': score, 'pos_news_representation': src_emb}

    def msgfunc_score_neg_edc(self, edges):
        src_emb = edges.src['Representation']
        # dst_emb = edges.dst['Representation']
        Last_Update_Time = torch.zeros(edges.dst['Last_Update_Time'].shape[0], edges.dst['Last_Update_Time'].shape[2])
        News_Pref = torch.zeros(edges.dst['News_Pref'].shape[0], edges.dst['News_Pref'].shape[2], edges.dst['News_Pref'].shape[3])
        for i, cate in enumerate(edges.src['CateID']):
            Last_Update_Time[i] = edges.dst['Last_Update_Time'][i][cate]
            News_Pref[i] = edges.dst['News_Pref'][i][cate]
        # delta-t and decay
        delta_t = edges.data['Time'][0] - Last_Update_Time
        decaying_weight = self.get_decay_weight(delta_t)
        # reparametrize pref
        pref_shape = News_Pref.shape[:-1]
        decaying_target = edges.dst['GNN_Emb'].unsqueeze(1).repeat(1, News_Pref.shape[1], 1)
        distribution_decayed_pref = torch.mul(News_Pref, decaying_weight.unsqueeze(-1)) + torch.mul(decaying_target, 1 - decaying_weight.unsqueeze(-1))
        if hparams['sample'] <= 0:
            pref_emb = distribution_decayed_pref.reshape(-1, 2 * hparams['gnn_out_features'])[:, :hparams['gnn_out_features']].reshape(pref_shape[0], pref_shape[1], hparams['gnn_out_features'])
        else:
            pref_emb = []
            for i in range(hparams['sample']):
                sub_pref_emb = reparametrize(
                        mu=distribution_decayed_pref.reshape(-1, 2 * hparams['gnn_out_features'])[:, :hparams['gnn_out_features']],
                        logvar=distribution_decayed_pref.reshape(-1, 2 * hparams['gnn_out_features'])[:, hparams['gnn_out_features']:]
                    ).reshape(pref_shape[0], pref_shape[1], hparams['gnn_out_features'])
                pref_emb.append(sub_pref_emb)
            pref_emb = torch.cat(pref_emb, dim=-1)
        dst_pref_emb, _ = attention(src_emb.unsqueeze(1), pref_emb, pref_emb)
        dst_pref_emb = dst_pref_emb.squeeze(1)
        return {
            'neg_score': (src_emb * dst_pref_emb).sum(dim=1), 
            'neg_news_representation': src_emb,
            'src_repr': edges.src['GNN_Emb'], 
            'time': edges.data['Time'],
            'cate': edges.src['CateID']
        }

    def msgfunc_score_pos_edc(self, edges):
        src_emb = edges.src['Representation']
        # dst_emb = edges.dst['Representation']
        Last_Update_Time = torch.zeros(edges.dst['Last_Update_Time'].shape[0], edges.dst['Last_Update_Time'].shape[2])
        News_Pref = torch.zeros(edges.dst['News_Pref'].shape[0], edges.dst['News_Pref'].shape[2], edges.dst['News_Pref'].shape[3])
        for i, cate in enumerate(edges.src['CateID']):
            Last_Update_Time[i] = edges.dst['Last_Update_Time'][i][cate]
            News_Pref[i] = edges.dst['News_Pref'][i][cate]
        # delta-t and decay
        delta_t = edges.data['Time'][0] - Last_Update_Time
        decaying_weight = self.get_decay_weight(delta_t)
        # reparametrize pref
        pref_shape = News_Pref.shape[:-1]
        decaying_target = edges.dst['GNN_Emb'].unsqueeze(1).repeat(1, News_Pref.shape[1], 1)
        distribution_decayed_pref = torch.mul(News_Pref, decaying_weight.unsqueeze(-1)) + torch.mul(decaying_target, 1 - decaying_weight.unsqueeze(-1))
        if hparams['sample'] <= 0:
            pref_emb = distribution_decayed_pref.reshape(-1, 2 * hparams['gnn_out_features'])[:, :hparams['gnn_out_features']].reshape(pref_shape[0], pref_shape[1], hparams['gnn_out_features'])
        else:
            pref_emb = []
            for i in range(hparams['sample']):
                sub_pref_emb = reparametrize(
                        mu=distribution_decayed_pref.reshape(-1, 2 * hparams['gnn_out_features'])[:, :hparams['gnn_out_features']],
                        logvar=distribution_decayed_pref.reshape(-1, 2 * hparams['gnn_out_features'])[:, hparams['gnn_out_features']:]
                    ).reshape(pref_shape[0], pref_shape[1], hparams['gnn_out_features'])
                pref_emb.append(sub_pref_emb)
            pref_emb = torch.cat(pref_emb, dim=-1)
        dst_pref_emb, _ = attention(src_emb.unsqueeze(1), pref_emb, pref_emb)
        dst_pref_emb = dst_pref_emb.squeeze(1)
        return {
            'pos_score': (src_emb * dst_pref_emb).sum(dim=1), 
            'pos_news_representation': src_emb,
            'src_repr': edges.src['GNN_Emb'], 
            'time': edges.data['Time'],
            'cate': edges.src['CateID']
        }
    
    def msgfunc_score_vgnn(self, edges):  # for training
        src_emb = edges.src['Representation']
        dst_emb = edges.dst['Representation']
        if hparams['cross_scorer']: 
            src_emb = src_emb.to(self.hparams['device'])
            dst_emb = src_emb.to(self.hparams['device'])
            if hparams['sample'] <= 0:
                crossed_emb = self.cross(torch.cat([src_emb, dst_emb], 1))
                crossed_src_emb = crossed_emb[:, :hparams['gnn_out_features']]
                crossed_dst_emb = crossed_emb[:, hparams['gnn_out_features']:]
                score = (crossed_src_emb * crossed_dst_emb).sum(dim=1).detach().cpu()
            else:
                scores = []
                for i in range(hparams['sample']):
                    sub_src_emb = src_emb[i*hparams['gnn_out_features'] : (i+1)*hparams['gnn_out_features']]
                    sub_dst_emb = dst_emb[i*hparams['gnn_out_features'] : (i+1)*hparams['gnn_out_features']]
                    crossed_emb = self.cross(torch.cat([sub_src_emb, sub_dst_emb], 1))
                    crossed_src_emb = crossed_emb[:, :hparams['gnn_out_features']]
                    crossed_dst_emb = crossed_emb[:, hparams['gnn_out_features']:]
                    score = (crossed_src_emb * crossed_dst_emb).sum(dim=1).detach().cpu()
                    scores.append(score)
                score = sum(scores) / len(scores)
        else:
            if hparams['sample'] <= 0:
                score = (src_emb * dst_emb).sum(dim=1)
            else:
                score = (src_emb * dst_emb).sum(dim=1) / hparams['sample']
        return {'score': score}

    def reduce_score_pos(self, nodes):
        pos_score = nodes.mailbox['pos_score']
        pos_news_representation = nodes.mailbox['pos_news_representation']
        return {'pos_score': pos_score, 'pos_news_representation': pos_news_representation}

    def reduce_score_neg(self, nodes):
        neg_score = nodes.mailbox['neg_score']
        neg_news_representation = nodes.mailbox['neg_news_representation']
        return {'neg_score': neg_score, 'neg_news_representation': neg_news_representation}

    def reduce_score_pos_edc(self, nodes):
        pos_score = nodes.mailbox['pos_score']
        pos_news_representation = nodes.mailbox['pos_news_representation']
        src_repr = nodes.mailbox['src_repr']
        time = nodes.mailbox['time']
        cate = nodes.mailbox['cate']
        new_pref = deepcopy(nodes.data['News_Pref'])
        new_lut = deepcopy(nodes.data['Last_Update_Time'])
        for dst_node in range(src_repr.shape[0]):
            for src_node in range(src_repr.shape[1]):
                i = new_lut[dst_node][cate[dst_node][src_node]].argmin()
                new_pref[dst_node][cate[dst_node][src_node]][i] = src_repr[dst_node][src_node]
                new_lut[dst_node][cate[dst_node][src_node]][i] = time[dst_node][src_node]
        return {
            'pos_score': pos_score, 
            'pos_news_representation': pos_news_representation, 
            'pref': new_pref, 
            'lut': new_lut
        }

    def reduce_score_neg_edc(self, nodes):
        neg_score = nodes.mailbox['neg_score']
        neg_news_representation = nodes.mailbox['neg_news_representation']
        src_repr = nodes.mailbox['src_repr']
        time = nodes.mailbox['time']
        cate = nodes.mailbox['cate']
        new_pref = deepcopy(nodes.data['News_Pref'])
        new_lut = deepcopy(nodes.data['Last_Update_Time'])
        for dst_node in range(src_repr.shape[0]):
            for src_node in range(src_repr.shape[1]):
                i = new_lut[dst_node][cate[dst_node][src_node]].argmin()
                new_pref[dst_node][cate[dst_node][src_node]][i] = src_repr[dst_node][src_node]
                new_lut[dst_node][cate[dst_node][src_node]][i] = time[dst_node][src_node]
        return {
            'neg_score': neg_score, 
            'neg_news_representation': neg_news_representation, 
            'pref': new_pref, 
            'lut': new_lut
        }
    
    def get_decay_weight(self, delta_t):
        shape = delta_t.shape
        return torch.Tensor([math.exp(- hparams['beta'] * math.pow(dt, hparams['rho'])) for dt in delta_t.reshape(-1)]).reshape(shape)

    def forward(self, edge_subgraph, x, scoring_edge):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['GNN_Emb'] = x
            edge_subgraph.apply_nodes(self.get_representation, ntype='user')
            edge_subgraph.apply_nodes(self.get_representation, ntype='news')
            edge_subgraph.apply_edges(self.msgfunc_score_vgnn, etype=scoring_edge)
            return edge_subgraph.edata['score'][scoring_edge]


class MultiLayerProcessorAdaptor(nn.Module):
    def __init__(self, features):
        super().__init__()
        if len(features) < 2:
            raise Exception('At least 2 is needed to build MultiLayerProcessorAdaptor')
        self.adaptor = nn.ModuleList()
        for i in range(len(features)-1):
            self.adaptor.append(nn.Linear(features[i], features[i+1]))
    
    def forward(self, x):
        x = x.type(torch.float)
        for i in range(len(self.adaptor)):
            x = self.adaptor[i](x)
        return x


class VGNN(nn.Module):
    def __init__(self, gnn_in_features, gnn_hidden_features, gnn_out_features, base_canonical_etypes, node_emb_meta, pruning_masks=None):
        super().__init__()
        feat_dims = [gnn_in_features]
        for i in range(len(hparams['train_sampler_param'])):
            feat_dims.append(gnn_hidden_features)
        
        self.rgcn = StochasticMultiLayerRGCN(feat_dims, base_canonical_etypes, node_emb_meta, pruning_masks)

        self.denser = nn.Linear(sum(feat_dims), gnn_out_features * 2)

        self.node_emb_meta = node_emb_meta
        
        self.adaptor_align = nn.ModuleDict()
        self.attr_set = {}

        self.embeddings = torch.nn.ModuleDict()

        feat_hidden = hparams['adaptor_feat_hidden']
        feat_out = gnn_in_features

        for node_type in node_emb_meta:
            for emb_type in node_emb_meta[node_type]:
                if emb_type in self.adaptor_align:
                    continue
                else:
                    self.adaptor_align[emb_type] = MultiLayerProcessorAdaptor([node_emb_meta[node_type][emb_type], feat_hidden, feat_out])
                    self.attr_set[emb_type] = len(self.attr_set)
        
        fusioner_router = {}
        for node_type in node_emb_meta:
            fusioner_router[node_type] = torch.zeros([len(self.attr_set), len(node_emb_meta[node_type])])  # ALL_ATTR_NUM * CUR_NODE_ATTR_NUM
            for i, emb_type in enumerate(node_emb_meta[node_type]):
                fusioner_router[node_type][self.attr_set[emb_type]][i] = 1
        self.fusioner_router = nn.ParameterDict({
            node_type: nn.Parameter(fusioner_router[node_type])
            for node_type in fusioner_router
        })
        
        self.fusioner = nn.Sequential(
            nn.Linear(len(self.attr_set) * feat_out, 2 * feat_out),
            nn.Linear(2 * feat_out, feat_out),
        )
        
        self.scorer = ScorePredictor(hparams)
    
    def init_embedding(self, graph):
        for node_type in self.node_emb_meta:
            self.embeddings[node_type] = torch.nn.ModuleDict()
            for emb_type in self.node_emb_meta[node_type]:
                self.embeddings[node_type][emb_type] = torch.nn.Embedding(
                    num_embeddings=graph.nodes[node_type].data[emb_type].shape[0],
                    embedding_dim=graph.nodes[node_type].data[emb_type].shape[1]
                )
                self.embeddings[node_type][emb_type].weight.data.copy_(graph.nodes[node_type].data[emb_type])
    
    def update_graph_attr(self, graph):
        for node_type in self.node_emb_meta:
            for emb_type in self.node_emb_meta[node_type]:
                graph.nodes[node_type].data[emb_type] = self.embeddings[node_type][emb_type].weight.data
    
    def adapt(self, blocks):
        input_features = {}
        for node_type in self.node_emb_meta:
            node_attr = []
            for emb_type in self.node_emb_meta[node_type]:
                if hparams['embedding']:
                    node_attr.append(self.adaptor_align[emb_type](
                        self.embeddings[node_type][emb_type](blocks[0].srcdata['_ID'][node_type])
                    ).unsqueeze(1))
                else:
                    node_attr.append(self.adaptor_align[emb_type](
                        blocks[0].srcdata[emb_type][node_type].to(hparams['device'])
                    ).unsqueeze(1))
            node_attr = torch.cat(node_attr, dim=1)
            node_attr, _ = attention(node_attr, node_attr, node_attr)
            input_features[node_type] = node_attr
        return input_features
    
    def fusion(self, adapted_features):
        input_features = {}
        for node_type in adapted_features:
            if adapted_features[node_type].shape[0] == 0:
                continue
            else:
                input_features[node_type] = self.fusioner(
                    torch.matmul(self.fusioner_router[node_type], adapted_features[node_type]).reshape(adapted_features[node_type].shape[0], -1)
                )
        return input_features

    def forward(self, edge_subgraph, blocks, scoring_edge):
        adapted_features = self.adapt(blocks)
        input_features = self.fusion(adapted_features)
        output_features, _ = self.rgcn(blocks, input_features)
        for node_type in output_features:
            output_features[node_type] = self.denser(output_features[node_type])
        kls = []
        for node_type in output_features:
            kls.append(kl(
                output_features[node_type][:, :hparams['gnn_out_features']], 
                output_features[node_type][:, hparams['gnn_out_features']:]
            ))
        return self.scorer(edge_subgraph, output_features, scoring_edge), output_features, kls

    def encode(self, blocks):
        adapted_features = self.adapt(blocks)
        input_features = self.fusion(adapted_features)
        output_features, attn_maps = self.rgcn(blocks, input_features)
        for node_type in output_features:
            output_features[node_type] = self.denser(output_features[node_type])
        kls = []
        for node_type in output_features:
            kls.append(kl(
                output_features[node_type][:, :hparams['gnn_out_features']], 
                output_features[node_type][:, hparams['gnn_out_features']:]
            ))
        return output_features, attn_maps
    
    def reweight_edges(self, mind_dgl):
        with torch.no_grad():
            if hparams['train_sampler'] == 'SimilarityWeightedMultiLayer':
                adapted_features = self.adapt([mind_dgl.graph])
                input_features = self.fusion(adapted_features)
                for ntype in input_features:
                    mind_dgl.graph.nodes[ntype].data['Fused_Attr'] = input_features[ntype].cpu()
                for etype in mind_dgl.num_relation:
                    mind_dgl.graph.apply_edges(lambda edges: {'Sampling_Weight' : (edges.src['Fused_Attr'] * edges.dst['Fused_Attr']).sum(1)}, etype=etype)
            elif hparams['train_sampler'] == 'InverseDegreeWeightedMultiLayer':
                for src_ntype, etype, dst_ntype in mind_dgl.graph.canonical_etypes:
                    mind_dgl.graph.nodes[src_ntype].data['Out_Degree'] = mind_dgl.graph.out_degrees(etype=etype)
                    mind_dgl.graph.apply_edges(lambda edges: {'Sampling_Weight' : 1 / (1 + edges.src['Out_Degree'])}, etype=etype)

    def collecting_metapath_utility(self):
        self.rgcn.collecting_metapath_utility()

    def pruning_metapath(self, epoch):
        self.rgcn.pruning_metapath(epoch)

