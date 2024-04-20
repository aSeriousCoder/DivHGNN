# to GNN.py

import torch
import torch.nn.functional as F
import math
from copy import deepcopy

from config.Config import hparams
from model.Utils import reparametrize, attention


def get_decay_weight(delta_t):
    shape = delta_t.shape
    return torch.Tensor([math.exp(- hparams['beta'] * math.pow(dt, hparams['rho'])) for dt in delta_t.reshape(-1)]).reshape(shape)


# GNN scoring function
def msgfunc_gnn(edges):
    src_emb = edges.src['x']
    dst_emb = edges.dst['x']
    score = (src_emb * dst_emb).sum(dim=1)
    return {'score': score}

# move: model build-in
def msgfunc_score_vgnn(edges):
    if hparams['sample'] <= 0:
        src_emb = edges.src['x'][:, :hparams['gnn_out_features']]
        dst_emb = edges.dst['x'][:, :hparams['gnn_out_features']]
        score = (src_emb * dst_emb).sum(dim=1)
        return {'score': score}
    else:
        scores = []
        for i in range(hparams['sample']):
            src_emb = reparametrize(edges.src['x'][:, :hparams['gnn_out_features']], edges.src['x'][:, hparams['gnn_out_features']:])
            dst_emb = reparametrize(edges.dst['x'][:, :hparams['gnn_out_features']], edges.dst['x'][:, hparams['gnn_out_features']:])
            score = (src_emb * dst_emb).sum(dim=1)
            scores.append(score)
        score = sum(scores) / len(scores)
        return {'score': score}


def msgfunc_score_neg(edges):
    if hparams['sample'] <= 0:
        src_emb = edges.src['GNN_Emb'][:, :hparams['gnn_out_features']]
        dst_emb = edges.dst['GNN_Emb'][:, :hparams['gnn_out_features']]
        score = (src_emb * dst_emb).sum(dim=1)
        return {'score': score}
    else:
        scores = []
        for i in range(hparams['sample']):
            src_emb = reparametrize(edges.src['GNN_Emb'][:, :hparams['gnn_out_features']], edges.src['GNN_Emb'][:, hparams['gnn_out_features']:])
            dst_emb = reparametrize(edges.dst['GNN_Emb'][:, :hparams['gnn_out_features']], edges.dst['GNN_Emb'][:, hparams['gnn_out_features']:])
            score = (src_emb * dst_emb).sum(dim=1)
            scores.append(score)
        score = sum(scores) / len(scores)
        return {'neg_score': score}


def msgfunc_score_pos(edges):
    if hparams['sample'] <= 0:
        src_emb = edges.src['GNN_Emb'][:, :hparams['gnn_out_features']]
        dst_emb = edges.dst['GNN_Emb'][:, :hparams['gnn_out_features']]
        score = (src_emb * dst_emb).sum(dim=1)
        return {'score': score}
    else:
        scores = []
        for i in range(hparams['sample']):
            src_emb = reparametrize(edges.src['GNN_Emb'][:, :hparams['gnn_out_features']], edges.src['GNN_Emb'][:, hparams['gnn_out_features']:])
            dst_emb = reparametrize(edges.dst['GNN_Emb'][:, :hparams['gnn_out_features']], edges.dst['GNN_Emb'][:, hparams['gnn_out_features']:])
            score = (src_emb * dst_emb).sum(dim=1)
            scores.append(score)
        score = sum(scores) / len(scores)
        return {'pos_score': score}


def msgfunc_reco_neg(edges):
    if hparams['sample'] <= 0:
        src_emb = edges.src['GNN_Emb'][:, :hparams['gnn_out_features']]
        dst_emb = edges.dst['GNN_Emb'][:, :hparams['gnn_out_features']]
        Last_Update_Time = torch.zeros(edges.dst['Last_Update_Time'].shape[0], edges.dst['Last_Update_Time'].shape[2])
        News_Pref = torch.zeros(edges.dst['News_Pref'].shape[0], edges.dst['News_Pref'].shape[2], edges.dst['News_Pref'].shape[3])
        for i, cate in enumerate(edges.src['CateID']):
            Last_Update_Time[i] = edges.dst['Last_Update_Time'][i][cate]
            News_Pref[i] = edges.dst['News_Pref'][i][cate]
        # delta-t and decay
        delta_t = edges.data['Time'][0] - Last_Update_Time
        decaying_weight = get_decay_weight(delta_t)
        # reparametrize pref
        pref_shape = News_Pref.shape[:-1]
        # distribution_decayed_pref = torch.mul(News_Pref, decaying_weight.unsqueeze(-1))
        decaying_target = edges.dst['GNN_Emb'].unsqueeze(1).repeat(1, News_Pref.shape[1], 1)
        distribution_decayed_pref = torch.mul(News_Pref, decaying_weight.unsqueeze(-1)) + torch.mul(decaying_target, 1 - decaying_weight.unsqueeze(-1))
        # distribution_decayed_pref = News_Pref
        pref_emb = distribution_decayed_pref.reshape(-1, 2 * hparams['gnn_out_features'])[:, :hparams['gnn_out_features']].reshape(pref_shape[0], pref_shape[1], hparams['gnn_out_features'])
        dst_pref_emb, attn = attention(src_emb.unsqueeze(1), pref_emb, pref_emb)
        dst_pref_emb = dst_pref_emb.squeeze(1)
        dst_pref_emb_decaying_weight = (attn.squeeze(1) * decaying_weight).sum(-1)
        return {
            'neg_score_un': (src_emb * dst_emb).sum(dim=1), 
            'neg_score_nn': (src_emb * dst_pref_emb).sum(dim=1), 
        }
    else:
        scores_un = []
        scores_nn = []
        for i in range(hparams['sample']):
            src_emb = reparametrize(edges.src['GNN_Emb'][:, :hparams['gnn_out_features']], edges.src['GNN_Emb'][:, hparams['gnn_out_features']:])
            dst_emb = reparametrize(edges.dst['GNN_Emb'][:, :hparams['gnn_out_features']], edges.dst['GNN_Emb'][:, hparams['gnn_out_features']:])
            Last_Update_Time = torch.zeros(edges.dst['Last_Update_Time'].shape[0], edges.dst['Last_Update_Time'].shape[2])
            News_Pref = torch.zeros(edges.dst['News_Pref'].shape[0], edges.dst['News_Pref'].shape[2], edges.dst['News_Pref'].shape[3])
            for i, cate in enumerate(edges.src['CateID']):
                Last_Update_Time[i] = edges.dst['Last_Update_Time'][i][cate]
                News_Pref[i] = edges.dst['News_Pref'][i][cate]
            # delta-t and decay
            delta_t = edges.data['Time'][0] - Last_Update_Time
            decaying_weight = get_decay_weight(delta_t)
            # reparametrize pref
            pref_shape = News_Pref.shape[:-1]
            # distribution_decayed_pref = torch.mul(News_Pref, decaying_weight.unsqueeze(-1))
            decaying_target = edges.dst['GNN_Emb'].unsqueeze(1).repeat(1, News_Pref.shape[1], 1)
            distribution_decayed_pref = torch.mul(News_Pref, decaying_weight.unsqueeze(-1)) + torch.mul(decaying_target, 1 - decaying_weight.unsqueeze(-1))
            # distribution_decayed_pref = News_Pref
            pref_emb = reparametrize(
                mu=distribution_decayed_pref.reshape(-1, 2 * hparams['gnn_out_features'])[:, :hparams['gnn_out_features']],
                logvar=distribution_decayed_pref.reshape(-1, 2 * hparams['gnn_out_features'])[:, hparams['gnn_out_features']:]
            ).reshape(pref_shape[0], pref_shape[1], hparams['gnn_out_features'])
            dst_pref_emb, attn = attention(src_emb.unsqueeze(1), pref_emb, pref_emb)
            dst_pref_emb = dst_pref_emb.squeeze(1)
            dst_pref_emb_decaying_weight = (attn.squeeze(1) * decaying_weight).sum(-1)
            # total score
            scores_un.append((src_emb * dst_emb).sum(dim=1))
            scores_nn.append((src_emb * dst_pref_emb).sum(dim=1))
        score_un = sum(scores_un) / len(scores_un)
        score_nn = sum(scores_nn) / len(scores_nn)
        return {
            'neg_score_un': score_un, 
            'neg_score_nn': score_nn, 
        }


def msgfunc_reco_pos(edges):
    if hparams['sample'] <= 0:
        src_emb = edges.src['GNN_Emb'][:, :hparams['gnn_out_features']]
        dst_emb = edges.dst['GNN_Emb'][:, :hparams['gnn_out_features']]
        Last_Update_Time = torch.zeros(edges.dst['Last_Update_Time'].shape[0], edges.dst['Last_Update_Time'].shape[2])
        News_Pref = torch.zeros(edges.dst['News_Pref'].shape[0], edges.dst['News_Pref'].shape[2], edges.dst['News_Pref'].shape[3])
        for i, cate in enumerate(edges.src['CateID']):
            Last_Update_Time[i] = edges.dst['Last_Update_Time'][i][cate]
            News_Pref[i] = edges.dst['News_Pref'][i][cate]
        # delta-t and decay
        delta_t = edges.data['Time'][0] - Last_Update_Time
        decaying_weight = get_decay_weight(delta_t)
        # reparametrize pref
        pref_shape = News_Pref.shape[:-1]
        # distribution_decayed_pref = torch.mul(News_Pref, decaying_weight.unsqueeze(-1))
        decaying_target = edges.dst['GNN_Emb'].unsqueeze(1).repeat(1, News_Pref.shape[1], 1)
        distribution_decayed_pref = torch.mul(News_Pref, decaying_weight.unsqueeze(-1)) + torch.mul(decaying_target, 1 - decaying_weight.unsqueeze(-1))
        # distribution_decayed_pref = News_Pref
        pref_emb = distribution_decayed_pref.reshape(-1, 2 * hparams['gnn_out_features'])[:, :hparams['gnn_out_features']].reshape(pref_shape[0], pref_shape[1], hparams['gnn_out_features'])
        dst_pref_emb, attn = attention(src_emb.unsqueeze(1), pref_emb, pref_emb)
        dst_pref_emb = dst_pref_emb.squeeze(1)
        dst_pref_emb_decaying_weight = (attn.squeeze(1) * decaying_weight).sum(-1)
        return {
            'pos_score_un': (src_emb * dst_emb).sum(dim=1), 
            'pos_score_nn': (src_emb * dst_pref_emb).sum(dim=1), 
            'src_repr': edges.src['GNN_Emb'], 
            'time': edges.data['Time'],
            'cate': edges.src['CateID']
        }
    else:
        scores_un = []
        scores_nn = []
        for i in range(hparams['sample']):
            src_emb = reparametrize(edges.src['GNN_Emb'][:, :hparams['gnn_out_features']], edges.src['GNN_Emb'][:, hparams['gnn_out_features']:])
            dst_emb = reparametrize(edges.dst['GNN_Emb'][:, :hparams['gnn_out_features']], edges.dst['GNN_Emb'][:, hparams['gnn_out_features']:])
            Last_Update_Time = torch.zeros(edges.dst['Last_Update_Time'].shape[0], edges.dst['Last_Update_Time'].shape[2])
            News_Pref = torch.zeros(edges.dst['News_Pref'].shape[0], edges.dst['News_Pref'].shape[2], edges.dst['News_Pref'].shape[3])
            for i, cate in enumerate(edges.src['CateID']):
                Last_Update_Time[i] = edges.dst['Last_Update_Time'][i][cate]
                News_Pref[i] = edges.dst['News_Pref'][i][cate]
            # delta-t and decay
            delta_t = edges.data['Time'][0] - Last_Update_Time
            decaying_weight = get_decay_weight(delta_t)
            # reparametrize pref
            pref_shape = News_Pref.shape[:-1]
            # distribution_decayed_pref = torch.mul(News_Pref, decaying_weight.unsqueeze(-1))
            decaying_target = edges.dst['GNN_Emb'].unsqueeze(1).repeat(1, News_Pref.shape[1], 1)
            distribution_decayed_pref = torch.mul(News_Pref, decaying_weight.unsqueeze(-1)) + torch.mul(decaying_target, 1 - decaying_weight.unsqueeze(-1))
            # distribution_decayed_pref = News_Pref
            pref_emb = reparametrize(
                mu=distribution_decayed_pref.reshape(-1, 2 * hparams['gnn_out_features'])[:, :hparams['gnn_out_features']],
                logvar=distribution_decayed_pref.reshape(-1, 2 * hparams['gnn_out_features'])[:, hparams['gnn_out_features']:]
            ).reshape(pref_shape[0], pref_shape[1], hparams['gnn_out_features'])
            dst_pref_emb, attn = attention(src_emb.unsqueeze(1), pref_emb, pref_emb)
            dst_pref_emb = dst_pref_emb.squeeze(1)
            dst_pref_emb_decaying_weight = (attn.squeeze(1) * decaying_weight).sum(-1)
            # total score
            scores_un.append((src_emb * dst_emb).sum(dim=1))
            scores_nn.append((src_emb * dst_pref_emb).sum(dim=1))
        score_un = sum(scores_un) / len(scores_un)
        score_nn = sum(scores_nn) / len(scores_nn)
        return {
            'pos_score_un': score_un, 
            'pos_score_nn': score_nn, 
            'src_repr': edges.src['GNN_Emb'], 
            'time': edges.data['Time'],
            'cate': edges.src['CateID']
        }


def reduce_score_pos(nodes):
    pos_score = nodes.mailbox['pos_score']
    return {'pos_score': pos_score}


def reduce_score_neg(nodes):
    neg_score = nodes.mailbox['neg_score']
    return {'neg_score': neg_score}


def reduce_reco_pos(nodes):
    pos_score_un = nodes.mailbox['pos_score_un']
    pos_score_nn = nodes.mailbox['pos_score_nn']
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
        'pos_score_un': pos_score_un, 
        'pos_score_nn': pos_score_nn, 
        'pref': new_pref, 
        'lut': new_lut
    }


def reduce_reco_neg(nodes):
    neg_score_un = nodes.mailbox['neg_score_un']
    neg_score_nn = nodes.mailbox['neg_score_nn']
    return {
        'neg_score_un': neg_score_un, 
        'neg_score_nn': neg_score_nn
    }


def msgfunc_weighting_edges(edges):
    values = []
    for key in edges.src.keys():
        if key in ['CateID', 'SubCateID', '_ID', 'tmp_value', 'tmp_len']:
            continue
        values.append(edges.src[key])
    value = torch.cat(values, dim=1)
    return {'value': value}


def reduce_weighting_edges(nodes):
    if 'tmp_value' not in nodes.data.keys():
        tmp_len = torch.ones(nodes.mailbox['value'].shape[0]) * nodes.mailbox['value'].shape[1]
        tmp_value = nodes.mailbox['value'].sum(dim=1)
    else:
        tmp_len = nodes.data['tmp_len'] + torch.ones(nodes.mailbox['value'].shape[0]) * nodes.mailbox['value'].shape[1]
        tmp_value = nodes.data['tmp_value'] + nodes.mailbox['value'].sum(dim=1)
    return {
        'tmp_value': tmp_value,
        'tmp_len': tmp_len,
    }


def apply_edge_weighting_edges(edges):
    src_values = []
    for key in edges.src.keys():
        if key in ['CateID', 'SubCateID', '_ID', 'tmp_value', 'tmp_len']:
            continue
        src_values.append(edges.src[key])
    src_value = torch.cat(src_values, dim=1)
    dst_value = (edges.dst['tmp_value'] - src_value) / (edges.dst['tmp_value'] - 1)
    # weight = F.sigmoid((src_value * dst_value).mean(dim=1))
    weight = 1 / (((dst_value - src_value) ** 2).mean(dim=1) + 1)
    return {'weight': weight}


def apply_edge_weighting_edges_for_selfloop(edges):
    return {'weight': torch.ones(len(edges))}