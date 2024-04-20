# --------------------------------------------------------------------

import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------

from model.GNN import VGNN
from config.Config import *
from data.MIND import MIND_DGL
from utils.Metrics import auc, mrr, nDCG, ILAD
from model.DGL import msgfunc_score_neg, msgfunc_score_pos, reduce_score_pos, reduce_score_neg
from model.Utils import reparametrize
import dgl
import torch
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
import json
import time

# --------------------------------------------------------------------

if hparams['wandb']:
    import wandb
    wandb.login(key='your_wandb_key')
    wandb.init(project=hparams['name'])
    wandb.config.name = hparams['name']
    wandb.config.version = hparams['version']
    wandb.config.batch_size = hparams['batch_size']
    wandb.config.lr = hparams['lr']
    wandb.config.gnn_kl_weight = hparams['gnn_kl_weight']
    wandb.config.train_sampler = hparams['train_sampler']
    wandb.config.train_sampler_param = hparams['train_sampler_param']
    wandb.config.dev_sampler = hparams['dev_sampler']
    wandb.config.dev_sampler_param = hparams['dev_sampler_param']
    wandb.config.node_emb_meta = hparams['node_emb_meta']
    wandb.config.selfloop = hparams['self_loop']
    wandb.config.seed = hparams['seed']

from model.Utils import seed_everything
seed_everything(hparams['seed'])

# --------------------------------------------------------------------


def train():
    mind_dgl = MIND_DGL(force_reload=False)
    for ntype in mind_dgl.num_node:
        mind_dgl.graph.nodes[ntype].data['GNN_Emb'] = torch.zeros([mind_dgl.num_node[ntype], hparams['gnn_out_features'] * 2]).float()
    for etype in mind_dgl.num_relation:
        mind_dgl.graph.edges[etype].data['Sampling_Weight'] = torch.ones([mind_dgl.num_relation[etype]]).float() * 0.5

    base_canonical_etypes = sorted([canonical_etype for canonical_etype in mind_dgl.graph.canonical_etypes if canonical_etype[1] in hparams['base_etypes']])

    model = VGNN(
        hparams['gnn_in_features'], hparams['gnn_hidden_features'], hparams['gnn_out_features'], 
        base_canonical_etypes, hparams['node_emb_meta']
    )
    if hparams['embedding']:
        model.init_embedding(mind_dgl.graph)
    try:
        param_path = '{}/{}_{}_seed={}_ckpt={}.pth'.format(hparams['param_dir'], hparams['name'], hparams['version'], hparams['seed'], hparams['ckpt'])
        model.load_state_dict(torch.load(param_path))
        print('Model param loaded from {}!'.format(param_path))
    except Exception as e:
        print(e)
    model.to(hparams['device'])
    
    if hparams['wandb']:
        wandb.watch(model)

    opt = torch.optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=hparams['lr_step_size'], gamma=hparams['lr_step_gamma'])
    gradient_clip_norm = 1

    best_acc = 0.63
    best_model = None
    global_step = 0
    not_improved_count = 0
    acc_training_time = 0
    global_ct = []

    for epoch in range(int(hparams['ckpt']), int(hparams['ckpt']) + int(hparams['epochs'])):

        if hparams['train_sampler'] == 'SimilarityWeightedMultiLayer' or hparams['train_sampler'] == 'InverseDegreeWeightedMultiLayer':
            model.reweight_edges(mind_dgl)

        pos_dataloader, neg_dataloader = mind_dgl.get_gnn_train_loader()
        if hparams['debug']:
            trainloader = tqdm(enumerate(zip(pos_dataloader, neg_dataloader)))
        else:
            trainloader = enumerate(zip(pos_dataloader, neg_dataloader))

        model.train()
        epoch_loss = 0
        epoch_pred_loss = 0
        epoch_gnn_kl = 0
        epoch_score_diff = 0
        for i, ((pos_input_nodes, pos_sample_graph, pos_blocks), (neg_input_nodes, neg_sample_graph, neg_blocks)) in trainloader:

            iter_start_time = time.time()

            pos_sample_graph = pos_sample_graph.to(hparams['device'])
            pos_blocks = [b.to(hparams['device']) for b in pos_blocks]
            pos_scores, pos_output_features, pos_gnn_kls = model(pos_sample_graph, pos_blocks, ('user', 'pos_train', 'news'))
            neg_sample_graph = neg_sample_graph.to(hparams['device'])
            neg_blocks = [b.to(hparams['device']) for b in neg_blocks]
            neg_scores, neg_output_features, neg_gnn_kls = model(neg_sample_graph, neg_blocks, ('user', 'neg_train', 'news'))

            pred = torch.cat([pos_scores.unsqueeze(1), neg_scores.reshape(-1, hparams['gnn_neg_ratio'])], dim=1)
            score_diff = (F.sigmoid(pred)[:, 0] - F.sigmoid(pred)[:, 0:].mean(dim=1)).mean()
            
            if hparams['loss_func'] == 'log_sofmax':
                pred_loss = (-torch.log_softmax(pred, dim=1).select(1, 0)).mean()
            elif hparams['loss_func'] == 'cross_entropy':
                label = torch.cat([torch.ones([pred.shape[0], 1]), torch.zeros([pred.shape[0], hparams['gnn_neg_ratio']])], dim=1).to(hparams['device'])
                pred_loss = F.binary_cross_entropy(F.sigmoid(pred), label)
            elif hparams['loss_func'] == 'focal':
                label = torch.cat([torch.ones([pred.shape[0], 1]), torch.zeros([pred.shape[0], hparams['gnn_neg_ratio']])], dim=1).to(hparams['device'])
                alpha = torch.tensor([hparams['focal_alpha'], 1-hparams['focal_alpha']]).to(hparams['device'])
                gamma = hparams['focal_gamma']
                BCE_loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')        
                targets = label.type(torch.long)        
                at = alpha.gather(0, targets.data.view(-1))
                pt = torch.exp(-BCE_loss).view(-1)
                pred_loss = at*(1-pt)**gamma * BCE_loss.view(-1)   
                pred_loss = pred_loss.mean()
            else:
                raise Exception('Unexpected Loss Function')
            
            gnn_kl = (sum(pos_gnn_kls) / len(pos_gnn_kls) + sum(neg_gnn_kls) / len(neg_gnn_kls)).mean()
            
            loss = pred_loss + hparams['gnn_kl_weight'] * gnn_kl

            opt.zero_grad()
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            if hparams['pruning']:
                model.collecting_metapath_utility()
            opt.step()

            iter_end_time = time.time()
            iter_elapsed_time = iter_end_time - iter_start_time
            epoch_time = iter_elapsed_time * len(pos_dataloader)
            acc_training_time += iter_elapsed_time

            epoch_loss += loss.item()
            epoch_pred_loss += pred_loss.item()
            epoch_gnn_kl += gnn_kl.item()
            epoch_score_diff += score_diff.item()

            if hparams['wandb']:
                wandb.log({'loss': float(loss), 'pred_loss': float(pred_loss), 'gnn_kl': float(gnn_kl), 'score_diff': float(score_diff)}, global_step)
            global_step += 1
            if (i+1)%hparams['print_per_N_steps'] == 0:
                print('\nTrain Result @ Iter = {}\n- Training Loss = {}\n- Predict Loss = {}\n- KL = {}\n- Score Diff = {}\n'.format(
                    i, loss.item(), pred_loss.item(), gnn_kl.item(), score_diff.item()
                ))

        print("EPOCH: ", epoch)
        print("ELAPSED TIME: ", iter_elapsed_time)
        print("EPOCH TIME: ", epoch_time)
        print("ACC TRAINING TIME: ", acc_training_time)
        
        global_ct.append(sum(model.rgcn.acc_utility)/len(model.rgcn.acc_utility))  # epoch utility
        print(sum(global_ct))

        if hparams['pruning']:
            model.pruning_metapath(epoch)
        scheduler.step()

        epoch_loss /= len(pos_dataloader)
        epoch_pred_loss /= len(pos_dataloader)
        epoch_gnn_kl /= len(pos_dataloader)
        epoch_score_diff /= len(pos_dataloader)

        if hparams['wandb']:
            wandb.log({'epoch_loss': float(epoch_loss), 'epoch_pred_loss': float(epoch_pred_loss), 'epoch_gnn_kl': float(epoch_gnn_kl), 'epoch_score_diff': float(epoch_score_diff)}, global_step)
        
        print('Train Result @ Epoch = {}\n- Training Loss = {}\n- Predict Loss = {}\n- KL = {}\n- Score Diff = {}\n'.format(
            epoch, epoch_loss, epoch_pred_loss, epoch_gnn_kl, epoch_score_diff
        ))

        if hparams['pruning']:
            with open('{}/{}.json'.format('./pruning_log', hparams['version']), 'w') as pruning_log:
                pruning_log.write(json.dumps(model.rgcn.history_pruning))
        
        torch.save(model.state_dict(), '{}/{}_{}_seed={}_ckpt={}.pth'.format(hparams['param_dir'], hparams['name'], hparams['version'], hparams['seed'], epoch+1))

        del trainloader
        del pos_dataloader
        del neg_dataloader

        if (epoch + 1) % hparams['unit_epochs'] == 0:
            result = quick_eval(mind_dgl, model, epoch, global_step)
            this_acc = result[0]
            if this_acc > best_acc:
                best_acc = this_acc
                best_model = deepcopy(model.state_dict())
                torch.save(best_model, '{}/{}_{}_seed={}.pth'.format(
                    hparams['param_dir'], 
                    hparams['name'], 
                    hparams['version'], 
                    hparams['seed']
                ))
                not_improved_count = 0
            else:
                not_improved_count += 1
                if not_improved_count >= hparams['early_stop']:
                    break


def quick_eval(mind_dgl=None, model=None, epoch=0, global_step=0):
    if mind_dgl == None:
        mind_dgl = MIND_DGL(force_reload=False)
    for ntype in mind_dgl.num_node:
        mind_dgl.graph.nodes[ntype].data['GNN_Emb'] = torch.zeros([mind_dgl.num_node[ntype], hparams['gnn_out_features'] * 2]).float()
    for etype in mind_dgl.num_relation:
        mind_dgl.graph.edges[etype].data['Sampling_Weight'] = torch.ones([mind_dgl.num_relation[etype]]).float() * 0.5
    
    if model == None:
        base_canonical_etypes = sorted([canonical_etype for canonical_etype in mind_dgl.graph.canonical_etypes if canonical_etype[1] in hparams['base_etypes']])
        model = VGNN(
            hparams['gnn_in_features'], hparams['gnn_hidden_features'], hparams['gnn_out_features'], 
            base_canonical_etypes, hparams['node_emb_meta']
        )
        param = torch.load('{}/{}_{}_seed={}.pth'.format(hparams['param_dir'], hparams['name'], hparams['version'], hparams['seed']))
        remove_param_name = ['rgcn.pruning_masks']
        param = {p:param[p] for p in param if p not in remove_param_name}
        model.load_state_dict(param)
        model.to(hparams['device'])
        if hparams['pruning']:
            with open('{}/{}.json'.format('./pruning_log', hparams['version']), 'r') as f:
                pruning_history = json.load(f)
            model.rgcn.history_pruning = pruning_history
            for pruning_record in pruning_history:
                pruned_metapath = pruning_history[pruning_record].split('_')[:2]
                layer = int(pruned_metapath[0][5:])
                path_index = int(pruned_metapath[1][8:])
                model.rgcn.pruning_masks[layer][path_index] = 0

    encode_all_graph(model, mind_dgl)
    result = quick_rec(model, mind_dgl, epoch, global_step)
    return result


def full_eval(mind_dgl=None, model=None, epoch=0, global_step=0):
    if mind_dgl == None:
        mind_dgl = MIND_DGL(force_reload=False)
    for ntype in mind_dgl.num_node:
        mind_dgl.graph.nodes[ntype].data['GNN_Emb'] = torch.zeros([mind_dgl.num_node[ntype], hparams['gnn_out_features'] * 2]).float()
    for etype in mind_dgl.num_relation:
        mind_dgl.graph.edges[etype].data['Sampling_Weight'] = torch.ones([mind_dgl.num_relation[etype]]).float() * 0.5
    
    if model == None:
        base_canonical_etypes = sorted([canonical_etype for canonical_etype in mind_dgl.graph.canonical_etypes if canonical_etype[1] in hparams['base_etypes']])
        model = VGNN(
            hparams['gnn_in_features'], hparams['gnn_hidden_features'], hparams['gnn_out_features'], 
            base_canonical_etypes, hparams['node_emb_meta']
        )
        param = torch.load('{}/{}_{}_seed={}.pth'.format(hparams['param_dir'], hparams['name'], hparams['version'], hparams['seed']))
        remove_param_name = ['rgcn.pruning_masks']
        param = {p:param[p] for p in param if p not in remove_param_name}
        model.load_state_dict(param)
        model.to(hparams['device'])
        if hparams['pruning']:
            with open('{}/{}.json'.format('./pruning_log', hparams['version']), 'r') as f:
                pruning_history = json.load(f)            
            model.rgcn.history_pruning = pruning_history
            for pruning_record in pruning_history:
                pruned_metapath = pruning_history[pruning_record].split('_')[:2]
                layer = int(pruned_metapath[0][5:])
                path_index = int(pruned_metapath[1][8:])
                model.rgcn.pruning_masks[layer][path_index] = 0
                
    encode_all_graph(model, mind_dgl)
    result = full_rec(model, mind_dgl, epoch, global_step)
    return result


def encode_all_graph(model, mind_dgl):
    print('Generating GNN Representation')
    model.eval()
    with torch.no_grad():
        if hparams['dev_sampler'] == 'SimilarityWeightedMultiLayer' or hparams['dev_sampler'] == 'InverseDegreeWeightedMultiLayer':
            model.reweight_edges(mind_dgl)
        user_dataloader, news_dataloader = mind_dgl.get_gnn_dev_node_loader()
        if hparams['debug']:
            user_dataloader = tqdm(enumerate(user_dataloader))
            news_dataloader = tqdm(enumerate(news_dataloader))
        else:
            user_dataloader = enumerate(user_dataloader)
            news_dataloader = enumerate(news_dataloader)
        
        user_attn_maps = []
        news_attn_maps = []
        for i, (user_input_nodes, user_sample_graph, user_blocks) in user_dataloader:
            user_blocks = [b.to(hparams['device']) for b in user_blocks]
            user_output_features, attn_maps = model.encode(user_blocks)
            user_attn_maps.append({
                'l1-entity': torch.Tensor(attn_maps[0]['entity']).sum(0).reshape(-1, hparams['gnn_attention_head']).mean(-1),
                'l1-news': torch.Tensor(attn_maps[0]['news']).sum(0).reshape(-1, hparams['gnn_attention_head']).mean(-1),
                'l1-user': torch.Tensor(attn_maps[0]['user']).sum(0).reshape(-1, hparams['gnn_attention_head']).mean(-1),
                'l1-word': torch.Tensor(attn_maps[0]['word']).sum(0).reshape(-1, hparams['gnn_attention_head']).mean(-1),
                'l2-user': torch.Tensor(attn_maps[1]['user']).sum(0).reshape(-1, hparams['gnn_attention_head']).mean(-1),
            })
            mind_dgl.graph.nodes['user'].data['GNN_Emb'][user_blocks[-1].dstdata['_ID']['user'].long()] = user_output_features['user'].cpu()
        for i, (news_input_nodes, news_sample_graph, news_blocks) in news_dataloader:
            news_blocks = [b.to(hparams['device']) for b in news_blocks]
            news_output_features, attn_maps = model.encode(news_blocks)
            news_attn_maps.append({
                'l1-entity': torch.Tensor(attn_maps[0]['entity']).sum(0).reshape(-1, hparams['gnn_attention_head']).mean(-1),
                'l1-news': torch.Tensor(attn_maps[0]['news']).sum(0).reshape(-1, hparams['gnn_attention_head']).mean(-1),
                'l1-user': torch.Tensor(attn_maps[0]['user']).sum(0).reshape(-1, hparams['gnn_attention_head']).mean(-1),
                'l1-word': torch.Tensor(attn_maps[0]['word']).sum(0).reshape(-1, hparams['gnn_attention_head']).mean(-1),
                'l2-news': torch.Tensor(attn_maps[1]['news']).sum(0).reshape(-1, hparams['gnn_attention_head']).mean(-1),
            })
            mind_dgl.graph.nodes['news'].data['GNN_Emb'][news_blocks[-1].dstdata['_ID']['news'].long()] = news_output_features['news'].cpu()
    print('Generating GNN Representation Finished')


def quick_rec(model, mind_dgl, epoch, global_step):
    # testing performance w/o EDC using randomly sampled users
    
    dev_session_loader = mind_dgl.get_dev_session_loader(shuffle=True)
    epoch_auc_score = 0
    epoch_mrr = 0
    epoch_ndcg_5 = 0
    epoch_ndcg_10 = 0
    epoch_ilad_5 = 0
    epoch_ilad_10 = 0

    if hparams['debug']:
        devloader = tqdm(enumerate(dev_session_loader))
    else:
        devloader = enumerate(dev_session_loader)

    for i, (pos_links, neg_links) in devloader:
        if hparams['gnn_quick_dev_reco'] and i >= hparams['gnn_quick_dev_reco_size']:
            break
        sub_g = dgl.edge_subgraph(mind_dgl.graph, {('news', 'pos_dev_r', 'user'): pos_links, ('news', 'neg_dev_r', 'user'): neg_links})
        sub_g.apply_nodes(model.scorer.get_representation, ntype='user')
        sub_g.apply_nodes(model.scorer.get_representation, ntype='news')
        sub_g.update_all(model.scorer.msgfunc_score_neg, model.scorer.reduce_score_neg, etype=('news', 'neg_dev_r', 'user'))
        sub_g.update_all(model.scorer.msgfunc_score_pos, model.scorer.reduce_score_pos, etype=('news', 'pos_dev_r', 'user'))

        labels = torch.cat([torch.ones(sub_g.nodes['user'].data['pos_score'].shape), torch.zeros(sub_g.nodes['user'].data['neg_score'].shape)], dim=1).type(torch.int32).squeeze(0)
        scores = torch.cat([sub_g.nodes['user'].data['pos_score'], sub_g.nodes['user'].data['neg_score']], dim=1).squeeze(0)
        news_representation = torch.cat([sub_g.nodes['user'].data['pos_news_representation'], sub_g.nodes['user'].data['neg_news_representation']], dim=1).squeeze(0)
        
        auc_score = auc(labels.numpy(), scores.numpy())
        mrr_score = mrr(labels.numpy(), scores.numpy())
        ndcg_5 = nDCG(labels.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=5)
        ndcg_10 = nDCG(labels.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=10)

        epoch_auc_score += auc_score
        epoch_mrr += mrr_score
        epoch_ndcg_5 += ndcg_5
        epoch_ndcg_10 += ndcg_10

        if news_representation.shape[0] >= 5:
            top_5_news_representation = news_representation[torch.topk(scores, k=5).indices]
        else:
            top_5_news_representation = news_representation
        top_5_news_representation = (top_5_news_representation.T / top_5_news_representation.norm(dim=1)).T
        
        if news_representation.shape[0] >= 10:
            top_10_news_representation = news_representation[torch.topk(scores, k=10).indices]
        else:
            top_10_news_representation = news_representation
        top_10_news_representation = (top_10_news_representation.T / top_10_news_representation.norm(dim=1)).T

        epoch_ilad_5 += ILAD(top_5_news_representation.numpy())
        epoch_ilad_10 += ILAD(top_10_news_representation.numpy())

    if hparams['gnn_quick_dev_reco']:
        epoch_auc_score /= hparams['gnn_quick_dev_reco_size']
        epoch_mrr /= hparams['gnn_quick_dev_reco_size']
        epoch_ndcg_5 /= hparams['gnn_quick_dev_reco_size']
        epoch_ndcg_10 /= hparams['gnn_quick_dev_reco_size']
        epoch_ilad_5 /= hparams['gnn_quick_dev_reco_size']
        epoch_ilad_10 /= hparams['gnn_quick_dev_reco_size']
    else:
        epoch_auc_score /= len(dev_session_loader)
        epoch_mrr /= len(dev_session_loader)
        epoch_ndcg_5 /= len(dev_session_loader)
        epoch_ndcg_10 /= len(dev_session_loader)
        epoch_ilad_5 /= len(dev_session_loader)
        epoch_ilad_10 /= len(dev_session_loader)

    if hparams['wandb']:
        wandb.log({
            'epoch_auc_score': float(epoch_auc_score), 
            'epoch_mrr': float(epoch_mrr), 
            'epoch_ndcg_5': float(epoch_ndcg_5), 
            'epoch_ndcg_10': float(epoch_ndcg_10), 
            'epoch_ilad_5': float(epoch_ilad_5), 
            'epoch_ilad_10': float(epoch_ilad_10), 
        }, global_step)
    print('Testing Result @ Epoch = {}\n- AUC = {}\n- MRR = {}\n- nDCG@5 = {}\n- nDCG@10 = {}\n- ILAD@5 = {}\n- ILAD@10 = {}\n'.format(epoch, epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10))

    return [epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10]


def full_rec(model, mind_dgl, epoch, global_step):
    # testing performance w/z EDC using randomly sampled users
    
    dev_session_loader = mind_dgl.get_dev_session_loader(shuffle=False)
    mind_dgl.graph.nodes['user'].data['News_Pref'] = mind_dgl.graph.nodes['user'].data['GNN_Emb'].unsqueeze(1).unsqueeze(1).repeat(1, mind_dgl.graph.nodes['news'].data['CateID'].max()+1, hparams['cache_size'], 1)
    mind_dgl.graph.nodes['user'].data['Last_Update_Time'] = torch.zeros([mind_dgl.num_node['user'], mind_dgl.graph.nodes['news'].data['CateID'].max()+1, hparams['cache_size']])
    
    epoch_auc_score = 0
    epoch_mrr = 0
    epoch_ndcg_5 = 0
    epoch_ndcg_10 = 0
    epoch_ilad_5 = 0
    epoch_ilad_10 = 0

    if hparams['debug']:
        devloader = tqdm(enumerate(dev_session_loader))
    else:
        devloader = enumerate(dev_session_loader)

    for i, (pos_links, neg_links) in devloader:
        if hparams['gnn_quick_dev_reco'] and i >= hparams['gnn_quick_dev_reco_size']:
            break
        sub_g = dgl.edge_subgraph(mind_dgl.graph, {('news', 'pos_dev_r', 'user'): pos_links, ('news', 'neg_dev_r', 'user'): neg_links})
        # sub_g.apply_nodes(model.scorer.get_representation, ntype='user')
        sub_g.apply_nodes(model.scorer.get_representation, ntype='news')
        sub_g.update_all(model.scorer.msgfunc_score_neg_edc, model.scorer.reduce_score_neg_edc, etype=('news', 'neg_dev_r', 'user'))
        sub_g.update_all(model.scorer.msgfunc_score_pos_edc, model.scorer.reduce_score_pos_edc, etype=('news', 'pos_dev_r', 'user'))
        mind_dgl.graph.nodes['user'].data['News_Pref'][sub_g.dstdata['_ID']['user'].long()] = sub_g.dstdata['pref']['user']  # write back to mind.graph
        mind_dgl.graph.nodes['user'].data['Last_Update_Time'][sub_g.dstdata['_ID']['user'].long()] = sub_g.dstdata['lut']['user']

        labels = torch.cat([torch.ones(sub_g.nodes['user'].data['pos_score'].shape), torch.zeros(sub_g.nodes['user'].data['neg_score'].shape)], dim=1).type(torch.int32).squeeze(0)
        scores = torch.cat([sub_g.nodes['user'].data['pos_score'], sub_g.nodes['user'].data['neg_score']], dim=1).squeeze(0)
        news_representation = torch.cat([sub_g.nodes['user'].data['pos_news_representation'], sub_g.nodes['user'].data['neg_news_representation']], dim=1).squeeze(0)
        
        auc_score = auc(labels.numpy(), scores.numpy())
        mrr_score = mrr(labels.numpy(), scores.numpy())
        ndcg_5 = nDCG(labels.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=5)
        ndcg_10 = nDCG(labels.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=10)

        epoch_auc_score += auc_score
        epoch_mrr += mrr_score
        epoch_ndcg_5 += ndcg_5
        epoch_ndcg_10 += ndcg_10

        if news_representation.shape[0] >= 5:
            top_5_news_representation = news_representation[torch.topk(scores, k=5).indices]
        else:
            top_5_news_representation = news_representation
        top_5_news_representation = (top_5_news_representation.T / top_5_news_representation.norm(dim=1)).T
        
        if news_representation.shape[0] >= 10:
            top_10_news_representation = news_representation[torch.topk(scores, k=10).indices]
        else:
            top_10_news_representation = news_representation
        top_10_news_representation = (top_10_news_representation.T / top_10_news_representation.norm(dim=1)).T

        epoch_ilad_5 += ILAD(top_5_news_representation.numpy())
        epoch_ilad_10 += ILAD(top_10_news_representation.numpy())

    if hparams['gnn_quick_dev_reco']:
        epoch_auc_score /= hparams['gnn_quick_dev_reco_size']
        epoch_mrr /= hparams['gnn_quick_dev_reco_size']
        epoch_ndcg_5 /= hparams['gnn_quick_dev_reco_size']
        epoch_ndcg_10 /= hparams['gnn_quick_dev_reco_size']
        epoch_ilad_5 /= hparams['gnn_quick_dev_reco_size']
        epoch_ilad_10 /= hparams['gnn_quick_dev_reco_size']
    else:
        epoch_auc_score /= len(dev_session_loader)
        epoch_mrr /= len(dev_session_loader)
        epoch_ndcg_5 /= len(dev_session_loader)
        epoch_ndcg_10 /= len(dev_session_loader)
        epoch_ilad_5 /= len(dev_session_loader)
        epoch_ilad_10 /= len(dev_session_loader)
    
    # 主动推荐，用于ablation/case
    # user_repr = []
    # news_repr = []
    # for i in range(hparams['sample']):
    #     user_repr.append(reparametrize(mind_dgl.graph.nodes['user'].data['GNN_Emb'][:, :hparams['gnn_out_features']], mind_dgl.graph.nodes['user'].data['GNN_Emb'][:, hparams['gnn_out_features']:]))
    #     news_repr.append(reparametrize(mind_dgl.graph.nodes['news'].data['GNN_Emb'][:, :hparams['gnn_out_features']], mind_dgl.graph.nodes['news'].data['GNN_Emb'][:, hparams['gnn_out_features']:]))
    # user_repr = torch.cat(user_repr, dim=1)
    # user_repr = (user_repr.T / user_repr.norm(dim=1)).T
    # news_repr = torch.cat(news_repr, dim=1)
    # news_repr = (news_repr.T / news_repr.norm(dim=1)).T
    # rcmd_score = torch.matmul(user_repr, news_repr.T)
    # recommendation_lists = torch.topk(rcmd_score, k=10).indices
    # epoch_ils_5 = ils(news_repr[recommendation_lists[:, :5]])
    # epoch_ils_10 = ils(news_repr[recommendation_lists[:, :10]])
    # epoch_ilad_5 = ilad(news_repr[recommendation_lists[:, :5]])
    # epoch_ilad_10 = ilad(news_repr[recommendation_lists[:, :10]])

    print('Testing Result @ Epoch = {}\n- AUC = {}\n- MRR = {}\n- nDCG@5 = {}\n- nDCG@10 = {}\n- ILAD@5 = {}\n- ILAD@10 = {}\n'.format(epoch, epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10))

    return [epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10]


if __name__ == '__main__':
    train()
    # quick_eval()
    # full_eval()
