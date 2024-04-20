hparams = {
    'name': 'DivHGNN',
    'description': 'Codes for Paper - Heterogeneous Graph Neural Network with Personalized and Adaptive Diversity for News Recommendation',
    'optimizer': 'adam',
    'data_dir': 'data/MIND/MINDsmall',
    'mind_version': 'small',
    'batch_size': 128,
    'epochs': 50, 
    'unit_epochs': 1,
    'lr': 1e-3,
    'lr_step_size': 1,
    'lr_step_gamma': 0.9,
    'weight_decay': 0,
    'loss_func': 'log_sofmax',
    'focal_alpha': 0.25,  
    'focal_gamma': 2,  
    'seed': 48, 
    'debug': True, 
    'ckpt': 0, 
    'early_stop': 3,

    'version': 'v8.0.0', 
    'device': 'cuda:5',
    'embedding': False, 
    'pruning': True, 

    # Static Config
    'param_dir': 'out/param',
    'print_per_N_steps': 100,
    'gnn_quick_dev_reco': False,
    'gnn_quick_dev_reco_size': 1000,

    # GNN mete-info
    'gnn_neg_ratio': 4,
    'node_emb_meta': {
        'user': {
            'Category': 384,
            'SubCategory': 384,
            'Node2Vec': 128,
        },
        'news': {
            'News_Title_Embedding': 384,
            'News_Abstract_Embedding': 384,
            'Category': 384,
            'SubCategory': 384,
            'Node2Vec': 128,
        },
        'entity': {
            'Entity_Embedding': 100,
            'Node2Vec': 128,
        },
        'word': {
            'Word_Embedding': 384,
            'Node2Vec': 128,
        },
    },
    'base_etypes': ['history', 'ne_link', 'nw_link', 'ue_link', 'uw_link'],

    # GNN config
    'gnn_kl_weight': 1e-4,
    'gnn_in_features': 128,
    'gnn_hidden_features': 128,
    'gnn_out_features': 128,
    'gnn_attention_head': 4,
    'adaptor_feat_hidden': 256,

    'train_sampler': 'MultiLayer',  # MultiLayer / MultiLayerFull / SimilarityWeightedMultiLayer / InverseDegreeWeightedMultiLayer
    'train_sampler_param': [5,5],  # for MultiLayerFull, the number is not used, only the length determines the num of GNN layers
    'dev_sampler': 'MultiLayer',  # Same as train_sampler or "MultiLayerFull"
    'dev_sampler_param': [5,5],  # Same as train_sampler_param

    'sample': 3,  # representation
    'self_loop': True, 
    'cross_scorer': False, 
    'cold_start': False, 

    'pruning_order': 2,  # 1 using parameters, 2 using grad 
    'pruning_thres': {
        0: 0.2,
        # 0: 0.01,
        # 20: 0.2,
        # 35: 0.01,
    },  
    
    # Decaying config
    'group_type': 'CateID',
    'cache_size': 5,
    'rho': 0.5,
    'beta': 0.05,
}

hparams['wandb'] = not hparams['debug']

from copy import deepcopy
hparams['reverse'] = {}
cur_etypes = deepcopy(hparams['base_etypes'])
for etype in cur_etypes:
    hparams['base_etypes'].append('{}_r'.format(etype))
    hparams['reverse'][etype] = '{}_r'.format(etype)
    hparams['reverse']['{}_r'.format(etype)] = etype
if hparams['self_loop']:
    hparams['base_etypes'].append('user_selfloop')
    hparams['base_etypes'].append('news_selfloop')
    hparams['base_etypes'].append('word_selfloop')
    hparams['base_etypes'].append('entity_selfloop')
