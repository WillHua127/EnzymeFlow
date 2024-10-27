class Args:
    class metadata:
        csv_path = 'data/metadata_eval.csv'
        filter_num_ligand_atom = None
        filter_num_protein_atom = None
        filter_num_protein_aa = None
        subset = None

    epochs = 5000
    lr = 1e-5
    lr_min = 3e-6
    warmup_steps = 3000
    weight_decay = 0.01
    clip_norm = 1.0
    trn_batch_size = 90
    val_batch_size = 90
    num_worker = 0
    logger_dir = 'logger'
    ckpt_dir = 'checkpoint'
    ckpt_from_pretrain = False
    pretrain_ckpt_path = 'checkpoint/enzymeflow_mini.ckpt'
    ckpt_path = 'checkpoint/enzymeflow_mini.ckpt'
    early_stopping = 50
    seed = 42
    
    ######
    #1 data arguments
    min_t = 0.0
    max_t = 1.
    num_t = 100
    max_ot_res = 10
    max_mol = 200
    ######

    ######
    #2 flow matcher arguments
    ot_plan = True
    flow_trans = True
    flow_rot = True
    ot_fn = 'exact'
    ot_reg = 0.05

    # amino-acid flow
    flow_aa = True
    aa_ot = True
    discrete_flow_type = 'masking'
    # msa flow
    flow_msa = True
    msa_ot = False

    # ec flow
    flow_ec = False

    class r3:
        min_b = 0.01
        min_sigma = 0.01
        max_b = 20.0
        coordinate_scaling = .1 #dont scale coordinates
        g=0.1

    class so3:
        min_sigma = 0.01
        max_sigma = 1.5
        axis_angle = True
        inference_scaling = 0.01
        g = 0.1
    ######

    ######
    #3 model arguments
    guide_by_condition = True
    num_aa_type = 20 #fixed
    num_atom_type = 95 #fixed
    node_embed_size = 256
    edge_embed_size = 128
    dropout = 0.
    num_rbf_size = 16
    ligand_rbf_d_min = 0.05
    ligand_rbf_d_max = 6.
    bb_ligand_rbf_d_min = 0.5
    bb_ligand_rbf_d_max = 6.

    #4 MSA arguments
    class msa:
        num_msa = 1
        num_msa_vocab = 64 #fixed
        num_msa_token = 500
        msa_layers = 2
        msa_heads = 4
        msa_hidden_size = 128
        msa_embed_size = 32 #node_embed_size

    class ec:
        num_ec_class = 6 #fixed
        ec_heads = 4
        ec_embed_size = 32 #node_embed_size
        

    class mpnn:
        num_edge_type = 4
        dropout = 0.
        n_timesteps = 16
        mpnn_layers = 3
        mpnn_node_embed_size = 256 #node_embed_size
        mpnn_edge_embed_size = 128 #edge_embed_size
        

    class embed:
        c_s = 256 #node_embed_size
        c_pos_emb = 128
        c_timestep_emb = 128
        timestep_int = 1000
    
        c_z = 128 #edge_embed_size
        embed_self_conditioning = True
        relpos_k = 64
        feat_dim = 64
        num_bins = 22
        
    class ipa:
        c_s = 256 #node_embed_size
        c_z = 128 #edge_embed_size
        c_hidden = 16
        # c_skip = 16
        no_heads = 8
        no_qk_points = 8
        no_v_points = 12
        seq_tfmr_num_heads = 4
        seq_tfmr_num_layers = 4
        num_blocks = 20
        coordinate_scaling = .1 #r3.coordinate_scaling
    ######


    ######
    #4 experiment arguments
    class exp:
        dist_loss_filter = 8.
        aa_loss_weight = 1.0
        msa_loss_weight = 1.0
        ec_loss_weight = 1.0
        trans_loss_weight = 1.0
        rot_loss_weight = 1.0
        trans_x1_threshold = 1.0
        coordinate_scaling = 0.1
        bb_atom_loss_weight = 1.0
        dist_mat_loss_weight = 1.0
        aux_loss_weight = 0.25
        aux_loss_t_filter = 0.6
        bb_aux_loss_weight = 0.25
    ######

    
    ######
    #5 evaluation arguments
    class eval:
        noise_scale = 1.
        dist_loss_filter = 6.
        sample_from_multinomial = False
        eval_dir = 'generated'
        record_extra = False
        samples_per_eval_ec = 10
        eval_freq = 1000
        discrete_purity = False
        discrete_temp = .1
        aa_noise = 20.
        msa_noise = 64.
        ec_noise = 6.
        rot_sample_schedule = 'exp'
        trans_sample_schedule = 'linear'
    ######
