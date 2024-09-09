class Args:
    class metadata:
        csv_path = './data/metadata_100.csv'
        filter_num_ligand_atom = None
        filter_num_protein_atom = None
        filter_num_protein_aa = None
        subset = None

    epochs = 10000
    lr = 3e-6
    weight_decay = 5e-10
    clip_norm = 1.0
    trn_batch_size = 18
    val_batch_size = 18
    num_worker = 0
    logger_dir = './logger'
    ckpt_dir = './checkpoint'
    pretrain_ckpt_path = './checkpoint/2024-09-02-21-52-16/epoch882'
    ckpt_from_foldflow = True
    foldflow_ckpt = './weights/foldflow.ckpt'
    early_stopping = 50
    seed = 42
    
    ######
    #1 data arguments
    min_t = 0.01
    max_t = 1.
    num_t = 100
    max_ot_res = 10
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
    flow_msa = False
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
        inference_scaling = 0.1
        g = 0.1
    ######

    ######
    #3 model arguments
    guide_by_condition = False
    pretrain_kd_pred = True
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
        # p_uncond = 0.2

    class pairformer:
        dim_single = 128 #node_embed_size
        num_layers = 2
        tri_dim = 64
        tri_num_heads = 8
        tri_head_dim = 32
        row_dropout = 0.
        col_dropout = 0.
        padding = True
        expand_method = 'repeat' #repeat or outer_sum
    ######


    ######
    #4 experiment arguments
    class exp:
        dist_loss_filter = 8.
        aa_loss_weight = 1.0
        msa_loss_weight = 1.0
        kd_loss_weight = 1.0
        trans_loss_weight = 1.0
        rot_loss_weight = 1.0
        trans_x1_threshold = 1.0
        coordinate_scaling = .1 #r3.coordinate_scaling
        bb_atom_loss_weight = 1.0
        dist_mat_loss_weight = 1.0
        aux_loss_weight = 0.25
        aux_loss_t_filter = 0.6
        rot_loss_t_threshold = 1.
        bb_intersection_loss = True
        bb_inter_loss_weight = 1.0
        bb_lig_dist_loss = True
        bb_lig_dist_loss_weight = 1.0
        bb_aux_loss_weight = 0.25
        bb_aux_loss_t_filter = 0.

    ######

    
    ######
    #5 evaluation arguments
    class eval:
        noise_scale = 1.
        dist_loss_filter = 6.
        sample_from_multinomial = False
        eval_dir = './generated'
        record_extra = False
        samples_per_eval_ec = 10
        eval_freq = 100
    ######
