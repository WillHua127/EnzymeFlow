import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ofold.utils import rigid_utils as ru

from flowmatch.data import utils as du
from flowmatch.data import all_atom

def self_conditioning_fn(args, model, batch):
    model_sc = model.forward_frame(batch)
    batch["sc_ca_t"] = model_sc["rigids"][..., 4:]
    sc_aa_t = model_sc["amino_acid"]
    batch["sc_aa_t"] = sc_aa_t
    return batch

def set_t_feats(feats, t, t_placeholder):
    feats["t"] = t * t_placeholder
    return feats

def inference_fn(
    args,
    init_feats,
    gen_model,
    main_network,
    min_t = 0.,
    max_t = 1.,
    num_t = 100,
    center = True,
    self_condition = True,
    aa_do_purity = True,
    msa_do_purity = False,
    ec_do_purity = False,
    rot_sample_schedule = 'exp',
    trans_sample_schedule = 'linear',
    
):

    sample_feats = copy.deepcopy(init_feats)
    if sample_feats["rigids_t"].ndim == 2:
        t_placeholder = torch.ones((1,)).to(args.device)
    else:
        t_placeholder = torch.ones((sample_feats["rigids_t"].shape[0],)).to(args.device)

    forward_steps = np.linspace(min_t, max_t, num_t)
    all_rigids = [du.move_to_np(copy.deepcopy(sample_feats["rigids_t"]))]
    all_aa = [du.move_to_np(copy.deepcopy(sample_feats["aatype_t"]))]
    all_bb_atom37 = [du.move_to_np(all_atom.to_atom37(ru.Rigid.from_tensor_7(sample_feats["rigids_t"].type(torch.float32)))[0])]
    if args.flow_ec:
        all_ec = [du.move_to_np(copy.deepcopy(sample_feats["ec_t"]))]
    
    t_1 = forward_steps[0]
    with torch.no_grad():
        for t_2 in forward_steps[1:]:
            if args.embed.embed_self_conditioning and self_condition:
                sample_feats["t"] = t_1 * t_placeholder
                sample_feats = self_conditioning_fn(args, main_network, sample_feats)
            
            sample_feats["t"] = t_1 * t_placeholder
            dt = t_2 - t_1
            
            model_out = main_network(sample_feats)
            aa_pred = model_out["amino_acid"]
            rot_pred = model_out["rigids_tensor"].get_rots().get_rot_mats()
            trans_pred = model_out["rigids_tensor"].get_trans()
        
            if args.embed.embed_self_conditioning:
                sample_feats["sc_ca_t"] = model_out["rigids"][..., 4:]
                sample_feats["sc_aa_t"] = model_out["amino_acid"]
        
            if args.flow_msa:
                msa_pred = model_out["msa"]
            if args.flow_ec:
                ec_pred = model_out["ec"]

        
            rots_t, trans_t, rigids_t = gen_model.reverse_euler(
                        rigid_t=ru.Rigid.from_tensor_7(sample_feats["rigids_t"]),
                        rot=du.move_to_np(rot_pred),
                        trans=du.move_to_np(trans_pred),
                        flow_mask=None,
                        t=t_1,
                        dt=dt,
                        center=center,
                        center_of_mass=None,
                        rot_sample_schedule=rot_sample_schedule,
                        trans_sample_schedule=trans_sample_schedule,
                    )
        
            if args.eval.discrete_purity and aa_do_purity:
                aa_t = gen_model.reverse_masking_euler_purity(
                    feat_t=sample_feats["aatype_t"],
                    feat=aa_pred,
                    flow_mask=None,
                    t=t_1,
                    dt=dt,
                    n_token=args.num_aa_type,
                    mask_token_idx=args.masked_aa_token_idx,
                    temp=args.eval.aa_temp,
                    noise=args.eval.aa_noise,
                )
        
            else:
                aa_t = gen_model.reverse_masking_euler(
                    feat_t=sample_feats["aatype_t"],
                    feat=aa_pred,
                    flow_mask=None,
                    t=t_1,
                    dt=dt,
                    n_token=args.num_aa_type,
                    mask_token_idx=args.masked_aa_token_idx,
                    temp=args.eval.aa_temp,
                    noise=args.eval.aa_noise,
                )
        
            
            if args.flow_msa:
                if args.eval.discrete_purity and msa_do_purity:
                    msa_t = gen_model.reverse_masking_euler_purity(
                        feat_t=sample_feats["msa_t"],
                        feat=msa_pred,
                        flow_mask=None,
                        t=t_1,
                        dt=dt,
                        n_token=args.msa.num_msa_vocab,
                        mask_token_idx=args.msa.masked_msa_token_idx,
                        temp=args.eval.msa_temp,
                        noise=args.eval.msa_noise,
                    )
            
                else:
                    msa_t = gen_model.reverse_masking_euler(
                        feat_t=sample_feats["msa_t"],
                        feat=msa_pred,
                        flow_mask=None,
                        t=t_1,
                        dt=dt,
                        n_token=args.msa.num_msa_vocab,
                        mask_token_idx=args.msa.masked_msa_token_idx,
                        temp=args.eval.msa_temp,
                        noise=args.eval.msa_noise,
                    )
        
            if args.flow_ec:
                if args.eval.discrete_purity and ec_do_purity:
                    ec_t = gen_model.reverse_masking_euler_purity(
                        feat_t=sample_feats["ec_t"],
                        feat=ec_pred,
                        flow_mask=None,
                        t=t_1,
                        dt=dt,
                        n_token=args.ec.num_ec_class,
                        mask_token_idx=args.ec.masked_ec_token_idx,
                        temp=args.eval.ec_temp,
                        noise=args.eval.ec_noise,
                    )
        
                else:
                    ec_t = gen_model.reverse_masking_euler(
                        feat_t=sample_feats["ec_t"],
                        feat=ec_pred,
                        flow_mask=None,
                        t=t_1,
                        dt=dt,
                        n_token=args.ec.num_ec_class,
                        mask_token_idx=args.ec.masked_ec_token_idx,
                        temp=args.eval.ec_temp,
                        noise=args.eval.ec_noise,
                    )
                    
        
            sample_feats["rigids_t"] = rigids_t.to_tensor_7().to(args.device)
            sample_feats["aatype_t"] = aa_t.long().to(args.device)
            if args.flow_msa:
                sample_feats["msa_t"] = msa_t.long().to(args.device)
            if args.flow_ec:
                sample_feats["ec_t"] = ec_t.long().to(args.device)
        
            all_aa.append(du.move_to_np(aa_t))
            all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))
            if args.flow_ec:
                all_ec.append(du.move_to_np(ec_t))
        
            atom37_t = all_atom.to_atom37(rigids_t)[0]
            all_bb_atom37.append(du.move_to_np(atom37_t))
            t_1 = t_2

    t_1 = max_t
    sample_feats["t"] = t_1 * t_placeholder
    n_batch, n_res = sample_feats["aatype_t"].size(0), sample_feats["aatype_t"].size(1)
    with torch.no_grad():
        model_out = main_network(sample_feats)
        aa_logits = model_out['amino_acid']
        aa_logits[..., args.masked_aa_token_idx] = -1e10
        aa_pred = aa_logits.argmax(-1)
        rigid_pred = model_out['rigids_tensor']
        atom37_pred = all_atom.to_atom37(rigid_pred)[0]
        if args.flow_ec:
            ec_logits = model_out['ec']
            ec_logits[..., args.ec.masked_ec_token_idx] = -1e10
            ec_pred = ec_logits.argmax(-1).reshape(-1, 1)

        all_aa.append(du.move_to_np(aa_pred))
        all_bb_atom37.append(du.move_to_np(atom37_pred))
        all_rigids.append(du.move_to_np(rigid_pred.to_tensor_7()))
        if args.flow_ec:
            all_ec.append(du.move_to_np(ec_pred))

    # Flip trajectory
    flip = lambda x: np.flip(np.stack(x), (0,))
    time_steps = flip(forward_steps)
    all_bb_atom37 = flip(all_bb_atom37)
    all_aa = flip(all_aa)
    all_rigids = flip(all_rigids)
    if args.flow_ec:
        all_ec = flip(all_ec)
        

    out = {
        "t": time_steps, 
        "coord_traj": all_bb_atom37,
        "aa_traj": all_aa,
        "rigid_traj": all_rigids
        }
    if args.flow_ec:
        out["ec_traj"] = all_ec
        
    return out