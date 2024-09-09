import os
from tqdm import tqdm
from datetime import datetime, timedelta

import torch
import numpy as np
import pandas as pd

from configs import *
from inference import *

from ofold.np import residue_constants

from flowmatch import flowmatcher

from model import main_network
from flowmatch.data import utils as du
from flowmatch.data import all_atom
from evaluation.metrics import *
from evaluation.loss import *
from data.utils import *
from data.loader import *
from data.data import *


CA_IDX = residue_constants.atom_order["CA"]
def train_epoch(args, model, flow_matcher, optimizer, dataloader):
    model.train()
    optimizer.zero_grad()

    n_data = 0
    avg_sample_time = 0
    total_loss = 0
    aa_loss = 0
    msa_loss = 0
    kd_loss = 0
    rot_loss = 0
    trans_loss = 0
    bb_atom_loss = 0
    dist_mat_loss = 0
    bb_inter_loss = 0
    bb_lig_dist_loss = 0
    trained_step = 0

    for train_feats in tqdm(dataloader):
        train_feats = {
                k: v.to(args.device) if torch.is_tensor(v) else v for k, v in train_feats.items()
            }

        if (
            args.embed.embed_self_conditioning
            and trained_step % 2 == 1
            ):
            with torch.no_grad():
                train_feats = self_conditioning_fn(model, train_feats)
        
        model_out = model(train_feats)
        loss, aux_data = loss_fn(args, train_feats, model_out, flow_matcher)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
    
        n_data += aux_data['examples_per_step']
        avg_sample_time += aux_data['batch_time'].sum().item()
        total_loss += aux_data['total_loss'] * aux_data['examples_per_step']
        aa_loss += aux_data['aa_loss'] * aux_data['examples_per_step']
        kd_loss += aux_data['kd_loss'] * aux_data['examples_per_step']
        rot_loss += aux_data['rot_loss'] * aux_data['examples_per_step']
        trans_loss += aux_data['trans_loss'] * aux_data['examples_per_step']
        bb_atom_loss += aux_data['bb_atom_loss'] * aux_data['examples_per_step']
        dist_mat_loss += aux_data['dist_mat_loss'] * aux_data['examples_per_step']
        bb_inter_loss += aux_data['bb_inter_loss'] * aux_data['examples_per_step']
        bb_lig_dist_loss += aux_data['bb_lig_dist_loss'] * aux_data['examples_per_step']
        trained_step += 1
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    

    total_loss = total_loss / n_data
    avg_sample_time = avg_sample_time / n_data
    aa_loss = aa_loss / n_data
    kd_loss = kd_loss / n_data
    rot_loss = rot_loss / n_data
    trans_loss = trans_loss / n_data
    bb_atom_loss = bb_atom_loss / n_data
    dist_mat_loss = dist_mat_loss / n_data
    bb_inter_loss += bb_inter_loss / n_data
    bb_lig_dist_loss += bb_lig_dist_loss / n_data

    return total_loss, avg_sample_time, aa_loss, kd_loss, rot_loss, trans_loss, bb_atom_loss, dist_mat_loss, bb_inter_loss, bb_lig_dist_loss


def eval_epoch(args, epoch, model, flow_matcher, dataloader, min_t=None, num_t=None, noise_scale=1.0, context=None):
    ckpt_eval_metrics = []
    for valid_feats, pdb_names in tqdm(dataloader):
        res_mask = du.move_to_np(valid_feats["res_mask"].bool())
        fixed_mask = du.move_to_np(valid_feats["fixed_mask"].bool())
        gt_aatype = du.move_to_np(valid_feats["aatype"])
        gt_protein_pos = du.move_to_np(all_atom.to_atom37(ru.Rigid.from_tensor_7(valid_feats["rigids_1"].type(torch.float32)))[0])
    
        ligand_pos = du.move_to_np(valid_feats["ligand_pos"])
        ligand_atom = du.move_to_np(valid_feats["ligand_atom"])
        ligand_mask = du.move_to_np(valid_feats["ligand_mask"].bool())
        batch_size = res_mask.shape[0]
    
        valid_feats = {
                k: v.to(args.device) if torch.is_tensor(v) else v for k, v in valid_feats.items()
            }
    
        # Run inference
        infer_out = inference_fn(
            args,
            model,
            flow_matcher,
            valid_feats,
            min_t=min_t,
            num_t=num_t,
            noise_scale=noise_scale,
            context=context,
        )
        
        final_prot = {
            "t_1": infer_out["t"][0],
            "pos_1": infer_out["coord_traj"][0],
            "aa_1": infer_out["aa_traj"][0],
        }
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
        for i in range(batch_size):
            num_res = int(np.sum(res_mask[i]).item())
            unpad_fixed_mask = fixed_mask[i][res_mask[i]]
            unpad_flow_mask = 1 - unpad_fixed_mask
            unpad_protein = {
                "pos": final_prot['pos_1'][i][res_mask[i]],
                "aatype": final_prot['aa_1'][i][res_mask[i]],
            }
            
            pred_aatype = unpad_protein["aatype"]
            pred_portein_pos = unpad_protein["pos"]
            
            unpad_gt_protein_pos = gt_protein_pos[i][res_mask[i]]
            unpad_gt_aatype = gt_aatype[i][res_mask[i]]
    
            unpad_gt_ligand_pos = ligand_pos[i][ligand_mask[i]]
            unpad_gt_ligand_atom = ligand_atom[i][ligand_mask[i]]
    
            prot_path = os.path.join(
                            args.evaluation_dir,
                            f"{pdb_names[i]}_len_{num_res}_sample_{i}.pdb",
                        )
    
            saved_path = write_prot_to_pdb(
                            prot_pos=pred_portein_pos,
                            file_path=prot_path,
                            aatype=pred_aatype,
                            no_indexing=True,
                            b_factors=np.tile(1 - unpad_fixed_mask[..., None], 37) * 100,
                        )
    
            try:
                sample_metrics, tm = protein_metrics(
                    pdb_path=saved_path,
                    atom37_pos=pred_portein_pos,
                    pred_aatype=pred_aatype,
                    gt_atom37_pos=unpad_gt_protein_pos,
                    gt_aatype=unpad_gt_aatype,
                    flow_mask=unpad_flow_mask,
                )
            except ValueError as e:
                print(f"Failed evaluation of length {num_res} sample {i}: {e}")
                continue

            
            n_bb_atom = 3
            amino_acid_recovery = compute_amino_acid_recovery_rate(pred_aatype, unpad_gt_aatype, res_mask[i])
            bb_pred_dist = compute_protein_ligand_dist(pred_portein_pos[..., :n_bb_atom, :].reshape(-1, 3), unpad_gt_ligand_pos)
            bb_gt_dist = compute_protein_ligand_dist(unpad_gt_protein_pos[..., :n_bb_atom, :].reshape(-1, 3), unpad_gt_ligand_pos)
            bb_gt_dist_mask = (bb_gt_dist > 0.) * (bb_gt_dist < args.eval.dist_loss_filter)
            ca_pred_dist = bb_pred_dist.reshape(num_res, n_bb_atom, -1)[..., CA_IDX, :]
            ca_gt_dist = bb_gt_dist.reshape(num_res, n_bb_atom, -1)[..., CA_IDX, :]
            ca_gt_dist_mask = (ca_gt_dist > 0.) * (ca_gt_dist < args.eval.dist_loss_filter)

    
            ca_rmsd = compute_rmsd(ca_pred_dist, ca_gt_dist, ca_gt_dist_mask)
            bb_rmsd = compute_rmsd(bb_pred_dist, bb_gt_dist, bb_gt_dist_mask)


            eval_metric = {}
            eval_metric["epoch"] = epoch
            eval_metric["gt_pdb"] = pdb_names[i]
            eval_metric["amino_acid_recovery"] = amino_acid_recovery
            eval_metric["ca_rmsd"] = ca_rmsd
            eval_metric["bb_rmsd"] = bb_rmsd
            
            eval_metric["sample_path"] = saved_path
            eval_metric.update(sample_metrics)
            ckpt_eval_metrics.append(eval_metric)
        
    # Save metrics as CSV.
    eval_metrics_csv_path = os.path.join(args.evaluation_dir, "metrics.csv")
    if not os.path.exists(eval_metrics_csv_path):
        ckpt_eval_metrics = pd.DataFrame(ckpt_eval_metrics)
        ckpt_eval_metrics.to_csv(eval_metrics_csv_path, index=False)

    else:
        with open(eval_metrics_csv_path, 'a') as eval_csv:
            ckpt_eval_metrics = pd.DataFrame(ckpt_eval_metrics)
            ckpt_eval_metrics.to_csv(eval_csv, index=False)
        

    return ckpt_eval_metrics


def main(args):
    flow_matcher = flowmatcher.SE3FlowMatcher(args)
    model = main_network.VectorFieldNetwork(args, flow_matcher)
    model = model.to(args.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters {num_parameters}")

    print('loading data...')
    trn_data = PdbDataset(
        args = args,
        gen_model = flow_matcher,
        is_training = True,
    )

    val_data = PdbDataset(
        args = args,
        gen_model = flow_matcher,
        is_training = False,
    )

    trn_loader = create_data_loader(
                    trn_data,
                    sampler=None,
                    length_batch=True,
                    batch_size=args.trn_batch_size,
                    shuffle=True,
                    num_workers=args.num_worker,
                    drop_last=False,
                )

    val_loader = create_data_loader(
                    val_data,
                    sampler=None,
                    length_batch=True,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=0,
                    drop_last=False,
                )

    
    current_pointer = 0
    best_tm_score = 0
    for epoch in range(args.epochs):
        ### Train
        print(f'#### TRAINING epoch {epoch}')
        total_loss, avg_sample_time, aa_loss, kd_loss, rot_loss, trans_loss, bb_atom_loss, dist_mat_loss, bb_inter_loss, bb_lig_dist_loss = train_epoch(args, model, flow_matcher, optimizer, trn_loader)
        
        print(f'Train epoch: {epoch}, total_loss: {total_loss:.5f}, avg_time: {avg_sample_time:.5f}, aa_loss: {aa_loss:.5f}, kd_loss: {kd_loss:.5f}, rot_loss: {rot_loss:.5f}, trans_loss: {trans_loss:.5f}, bb_loss: {bb_atom_loss:.5f}, dist_mat_loss: {dist_mat_loss:.5f}, inter_loss: {bb_inter_loss:.5f}, lig_dist_loss: {bb_lig_dist_loss:.5f}')

        with open(f'{args.logger_dir}/{args.date}.txt', 'a') as logger:
            logger.write(f'Train epoch: {epoch}, total_loss: {total_loss:.5f}, avg_time: {avg_sample_time:.5f}, aa_loss: {aa_loss:.5f}, kd_loss: {kd_loss:.5f}, rot_loss: {rot_loss:.5f}, trans_loss: {trans_loss:.5f}, bb_loss: {bb_atom_loss:.5f}, dist_mat_loss: {dist_mat_loss:.5f}, inter_loss: {bb_inter_loss:.5f}, lig_dist_loss: {bb_lig_dist_loss:.5f}\n')
            logger.close()


        
        ### Eval
        if (epoch+1) % args.eval.eval_freq == 0:
            print(f'#### EVALUATION epoch {epoch}')
            eval_metrics = eval_epoch(args, epoch, model, flow_matcher, val_loader)
            eval_aar = np.array(eval_metrics["amino_acid_recovery"]).mean()
            eval_ca_rmsd = np.array(eval_metrics["ca_rmsd"]).mean()
            eval_bb_rmsd = np.array(eval_metrics["bb_rmsd"]).mean()
            eval_tm_score = np.array(eval_metrics["tm_score"]).mean()
            eval_tm_rmsd = np.array(eval_metrics["tm_rmsd"]).mean()
            print(f'Eval epoch: {epoch}, amino_acid_recovery: {eval_aar:.5f}, ca_rmsd: {eval_ca_rmsd:.5f}, bb_rmsd: {eval_bb_rmsd:.5f}, tm_score: {eval_tm_score:.5f}, tm_rmsd: {eval_tm_rmsd:.5f}')
    
            with open(f'{args.logger_dir}/{args.date}.txt', 'a') as logger:
                logger.write(f'Eval epoch: {epoch}, amino_acid_recovery: {eval_aar:.5f}, ca_rmsd: {eval_ca_rmsd:.5f}, bb_rmsd: {eval_bb_rmsd:.5f}, tm_score: {eval_tm_score:.5f}, tm_rmsd: {eval_tm_rmsd:.5f}\n')
                logger.close()
    
            
            current_pointer += 1
            if eval_tm_score > best_tm_score:
                best_tm_score = eval_tm_score
                current_pointer = 0

            torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_tm_score": best_tm_score,
                        },
                        f'{args.checkpoint_dir}/epoch{epoch}',
                    )

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if current_pointer == args.early_stopping:
                break

    
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    
    args = Args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.logger_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.eval.eval_dir, exist_ok=True)

    args.date = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

    args.evaluation_dir = os.path.join(args.eval.eval_dir, args.date)
    os.makedirs(args.evaluation_dir, exist_ok=True)
    
    args.checkpoint_dir = os.path.join(args.ckpt_dir, args.date)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.discrete_flow_type == 'uniform':
        args.num_aa_type = 20
        args.masked_aa_token_idx = None
        
    elif args.discrete_flow_type == 'masking':
        args.num_aa_type = 21
        args.masked_aa_token_idx = 20
        args.aa_ot = False

    else:
        raise ValueError(f'Unknown discrete flow type {args.discrete_flow_type}')
        
    
    with open(f'{args.logger_dir}/{args.date}.txt', 'a') as logger:
        logger.write(f'{args}\n')
        logger.close()

    main(args)
