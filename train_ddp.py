import os
os.environ['NCCL_BLOCKING_WAIT'] = '0'
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '0'

import argparse
import random
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import OrderedDict

import torch
import numpy as np
import pandas as pd
import pytorch_warmup as warmup

import torch.distributed as dist

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


def train_epoch(args, model, flow_matcher, optimizer, lr_scheduler, warmup_scheduler, dataloader):
    model.train()
    optimizer.zero_grad()

    n_data = 0
    avg_sample_time = 0
    total_loss = 0
    aa_loss = 0
    msa_loss = 0
    ec_loss = 0
    rot_loss = 0
    trans_loss = 0
    bb_atom_loss = 0
    dist_mat_loss = 0
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
                train_feats = self_conditioning_fn(args, model, train_feats)
        
        model_out = model(train_feats)
        loss, aux_data = loss_fn(args, train_feats, model_out, flow_matcher)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        with warmup_scheduler.dampening():
            lr_scheduler.step()
    
        n_data += aux_data['examples_per_step']
        avg_sample_time += aux_data['batch_time'].sum().item()
        total_loss += aux_data['total_loss'] * aux_data['examples_per_step']
        aa_loss += aux_data['aa_loss'] * aux_data['examples_per_step']
        msa_loss += aux_data['msa_loss'] * aux_data['examples_per_step']
        ec_loss += aux_data['ec_loss'] * aux_data['examples_per_step']
        rot_loss += aux_data['rot_loss'] * aux_data['examples_per_step']
        trans_loss += aux_data['trans_loss'] * aux_data['examples_per_step']
        bb_atom_loss += aux_data['bb_atom_loss'] * aux_data['examples_per_step']
        dist_mat_loss += aux_data['dist_mat_loss'] * aux_data['examples_per_step']
        trained_step += 1
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    

    total_loss = total_loss / n_data
    avg_sample_time = avg_sample_time / n_data
    aa_loss = aa_loss / n_data
    msa_loss = msa_loss / n_data
    ec_loss = ec_loss / n_data
    rot_loss = rot_loss / n_data
    trans_loss = trans_loss / n_data
    bb_atom_loss = bb_atom_loss / n_data
    dist_mat_loss = dist_mat_loss / n_data

    return total_loss, avg_sample_time, aa_loss, msa_loss, ec_loss, rot_loss, trans_loss, bb_atom_loss, dist_mat_loss


def eval_epoch(args, epoch, model, flow_matcher, dataloader, min_t=None, num_t=None, noise_scale=1.0, context=None):
    ckpt_eval_metrics = []
    CA_IDX = residue_constants.atom_order["CA"]
    for valid_feats, pdb_names in tqdm(dataloader):
        res_mask = du.move_to_np(valid_feats["res_mask"].bool())
        flow_mask = du.move_to_np(valid_feats["flow_mask"].bool())
        gt_aatype = du.move_to_np(valid_feats["aatype"])
        gt_protein_pos = du.move_to_np(all_atom.to_atom37(ru.Rigid.from_tensor_7(valid_feats["rigids_1"].type(torch.float32)))[0])
        gt_ec = du.move_to_np(valid_feats["ec_1"])
    
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
                    init_feats = valid_feats,
                    gen_model = flow_matcher,
                    main_network = model,
                    min_t = min_t,
                    max_t = 1.0,
                    num_t = num_t,
                    self_condition = False,
                    center = True,
                    aa_do_purity = False,
                    msa_do_purity = False,
                    ec_do_purity = False,
                    rot_sample_schedule = 'linear',
                    trans_sample_schedule = 'linear',
                )    
    

        final_prot = {
            "t_1": infer_out["t"][0],
            "pos_1": infer_out["coord_traj"][0],
            "aa_1": infer_out["aa_traj"][0],
            "ec_1": infer_out["ec_traj"][0],
        }
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
        for i in range(batch_size):
            num_res = int(np.sum(res_mask[i]).item())
            unpad_flow_mask = flow_mask[i][res_mask[i]]
            unpad_protein = {
                "pos": final_prot['pos_1'][i][res_mask[i]],
                "aatype": final_prot['aa_1'][i][res_mask[i]],
                "ec": final_prot['ec_1'][i],
            }
            
            pred_aatype = unpad_protein["aatype"]
            pred_ec = unpad_protein["ec"].item()
            pred_portein_pos = unpad_protein["pos"]
            
            unpad_gt_protein_pos = gt_protein_pos[i][res_mask[i]]
            unpad_gt_aatype = gt_aatype[i][res_mask[i]]
            unpad_gt_ec = gt_ec[i][0]
    
            unpad_gt_ligand_pos = ligand_pos[i][ligand_mask[i]]
            unpad_gt_ligand_atom = ligand_atom[i][ligand_mask[i]]
    
            prot_dir = os.path.join(args.evaluation_dir, pdb_names[i])
            if not os.path.isdir(prot_dir):
                os.makedirs(prot_dir, exist_ok=True)
                        
            prot_path = os.path.join(
                            prot_dir, f"{pdb_names[i]}_sample_{i}_epoch{epoch}.pdb",
                        )
    
            saved_path = write_prot_to_pdb(
                            prot_pos=pred_portein_pos,
                            file_path=prot_path,
                            aatype=pred_aatype,
                            no_indexing=True,
                            b_factors=np.tile(unpad_flow_mask[..., None], 37) * 100,
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

            ec_acc = unpad_gt_ec == pred_ec

            eval_metric = {}
            eval_metric["epoch"] = epoch
            eval_metric["gt_ec"] = unpad_gt_ec
            eval_metric["pred_ec"] = pred_ec
            eval_metric["ec_accuracy"] = ec_acc
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
    print('initializing muti-gpu training...')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    args.device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl' if dist.is_nccl_available() else 'gloo', timeout=timedelta(seconds=7200000000))
    rank = dist.get_rank()
    
    
    flow_matcher = flowmatcher.SE3FlowMatcher(args)
    model = main_network.ProteinLigandNetwork(args)
    current_pointer = 0
    best_tm_score = 0
    best_epoch = 0
    starting_epoch = 0
    
    if args.ckpt_from_pretrain and args.pretrain_ckpt_path is not None:
        print(f'loading pretrained model from checkpoint {args.pretrain_ckpt_path}')
        checkpoint = torch.load(args.pretrain_ckpt_path, map_location='cpu')
        model_state_dict = checkpoint["model_state_dict"]
    
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        
    
    if args.ckpt_path is not None:
        print(f'resume training for {args.ckpt_path}')
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        model_state_dict = checkpoint["model_state_dict"]
    
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=True)
        starting_epoch = checkpoint["epoch"]
        best_tm_score  = checkpoint["best_tm_score"]
        starting_epoch += 1 
        
    model.cuda(local_rank)
    
    if rank == 0 and local_rank == 0:
        num_parameters = sum(p.numel() for p in model.parameters())
        print(f"Number of model parameters {num_parameters}")
        with open(f'{args.logger_dir}/{args.date}.txt', 'a') as logger:
                logger.write(f"Number of model parameters {num_parameters}\n")
                logger.close()
    
    model_dp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    print('loading data...')
    trn_data = PdbDataset(
        args = args,
        gen_model = flow_matcher,
        is_training = True,
    )
    
    trn_sampler = torch.utils.data.distributed.DistributedSampler(trn_data)

    trn_loader = create_data_loader(
                    trn_data,
                    sampler=trn_sampler,
                    length_batch=True,
                    batch_size=args.trn_batch_size,
                    shuffle=False,
                    num_workers=args.num_worker,
                    drop_last=False,
                )
    
    optimizer = torch.optim.AdamW(model_dp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = len(trn_loader) * args.epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup_steps, eta_min=args.lr_min)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    
    if rank == 0 and local_rank == 0:
        val_data = PdbDataset(
            args = args,
            gen_model = flow_matcher,
            is_training = False,
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

    
    for epoch in range(starting_epoch, args.epochs):
        trn_sampler.set_epoch(epoch)
        ### Train
        print(f'#### TRAINING epoch {epoch}')
        total_loss, avg_sample_time, aa_loss, msa_loss, ec_loss, rot_loss, trans_loss, bb_atom_loss, dist_mat_loss = train_epoch(args, model, flow_matcher, optimizer, lr_scheduler, warmup_scheduler, trn_loader)
        
        if rank == 0 and local_rank == 0:
            print(f'Train epoch: {epoch}, total_loss: {total_loss:.5f}, avg_time: {avg_sample_time:.5f}, aa_loss: {aa_loss:.5f}, msa_loss: {msa_loss:.5f}, ec_loss: {ec_loss:.5f}, rot_loss: {rot_loss:.5f}, trans_loss: {trans_loss:.5f}, bb_loss: {bb_atom_loss:.5f}, dist_mat_loss: {dist_mat_loss:.5f}')

            with open(f'{args.logger_dir}/{args.date}.txt', 'a') as logger:
                logger.write(f'Train epoch: {epoch}, total_loss: {total_loss:.5f}, avg_time: {avg_sample_time:.5f}, aa_loss: {aa_loss:.5f}, msa_loss: {msa_loss:.5f}, ec_loss: {ec_loss:.5f}, rot_loss: {rot_loss:.5f}, trans_loss: {trans_loss:.5f}, bb_loss: {bb_atom_loss:.5f}, dist_mat_loss: {dist_mat_loss:.5f}\n')
            logger.close()

        
        ### Eval
        if rank == 0 and local_rank == 0:
            if (epoch+1) % args.eval.eval_freq == 0:
                print(f'#### EVALUATION epoch {epoch}')
                eval_metrics = eval_epoch(args, epoch, model_dp.module, flow_matcher, val_loader)
                eval_aar = np.array(eval_metrics["amino_acid_recovery"]).mean()
                eval_ca_rmsd = np.array(eval_metrics["ca_rmsd"]).mean()
                eval_bb_rmsd = np.array(eval_metrics["bb_rmsd"]).mean()
                eval_tm_score = np.array(eval_metrics["tm_score"]).mean()
                eval_tm_rmsd = np.array(eval_metrics["tm_rmsd"]).mean()
                eval_ec_accuracy = np.array(eval_metrics["ec_accuracy"]).mean()
                print(f'Eval epoch: {epoch}, amino_acid_recovery: {eval_aar:.5f}, ca_rmsd: {eval_ca_rmsd:.5f}, bb_rmsd: {eval_bb_rmsd:.5f}, tm_score: {eval_tm_score:.5f}, tm_rmsd: {eval_tm_rmsd:.5f}, ec_accuracy: {eval_ec_accuracy:.5f}')

                with open(f'{args.logger_dir}/{args.date}.txt', 'a') as logger:
                    logger.write(f'Eval epoch: {epoch}, amino_acid_recovery: {eval_aar:.5f}, ca_rmsd: {eval_ca_rmsd:.5f}, bb_rmsd: {eval_bb_rmsd:.5f}, tm_score: {eval_tm_score:.5f}, tm_rmsd: {eval_tm_rmsd:.5f}, ec_accuracy: {eval_ec_accuracy:.5f}\n')
                    logger.close()


                current_pointer += 1
                if eval_tm_score > best_tm_score:
                    best_tm_score = eval_tm_score
                    best_epoch = epoch
                    current_pointer = 0

            torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model_dp.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_tm_score": best_tm_score,
                        },
                        f'{args.checkpoint_dir}/epoch{epoch}',
                    )

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if current_pointer == args.early_stopping:
                break
                    
        if not (rank == 0 and local_rank == 0):
            dist.barrier()
            
        if (rank == 0 and local_rank == 0):
            dist.barrier()
                    
    dist.destroy_process_group()

    
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    ddp_parser = argparse.ArgumentParser()
    ddp_parser.add_argument("--local-rank", type=int, default=-1)
    args_ddp = ddp_parser.parse_args()
    
    args = Args()
    args.local_rank = args_ddp.local_rank
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        
    os.makedirs(args.logger_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.eval.eval_dir, exist_ok=True)
    
    # uniform
    args.flow_ec = True
    if args.discrete_flow_type == 'uniform':
        args.num_aa_type = 20
        args.masked_aa_token_idx = None


        if args.flow_msa:
            args.msa.num_msa_vocab = 64
            args.msa.masked_msa_token_idx = None

        if args.flow_ec:
            args.ec.num_ec_class = 6
            args.ec.masked_ec_token_idx = None
            

    # discrete
    elif args.discrete_flow_type == 'masking':
        args.num_aa_type = 21
        args.masked_aa_token_idx = 20
        args.aa_ot = False


        if args.flow_msa:
            args.msa.num_msa_vocab = 65
            args.msa.masked_msa_token_idx = 64
            args.msa_ot = False

        if args.flow_ec:
            args.ec.num_ec_class = 7
            args.ec.masked_ec_token_idx = 6

    else:
        raise ValueError(f'Unknown discrete flow type {args.discrete_flow_type}')

    
    if args.local_rank == 0:
        args.date = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

        args.evaluation_dir = os.path.join(args.eval.eval_dir, args.date)
        os.makedirs(args.evaluation_dir, exist_ok=True)

        args.checkpoint_dir = os.path.join(args.ckpt_dir, args.date)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        with open(f'{args.logger_dir}/{args.date}.txt', 'a') as logger:
            logger.write(f'{args}\n')
            logger.close()

    main(args)


