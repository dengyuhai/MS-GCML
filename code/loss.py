import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle

class Momentum_Update(nn.Module):
    """Log ratio loss function. """
    def __init__(self, momentum):
        super(Momentum_Update, self).__init__()
        self.momentum = momentum
        
    @torch.no_grad()
    def forward(self, model_student, model_teacher):
        """
        Momentum update of the key encoder
        """
        m = self.momentum

        state_dict_s = model_student.state_dict()
        state_dict_t = model_teacher.state_dict()
        for (k_s, v_s), (k_t, v_t) in zip(state_dict_s.items(), state_dict_t.items()):
            if 'num_batches_tracked' in k_s:
                v_t.copy_(v_s)
            else:
                v_t.copy_(v_t * m + (1. - m) * v_s)  
    
class RC_STML(nn.Module):
    def __init__(self, sigma, delta, view, disable_mu, topk):
        super(RC_STML, self).__init__()
        self.sigma = sigma
        self.delta = delta
        self.view = view
        self.disable_mu = disable_mu
        self.topk = topk
        
    def k_reciprocal_neigh(self, initial_rank, i, topk):
        forward_k_neigh_index = initial_rank[i,:topk]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:topk]
        fi = np.where(backward_k_neigh_index==i)[0]
        return forward_k_neigh_index[fi]

    def forward(self, s_emb, t_emb, idx):
        if self.disable_mu:
            s_emb = F.normalize(s_emb)

        t_emb = F.normalize(t_emb)
        device = t_emb.device
        N = len(s_emb)        
        S_dist = torch.cdist(s_emb, s_emb)
        S_dist = S_dist / S_dist.mean(1, keepdim=True)
        
        with torch.no_grad():
            T_dist = torch.cdist(t_emb, t_emb) 
            W_P = torch.exp(-T_dist.pow(2) / self.sigma)
            
            batch_size = len(s_emb) // self.view
            W_P_copy = W_P.clone()
            W_P_copy[idx.unsqueeze(1) == idx.unsqueeze(1).t()] = 1
            dim = np.shape(W_P_copy)[0]
            if dim > self.topk:
                topk = self.topk
            else:
                topk = dim
            topk_index = torch.topk(W_P_copy, topk)[1]
            topk_half_index = topk_index[:, :int(np.around(self.topk/2))]

            W_NN = torch.zeros_like(W_P).scatter_(1, topk_index, torch.ones_like(W_P))
            V = ((W_NN + W_NN.t())/2 == 1).float()

            W_C_tilda = torch.zeros_like(W_P)
            # vars = []
            for i in range(N):
                indNonzero = torch.where(V[i, :]!=0)[0]
                # ind_zero = torch.where(V[i, :]==0)[0]
                # vars.append(torch.var(T_dist[i,ind_zero]))
                # vars.append(torch.var(S_dist1[i, ind_zero]))
                W_C_tilda[i, indNonzero] = (V[:,indNonzero].sum(1) / len(indNonzero))[indNonzero]

            # vars = torch.stack(vars)
            # not_NN_var = torch.mean(vars)
            W_C_hat = W_C_tilda[topk_half_index].mean(1)
            W_C = (W_C_hat + W_C_hat.t())/2
            W = (W_P + W_C)/2

            identity_matrix = torch.eye(N).to(device)
            pos_weight = (W) * (1 - identity_matrix)
            neg_weight = (1 - W) * (1 - identity_matrix)
        
        pull_losses = torch.relu(S_dist).pow(2) * pos_weight
        push_losses = torch.relu(self.delta - S_dist).pow(2) * neg_weight
        
        loss = (pull_losses.sum() + push_losses.sum()) / (len(s_emb) * (len(s_emb)-1))
        
        return loss
    
class KL_STML(nn.Module):
    def __init__(self, disable_mu, temp=1):
        super(KL_STML, self).__init__()
        self.disable_mu = disable_mu
        self.temp = temp
    
    def kl_div(self, A, B, T = 1):
        log_q = F.log_softmax(A/T, dim=-1)
        p = F.softmax(B/T, dim=-1)
        kl_d = F.kl_div(log_q, p, reduction='sum') * T**2 / A.size(0)
        return kl_d

    def forward(self, s_f, s_g):
        if self.disable_mu:
            s_f, s_g = F.normalize(s_f), F.normalize(s_g)

        N = len(s_f)
        S_dist = torch.cdist(s_f, s_f)
        S_dist = S_dist / S_dist.mean(1, keepdim=True)
        
        S_bg_dist = torch.cdist(s_g, s_g)
        S_bg_dist = S_bg_dist / S_bg_dist.mean(1, keepdim=True)
        
        loss = self.kl_div(-S_dist, -S_bg_dist.detach(), T=1)
        
        return loss
    
class STML_loss(nn.Module):
    def __init__(self, sigma, delta, view, disable_mu, topk):
        super(STML_loss, self).__init__()
        self.sigma = sigma
        self.delta = delta
        self.view = view
        self.disable_mu = disable_mu
        self.topk = topk
        self.RC_criterion = RC_STML(sigma, delta, view, disable_mu, topk)
        self.KL_criterion = KL_STML(disable_mu, temp=1)

    def forward(self, s_f, s_g, t_g, idx):
        # Relaxed contrastive loss for STML
        loss_RC_f = self.RC_criterion(s_f, t_g, idx)
        loss_RC_g = self.RC_criterion(s_g, t_g, idx)
        loss_RC = (loss_RC_f + loss_RC_g)/2
        
        # Self-Distillation for STML
        loss_KL = self.KL_criterion(s_f, s_g)
        
        loss = loss_RC + loss_KL
        
        total_loss = dict(RC=loss_RC, KL=loss_KL, loss=loss)
        
        return total_loss


class STML_GCL_loss(nn.Module):
    def __init__(self, sigma, delta, view, disable_mu,topk, center_file,cluster_num = 400,logger=None):
        super(STML_GCL_loss, self).__init__()
        self.sigma = sigma
        self.delta = delta
        self.view = view
        self.disable_mu = disable_mu
        self.topk = topk
        self.RC_criterion = RC_STML(sigma, delta, view, self.disable_mu, topk)
        self.KL_criterion = KL_STML(self.disable_mu, temp=1)
        self.logger = logger
        print(center_file)
        with open(center_file, "rb") as f:
            center = pickle.load(f)
        self.center = torch.from_numpy(center).cuda()
        # self.center = torch.from_numpy(center).cuda(1)


    def forward(self, s_f_groups, s_g_groups, t_g_groups, s_idxs,iteration):
        # Relaxed contrastive loss for STML
        loss_RC_all = 0
        loss_KL_all = 0
        loss_KL_groups_all = []
        device = s_f_groups[0].device
        for i,(s_f,t_g,s_g,s_idx) in enumerate(zip(s_f_groups,t_g_groups,s_g_groups, s_idxs)):
            if s_idx.size()[0] == 0:
                continue
            # s_idx = torch.tensor(np.array(s_idx)).squeeze().cuda(non_blocking=True)
            if (s_idx.device.type == 'cpu'):
                s_idx = s_idx.to(device)
            j = i
            while(j > 0):
                half_num = int(np.shape(s_idxs[j - 1])[0] / 2)
                last_group_sample_num = min(int(len(s_idxs[j-1])/10 + 1),half_num)
                last_group_idx = torch.cat([s_idxs[j-1][:last_group_sample_num],
                                            s_idxs[j-1][half_num : half_num + last_group_sample_num]])
                s_idx = torch.cat([s_idx,last_group_idx.to(device)])
                # s_idx = torch.cat([s_idx, last_group_idx])
                j -= 1
            s_f_cur = s_f[s_idx]
            t_g_cur = t_g[s_idx]
            s_g_cur = s_g[s_idx]
            loss_RC_f = self.RC_criterion(s_f_cur, t_g_cur, s_idx)
            loss_RC_g = self.RC_criterion(s_g_cur, t_g_cur, s_idx)
            loss_RC = (loss_RC_f + loss_RC_g) / 2
            loss_RC_all +=loss_RC

            # Self-Distillation for STML
            loss_KL = self.KL_criterion(s_f_cur, s_g_cur)
            loss_KL_all += loss_KL

            # 组蒸馏
            if iteration > -1:
                loss_KL_groups = []
                for j in range(i):
                    num_batch = int(len(s_idxs[j])/10+1)
                    s_f_prev = s_f_groups[j][s_idxs[j][:num_batch]]  # 之前组的特征
                    s_f_cur_prev = s_f_groups[i][s_idxs[j][:num_batch]]  # 当前embedding提取的之前组的特征

                    dist_emb = s_f_prev.pow(2).sum(1) + (-2) * self.center.mm(s_f_prev.t())
                    s_f_logits = self.center.pow(2).sum(1) + dist_emb.t()
                    s_f_logits = torch.sqrt(s_f_logits)
                    dist_emb = s_f_cur_prev.pow(2).sum(1) + (-2) * self.center.mm(s_f_cur_prev.t())
                    s_f_cur_logits = self.center.pow(2).sum(1) + dist_emb.t()
                    s_f_cur_logits = torch.sqrt(s_f_cur_logits)

                    # loss_kl_group = self.KL_criterion(s_f_cur_prev, s_f_prev)#KL_divergence(s_f_prev, s_f_cur_prev)
                    loss_kl_group = KL_divergence(s_f_logits, s_f_cur_logits)

                    loss_KL_groups.append(loss_kl_group)
                if len(loss_KL_groups) != 0:
                    loss_KL_groups_all.append(torch.mean(torch.stack(loss_KL_groups)))



        loss_KL_final = loss_KL_all / 4
        loss_RC_final = loss_RC_all / 4
        if iteration > -1:
            loss_KL_groups_final = torch.mean(torch.stack(loss_KL_groups_all))
            loss = loss_RC_final + loss_KL_final + loss_KL_groups_final
        else:
            loss_KL_groups_final = 0
            loss = loss_RC_final + loss_KL_final

        total_loss = dict(RC=loss_RC_final.cpu(), KL=loss_KL_final.cpu(), KL_groups =loss_KL_groups_final.cpu() ,loss=loss)
        return total_loss


def KL_divergence(logits_p, logits_q, reduce=True):
    # p = softmax(logits_p)
    # q = softmax(logits_q)
    # KL(p||q)
    # suppose that p/q is in shape of [bs, num_classes]
    device = logits_q.device
    p = F.softmax(logits_p, dim=1)
    q = F.softmax(logits_q, dim=1)

    shape = list(p.size())
    _shape = list(q.size())
    assert shape == _shape
    # print(shape)
    # num_classes = shape[1]
    epsilon = 1e-8
    # _p = (p + epsilon * Variable(torch.ones(*shape).cuda())) / (1.0 + num_classes * epsilon)
    # _q = (q + epsilon * Variable(torch.ones(*shape).cuda())) / (1.0 + num_classes * epsilon)
    _p = (p + epsilon * Variable(torch.ones(*shape).to(device))) / (1.0 + epsilon)
    _q = (q + epsilon * Variable(torch.ones(*shape).to(device))) / (1.0 + epsilon)
    if reduce:
        return torch.mean(torch.sum(_p * torch.log(_p / _q), 1))
    else:
        return torch.sum(_p * torch.log(_p / _q), 1)