'''
Reference:
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
'''
import math
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from timm.models.layers.weight_init import trunc_normal_


class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return {'logits': F.linear(input, self.weight, self.bias)}


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}


class SplitCosineLinear(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, nb_proxy=1, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = (out_features1 + out_features2) * nb_proxy
        self.nb_proxy = nb_proxy
        self.fc1 = CosineLinear(in_features, out_features1, nb_proxy, False, False)
        self.fc2 = CosineLinear(in_features, out_features2, nb_proxy, False, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        out = torch.cat((out1['logits'], out2['logits']), dim=1)  # concatenate along the channel

        # Reduce_proxy
        out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {
            'old_scores': reduce_proxies(out1['logits'], self.nb_proxy),
            'new_scores': reduce_proxies(out2['logits'], self.nb_proxy),
            'logits': out
        }


class EaseCosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(EaseCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)
    
    def reset_parameters_to_zero(self):
        self.weight.data.fill_(0)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}
    
    def forward_reweight(self, input, cur_task, alpha=0.1, beta=0.0, init_cls=10, inc=10, out_dim=768, use_init_ptm=False):
        for i in range(cur_task + 1):
            if i == 0:
                start_cls = 0
                end_cls = init_cls
            else:
                start_cls = init_cls + (i - 1) * inc
                end_cls = start_cls + inc
            
            out = 0.0
            for j in range((self.in_features // out_dim)):
                # PTM feature
                if use_init_ptm and j == 0:
                    input_ptm = F.normalize(input[:, 0:out_dim], p=2, dim=1)
                    weight_ptm = F.normalize(self.weight[start_cls:end_cls, 0:out_dim], p=2, dim=1)
                    out_ptm = beta * F.linear(input_ptm, weight_ptm)
                    out += out_ptm
                    continue

                input1 = F.normalize(input[:, j*out_dim:(j+1)*out_dim], p=2, dim=1)
                weight1 = F.normalize(self.weight[start_cls:end_cls, j*out_dim:(j+1)*out_dim], p=2, dim=1)
                if use_init_ptm:
                    if j != (i+1):
                        out1 = alpha * F.linear(input1, weight1)
                        out1 /= cur_task
                    else:
                        out1 = F.linear(input1, weight1)
                else:
                    if j != i:
                        out1 = alpha * F.linear(input1, weight1)
                        out1 /= cur_task
                    else:
                        out1 = F.linear(input1, weight1)

                out += out1
            
            if i == 0:
                out_all = out
            else:
                out_all = torch.cat((out_all, out), dim=1) if i != 0 else out
                
        if self.to_reduce:
            # Reduce_proxy
            out_all = reduce_proxies(out_all, self.nb_proxy)

        if self.sigma is not None:
            out_all = self.sigma * out_all
        
        return {'logits': out_all}


def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)


class SimpleContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes, feat_expand=False, with_norm=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.feat_expand = feat_expand
        self.with_norm = with_norm
        heads = []
        single_head = []
        if with_norm:
            single_head.append(nn.LayerNorm(embed_dim))

        single_head.append(nn.Linear(embed_dim, nb_classes, bias=True))
        head = nn.Sequential(*single_head)

        heads.append(head)
        self.heads = nn.ModuleList(heads)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02) 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) 

    def backup(self):
        self.old_state_dict = deepcopy(self.state_dict())

    def recall(self):
        self.load_state_dict(self.old_state_dict)

    def update(self, nb_classes, freeze_old=True):
        single_head = []
        if self.with_norm:
            single_head.append(nn.LayerNorm(self.embed_dim))
            
        _fc = nn.Linear(self.embed_dim, nb_classes, bias=True)
        trunc_normal_(_fc.weight, std=.02)
        nn.init.constant_(_fc.bias, 0) 
        single_head.append(_fc)
        new_head = nn.Sequential(*single_head)

        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad=False

        self.heads.append(new_head)

    def forward(self, x):
        out = []
        for ti in range(len(self.heads)):
            fc_inp = x[ti] if self.feat_expand else x
            out.append(self.heads[ti](fc_inp))
        out = {'logits': torch.cat(out, dim=1)}
        return out

# 路由器层
class LinearRouter(nn.Module):
    def __init__(self, embed_dim, depth):
        super(LinearRouter, self).__init__()
        self.Router = nn.Linear(embed_dim, depth)
        self.depth = depth
        self.embed_dim = embed_dim
        # 可训练的高斯噪声参数
        self.noise_mean = nn.Parameter(torch.zeros(1))            # 初始化为 0
        self.noise_std = nn.Parameter(torch.ones(1) * 0.1)         # 初始化为较小扰动
        self.register_buffer('score', torch.zeros(depth))

    def get_running_score(self):
        """获取当前的 EMA 平滑层重要性得分（用于可视化）"""
        return self.score.detach().cpu().numpy()
    
    def forward(self, x_embed):
        gate_logits = self.Router(x_embed)
        noise = torch.randn_like(gate_logits) * self.noise_std + self.noise_mean
        gate_logits = gate_logits + noise
        gate = F.softmax(gate_logits, dim=-1)        # [B, num_layers]
        # gate_avg = gate.mean(dim=0)                  # [num_layers]
        with torch.no_grad():
            batch_score = gate.mean(dim=0)
            self.score.copy_(batch_score)
        return gate


class KlLinearRouter(nn.Module):
    def __init__(self, embed_dim, depth):
        super(KlLinearRouter, self).__init__()
        self.Router = nn.Linear(embed_dim, depth)
        self.depth = depth
        # 可训练的高斯噪声参数
        self.noise_mean = nn.Parameter(torch.zeros(1))            
        self.noise_std = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x_embed, train=False):
        """
        Args:
            x_embed: [B, input_dim] image-level representation
        Returns:
            gate: [B, num_layers], soft importance for each layer
            gate_avg: [num_layers], average importance across batch
            kl_loss: scalar, load-balancing loss encouraging uniformity
        """
        gate_logits = self.Router(x_embed)

        if train:
            noise = torch.randn_like(gate_logits) * self.noise_std + self.noise_mean
            gate_logits = gate_logits + noise

        gate = F.softmax(gate_logits, dim=-1)        # [B, num_layers]
        gate_avg = gate.mean(dim=0)                  # [num_layers]

        # 计算 KL 散度，目标是均匀分布
        uniform = torch.full_like(gate_avg, 1.0 / self.depth)
        kl_loss = F.kl_div(gate_avg.log(), uniform, reduction='batchmean')

        return gate, gate_avg, kl_loss

class KlLinearRouter(nn.Module):
    def __init__(self, embed_dim, depth):
        super(KlLinearRouter, self).__init__()
        self.Router = nn.Linear(embed_dim, depth)
        self.embed_dim = embed_dim
        self.depth = depth
        # 可训练的高斯噪声参数
        self.noise_mean = nn.Parameter(torch.zeros(1))            
        self.noise_std = nn.Parameter(torch.ones(1) * 0.1)
        # 路由任务状态存储
        self.task_router_dict = {}

    def save_task_state(self, task_id):
        self.task_router_dict[task_id] = {
            'router_weight': self.Router.state_dict(),
        }
        self.Router = nn.Linear(self.embed_dim, self.depth)

    def forward_train(self, x_embed):
        gate_logits = self.Router(x_embed)
        noise = torch.randn_like(gate_logits) * self.noise_std + self.noise_mean
        gate_logits = gate_logits + noise
        gate = F.softmax(gate_logits, dim=-1)        # [B, num_layers]
        gate_avg = gate.mean(dim=0)                  # [num_layers]
        # 计算 KL 散度，目标是均匀分布
        uniform = torch.full_like(gate_avg, 1.0 / self.depth)
        kl_loss = F.kl_div(gate_avg.log(), uniform, reduction='batchmean')

        return gate, gate_avg, kl_loss
    

    def forward_test(self, x_embed, taskids):
        B = x_embed.size(0)
        device = x_embed.device
        gate_out = torch.zeros(B, self.depth, device=device)

        
        # 处理各种 taskids 类型为 List[int]
        if isinstance(taskids, torch.Tensor):
            taskids = taskids.view(-1).tolist()
        elif isinstance(taskids, list) and isinstance(taskids[0], (list, torch.Tensor)):
            taskids = [int(t[0]) if isinstance(t, (list, tuple)) else int(t.item()) for t in taskids]
        else:
            taskids = [int(t) for t in taskids]

        unique_task_ids = set(taskids)

        for task_id in unique_task_ids:
            # 找出属于当前任务的索引
            indices = [i for i, tid in enumerate(taskids) if tid == task_id]
            x_sub = x_embed[indices]  # [B_sub, D]

            # 构造一个新的 Router，载入参数
            temp_router = nn.Linear(x_embed.size(1), self.depth).to(device)
            temp_router.load_state_dict(self.task_router_dict[task_id]['router_weight'])

            with torch.no_grad():
                gate_logits = temp_router(x_sub)  # [B_sub, depth]
                gate = F.softmax(gate_logits, dim=-1)
                gate_out[indices] = gate

        return gate_out

    def forward(self, x_embed, train=False, taskids=None):
        if train:
            gate, gate_avg, kl_loss = self.forward_train(x_embed)
            return gate, gate_avg, kl_loss
        else:
            gate = self.forward_test(x_embed, taskids)
            return gate

class UniformLinearRouter(nn.Module):
    def __init__(self, embed_dim, depth, ema_decay=0.9):
        super(UniformLinearRouter, self).__init__()
        self.Router = nn.Linear(embed_dim, depth)
        self.depth = depth
        self.embed_dim = embed_dim
        self.ema_decay = ema_decay
        self.noise_mean = nn.Parameter(torch.zeros(1))            
        self.noise_std = nn.Parameter(torch.ones(1) * 0.1)
        self.task_router_dict = {}
        self.register_buffer('score_ema', torch.zeros(depth))  # [L]
        self.ema_initialized = False

    def save_task_state(self, task_id):
        self.task_router_dict[task_id] = {
            'router_weight': self.Router.state_dict(),
        }
        self.Router = nn.Linear(self.embed_dim, self.depth).to(self.noise_std.device)

    def get_running_score(self):
        """获取当前的 EMA 平滑层重要性得分（用于可视化）"""
        return self.score_ema.detach().cpu().numpy()
    
    def forward_train(self, x_embed):
        gate_logits = self.Router(x_embed)
        noise = torch.randn_like(gate_logits) * self.noise_std + self.noise_mean
        gate_logits = gate_logits + noise

        gate = F.softmax(gate_logits, dim=-1)        # [B, num_layers]
        gate_avg = gate.mean(dim=0)                  # [num_layers]

        # 直接用均方误差约束为均匀分布
        uniform = torch.full_like(gate_avg, 1.0 / self.depth)
        uniform_loss = F.mse_loss(gate_avg, uniform)

        with torch.no_grad():
            batch_score = gate.mean(dim=0)  # [L]
            if not self.ema_initialized:
                self.score_ema.copy_(batch_score)
                self.ema_initialized = True
            else:
                self.score_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * batch_score)

        return gate, uniform_loss

    def forward_test(self, x_embed, task_ids):
        B = x_embed.size(0)
        device = x_embed.device
        gate_out = torch.zeros(B, self.depth, device=device)

        
        # 处理各种 taskids 类型为 List[int]
        if isinstance(task_ids, torch.Tensor):
            task_ids = task_ids.view(-1).tolist()
        elif isinstance(task_ids, list) and isinstance(task_ids[0], (list, torch.Tensor)):
            task_ids = [int(t[0]) if isinstance(t, (list, tuple)) else int(t.item()) for t in task_ids]
        else:
            task_ids = [int(t) for t in task_ids]

        unique_task_ids = set(task_ids)

        for task_id in unique_task_ids:
            # 找出属于当前任务的索引
            indices = [i for i, tid in enumerate(task_ids) if tid == task_id]
            x_sub = x_embed[indices]  # [B_sub, D]

            # 构造一个新的 Router，载入参数
            temp_router = nn.Linear(x_embed.size(1), self.depth).to(device)
            temp_router.load_state_dict(self.task_router_dict[task_id]['router_weight'])

            with torch.no_grad():
                gate_logits = temp_router(x_sub)  # [B_sub, depth]
                gate = F.softmax(gate_logits, dim=-1)
                gate_out[indices] = gate

        return gate_out

    def forward(self, x_embed, task_ids=None, train=False):
        if train:
            gate, uniform_loss = self.forward_train(x_embed)
            return gate, uniform_loss
        else:
            gate = self.forward_test(x_embed, task_ids)
            return gate

# class RandomLinearRouter(nn.Module):
#     def __init__(self, embed_dim, depth):
#         super(RandomLinearRouter, self).__init__()
#         self.Router = nn.Linear(embed_dim, depth)
#         self.depth = depth  # 保存层数
#         # 可训练的高斯噪声参数（保留以备后用）
#         self.noise_mean = nn.Parameter(torch.zeros(1))           
#         self.noise_std = nn.Parameter(torch.ones(1) * 0.1)       

#     def forward(self, x_embed, train=False):
#         """
#         Args:
#             x_embed: [B, input_dim] image-level representation
#         Returns:
#             gate: [B, num_layers], one-hot (or multi-hot) gate vector
#             gate_avg: [num_layers], mean over batch
#         """
#         B = x_embed.size(0)
#         gate = torch.zeros(B, self.depth, device=x_embed.device)

#         for i in range(B):
#             topk_indices = torch.randperm(self.depth)[:3]  # 随机选择3个不重复层
#             gate[i, topk_indices] = 1.0  # 将这三个位置设为1（multi-one-hot）

#         gate_avg = gate.mean(dim=0)  # 用于记录每层的使用频率
#         return gate, gate_avg
    

class RandomLinearRouter(nn.Module):
    def __init__(self, embed_dim, depth, selected_layers=[4,5,6]):
        """
        Args:
            embed_dim: 输入特征维度
            depth: 总层数（例如 ViT 有 12 层）
            selected_layers: List[int]，你想激活的层，比如 [2, 5, 10]
        """
        super(RandomLinearRouter, self).__init__()
        self.depth = depth
        self.selected_layers = selected_layers

        # 构造静态 one-hot 层权重（multi-hot）
        gate_vec = torch.zeros(depth)
        gate_vec[selected_layers] = 1.0
        self.register_buffer('gate_template', gate_vec.view(1, -1))  # shape [1, depth]

    def forward(self, x_embed, train=False):
        """
        Args:
            x_embed: [B, input_dim]，但这里我们不使用 x_embed
        Returns:
            gate: [B, depth]，每个样本都使用固定的 gate
            gate_avg: [depth]，等于 gate[0]，因为每个样本都一样
        """
        B = x_embed.size(0)
        gate = self.gate_template.expand(B, -1)  # shape: [B, depth]
        gate_avg = self.gate_template.squeeze(0)
        return gate, gate_avg
    


class AttnLayerRouter(nn.Module):
    def __init__(self, embed_dim, depth, ema_decay=0.9):
        super(AttnLayerRouter, self).__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        self.ema_decay = ema_decay

        self.layer_tokens = nn.Parameter(torch.randn(depth, embed_dim))
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.proto_proj = nn.Linear(embed_dim, embed_dim)
        self.noise_mean = nn.Parameter(torch.zeros(1))
        self.noise_std = nn.Parameter(torch.ones(1) * 0.1)
        self.register_buffer('score_ema', torch.zeros(depth))  # [L]
        self.ema_initialized = False
    
    def get_running_score(self):
        """获取当前的 EMA 平滑层重要性得分（用于可视化）"""
        return self.score_ema.detach().cpu().numpy()
    
    def forward(self, x_embed, task_proto, train=False):
    # def forward(self, x_embed, train=False):
        B, D = x_embed.shape
        device = x_embed.device

        # 投影
        Q_x = self.query_proj(x_embed)           # [B, D]
        Q_p = self.proto_proj(task_proto)        # [B, D]

        Q = Q_x + Q_p                            # [B, D]，融合样本 & 任务语义
        Q = Q.unsqueeze(1)                       # [B, 1, D]

        # Key 是 layer tokens 投影后结果
        K = self.key_proj(self.layer_tokens)     # [L, D]
        K = K.unsqueeze(0).expand(B, -1, -1)     # [B, L, D]

        # Attention logits
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)).squeeze(1)  # [B, L]
                # # 加噪声（仅在训练时）
        if train:
            noise = torch.randn_like(attn_logits) * self.noise_std + self.noise_mean
            attn_logits = attn_logits + noise
        gate = F.softmax(attn_logits, dim=-1)  # [B, L]

        # === EMA 平滑：score_ema ← α * ema + (1 - α) * batch_mean ===
        with torch.no_grad():
            batch_score = gate.mean(dim=0)  # [L]
            if not self.ema_initialized:
                self.score_ema.copy_(batch_score)
                self.ema_initialized = True
            else:
                self.score_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * batch_score)
        

        return gate
    

