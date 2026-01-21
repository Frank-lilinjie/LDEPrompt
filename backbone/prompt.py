import torch
import torch.nn as nn
import copy
from torch.nn import functional as F
import random
from typing import Optional

class Task_MultiPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 task_prompt_key=False, task_pool_size=None, task_topk=None, batchwise_prompt=False, prompt_key_init='uniform', 
                 num_layers=1, use_prefix_tune_for_prompt=False, num_heads=-1, same_key_value=False):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.num_layers = num_layers
        self.num_heads=num_heads
        self.task_topk = task_topk
        self.embedding_key = embedding_key
        self.batchwise_prompt = batchwise_prompt
        self.task_prompt_key = task_prompt_key
        self.prompt_key_init = prompt_key_init
        self.task_pool_size = task_pool_size
        self.use_prefix_tune_for_prompt = use_prefix_tune_for_prompt
        self.prompt_init = prompt_init
        self.same_key_value = same_key_value
        self.pools = nn.ModuleList()
        self.add_prompt()

    def add_prompt(self):
        prompt = Prompt_Simple(length=self.length,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                prompt_init=self.prompt_init,
                use_prefix_tune_for_prompt=self.use_prefix_tune_for_prompt,
                same_key_value=self.same_key_value,
                prompt_key_init=self.prompt_key_init
                                      )
        self.pools.append(prompt)
        


class ADA_Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', 
                prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform', 
                num_layers=1, use_prefix_tune_for_prompt=False, num_heads=-1, same_key_value=False):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads=num_heads
        self.top_k = top_k
        self.embedding_key = embedding_key
        self.batchwise_prompt = batchwise_prompt
        self.prompt_key = prompt_key
        self.prompt_key_init = prompt_key_init
        self.pool_size = pool_size
        self.use_prefix_tune_for_prompt = use_prefix_tune_for_prompt
        self.prompt_init = prompt_init
        self.same_key_value = same_key_value
        self.train_layer_idx = [] # 这里存单次训练需要添加的 prompt 的 layer_idx
        self.prompt_layer_idx = [] # 这里存所有添加过 prompt 的 layer_idx
        self.placeholder_idx = []
        self.prompt_train_pools =  nn.ModuleDict()
        self.prompt_all_pools = nn.ModuleDict()# 
        self.place_holder = nn.ModuleDict()

    def update_placeholder(self, layer_idx, num_placeholder):
        for i in layer_idx:
            layer_key = str(i)

            # 如果该层已经存在占位符，则跳过
            if i not in self.placeholder_idx:
                self.placeholder_idx.append(i)

            # 否则新增 num_placeholder 个占位符
            placeholder_list = nn.ModuleList()
            for _ in range(num_placeholder):
                placeholder_list.append(Placeholder())

            # 注册到 self.place_holder 中
            self.place_holder[layer_key] = placeholder_list



    # 更新训练 prompt
    def update_train_prompt_layer(self, layer_idx, old_prompt_ids, add_num):

        layer_key = str(layer_idx)

        if layer_idx not in self.train_layer_idx:
            self.train_layer_idx.append(layer_idx)
        reused_prompt_list = nn.ModuleList()

        # ✅ 尝试从已有 pool 中复用旧 prompt
        if layer_key in self.prompt_all_pools:
            old_prompt_list = self.prompt_all_pools[layer_key]
            assert isinstance(old_prompt_list, nn.ModuleList)
            
            for idx in old_prompt_ids:
                reused_prompt = copy.deepcopy(old_prompt_list[idx])
                reused_prompt_list.append(reused_prompt)

        # ✅ 新增若干 Prompt_Simple（按 add_num）
        for _ in range(add_num):
            new_prompt = Prompt_Simple(
                length=self.length,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                prompt_init=self.prompt_init,
                use_prefix_tune_for_prompt=self.use_prefix_tune_for_prompt,
                same_key_value=self.same_key_value,
                use_prompt_key=self.prompt_key,
                prompt_key_init=self.prompt_key_init
            )

            reused_prompt_list.append(new_prompt)

        # ✅ 更新 train pool（List[Prompt_Simple]）
        self.prompt_train_pools[layer_key] = reused_prompt_list

    
    # 更新所有的 prompt
    def update_all_prompt_pools(self, selected_blocks, add_nums):
        """
        将训练结束的 prompt 合并到 prompt_all_pools 中。

        Args:
            selected_blocks (List[int]): 本轮训练的层索引
            add_nums (List[int]): 对应每层新添加的数量
        """
        assert len(selected_blocks) == len(add_nums), "selected_blocks 和 add_nums 长度不一致"
        
        # 添加index 到 self.prompt_layer_idx
        for i in selected_blocks:
            if i not in self.prompt_layer_idx:
                self.prompt_layer_idx.append(i)

        for i, layer_idx in enumerate(selected_blocks):
            layer_key = str(layer_idx)
            new_prompts = self.prompt_train_pools[layer_key]  # List[Prompt_Simple]
            add_num = add_nums[i]

            # 仅保留新添加的尾部 prompt
            new_added_prompts = new_prompts[-add_num:]
            
            if layer_key in self.prompt_all_pools:
                self.prompt_all_pools[layer_key].extend(new_added_prompts)
            else:
                self.prompt_all_pools[layer_key] = new_added_prompts

            # 只冻结本轮新加的 prompt
            for prompt in new_prompts:
                for param in prompt.parameters():
                    param.requires_grad = False
        

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True) # 计算L2范数
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device))) # 使用反平方根进行归一化
        return x * x_inv_norm
    
    def forward_train(self, x_embed, layer_idx, cls_features):
        out = dict()

        # 1. 获取样本嵌入
        if self.embedding_key == 'mean':
            x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'max':
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'cls':
            x_embed_mean = cls_features if cls_features is not None else torch.max(x_embed, dim=1)[0]
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")

        # 2. 获取该层的 prompt pool
        prompt_pool = self.prompt_train_pools[layer_idx]
        prompt_keys = torch.cat([p.prompt_key for p in prompt_pool], dim=0)  # [prompt_num, C]

        # 3. 计算相似度
        prompt_key_norm = self.l2_normalize(prompt_keys, dim=1).to(x_embed.device)  # [P, C]
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)                       # [B, C]
        similarity = torch.matmul(x_embed_norm, prompt_key_norm.T)                  # [B, P]
        out['similarity'] = similarity

        # 4. Top-K prompt 选择
        topk_values, topk_idx = torch.topk(similarity, k=self.top_k, dim=1)  # [B, top_k]

        # 如果 batchwise 共享 prompt：选出全局 top-K
        if self.batchwise_prompt:
            prompt_id, id_counts = torch.unique(topk_idx, return_counts=True, sorted=True)
            if prompt_id.shape[0] < self.pool_size:
                fill_n = self.pool_size - prompt_id.shape[0]
                pad_ids = torch.full((fill_n,), torch.min(topk_idx.flatten()), device=topk_idx.device)
                prompt_id = torch.cat([prompt_id, pad_ids])
                id_counts = torch.cat([id_counts, torch.zeros(fill_n, device=topk_idx.device)])
            _, major_idx = torch.topk(id_counts, k=self.top_k)
            major_prompt_id = prompt_id[major_idx]
            topk_idx = major_prompt_id.expand(x_embed.shape[0], -1)  # [B, top_k]

        out['prompt_idx'] = topk_idx

        # 5. 构造 prompt batch（无占位逻辑）
        if self.use_prefix_tune_for_prompt:
            B, top_k = topk_idx.shape
            prompt_list = []

            for b in range(B):
                prompt_per_sample = []
                for k in range(top_k):
                    p_idx = topk_idx[b, k].item()
                    prompt_tensor = prompt_pool[p_idx].prompt.to(x_embed.device)
                    prompt_per_sample.append(prompt_tensor.unsqueeze(0))
                # 拼成 [top_k, dual, prompt_len, num_heads, head_dim]
                prompt_per_sample = torch.cat(prompt_per_sample, dim=0)
                prompt_list.append(prompt_per_sample.unsqueeze(0))  # [1, top_k, ...]

            batched_prompt_raw = torch.cat(prompt_list, dim=0)  # [B, top_k, dual, prompt_len, num_heads, head_dim]

            # 变换维度：[dual, B, top_k * prompt_len, num_heads, head_dim]
            B, top_k, num_layers, dual, prompt_len, num_heads, head_dim = batched_prompt_raw.shape
            batched_prompt_raw = batched_prompt_raw.reshape(
                num_layers, B, dual, top_k * prompt_len, num_heads, head_dim
            )  # [dual, B, L, H, D]
        else:
            raise NotImplementedError("Only prefix-tuning is supported currently.")

        # 6. 返回结果
        batched_key_norm = prompt_key_norm[topk_idx]  # [B, top_k, dim]
        batched_prompt = batched_prompt_raw
        out['selected_key'] = batched_key_norm
        out['prompt_key_norm'] = prompt_key_norm
        out['x_embed_norm'] = x_embed_norm
        out['batched_prompt'] = batched_prompt

        return out

    def forward_test(self, x_embed, layer_idx, cls_features):
        out = dict()

        # 1. 获取样本嵌入
        if self.embedding_key == 'mean':
            x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'max':
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'cls':
            if cls_features is None:
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            else:
                x_embed_mean = cls_features
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")

        # 2. 获取该层的 prompt pool 和 placeholder
        prompt_pool = self.prompt_all_pools[layer_idx]

        placeholder_pool = self.place_holder[layer_idx]

        prompt_keys = torch.cat([p.prompt_key for p in prompt_pool], dim=0)           # [prompt_num, C]
        prompt_keys_length = len(prompt_keys)
        placeholder_keys = torch.cat([p.placeholder for p in placeholder_pool], dim=0) # [placeholder_num, C]

        # 拼接 key：prompt key + 占位符 key
        all_keys = torch.cat([prompt_keys, placeholder_keys], dim=0)                  # [total_num, C]

        # 3. 计算相似度
        prompt_key_norm = self.l2_normalize(all_keys, dim=1).to(x_embed.device)  # [total, C]
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)                    # [B, C]
        similarity = torch.matmul(x_embed_norm, prompt_key_norm.T)               # [B, total]
        out['similarity'] = similarity

        # 4. 判断是否跳过 prompt（如果 top-1 是 placeholder）
        top1_value, top1_idx = torch.topk(similarity, k=1, dim=1)                # [B, 1]
        need_prompt_mask = top1_idx.squeeze(1) < prompt_keys_length              # [B]，True 表示需要添加 prompt

        # 为所有样本计算 Top-K，只从 prompt_key 中取（排除 placeholder）
        prompt_similarity = similarity[:, :prompt_keys_length]                  # [B, prompt_num]
        topk_values, topk_idx = torch.topk(prompt_similarity, k=self.top_k, dim=1)  # [B, top_k]

        # 如果 batchwise prompt：选出全局 top-K
        if self.batchwise_prompt:
            prompt_id, id_counts = torch.unique(topk_idx, return_counts=True, sorted=True)
            if prompt_id.shape[0] < self.pool_size:
                prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(topk_idx.flatten()), device=prompt_id.device)])
                id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
            _, major_idx = torch.topk(id_counts, k=self.top_k)
            major_prompt_id = prompt_id[major_idx]
            topk_idx = major_prompt_id.expand(x_embed.shape[0], -1)  # [B, top_k]

        out['prompt_idx'] = topk_idx

        # 5. 仅对需要添加 prompt 的样本组装 prompt
        if self.use_prefix_tune_for_prompt:
            B, top_k = topk_idx.shape
            prompt_list = []
            for b in range(B):
                if need_prompt_mask[b]:
                    prompt_per_sample = []
                    for k in range(top_k):
                        p_idx = topk_idx[b, k].item()
                        prompt_tensor = prompt_pool[p_idx].prompt.to(x_embed.device)
                        prompt_per_sample.append(prompt_tensor.unsqueeze(0))
                    prompt_per_sample = torch.cat(prompt_per_sample, dim=0)  # [top_k, dual, prompt_len, num_heads, head_dim]
                    prompt_list.append(prompt_per_sample.unsqueeze(0))       # [1, top_k, ...]
                else:
                    # 若不需要 prompt，则占位（全 0），稍后外部 forward 判断是否启用
                    dummy = torch.zeros((1, top_k, *prompt_pool[0].prompt.shape), device=x_embed.device)
                    prompt_list.append(dummy)

            batched_prompt_raw = torch.cat(prompt_list, dim=0)  # [B, top_k, dual, prompt_len, num_heads, head_dim]

            # 转换维度：[num_layers, B, dual, top_k * prompt_len, num_heads, head_dim]
            B, top_k, num_layers, dual, prompt_len, num_heads, head_dim = batched_prompt_raw.shape
            # batched_prompt_raw = batched_prompt_raw.permute(2, 0, 1, 3, 4, 5)  # [dual, B, top_k, prompt_len, num_heads, head_dim]
            batched_prompt_raw = batched_prompt_raw.reshape(
                num_layers, B, dual, top_k * prompt_len, num_heads, head_dim
            )

        batched_prompt = batched_prompt_raw
        batched_key_norm = prompt_key_norm[topk_idx]  # [B, top_k, dim]

        out['selected_key'] = batched_key_norm
        out['prompt_key_norm'] = prompt_key_norm
        out['x_embed_norm'] = x_embed_norm
        out['batched_prompt'] = batched_prompt
        out['need_prompt_mask'] = need_prompt_mask  # 加上这个字段，方便外部使用

        return out
    
    def forward(self, x_embed, layer_idx, test=False, cls_features=None):
        if test:
            out = self.forward_test(x_embed, layer_idx, cls_features)
        else:
            out = self.forward_train(x_embed, layer_idx, cls_features)
        return out


class Prompt_Simple(nn.Module):
    def __init__(self, length = 5, embed_dim = 768, 
                 num_heads = -1, prompt_init = 'uniform',
                 use_prefix_tune_for_prompt = False, same_key_value = False, 
                 use_prompt_key = True, prompt_key_init = 'uniform'):
        """
        Simple prompt module that holds prompt and optional key.

        Args:
            num_prompts: Number of prompts.
            prompt_length: Length of each prompt (token-wise).
            embed_dim: Embedding dimension of each token.
            use_key: Whether to initialize key for each prompt.
        """
        super().__init__()
        self.use_prefix_tune_for_prompt = use_prefix_tune_for_prompt
        # 初始化 prompt
        if use_prefix_tune_for_prompt:
            assert embed_dim % num_heads == 0
            # key 和 value 使用同一份参数
            if same_key_value:
                prompt_pool_shape = (1, 1, length, num_heads, embed_dim // num_heads)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
            else:
                prompt_pool_shape = (1, 2, length, num_heads, embed_dim // num_heads)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
        else:
            prompt_pool_shape = (1, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
        
        # 初始化key
        if use_prompt_key:
            key_shape = (1, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape)) # 随机的key
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def forward_features(self):
        # Flatten 所有 prompt token
        if self.use_prefix_tune_for_prompt:
            return self.prompt[0,0,:,:,:]
        else:
            return self.prompt.mean(dim=0)
        
class Placeholder(nn.Module):
    def __init__(self):
        super().__init__()
        self.placeholder = nn.Parameter(torch.randn(1, 768))