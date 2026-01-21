import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import LDEnet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from collections import defaultdict
import random
import math

from torch.distributions.multivariate_normal import MultivariateNormal
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
    
        self._network = LDEnet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.inc = args["increment"]
        self.pool_size = args["pool_size"]
        self.num_placeholder = args["num_placeholder"]
        self.ca_lr = args["ca_lr"]
        self.crct_epochs = args["crct_epochs"]
        self.cls_mean = dict()
        self.cls_cov = dict()
        self.cls2task = dict()


        # Freeze the parameters for ViT.
        if self.args["freeze"]:
            for p in self._network.original_backbone.parameters():
                p.requires_grad = False
        
            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in self._network.backbone.named_parameters():
                if n.startswith(tuple(self.args["freeze"])):
                    p.requires_grad = False
        
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

    def after_task(self):
        self._known_classes = self._total_classes

    def replace_indicator(self, train_loader):
        original_model = self._network.backbone
        with torch.no_grad():
            start_idx = 0
            embedding_list, label_list = [], []
            for i, batch in enumerate(train_loader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = original_model.forward_proto(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)
            class_list = np.unique(train_loader.dataset.labels)
            for class_index in class_list:
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                embedding = embedding_list[data_index]
                proto = embedding.mean(0)
                self._network.task_indicator.weight.data[class_index, :] = proto
        self._network.task_indicator.requires_grad = False

    def incremental_train(self, data_manager):
        self._network.to(self._device)
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_task_indicator(self._total_classes)
        
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        
        self.replace_indicator(self.train_loader_for_protonet)


        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.backbone.add_taskprompt_to_pool()
        
    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        selected_blocks = self.select_blocks_by_info_gain(train_loader)
        # selected_blocks = self.compute_block_sensitivity(train_loader)
        logging.info("Selected_blocks {}".format(selected_blocks))
        # 组成本次训练的prompt
        self.group_training_prompt(train_loader, selected_blocks, self.pool_size)
        self._network.backbone.adaprompt.update_placeholder(selected_blocks, self.num_placeholder)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        self._init_train(train_loader, test_loader, optimizer, scheduler)
        self._network.backbone.adaprompt.update_all_prompt_pools(selected_blocks, self.add_nums)

        self._compute_mean(self._network.backbone)
        if self._cur_task > 0:
            self.classifer_align(self._network.backbone)
        
    def get_optimizer(self):
        # 获取所有参与训练的参数，并打印参数名
        trainable_params = []
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                print(f"Trainable parameter: {name} | shape: {tuple(param.shape)}")
                trainable_params.append(param)

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                trainable_params, 
                momentum=0.9, 
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                trainable_params,
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                trainable_params,
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )
        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None
        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.to(self._device)
            self._network.backbone.train()
            losses = 0.0
            losses_clf, losses_div, losses_placeholder = 0.0, 0.0, 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                output = self._network(inputs)
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')

                loss_clf = F.cross_entropy(logits, targets.long())
                loss_div = self.prompt_diversity_loss(
                        self._network.backbone.adaprompt.prompt_train_pools, self.add_nums, device=self._device
                    )
                loss_placeholder = self.placeholder_orthogonality_loss(
                    self._network.backbone.adaprompt.prompt_train_pools, device=self._device
                )
                loss = loss_clf - self.args["diversity_loss_coeff"] * loss_div + self.args["placeholder_loss_coeff"] * loss_placeholder
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_div += loss_div.item()
                losses_placeholder += loss_placeholder.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)


            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_div {:.3f}, Loss_plc {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                losses_clf / len(train_loader),
                -losses_div / len(train_loader),
                losses_placeholder / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    @torch.no_grad()
    def _compute_mean(self, model):
        model.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            task_id = class_idx // self.inc
            self.cls2task[class_idx] = task_id

            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size*3, shuffle=False, num_workers=4
            )
            
            vectors = []
            for _, _inputs, _targets in idx_loader:
                batch_size = _inputs.size(0)
                taskids = [self._cur_task] * batch_size
                _vectors = model(_inputs.to(self._device), taskids = taskids)['features']
                vectors.append(_vectors)
            vectors = torch.cat(vectors, dim=0)

            if self.args["ca_storage_efficient_method"] == 'covariance':
                features_per_cls = vectors
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
                self.cls_cov[class_idx] = torch.cov(features_per_cls.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device)
            elif self.args["ca_storage_efficient_method"] == 'variance':
                features_per_cls = vectors
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
                self.cls_cov[class_idx] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device))
            elif self.args["ca_storage_efficient_method"] == 'multi-centroid':
                from sklearn.cluster import KMeans
                n_clusters = self.args["n_centroids"] # 10
                features_per_cls = vectors.cpu().numpy()
                kmeans = KMeans(n_clusters=n_clusters, n_init=10)
                kmeans.fit(features_per_cls)
                cluster_lables = kmeans.labels_
                cluster_means = []
                cluster_vars = []
                for i in range(n_clusters):
                    cluster_data = features_per_cls[cluster_lables == i]
                    cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(self._device)
                    cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(self._device)
                    cluster_means.append(cluster_mean)
                    cluster_vars.append(cluster_var)
                
                self.cls_mean[class_idx] = cluster_means
                self.cls_cov[class_idx] = cluster_vars

    def classifer_align(self, model):
        model.train()
        
        run_epochs = self.crct_epochs
        param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'lspromptPool' and 'task_id_embedding' not in n]
        network_params = [{'params': param_list, 'lr': self.ca_lr, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.ca_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        prog_bar = tqdm(range(run_epochs))
        for epoch in prog_bar:

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            if self.args["ca_storage_efficient_method"] in ['covariance', 'variance']:
                for class_idx in range(self._total_classes):
                    mean = self.cls_mean[class_idx].to(self._device)
                    cov = self.cls_cov[class_idx].to(self._device)
                    if self.args["ca_storage_efficient_method"] == 'variance':
                        cov = torch.diag(cov)
                    
                    # 检查是否包含 NaN 或 inf
                    if torch.isnan(mean).any() or torch.isinf(mean).any():
                        raise ValueError("mean contains NaN or Inf")
                    if torch.isnan(cov).any() or torch.isinf(cov).any():
                        raise ValueError("cov contains NaN or Inf")

                    # 添加小扰动以确保协方差矩阵正定
                    cov = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device)
                    
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)

                    sampled_label.extend([class_idx] * num_sampled_pcls)

            elif self.args["ca_storage_efficient_method"] == 'multi-centroid':
                for class_idx in range(self._total_classes):
                    for cluster in range(len(self.cls_mean[class_idx])):
                        mean = self.cls_mean[class_idx][cluster]
                        var = self.cls_cov[class_idx][cluster]
                        if var.mean() == 0:
                            continue
                        m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                        sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                        sampled_data.append(sampled_data_single)
                        sampled_label.extend([class_idx] * num_sampled_pcls)
            else:
                raise NotImplementedError


            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            if epoch == 0:
                print("sampled data shape: ", sampled_data.shape)

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            losses = 0.0
            correct, total = 0, 0
            for _iter in range(self._total_classes):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                outputs = model(inp, fc_only=True)
                logits = outputs['logits'][:, :self._total_classes]

                loss = F.cross_entropy(logits, tgt)
                
                _, preds = torch.max(logits, dim=1)
                
                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss

            scheduler.step()
            ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.crct_epochs,
                losses / self._total_classes,
                ca_acc,
            )
            prog_bar.set_description(info)
         
        logging.info(info)
    # # 方案 2：记录信息增益
    # @torch.no_grad()
    # def select_blocks_by_info_gain(self, train_loader):
    #     model = self._network.original_backbone.to(self._device)
    #     model.eval()
    #     num_blocks = len(model.blocks)
    #     total_info_gain = torch.zeros(num_blocks).to(self._device)
    #     total_count = 0

    #     for i, (_, inputs, targets) in enumerate(train_loader):
    #         inputs, targets = inputs.to(self._device), targets.to(self._device)
    #         x = model.patch_embed(inputs)
    #         if model.cls_token is not None:
    #             cls_token = model.cls_token.expand(x.shape[0], -1, -1)
    #             x = torch.cat((cls_token, x), dim=1)
    #         x = model.pos_drop(x + model.pos_embed)

    #         for i, block in enumerate(model.blocks):
    #             x_in = x.clone()
    #             # x_in_cls = x_in[:,0]
    #             x = block(x)
    #             x_out = x
    #             # x_out_cls = x_out[:,0]

    #             ent_in = entropy(x_in)
    #             ent_out = entropy(x_out)
    #             gain = torch.abs(ent_out - ent_in)
    #             total_info_gain[i] += gain.item() * x.shape[0]  # 加权总信息增益
    #         total_count += x.shape[0]

    #     avg_info_gain = total_info_gain / total_count

    #     # === Layer-wise Normalization ===
    #     mean = avg_info_gain.mean()
    #     std = avg_info_gain.std()
    #     normed_info_gain = (avg_info_gain - mean) / (std + 1e-6)

    #     # 打印原始信息增益
    #     print("\n[Info Gain per Block Layer]:")
    #     for i, gain in enumerate(avg_info_gain.tolist()):
    #         print(f"Block {i:02d}: Info Gain = {gain:.6f}")

    #     # 打印归一化后信息增益
    #     print("\n[Normalized Info Gain per Block Layer]:")
    #     for i, gain in enumerate(normed_info_gain.tolist()):
    #         print(f"Block {i:02d}: Normed Info Gain = {gain:.6f}")

    #     positive_info_gain_indices = (normed_info_gain > 0).nonzero(as_tuple=False).flatten().tolist()
    #     return positive_info_gain_indices

    def compute_block_sensitivity(self, data_loader):
        model = self._network.original_backbone.to(self._device)
        model.train()
        
        # 确保梯度计算开启
        for param in model.parameters():
            param.requires_grad = True

        num_blocks = len(model.blocks)
        
        # ❗只计算中间层（排除首尾）
        valid_block_indices = list(range(1, num_blocks - 1))
        sensitivity = {f'Block {i}': 0.0 for i in valid_block_indices}

        total_batches = 0

        for batch in data_loader:
            _, data, label = batch
            data, label = data.to(self._device), label.to(self._device)
            
            # 前向传播
            output = model(data)
            logits = output
            aux_targets = label.clone()
            aux_targets = torch.where(
                aux_targets - self._known_classes >= 0,
                aux_targets - self._known_classes,
                -1,
            )
            loss = F.cross_entropy(logits, aux_targets)
            
            self._network.zero_grad()
            loss.backward()

            # 计算中间层 block 的梯度敏感性
            for i in valid_block_indices:
                block = model.blocks[i]
                block_grad_norm = 0.0
                for param in block.parameters():
                    if param.grad is not None:
                        block_grad_norm += torch.norm(param.grad, p=2).item()
                sensitivity[f'Block {i}'] += block_grad_norm

            total_batches += 1

        avg_sensitivity = []
        for i in valid_block_indices:
            key = f'Block {i}'
            sensitivity[key] /= (total_batches + 1e-6)
            avg_sensitivity.append(sensitivity[key])

        global_avg = np.mean(avg_sensitivity)
        
        # ✅ 只选中中间层中高于平均值的 block
        selected_blocks = [
            int(k.split()[-1]) 
            for k, v in sensitivity.items() 
            if v > global_avg
        ]

        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

        # 打印结果（只打印中间层）
        print("Grad sensitivity (normalized):")
        for i in valid_block_indices:
            print(f"Block {i}: {sensitivity[f'Block {i}']:.6f}")
        print(f"Global average: {global_avg:.6f}")
        print(f"Selected blocks: {selected_blocks}")
        
        return selected_blocks

    
    @torch.no_grad()
    def select_blocks_by_info_gain(self, train_loader):
        model = self._network.original_backbone.to(self._device)
        model.eval()
        num_blocks = len(model.blocks)
        total_info_gain = torch.zeros(num_blocks).to(self._device)
        total_count = 0

        for _, inputs, targets in train_loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            x = model.patch_embed(inputs)
            if model.cls_token is not None:
                cls_token = model.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_token, x), dim=1)
            x = model.pos_drop(x + model.pos_embed)

            for i, block in enumerate(model.blocks):
                if i == 0 or i == num_blocks - 1:
                    x = block(x)  # 仍需前向传播以维持状态一致
                    continue

                x_in = x.clone()
                x = block(x)
                x_out = x

                ent_in = entropy(x_in)
                ent_out = entropy(x_out)
                gain = torch.abs(ent_out - ent_in)
                total_info_gain[i] += gain.item() * x.shape[0]
            total_count += x.shape[0]

        avg_info_gain = total_info_gain / total_count

        # === Layer-wise Normalization (Exclude first and last blocks) ===
        mid_indices = list(range(1, num_blocks - 1))
        mid_info_gain = avg_info_gain[mid_indices]

        mean = mid_info_gain.mean()
        std = mid_info_gain.std()
        normed_info_gain = (avg_info_gain - mean) / (std + 1e-6)

        # 打印原始信息增益（全部 block）
        logging.info("\n[Info Gain per Block Layer]:")
        for i, gain in enumerate(avg_info_gain.tolist()):
            logging.info(f"Block {i:02d}: Info Gain = {gain:.6f}")

        # 打印归一化后信息增益（仅中间 block）
        logging.info("\n[Normalized Info Gain per Block Layer (excluding Block 0 and last)]:")
        for i in range(1, num_blocks - 1):
            gain = normed_info_gain[i].item()
            logging.info(f"Block {i:02d}: Normed Info Gain = {gain:.6f}")

        # 选取正的归一化信息增益对应的层（剔除首尾层）
        selected_indices = (
            torch.nonzero(normed_info_gain > 0, as_tuple=True)[0]
            .tolist()
        )

        # ✅ 移除首尾 block
        selected_indices = [i for i in selected_indices if i != 0 and i != num_blocks - 1]

        return selected_indices
    
    # @torch.no_grad()
    # def select_blocks_by_info_gain(self, train_loader):
    #     model = self._network.original_backbone.to(self._device)
    #     model.eval()
    #     num_blocks = len(model.blocks)

    #     total_info_gain = torch.zeros(num_blocks).to(self._device)
    #     total_count = 0

    #     for _, inputs, targets in train_loader:
    #         inputs, targets = inputs.to(self._device), targets.to(self._device)
    #         x = model.patch_embed(inputs)
    #         if model.cls_token is not None:
    #             cls_token = model.cls_token.expand(x.shape[0], -1, -1)
    #             x = torch.cat((cls_token, x), dim=1)
    #         x = model.pos_drop(x + model.pos_embed)

    #         for i, block in enumerate(model.blocks):
    #             x_in = x.clone()
    #             x = block(x)
    #             x_out = x

    #             ent_in = entropy(x_in)
    #             ent_out = entropy(x_out)
    #             gain = torch.abs(ent_out - ent_in)
    #             total_info_gain[i] += gain.item() * x.shape[0]  # 加权求和
    #         total_count += x.shape[0]

    #     avg_info_gain = total_info_gain / total_count

    #     # === 打印原始信息增益 ===
    #     print("\n[Avg Info Gain per Block Layer]:")
    #     for i, gain in enumerate(avg_info_gain.tolist()):
    #         print(f"Block {i:02d}: Info Gain = {gain:.6f}")

    #     # === 去除首层和末层 ===
    #     mid_layers = list(range(1, num_blocks - 1))  # 去掉第0层和最后一层
    #     gain_group = [(i, avg_info_gain[i].item()) for i in mid_layers]
    #     best = max(gain_group, key=lambda x: x[1])[0]
    #     # # === 按层分组 ===
    #     # half = len(mid_layers) // 2
    #     # group1 = mid_layers[:half]     # 前一半层
    #     # group2 = mid_layers[half:]     # 后一半层

    #     # # === 分别选出每组中信息增益最大的层 ===
    #     # gain_group1 = [(i, avg_info_gain[i].item()) for i in group1]
    #     # gain_group2 = [(i, avg_info_gain[i].item()) for i in group2]

    #     # best1 = max(gain_group1, key=lambda x: x[1])[0]
    #     # best2 = max(gain_group2, key=lambda x: x[1])[0]

    #     # print(f"\n[Selected Block Layers]: {best1} (front half), {best2} (back half)")
    #     return [best]



    def group_training_prompt(self, data_loader, selected_blocks, pool_size=10):
        model = self._network.to(self._device)
        model.eval()
        self.add_nums = []
        # 初始化
        self._network.backbone.adaprompt.train_layer_idx = []
        self._network.backbone.adaprompt.prompt_train_pools =  nn.ModuleDict()
        with torch.no_grad():
            for layer_idx in selected_blocks:
                layer_key = str(layer_idx)
                selected_prompt_ids = []

                has_old_prompt = (
                    hasattr(self._network.backbone.adaprompt, "prompt_all_pools") and
                    layer_key in self._network.backbone.adaprompt.prompt_all_pools and
                    len(self._network.backbone.adaprompt.prompt_all_pools[layer_key]) > 0
                )

                if has_old_prompt:
                    all_prompts = self._network.backbone.adaprompt.prompt_all_pools[layer_key]

                    # 拼接所有 prompt_key 和 prompt
                    prompt_keys = torch.cat([p.prompt_key for p in all_prompts], dim=0)  # [P_total, key_dim]
                    prompts = torch.cat([p.prompt for p in all_prompts], dim=2)          # [B, L, P_total, H, D]

                    # 获取当前任务所有样本的特征
                    feat_list = []
                    for batch in data_loader:
                        _, data, label = batch
                        data, label = data.to(self._device), label.to(self._device)
                        layer_feat = self._network.backbone.get_layer_feature(data, layer_idx)  # [B, key_dim]
                        layer_feat = layer_feat[:, 0, :]  # 只取第一个 token
                        feat_list.append(layer_feat)

                    feat_all = torch.cat(feat_list, dim=0)  # [N, key_dim]
                    feat_all = F.normalize(feat_all, dim=-1)
                    norm_keys = F.normalize(prompt_keys, dim=-1)

                    sim_matrix = torch.matmul(feat_all, norm_keys.t())  # [N, P_total]
                    sim_mean = sim_matrix.mean(dim=0)  # [P_total]

                    # 只保留大于0的相似度再计算均值
                    positive_sim = sim_mean[sim_mean > 0]
                    if positive_sim.numel() > 0:
                        pos_mean = positive_sim.mean()
                        selected_idxs = (sim_mean > pos_mean).nonzero().squeeze(-1)
                    else:
                        # 没有正相似度时，保守处理：全部不选，或者任选一个（看你需求）
                        selected_idxs = torch.tensor([], dtype=torch.long, device=sim_mean.device)

                    # 限制最多选择 pool_size - 1 个旧 prompt
                    max_old_prompt = pool_size - 1
                    if selected_idxs.size(0) > max_old_prompt:
                        top_scores, top_ids = torch.topk(sim_mean[selected_idxs], k=max_old_prompt)
                        selected_idxs = selected_idxs[top_ids]

                    selected_prompt_ids = selected_idxs.tolist()

                # ✅ 动态计算新增 prompt 数量
                add_num = pool_size - len(selected_prompt_ids)
                assert add_num > 0, f"add_num <= 0! You selected too many old prompts for layer {layer_idx}"
                logging.info(f"[Layer {layer_idx}] Selected old prompt IDs: {selected_prompt_ids}, Add new: {add_num}")
                self.add_nums.append(add_num)

                # ✅ 更新训练池，传入 selected_prompt_ids（已经是全局索引）
                self._network.backbone.adaprompt.update_train_prompt_layer(
                    layer_idx=layer_idx,
                    old_prompt_ids=selected_prompt_ids,
                    add_num=add_num
                )

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, test=True)["logits"][:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs, test=True)["logits"][:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def prompt_diversity_loss(self, prompt_pools: nn.ModuleDict, add_nums, device):
        loss = 0.0
        num_pairs = 0

        # 当前任务用全部的 prompt 池，如果是首任务，fallback 到 train_pools
        if self._cur_task > 0:
            all_prompt_pools = self._network.backbone.adaprompt.prompt_all_pools
        else:
            all_prompt_pools = self._network.backbone.adaprompt.prompt_train_pools

        for i, (layer_key, train_prompt_list) in enumerate(prompt_pools.items()):
            add_num = add_nums[i]
            if add_num == 0:
                continue

            # 获取新添加的 prompt（无论后面对谁）
            new_prompts = train_prompt_list[-add_num:]

            # 判定当前层有没有可对比的全局 prompt
            if layer_key not in all_prompt_pools or len(all_prompt_pools[layer_key]) <= add_num:
                # 没有旧 prompt 或 prompt 数太少 => 自己和自己比
                old_prompts = new_prompts
            else:
                old_prompts = all_prompt_pools[layer_key]

            # 提取特征并 normalize
            new_features = [p.forward_features().to(device) for p in new_prompts]
            old_features = [p.forward_features().to(device) for p in old_prompts]

            new_features = F.normalize(torch.stack(new_features).view(len(new_features), -1), dim=1)
            old_features = F.normalize(torch.stack(old_features).view(len(old_features), -1), dim=1)

            # 相似度矩阵（越相似，loss 越大）
            sim_matrix = torch.matmul(new_features, old_features.T)
            sim_loss = sim_matrix.sum()
            loss += sim_loss
            num_pairs += sim_matrix.numel()

        if num_pairs == 0:
            return torch.tensor(0.0, device=device)

        return loss / num_pairs

    def placeholder_orthogonality_loss(self, prompt_pools: nn.ModuleDict, device, lambda_between_placeholders=0.1):
        total_loss = 0.0
        total_count = 0

        for layer_key, prompt_list in prompt_pools.items():
            # 如果该层没有占位符，跳过
            if layer_key not in self._network.backbone.adaprompt.place_holder:
                continue

            # 获取该层的 prompt keys（取每个 prompt 的 key 向量）
            keys = [p.prompt_key.to(device) for p in prompt_list]
            if len(keys) == 0:
                continue
            keys = torch.stack(keys)           # [N_prompt, 1, D]
            keys = keys.view(keys.size(0), -1) # [N_prompt, D]
            keys = F.normalize(keys, dim=1)    # 单位化处理

            # 获取占位符向量
            placeholders = [p.placeholder.to(device) for p in self._network.backbone.adaprompt.place_holder[layer_key]]
            placeholders = torch.stack(placeholders)           # [N_ph, 1, D]
            placeholders = placeholders.view(placeholders.size(0), -1)  # [N_ph, D]
            placeholders = F.normalize(placeholders, dim=1)

            # --- [1] 占位符和 keys 的正交性：希望 dot product 趋近于 0 ---
            sim_pk = torch.matmul(placeholders, keys.T)  # [N_ph, N_prompt]
            loss_pk = (sim_pk ** 2).mean()  # 越小越正交

            # --- [2] 占位符之间的距离（也希望它们彼此远离） ---
            if len(placeholders) > 1:
                sim_pp = torch.matmul(placeholders, placeholders.T)  # [N_ph, N_ph]
                mask = torch.triu(torch.ones_like(sim_pp), diagonal=1)  # 上三角
                sim_pp_offdiag = sim_pp * mask
                loss_pp = (sim_pp_offdiag ** 2).sum() / mask.sum()
            else:
                loss_pp = 0.0

            layer_loss = loss_pk + lambda_between_placeholders * loss_pp
            total_loss += layer_loss
            total_count += 1

        if total_count == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / total_count
    
def entropy(tensor, eps=1e-8):
    # tensor: [B, N, D]
    probs = F.softmax(tensor, dim=-1)  # 特征维度归一化
    log_probs = torch.log(probs + eps)
    ent = -torch.sum(probs * log_probs, dim=-1)  # [B, N]
    return ent.mean()