import torch
import torch.nn as nn
from torch.nn import functional as F
import ipdb
try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

import logging

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)
    # logging.info(f"gather_features done done done done done done {all_image_features.shape}, {all_text_features.shape}")
    return all_image_features, all_text_features

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            moe_head=3,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.moe_head = moe_head 
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels                

    def gather_logits(self, x, return_both=True, return_text=False, device=None):
        # extract max on diagonals and the corresponding column
        _,n,m = x.shape #batch size
        if n != m:
            # local-loss situation
            # logging.info(f"n = {n}, m = {m}")
            assert device is not None
            min_n = min(n, m)
            ground_truth = self.get_ground_truth(device, min_n)
            # logging.info("ground_truth shape = ", ground_truth.shape)
            # logging.info("ground_truth = ", ground_truth)
            if not return_text:
                new_x = x[:,:,ground_truth]
                diagonals = new_x[:, torch.arange(min_n),torch.arange(min_n)]
                max_indices = diagonals.argmax(dim=0)
                expanded_x = x.permute(1, 2, 0)  # reshape to (n, m, 3)
                expanded_indices = max_indices.expand(m,n)
                selected_columns_image = torch.gather(expanded_x, 2, expanded_indices.T.unsqueeze(2)).squeeze(2)
                return selected_columns_image
            else:
                new_x = x[:,ground_truth,:]
                diagonals = new_x[:, torch.arange(min_n),torch.arange(min_n)]
                max_indices = diagonals.argmax(dim=0)
                expanded_x = x.permute(1, 2, 0)
                expanded_indices = max_indices.expand(n,m)
                selected_columns_text = torch.gather(expanded_x, 2, expanded_indices.unsqueeze(2)).squeeze(2)
                selected_columns_text = selected_columns_text.T
                return selected_columns_text
        else:
            diagonals = x[:, torch.arange(n), torch.arange(n)]
            max_indices = diagonals.argmax(dim=0)
            expanded_x = x.permute(1, 2, 0)  # reshape to (n, n, 3) 
            expanded_indices = max_indices.expand(n, n)
            if return_both:
                selected_columns_image = torch.gather(expanded_x, 2, expanded_indices.T.unsqueeze(2)).squeeze(2)
                selected_columns_text = torch.gather(expanded_x, 2, expanded_indices.unsqueeze(2)).squeeze(2)
                selected_columns_text = selected_columns_text.T        
                return selected_columns_image, selected_columns_text
            else:
                if return_text:
                    selected_columns_text = torch.gather(expanded_x, 2, expanded_indices.unsqueeze(2)).squeeze(2)
                    selected_columns_text = selected_columns_text.T
                    return selected_columns_text
                else:
                    selected_columns_image = torch.gather(expanded_x, 2, expanded_indices.T.unsqueeze(2)).squeeze(2)
                    return selected_columns_image

    def logits_assign_hard(self, image_feature, text_feature, logit_scale, return_both=True, return_text=False, device=None):
        # image_feature: (moe_head, batch_size, embed_dim)
        # text_feature: (num_caption, batch_size, embed_dim)
        # return_both and return_text can not all be True at the same time
        logits_per_image_list = []
        logits_per_text_list = []
        logits_per_image_list_full = []
        logits_per_text_list_full = []
        if image_feature.shape[0] == text_feature.shape[0]:
            for i in range(image_feature.shape[0]):
                logits = logit_scale * image_feature[i] @ text_feature[i].T
                if return_both:
                    text_feature_concat = torch.cat((text_feature[i:i+1],text_feature[:i],text_feature[i+1:]),dim=0)
                    text_feature_concat = text_feature_concat.reshape(-1, text_feature.shape[-1])
                    logits_image = logit_scale * image_feature[i] @ text_feature_concat.T                    
                    logits_per_image_list.append(logits)
                    logits_per_image_list_full.append(logits_image)
                    
                    image_feature_concat = torch.cat((image_feature[i:i+1],image_feature[:i],image_feature[i+1:]),dim=0)
                    image_feature_concat = image_feature_concat.reshape(-1, image_feature.shape[-1])
                    logits_text = logit_scale * text_feature[i] @ image_feature_concat.T             
                    logits_per_text_list.append(logits.T)
                    logits_per_text_list_full.append(logits_text)                    
                elif return_text:
                    image_feature_concat = torch.cat((image_feature[i:i+1],image_feature[:i],image_feature[i+1:]),dim=0)
                    image_feature_concat = image_feature_concat.reshape(-1, image_feature.shape[-1])
                    logits_text = logit_scale * text_feature[i] @ image_feature_concat.T             
                    logits_per_text_list.append(logits.T)
                    logits_per_text_list_full.append(logits_text)
                else:
                    text_feature_concat = torch.cat((text_feature[i:i+1],text_feature[:i],text_feature[i+1:]),dim=0)
                    text_feature_concat = text_feature_concat.reshape(-1, text_feature.shape[-1])
                    logits_image = logit_scale * image_feature[i] @ text_feature_concat.T                    
                    logits_per_image_list.append(logits)
                    logits_per_image_list_full.append(logits_image)
            if return_both:
                return torch.stack(logits_per_image_list), torch.stack(logits_per_text_list), torch.stack(logits_per_image_list_full), torch.stack(logits_per_text_list_full)
            elif return_text:
                return torch.stack(logits_per_text_list), torch.stack(logits_per_text_list_full)
            return torch.stack(logits_per_image_list), torch.stack(logits_per_image_list_full)
        elif image_feature.shape[0] < text_feature.shape[0]:
            for i in range(text_feature.shape[0]):
                if i < image_feature.shape[0]:
                    logits = logit_scale * image_feature[i] @ text_feature[i].T
                    if return_both:
                        logits_per_image_list.append(logits)
                        logits_per_text_list.append(logits.T)
                    elif return_text:
                        logits_per_text_list.append(logits.T)
                    else:
                        logits_per_image_list.append(logits)
                else:
                    logits = logit_scale * image_feature @ text_feature[i].T.unsqueeze(0)
                    if return_both:
                        logit_image, logit_text = self.gather_logits(logits, return_both=return_both, return_text=return_text, device=device)
                        logits_per_image_list.append(logit_image)
                        logits_per_text_list.append(logit_text)
                    elif return_text:
                        logits_per_text_list.append(self.gather_logits(logits, return_both=False, return_text=True, device=device))
                    else:
                        logits_per_image_list.append(self.gather_logits(logits, return_both=False, device=device))
            if return_both:
                return torch.stack(logits_per_image_list), torch.stack(logits_per_text_list), None, None
            elif return_text:
                return torch.stack(logits_per_text_list), None
            return torch.stack(logits_per_image_list), None   
        else: # image_feature.shape[0] > text_feature.shape[0]
            for i in range(text_feature.shape[0]):
                logits = logit_scale * image_feature @ text_feature[i].T.unsqueeze(0)
                if return_both:
                    logit_image, logit_text = self.gather_logits(logits, return_both=return_both, return_text=return_text, device=device)
                    logits_per_image_list.append(logit_image)
                    logits_per_text_list.append(logit_text)
                elif return_text:
                    logits_per_text_list.append(self.gather_logits(logits, return_both=False, return_text=True, device=device))
                else:
                    logits_per_image_list.append(self.gather_logits(logits, return_both=False, device=device))
            if return_both:
                return torch.stack(logits_per_image_list), torch.stack(logits_per_text_list), None, None
            elif return_text:
                return torch.stack(logits_per_text_list), None
            return torch.stack(logits_per_image_list), None


    def get_logits(self, image_features, text_features, logit_scale, device = None):
        logits_per_image_full = None
        logits_per_text_full = None
        if len(text_features.shape) == 2:  # one caption
            image_features = image_features.reshape(-1, self.moe_head, image_features.shape[-1]) # (batch_size, moe_head, embed_dim)
            if self.world_size > 1:
                all_image_features, all_text_features = gather_features(
                    image_features, text_features,
                    self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
                if self.local_loss:
                    image_features = image_features.transpose(0,1) # (moe_head, batch_size, embed_dim)
                    all_image_features = all_image_features.transpose(0,1) # (moe_head, batch_size, embed_dim)
                    logits_per_image = logit_scale * image_features @ all_text_features.T.unsqueeze(0)
                    logits_per_image = self.gather_logits(logits_per_image, return_both=False, device=device)
                    logits_per_text = logit_scale * all_image_features @ text_features.T.unsqueeze(0)
                    logits_per_text = self.gather_logits(logits_per_text, return_both=False, return_text=True, device=device)
                else:
                    image_features = image_features.transpose(0,1) # (moe_head, batch_size, embed_dim)
                    all_image_features = all_image_features.transpose(0,1) # (moe_head, batch_size, embed_dim)
                    logits_per_image = logit_scale * all_image_features @ all_text_features.T.unsqueeze(0)
                    logits_per_image, logits_per_text = self.gather_logits(logits_per_image, device=device)
            else:
                image_features = image_features.transpose(0,1) # (moe_head, batch_size, embed_dim)
                logits_per_image = logit_scale * image_features @ text_features.T.unsqueeze(0)
                logits_per_image, logits_per_text = self.gather_logits(logits_per_image, device=device)
            return logits_per_image, logits_per_text, logits_per_image_full, logits_per_text_full
        else: #multiple captions
            image_features = image_features.reshape(-1, self.moe_head, image_features.shape[-1]) # (batch_size, moe_head, embed_dim)
            if self.world_size > 1:
                all_image_features, all_text_features = gather_features(
                    image_features, text_features,
                    self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
                if self.local_loss:
                    image_features = image_features.transpose(0,1) # (moe_head, batch_size, embed_dim)
                    text_features = text_features.transpose(0,1) # (num_caption, batch_size, embed_dim)
                    all_image_features = all_image_features.transpose(0,1) # (moe_head, batch_size, embed_dim)
                    all_text_features = all_text_features.transpose(0,1) # (num_caption, batch_size, embed_dim)
                    logits_per_image, logits_per_image_full = self.logits_assign_hard(image_features, all_text_features, logit_scale=logit_scale, return_both=False, device=device)
                    logits_per_text, logits_per_text_full = self.logits_assign_hard(all_image_features, text_features, logit_scale=logit_scale, return_both=False, return_text=True, device=device)
                else:
                    image_features = image_features.transpose(0,1) # (moe_head, batch_size, embed_dim)
                    text_features = text_features.transpose(0,1) # (num_caption, batch_size, embed_dim)
                    all_image_features = all_image_features.transpose(0,1) # (moe_head, batch_size, embed_dim)
                    all_text_features = all_text_features.transpose(0,1) # (num_caption, batch_size, embed_dim)
                    logits_per_image, logits_per_text, logits_per_image_full, logits_per_text_full = self.logits_assign_hard(all_image_features, all_text_features, logit_scale=logit_scale, return_both=True, device=device)
            else:
                image_features = image_features.transpose(0,1) # (moe_head, batch_size, embed_dim)
                text_features = text_features.transpose(0,1) # (num_caption, batch_size, embed_dim)
                logits_per_image, logits_per_text, logits_per_image_full, logits_per_text_full = self.logits_assign_hard(image_features, text_features, logit_scale=logit_scale, return_both=True, device=device)
            return logits_per_image, logits_per_text, logits_per_image_full, logits_per_text_full # tensor stack

    def forward(self, image_features, text_features, logit_scale, output_dict=False, epoch=10):
        device = image_features.device
        logits_per_image, logits_per_text, logits_per_image_full, logits_per_text_full = self.get_logits(image_features, text_features, logit_scale, device=device)

        if len(logits_per_image.shape) == 2: # one caption
            labels = self.get_ground_truth(device, logits_per_image.shape[0]) #batch size
            contrastive_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2
        else: # multiple captions
            batch_size = logits_per_image.shape[-2]
            all_batch_size = logits_per_image.shape[-1]
            labels = self.get_ground_truth(device, batch_size) #batch size
            labels = labels.unsqueeze(0).expand(logits_per_image.shape[0], -1)
            labels = labels.reshape(-1)
            logits_per_image = logits_per_image.reshape(-1, logits_per_image.shape[-1])
            logits_per_text = logits_per_text.reshape(-1, logits_per_text.shape[-1])
            logits_per_image_full = None #not use image，text contrast
            if logits_per_image_full is None:
                contrastive_loss = (
                    F.cross_entropy(logits_per_image, labels) +
                    F.cross_entropy(logits_per_text, labels)
                ) / 2
            else:
                logits_per_image_full = logits_per_image_full.reshape(-1, logits_per_image_full.shape[-1])
                logits_per_text_full = logits_per_text_full.reshape(-1, logits_per_text_full.shape[-1])
                label_list = [labels + i*all_batch_size for i in range(self.moe_head)]

                contrastive_loss = (
                    0.99 * F.cross_entropy(logits_per_image, labels) +
                    0.99 * F.cross_entropy(logits_per_text, labels)) / 2
                for i in range(self.moe_head):
                    contrastive_loss += (0.01 * F.cross_entropy(logits_per_image_full, label_list[i]) + 0.01 * F.cross_entropy(logits_per_text_full, label_list[i]))/2
    
        rank_loss = torch.tensor(0.0,device=device)
        if self.moe_head >1 and epoch < 200:
            image_features = image_features.reshape(-1, self.moe_head, image_features.shape[-1])
            moe_sim_matrix = image_features @ image_features.transpose(1,2) # (bs, moe_head, moe_head)
            moe_sim_matrix = torch.mean(moe_sim_matrix, dim=0)
            diag_mask = torch.eye(moe_sim_matrix.size(0), dtype=torch.bool,device=moe_sim_matrix.device)
            non_diag_mask = ~diag_mask
            non_diag_elements = moe_sim_matrix[non_diag_mask]
            mean_non_diag = non_diag_elements.mean() 
            rank_loss = max(mean_non_diag-torch.tensor(0.7,device=mean_non_diag.device),torch.tensor(0.0,device=mean_non_diag.device))

        return {"contrastive_loss": contrastive_loss,"rank_loss": rank_loss} if output_dict else contrastive_loss+rank_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss
