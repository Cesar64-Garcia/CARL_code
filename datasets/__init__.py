import torch

def construct_dataloader(cfg, split, mode="auto", fileName = ""):
    assert split in ["train", "val", "test"]
    if split == "train":
        from datasets.penn_action import PennAction, ActionBatchSampler
        train_dataset = PennAction(cfg, split, mode="train", fileName=fileName)
        if not cfg.SSL and "tcc" in cfg.TRAINING_ALGO:
            train_sampler = ActionBatchSampler(cfg, train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE)
            # for indices in batch_sampler:
            #     yield collate_fn([dataset[i] for i in indices])
            train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=cfg.DATA.NUM_WORKERS, pin_memory=True, 
                                        batch_sampler=train_sampler)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if cfg.NUM_GPUS > 1 else None
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, 
                                                    shuffle=True if train_sampler is None else False,
                                                    num_workers=cfg.DATA.NUM_WORKERS, pin_memory=True, sampler=train_sampler,
                                                    drop_last=True)
        train_eval_loader = []
        for dataset_name in cfg.DATASETS:
            train_eval_dataset = PennAction(cfg, split, dataset_name, mode="eval", sample_all=True, fileName=fileName)
            train_eval_loader.append(torch.utils.data.DataLoader(train_eval_dataset, batch_size=1, shuffle=False,
                                                        num_workers=0, pin_memory=True, sampler=None))
        return train_loader, train_eval_loader

    elif split == "val":
        from datasets.penn_action import PennAction, ActionBatchSampler
        val_dataset = PennAction(cfg, split, mode="eval", fileName=fileName)
        if not cfg.SSL and "tcc" in cfg.TRAINING_ALGO:
            val_sampler = ActionBatchSampler(cfg, val_dataset, batch_size=cfg.EVAL.BATCH_SIZE)
            val_loader = torch.utils.data.DataLoader(val_dataset,num_workers=cfg.DATA.NUM_WORKERS, pin_memory=True, 
                                        batch_sampler=val_sampler)
        else:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if cfg.NUM_GPUS > 1 else None
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.EVAL.BATCH_SIZE, shuffle=False,
                                                    num_workers=cfg.DATA.NUM_WORKERS, pin_memory=True, sampler=None,
                                                    drop_last=True)
        val_eval_loader = []
        for dataset_name in cfg.DATASETS:
            val_eval_dataset = PennAction(cfg, split, dataset_name, mode="eval", sample_all=True, fileName=fileName)
            val_eval_loader.append(torch.utils.data.DataLoader(val_eval_dataset, batch_size=1, shuffle=False,
                                            num_workers=0, pin_memory=True, sampler=None))
        return val_loader, val_eval_loader

def unnorm(images, mean=[0.485, 0.456, 0.406], stddev=[0.229, 0.224, 0.225], raw=False):
    """
    Perform color unnomration on the given images.
    Args:
        images (tensor): images to perform color unnormalization. Dimension is
            `num frames` x `channel` x `height` x `width`.
        mean (list): mean values for normalization.
        stddev (list): standard deviations for normalization.
    Returns:
        out_images (tensor): the unnoramlized images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    assert len(mean) == images.shape[1], "channel mean not computed properly"
    assert (
        len(stddev) == images.shape[1]
    ), "channel stddev not computed properly"

    out_images = torch.zeros_like(images)
    for idx in range(len(mean)):
        out_images[:, idx] = images[:, idx] * stddev[idx] + mean[idx]

    if raw:
        out_images = (255*out_images).to(torch.uint8).permute(0,2,3,1)

    return out_images