import time

import torch
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.datasets import InferenceDataset
from utils.setup_distributed import setup_distributed_gloo


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.net = torchvision.models.resnet18(pretrained=False)

    def __call__(self, input_tensor):
        return self.net(input_tensor)


class Worker():

    def __init__(self, world_size, local_rank):
        self.world_size = world_size
        self.local_rank = local_rank

        # DDP or not DDP is nothing to do with this worker
        self.model = DDP(Model().eval())
        self.model = Model().eval()

    @torch.no_grad()
    def inference(self, input_paths):

        inference_dataset = InferenceDataset(input_paths)
        sampler = DistributedSampler(dataset=inference_dataset, shuffle=False)

        dataloader = DataLoader(inference_dataset,
                                batch_size=1,
                                sampler=sampler,
                                num_workers=1)

        if self.local_rank == 0:
            pbar = tqdm(total=len(dataloader),
                        desc=f"Inferencing (Rank {self.local_rank})")

        for item in dataloader:
            input_tensor, path = item
            try:
                self.model(input_tensor)
                # TODO:

            except Exception as e:
                print(e)

            if self.local_rank == 0:
                pbar.update(1)

        return


if __name__ == "__main__":

    local_rank, world_size = setup_distributed_gloo(backend="gloo", port=None)

    if local_rank == 0:
        t1 = time.time()

    worker = Worker(world_size, local_rank)
    input_paths = [f"{i}.jpg" for i in range(1000)]
    worker.inference(input_paths)

    if local_rank == 0:
        t2 = time.time()
        print(f"Inferencing time: {t2 - t1}")
