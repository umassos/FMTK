import torch

class EmbeddingCache:
    """
    Dynamic cache for backbone embeddings keyed by dataset index.
    Size and embed_dim are inferred from the first put(); supports batched get/put.

    #TODO: 1. The cache is currently saved on the CPU, we need logic to save it on disk
    # and fetch it if it gets too large.
    """
    def __init__(self, cache_device = "cpu", to_device = "cpu"):
        self._cache = {}
        self._cache_device = cache_device
        self._to_device = to_device

    def __len__(self):
        return len(self._cache)

    def put(self, idx, value):
        value = value.detach()
        for i in range(len(idx)):
            self._cache[idx[i].item()] = value[i].to(self._cache_device)

    def contains(self, idx):
        return all([k.item() in self._cache for k in idx])

    def get(self, idx, device = None):
        device = device or self._to_device
        return torch.stack([self._cache[k.item()].to(device) for k in idx], dim=0)

    def save(self, path):
        # TODO: Very inefficient way to save the cache,
        # A better way is to store sharded files for the cache.
        torch.save({"cache": self._cache}, path)

    def load(self, path, map_location = None):
        data = torch.load(path, map_location=map_location or str(self._cache_device))
        self._cache = data["cache"]

    def __str__(self):
        intro = f"EmbeddingCache(cache_size={len(self._cache)})"
        intro += f"\nCache: {list(self._cache.keys())}"
        return intro

    def __repr__(self):
        return self.__str__()
