from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import tiktoken
from typing import Union, List, Optional, Tuple
import random
import os

class GPTDatasetV1(Dataset):
    """
    PyTorch Dataset for GPT-style next-token prediction.
    
    - Uses tiktoken ("r50k_base") for tokenization (GPT-2 vocabulary).
    - Concatenates multiple documents into a single sequence with EOT token.
    - Splits the token sequence into fixed-length blocks (default 1024).
    - Each sample is a pair (input_ids, target_ids) where:
        - input_ids: token indices of the input sequence.
        - target_ids: are input_ids shifted by one postion.
    - No padding, no attention masks (matches GPT-2 training setup).
    - Optionally shuffles block order each epoch for better training variability.
    
    Args:
        txt (str | List[str]): input text. Can be single string or list of strings.
        enc (tiktoken.Encoding | None): defaults to "r50k_base" if not provided.
        max_length (int): max length of each input sequence (default 1024).
        stride (int | None): step size between blocks Defeaults to non-overlapping.
        eot_id (int): end-of-text token ID (default 50256 for GPT-2).
        shuffle_blocks (bool): whether to shuffle blocks each epoch (default True).
        seed (int): random seed for reproducibility of shuffling
        
    """
    def __init__(
        self, 
        txt: Union[str, List[str]], # input single string or list of strings
        enc: Optional[tiktoken.Encoding] = None, # optional: pass own or use default
        max_length: int = 1024, # GPT-2 default block size
        stride: Optional[int] = None, # non-overlapping by default
        eot_id: int = 50256, # GPT-2 EOT between docs
        shuffle_blocks: bool = True, # shuffle blocks each epoch
        seed: int = 42
    ):

        self.block_size = max_length
        self.stride = self.block_size if stride is None else stride
        self._rng = random.Random(seed)
        self.shuffle_blocks = shuffle_blocks

        # Build encoding if not provided
        self.enc = enc or tiktoken.get_encoding("r50k_base")
        
        # Normalize input: always treat as list of docs
        if isinstance(txt, str):
            txt = [txt] 
        else:
            txt = list(txt)
        
        # Encode all docs, inserting EOT token between them
        token_ids: List[int] = []
        for i, doc in enumerate(txt):
            token_ids.extend(self.enc.encode(doc))
            if i < len(txt) - 1:
                token_ids.append(eot_id)
        
        # Precompute fixed windows; no padding; drop tails        
        self.input_ids = []
        self.target_ids = []
        
        # Calculate valid starting indices for blocks
        # max_start is the last index where a full block can start
        max_start = (len(token_ids) - self.block_size - 1)
        
        # Calculate starting indices for all full blocks
        starts = [] if max_start < 0 else list(range(0, max_start + 1, self.stride))
        
        # Shuffle block orders if specified
        if self.shuffle_blocks:
            self._rng.shuffle(starts)
            
        # Generate input-target pairs for each block
        for i in starts:
            x = token_ids[i:i + self.block_size]
            y = token_ids[i + 1:i + self.block_size + 1]
            # dtype=torch.long as nn.Embedding and CrossEntropyLoss require long indices 
            self.input_ids.append(torch.tensor(x, dtype=torch.long))
            self.target_ids.append(torch.tensor(y, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(
    txt: Union[str, List[str]],
    *, # everything after this is a keyword-only argument
    batch_size: int = 8, 
    block_size: int = 1024, # GPT-2 block length
    stride: Optional[int] = None, # default: non-overlapping == block_size
    eot_id: int = 50256, # GPT-2 end-of-text token
    shuffle_blocks: bool = True, # shuffle dataset blocks each epoch
    seed: int = 42,
    num_workers: Optional[int] = None,
    drop_last: bool = True,
    prefetch_factor: Optional[int] = None, # only used when num_workers > 0
) -> Tuple[DataLoader, "GPTDatasetV1", str]:
    """
    Build a DataLoader for GPT-style next-token training.
    
    Design choices:
    - Use tiktoken 'r50k_base' (GPT-2/3 vocab) and keep shuffling at the dataset level.
    - Auto-detect device: CUDA if available; otherwise CPU.
    - Enable pinned memory on CUDA for faster host-> device copies.
    - Choose a reasonable default worker count
    - Return (loader, dataset, device)  
    """
    
    # Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Encoding: GPT-2-compatible byte-level BPE tokenizer
    enc = tiktoken.get_encoding("r50k_base")
    
    # Create the dataset: single source of shuffling is here
    dataset = GPTDatasetV1(
        txt=txt,
        max_length=block_size,
        stride=stride,
        eot_id=eot_id,
        shuffle_blocks=shuffle_blocks,
        seed=seed,
    )
    
    # Reproducibility
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    
    # Create a per-worker seed so that workers don't produce the same data
    def _seed_worker(worker_id: int):
        # Make worker RNG deterministic
        worker_seed = seed + worker_id
        torch.manual_seed(worker_seed)
        
    # Performance knobs tuned
    # if user didn't specify num_workers, pick modest worker count
    if num_workers is None:
        cpu_count = os.cpu_count() or 2
        num_workers = max(0, min(4, cpu_count // 2))
    
    # Enable pinned memory for faster (and asynchronous) transfers to CUDA 
    pin_memory = (device == "cuda")
    
    # Avoid overhaad of terminating workers after each epoch, i.e. keep them alive
    use_persistent = (num_workers > 0)
    
    # Control how many batches are prefetched ahead of training loop
    effective_prefetch = prefetch_factor if num_workers > 0 else None
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,            # dataset handles block shuffling
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=effective_prefetch,
        worker_init_fn=(_seed_worker if num_workers > 0 else None),
        generator=gen,
    )
    
    return loader, dataset, device