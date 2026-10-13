import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter


class HeadTieEmbedding(nn.Module):
    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    sparse: bool

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HeadTieEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight)

        self.sparse = sparse

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:

        if isinstance(input, torch.LongTensor) or isinstance(input, torch.cuda.LongTensor):
            return F.embedding(
                input, self.weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
        elif isinstance(input, torch.FloatTensor) or isinstance(input, torch.cuda.FloatTensor) or \
                isinstance(input, torch.cuda.BFloat16Tensor) or isinstance(input, torch.cuda.HalfTensor):
            return F.linear(input, self.weight, None)
        else:
            raise UserWarning("unknown inpute type in head tie embedding:", input.type())

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False):

        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        embedding.weight.requires_grad = not freeze
        return embedding


class TokenEmbedding(nn.Module):
    def __init__(self, vocab, model_dim, max_len, pos_embedding, rel_pos_enc, initializer_range):

        super().__init__()

        self.rel_pos_enc = rel_pos_enc
        self.pos_embedding = pos_embedding
        self.max_len = max_len

        pad_vocab_size_multiple = 8
        vocab = (math.ceil(vocab / pad_vocab_size_multiple) * pad_vocab_size_multiple)

        self.embed_seq = HeadTieEmbedding(vocab, model_dim)
        self.scale = nn.Parameter(
            torch.sqrt(torch.tensor([model_dim // 2], dtype=torch.float32)),
            requires_grad=False,
        )

        if pos_embedding:
            if rel_pos_enc:
                self.embed_pair_pos = nn.Linear(max_len, model_dim, bias=False)
            else:
                self.embed_pair_pos = nn.Linear(model_dim, model_dim, bias=False)

                pe = torch.zeros(max_len, model_dim, dtype=torch.float32)
                position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, model_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / model_dim))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                pe = torch.nn.Parameter(pe, requires_grad=False)
                self.register_buffer('pe', pe, persistent=False)

    def reset_non_trainable_state(self, device: torch.device | None = None) -> None:
        """Rebuild non-trainable tensors after torchtitan `to_empty()`."""
        if device is None:
            device = self.embed_seq.weight.device

        # `scale` is a frozen parameter; restore deterministic value.
        with torch.no_grad():
            self.scale.fill_(math.sqrt(self.embed_seq.embedding_dim // 2))

        # Rebuild sinusoidal absolute-position buffer.
        if self.pos_embedding and not self.rel_pos_enc and hasattr(self, "pe"):
            pe = torch.zeros(self.max_len, self.embed_seq.embedding_dim, dtype=torch.float32, device=device)
            position = torch.arange(0, self.max_len, dtype=torch.float32, device=device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.embed_seq.embedding_dim, 2, dtype=torch.float32, device=device)
                * -(math.log(10000.0) / self.embed_seq.embedding_dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.pe.copy_(pe)

    def relative_position_encoding(self, src_seq):

        residue_index = torch.arange(src_seq.size()[1], device=src_seq.device).expand(src_seq.size())
        rel_pos = F.one_hot(torch.clip(residue_index, min=0, max=self.max_len - 1), self.max_len)

        if isinstance(self.embed_pair_pos.weight, torch.cuda.BFloat16Tensor):
            rel_pos = rel_pos.type(torch.bfloat16)
        elif isinstance(self.embed_pair_pos.weight, torch.cuda.HalfTensor):
            rel_pos = rel_pos.half()
        else:
            rel_pos = rel_pos.type(torch.float32)

        pos_encoding = self.embed_pair_pos(rel_pos)
        return pos_encoding

    def forward(self, src_seq):

        seq_embed = self.embed_seq(src_seq) * self.scale

        if self.pos_embedding:
            if self.rel_pos_enc:
                seq_embed = seq_embed + self.relative_position_encoding(src_seq)
            else:
                pe_slice = self.pe[:, :src_seq.size(1)].to(
                    device=self.embed_pair_pos.weight.device,
                    dtype=self.embed_pair_pos.weight.dtype,
                )
                seq_embed = seq_embed + self.embed_pair_pos(pe_slice)

        return seq_embed
