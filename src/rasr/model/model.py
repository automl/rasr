from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.rasr.masks import make_rasr_masks
from src.rasr.model.args import RASRModelArgs
from src.rasr.model.module.decoder_block import DecoderBlock
from src.rasr.model.module.embedding import TokenEmbedding
from src.rasr.model.module.encoder_block import EncoderBlock
from src.rasr.model.module.norm import RMSNorm


class RASRTransformer(nn.Module):
    """Encoder-decoder transformer for symbolic regression.

    Encoder: axial attention over tabular data (realizations x features).
    Decoder: causal transformer generating token sequences, conditioned on the
    encoder output via cross-attention.
    """

    def __init__(self, model_args: RASRModelArgs) -> None:
        super().__init__()

        self.model_args = model_args

        self.model_dim = model_args.model_dim
        self.n_layers_enc = model_args.n_layers_enc
        self.n_layers_dec = model_args.n_layers_dec
        self.trg_vocab_size = model_args.trg_vocab_size
        self.num_head = model_args.num_head
        self.max_len = model_args.max_len
        self.ff_factor = model_args.ff_factor
        self.activation = model_args.activation
        self.glu = model_args.glu
        self.softmax_scale = model_args.softmax_scale
        self.pos_embedding = model_args.pos_embedding
        self.rel_pos_enc = model_args.rel_pos_enc
        self.initializer_range = model_args.initializer_range
        self.rms_norm = model_args.rms_norm
        self.resi_dropout = model_args.resi_dropout
        self.attn_dropout = model_args.attn_dropout
        self.ln_eps = model_args.ln_eps
        self.add_unit_offset = model_args.add_unit_offset
        self.use_bias = model_args.use_bias
        self.learn_ln = model_args.learn_ln
        self.last_zero = model_args.last_zero
        self.max_var_pos = model_args.max_var_pos
        self.soe_token_id = model_args.soe_token_id
        self.eoe_token_id = model_args.eoe_token_id

        self.var_embed = nn.Linear(model_args.model_dim, model_args.model_dim, bias=False)

        pe = torch.zeros(model_args.max_var_pos + 1, model_args.model_dim, dtype=torch.float32)
        position = torch.arange(0, model_args.max_var_pos + 1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_args.model_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / model_args.model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = torch.nn.Parameter(pe, requires_grad=False)
        self.register_buffer('pe', pe)

        self.trg_embed = TokenEmbedding(
            model_args.trg_vocab_size,
            model_args.model_dim,
            model_args.max_len,
            model_args.pos_embedding,
            model_args.rel_pos_enc,
            model_args.initializer_range,
        )
        self.trg_output = self.trg_embed.embed_seq

        self.pre_encoder = _make_pre_encoder()
        self.embed_cell = nn.Linear(32, model_args.model_dim)

        self.encoder = nn.ModuleList()
        for _ in range(model_args.n_layers_enc):
            self.encoder.append(EncoderBlock(model_args))

        self.decoder = nn.ModuleList()
        for _ in range(model_args.n_layers_dec):
            self.decoder.append(DecoderBlock(model_args))

        if model_args.rms_norm:
            self.trg_norm = RMSNorm(
                model_args.model_dim, eps=model_args.ln_eps, add_unit_offset=model_args.add_unit_offset
            )
        else:
            self.trg_norm = nn.LayerNorm(
                model_args.model_dim,
                eps=model_args.ln_eps,
                bias=model_args.use_bias,
                elementwise_affine=model_args.learn_ln,
            ).float()

        self.memory_encoder_latent = None
        self.memory_key_is_valid = None
        self.gradient_checkpointing = False

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize weights; called by torchtitan after meta-device instantiation."""
        self._reset_non_trainable_state(buffer_device=buffer_device)
        self.initialize(
            self.model_args.initializer_range,
            self.model_args.trg_vocab_size,
            self.model_args.last_zero,
        )
        # Zero-init qassmax query MLP output projection so modulation starts as identity.
        for module in self.modules():
            if hasattr(module, 'qassmax_query_mlp'):
                nn.init.zeros_(module.qassmax_query_mlp[-1].weight)

    def _reset_non_trainable_state(self, buffer_device: torch.device | None = None) -> None:
        """Rebuild frozen parameters/buffers that `to_empty()` can invalidate."""
        device = buffer_device if buffer_device is not None else self.var_embed.weight.device
        pe = torch.zeros(self.model_args.max_var_pos + 1, self.model_args.model_dim, dtype=torch.float32, device=device)
        position = torch.arange(0, self.model_args.max_var_pos + 1, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.model_args.model_dim, 2, dtype=torch.float32, device=device)
            * -(math.log(10000.0) / self.model_args.model_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        with torch.no_grad():
            self.pe.copy_(pe)
        self.trg_embed.reset_non_trainable_state(device=device)

        # RoPE caches are non-persistent buffers and can be invalid after meta-device materialization.
        for module in self.modules():
            rebuild_cache = getattr(module, "_rebuild_rope_cache", None)
            if callable(rebuild_cache):
                rebuild_cache(device=device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initialize(self, initializer_range, vocab_size, last_zero):
        # `to_empty()` discards default parameter values, so re-init norms explicitly.
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.zeros_(module.weight)

        for n, p in self.named_parameters():
            if 'bias' in n:
                nn.init.zeros_(p)
            elif 'norm' in n or 'ln' in n:
                continue
            elif p.shape == torch.Size([1]):
                continue
            elif 'out' in n and last_zero:
                nn.init.zeros_(p)
            elif 'embed_seq' in n:
                if initializer_range:
                    nn.init.normal_(p, mean=0.0, std=initializer_range)
                else:
                    nn.init.normal_(p, mean=0.0, std=1.0/math.sqrt(vocab_size))
            else:
                if initializer_range:
                    nn.init.normal_(p, mean=0.0, std=initializer_range)
                else:
                    nn.init.xavier_uniform_(p)

    def _create_memory(self, tbl, tbl_mask):
        tbl = torch.where(tbl_mask, tbl, 0.0)
        tbl_latent = self.pre_encoder(tbl)
        tbl_latent = tbl_latent.to(device=self.embed_cell.weight.device, dtype=self.embed_cell.weight.dtype)
        tbl_latent = self.embed_cell(tbl_latent)
        pe_slice = self.pe[:, :tbl.size(2)].to(
            device=self.var_embed.weight.device,
            dtype=self.var_embed.weight.dtype,
        )
        tbl_latent = tbl_latent + self.var_embed(pe_slice).unsqueeze(0)

        for i in range(len(self.encoder)):
            layer = self.encoder[i]
            if self.gradient_checkpointing and self.training:
                tbl_latent = checkpoint(layer, tbl_latent, tbl_mask, use_reentrant=False)
            else:
                tbl_latent = layer(tbl_states=tbl_latent, tbl_mask=tbl_mask)

        encoder_latent = tbl_latent[:, :, 0, :]
        key_is_valid = tbl_mask[:, :, 0]
        return encoder_latent, key_is_valid

    def forward(self, tbl, decoder_input_ids, trg_len,
                decoder_self_attn_mask,
                decoder_cross_attn_mask,
                tbl_mask=None,
                memory_encoder_latent=None, memory_key_is_valid=None):
        if memory_encoder_latent is not None and memory_key_is_valid is not None:
            encoder_latent = memory_encoder_latent
            key_is_valid = memory_key_is_valid
        elif tbl is not None:
            if tbl_mask is None:
                raise ValueError("`tbl_mask` must be provided when `tbl` is provided.")
            encoder_latent, key_is_valid = self._create_memory(tbl, tbl_mask)
            self.memory_encoder_latent = encoder_latent
            self.memory_key_is_valid = key_is_valid
        elif self.memory_encoder_latent is not None:
            encoder_latent = self.memory_encoder_latent
            key_is_valid = self.memory_key_is_valid
        else:
            raise ValueError(
                "Either `tbl` or `memory` and `memory_key_is_valid` must be provided for the first forward pass."
            )

        trg_latent = self.trg_embed(decoder_input_ids)

        for i in range(len(self.decoder)):
            layer = self.decoder[i]
            trg_latent = layer(
                self_states=trg_latent,
                cross_states=encoder_latent,
                self_attn_mask=decoder_self_attn_mask,
                cross_attn_mask=decoder_cross_attn_mask,
            )

        logits = self.trg_output(self.trg_norm(trg_latent))
        return logits

    def sample(
        self,
        tbl: torch.Tensor,
        num_candidates: int = 1,
        max_len: int | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        batch_size: int = 128,
        verbose: bool = False,
        extract_expressions: bool = True,
        unique: bool = False,
    ) -> Tuple[list[list[list[int]]], list[list[float]]]:
        """Autoregressively sample token ids from the decoder."""
        if max_len is None:
            max_len = self.max_len
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not (0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")

        from tqdm import tqdm

        with torch.no_grad():
            tbl_mask = torch.ones_like(tbl, dtype=torch.bool)
            memory_encoder_latent, memory_key_is_valid = self._create_memory(tbl, tbl_mask)
            sample_batch_size = memory_encoder_latent.size(0)
            decode_batch_size = max(1, int(batch_size))

            if num_candidates > 1:
                memory_encoder_latent = memory_encoder_latent.unsqueeze(1).expand(-1, num_candidates, -1, -1).reshape(
                    sample_batch_size * num_candidates, memory_encoder_latent.size(1), memory_encoder_latent.size(2)
                )
                memory_key_is_valid = memory_key_is_valid.unsqueeze(1).expand(-1, num_candidates, -1).reshape(
                    sample_batch_size * num_candidates, memory_key_is_valid.size(1)
                )

            flat_batch = sample_batch_size * num_candidates
            sequences = torch.full(
                (flat_batch, max_len + 1),
                self.eoe_token_id,
                dtype=torch.long,
                device=tbl.device,
            )
            sequences[:, 0] = self.soe_token_id
            scores = torch.zeros((flat_batch,), dtype=torch.float32, device=tbl.device)
            finished = torch.zeros((flat_batch,), dtype=torch.bool, device=tbl.device)

            progress = tqdm(total=max_len, desc="Generating", unit="tok", smoothing=0.0) if verbose else None

            for current_len in range(1, max_len + 1):
                if torch.all(finished):
                    break
                if progress is not None:
                    progress.update(1)

                active_indices = (~finished).nonzero(as_tuple=True)[0]
                if active_indices.numel() == 0:
                    break

                for start_idx in range(0, active_indices.numel(), decode_batch_size):
                    batch_indices = active_indices[start_idx : start_idx + decode_batch_size]

                    decoder_input_ids = sequences[batch_indices, :current_len]
                    trg_len = torch.full((batch_indices.size(0),), current_len, dtype=torch.long, device=tbl.device)
                    batch_key_is_valid = memory_key_is_valid[batch_indices]
                    decoder_self_attn_mask, decoder_cross_attn_mask = make_rasr_masks(
                        batch_key_is_valid, current_len, trg_len
                    )
                    logits = self.forward(
                        tbl=None,
                        decoder_input_ids=decoder_input_ids,
                        trg_len=trg_len,
                        decoder_self_attn_mask=decoder_self_attn_mask,
                        decoder_cross_attn_mask=decoder_cross_attn_mask,
                        memory_encoder_latent=memory_encoder_latent[batch_indices],
                        memory_key_is_valid=batch_key_is_valid,
                    )

                    next_token_logits = logits[:, -1, :]
                    # Keep log-prob bookkeeping in fp32.
                    original_log_probs = torch.log_softmax(next_token_logits.float(), dim=-1)

                    filtered_logits = next_token_logits.clone()

                    if top_k > 0:
                        k = min(top_k, filtered_logits.size(-1))
                        topk_values = torch.topk(filtered_logits, k, dim=-1).values
                        threshold = topk_values[:, -1].unsqueeze(-1)
                        filtered_logits = filtered_logits.masked_fill(filtered_logits < threshold, -float("inf"))

                    if temperature > 0 and temperature != 1.0:
                        filtered_logits = filtered_logits / temperature

                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_remove = cumulative_probs > top_p
                        sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
                        sorted_remove[:, 0] = False
                        remove_mask = torch.zeros_like(sorted_remove, dtype=torch.bool).scatter(
                            1, sorted_indices, sorted_remove
                        )
                        filtered_logits = filtered_logits.masked_fill(remove_mask, -float("inf"))

                    if temperature == 0:
                        sampled = torch.argmax(filtered_logits, dim=-1)
                    else:
                        probs = torch.softmax(filtered_logits, dim=-1)
                        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

                    sequences[batch_indices, current_len] = sampled
                    step_scores = original_log_probs.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
                    scores[batch_indices] += step_scores
                    finished[batch_indices] |= (sampled == self.eoe_token_id)

            if progress is not None:
                progress.total = progress.n
                progress.close()

            sequences = sequences[:, 1:].view(sample_batch_size, num_candidates, max_len)  # Drop SOE.
            scores = scores.view(sample_batch_size, num_candidates)

            def _extract_expression(sequence: torch.Tensor) -> list[int]:
                token_list = sequence.tolist()
                if self.eoe_token_id in token_list:
                    token_list = token_list[: token_list.index(self.eoe_token_id)]
                return token_list

            sorted_sequences: list[list[list[int]]] = []
            sorted_scores: list[list[float]] = []

            for sample_idx in range(sample_batch_size):
                pairs = []
                seen = set()
                for cand_idx in range(num_candidates):
                    seq_tokens = sequences[sample_idx, cand_idx]
                    token_list = _extract_expression(seq_tokens) if extract_expressions else seq_tokens.tolist()
                    key = tuple(token_list)
                    if unique and key in seen:
                        continue
                    seen.add(key)
                    pairs.append((token_list, float(scores[sample_idx, cand_idx].item())))

                pairs.sort(key=lambda x: x[1], reverse=True)
                sorted_sequences.append([p[0] for p in pairs])
                sorted_scores.append([p[1] for p in pairs])

            return sorted_sequences, sorted_scores


def _make_pre_encoder():
    """Lazily imported to avoid circular dependencies."""
    from src.rasr.model.module.pre_encoder import IEEE75432PreEncoder
    return IEEE75432PreEncoder()
