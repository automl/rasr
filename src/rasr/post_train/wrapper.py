from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import ModelOutput

from src.rasr.model.model import RASRTransformer


class RASRConfig(PretrainedConfig):
    model_type = "rasr_transformer"

    def __init__(
        self,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        max_length: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=True,
            **kwargs,
        )
        self.max_length = int(max_length)


@dataclass
class RASRModelOutput(ModelOutput):
    logits: torch.Tensor


class RASRPolicyWrapper(PreTrainedModel):
    config_class = RASRConfig
    supports_gradient_checkpointing = True

    def __init__(self, model: RASRTransformer, tokenizer, max_length: int) -> None:
        cfg = RASRConfig(
            pad_token_id=tokenizer.token_to_index["<PAD>"],
            bos_token_id=tokenizer.token_to_index["<SOE>"],
            eos_token_id=tokenizer.token_to_index["<EOE>"],
            max_length=max_length,
        )
        super().__init__(cfg)
        self.model = model

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing = False

    def forward(
        self,
        tbl: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        tbl_mask: torch.Tensor,
        trg_len: torch.Tensor,
        decoder_self_attn_mask: torch.Tensor,
        decoder_cross_attn_mask: torch.Tensor,
    ) -> RASRModelOutput:
        logits = self.model(
            tbl=tbl,
            decoder_input_ids=decoder_input_ids,
            trg_len=trg_len,
            decoder_self_attn_mask=decoder_self_attn_mask,
            decoder_cross_attn_mask=decoder_cross_attn_mask,
            tbl_mask=tbl_mask,
        )
        return RASRModelOutput(logits=logits)

    def generate(
        self,
        tbl: torch.Tensor,
        generation_config: GenerationConfig,
    ) -> torch.Tensor:
        num_return_sequences = int(generation_config.num_return_sequences or 1)
        max_new_tokens = int(generation_config.max_new_tokens or self.config.max_length)
        temperature = float(generation_config.temperature)
        top_k = int(generation_config.top_k)
        top_p = float(generation_config.top_p)

        sorted_sequences, _ = self.model.sample(
            tbl=tbl,
            num_candidates=max(1, num_return_sequences),
            max_len=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            # Keep fixed-length candidate rows so torch.tensor(batch_rows) is always valid.
            extract_expressions=False,
            unique=False,
        )

        batch_rows: list[list[int]] = []
        for sample_sequences in sorted_sequences:
            sample_sequences = sample_sequences[:num_return_sequences]
            for sequence in sample_sequences:
                row = [self.config.bos_token_id] + list(sequence)
                eos_index = len(row)
                if self.config.eos_token_id in row:
                    eos_index = row.index(self.config.eos_token_id) + 1
                for idx in range(eos_index, len(row)):
                    row[idx] = self.config.pad_token_id
                batch_rows.append(row)

        return torch.tensor(batch_rows, dtype=torch.long, device=tbl.device)

    def save_pretrained(self, save_directory, state_dict=None, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        if state_dict is None:
            state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        self.config.save_pretrained(save_directory)

