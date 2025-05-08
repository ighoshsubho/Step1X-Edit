import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .connector_edit import Qwen2Connector
from .layers import DoubleStreamBlock, EmbedND, LastLayer, MLPEmbedder, SingleStreamBlock


@dataclass
class Step1XParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool


class Step1XEdit(nn.Module):
    """
    Transformer model for flow matching on sequences with TeaCache optimization.
    """

    enable_teacache = False
    cnt = 0
    num_steps = 28  # Default value, will be set based on actual steps
    rel_l1_thresh = 0.6  # Same as Flux - 0.6 for 2.0x speedup
    accumulated_rel_l1_distance = 0
    previous_modulated_input = None
    previous_residual = None

    def __init__(self, params: Step1XParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

        self.connector = Qwen2Connector()

    @staticmethod
    def timestep_embedding(
        t: Tensor, dim, max_period=10000, time_factor: float = 1000.0
    ):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        t = time_factor * t
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(t.device)

        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        if torch.is_floating_point(t):
            embedding = embedding.to(t)
        return embedding

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # Process inputs - this happens regardless of TeaCache
        img_processed = self.img_in(img)
        vec = self.time_in(self.timestep_embedding(timesteps, 256))
        vec = vec + self.vector_in(y)
        txt_processed = self.txt_in(txt)
        
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        
        if self.enable_teacache:
            # Check if we should calculate or reuse
            modulated_inp = img_processed.clone()
            
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else:
                # Using the same coefficients as in Flux
                coefficients = [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
                rescale_func = np.poly1d(coefficients)
                
                if self.previous_modulated_input is not None:
                    relative_diff = ((modulated_inp - self.previous_modulated_input).abs().mean() / 
                                     self.previous_modulated_input.abs().mean()).cpu().item()
                    self.accumulated_rel_l1_distance += rescale_func(relative_diff)
                
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
                    
            self.previous_modulated_input = modulated_inp
            self.cnt += 1
            if self.cnt == self.num_steps:
                self.cnt = 0
                
            if not should_calc and self.previous_residual is not None:
                img_output = img_processed + self.previous_residual
                img_output = self.final_layer(img_output, vec)
                return img_output
            else:
                ori_img_processed = img_processed.clone()
                
                for block in self.double_blocks:
                    img_processed, txt_processed = block(img=img_processed, txt=txt_processed, vec=vec, pe=pe)

                img_combined = torch.cat((txt_processed, img_processed), 1)
                for block in self.single_blocks:
                    img_combined = block(img_combined, vec=vec, pe=pe)
                
                img_output = img_combined[:, txt_processed.shape[1]:, ...]
                
                self.previous_residual = img_output - ori_img_processed
                
                img_output = self.final_layer(img_output, vec)
                return img_output
        else:
            for block in self.double_blocks:
                img_processed, txt_processed = block(img=img_processed, txt=txt_processed, vec=vec, pe=pe)

            img_combined = torch.cat((txt_processed, img_processed), 1)
            for block in self.single_blocks:
                img_combined = block(img_combined, vec=vec, pe=pe)
            
            img_output = img_combined[:, txt_processed.shape[1]:, ...]
            img_output = self.final_layer(img_output, vec)
            return img_output
