"""
dinosiglip_vit.py

Vision backbone that returns concatenated features from both DINOv2 and SigLIP.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Tuple

import timm
import torch
from PIL import Image
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize

from prismatic.models.backbones.vision.base_vision import ImageTransform, LetterboxPad, VisionBackbone, unpack_tuple

# Registry =>> Supported DinoSigLIP Pairs (as TIMM identifiers)
DINOSigLIP_VISION_BACKBONES = {
    "dinosiglip-vit-so-224px": {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "siglip": "vit_so400m_patch14_siglip_224",
    },
    "dinosiglip-vit-so-ttf-224px": {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "siglip": "vit_so400m_patch14_siglip_224",
    },
    "dinosiglip-vit-so-384px": {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "siglip": "vit_so400m_patch14_siglip_384",
    },
}


@dataclass
class DinoSigLIPImageTransform:
    dino_image_transform: ImageTransform
    siglip_image_transform: ImageTransform
    is_prismatic: bool = True

    def __call__(self, img: Image, **kwargs: str) -> Dict[str, torch.Tensor]:
        return {"dino": self.dino_image_transform(img, **kwargs), "siglip": self.siglip_image_transform(img, **kwargs)}


class DinoSigLIPViTBackbone(VisionBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        self.dino_timm_path_or_url = DINOSigLIP_VISION_BACKBONES[vision_backbone_id]["dino"]
        self.siglip_timm_path_or_url = DINOSigLIP_VISION_BACKBONES[vision_backbone_id]["siglip"]

        # Initialize both Featurizers (ViTs) by downloading from HF / TIMM Hub if necessary
        self.dino_featurizer: VisionTransformer = timm.create_model(
            self.dino_timm_path_or_url, pretrained=True, num_classes=0, img_size=self.default_image_size
        )
        self.dino_featurizer.eval()

        self.siglip_featurizer: VisionTransformer = timm.create_model(
            self.siglip_timm_path_or_url, pretrained=True, num_classes=0, img_size=self.default_image_size
        )
        self.siglip_featurizer.eval()

        # Monkey-Patch the `forward()` function of the featurizers to ensure FSDP-compatibility
        #   => Note: By default set `get_intermediate_layers` to return the *SECOND-TO-LAST* layer patches!
        #   => TODO (siddk) Remove after resolution of https://github.com/pytorch/pytorch/issues/109385
        self.dino_featurizer.forward = unpack_tuple(
            partial(self.dino_featurizer.get_intermediate_layers, n={len(self.dino_featurizer.blocks) - 2})
        )
        self.siglip_featurizer.forward = unpack_tuple(
            partial(self.siglip_featurizer.get_intermediate_layers, n={len(self.siglip_featurizer.blocks) - 2})
        )

        # Get Configs for _both_ Featurizers =>> Note :: Override default image size for larger resolution models
        self.dino_data_cfg = timm.data.resolve_model_data_config(self.dino_featurizer)
        self.dino_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        self.siglip_data_cfg = timm.data.resolve_model_data_config(self.siglip_featurizer)
        self.siglip_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # Initialize *both* Transforms
        default_dino_transform = timm.data.create_transform(**self.dino_data_cfg, is_training=False)
        default_siglip_transform = timm.data.create_transform(**self.siglip_data_cfg, is_training=False)

        # Fix =>> SigLIP default transform resizes to *larger* than `self.default_image_size` (crops image)!!
        assert isinstance(default_siglip_transform, Compose), "Unexpected `default_image_transform`!"
        assert isinstance(default_siglip_transform.transforms[0], Resize)
        default_siglip_transform = Compose(
            [
                Resize(self.default_image_size, interpolation=default_siglip_transform.transforms[0].interpolation),
                *default_siglip_transform.transforms[1:],
            ]
        )

        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_dino_transform, Compose), "Unexpected `default_dino_image_transform`!"
            assert isinstance(default_siglip_transform, Compose), "Unexpected `default_siglip_image_transform`!"
            assert isinstance(default_dino_transform.transforms[0], Resize)
            assert isinstance(default_siglip_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            dino_transform = Compose(
                [
                    Resize(target_size, interpolation=default_dino_transform.transforms[0].interpolation),
                    *default_dino_transform.transforms[1:],
                ]
            )
            siglip_transform = Compose(
                [
                    Resize(target_size, interpolation=default_siglip_transform.transforms[0].interpolation),
                    *default_siglip_transform.transforms[1:],
                ]
            )

            self.image_transform = DinoSigLIPImageTransform(dino_transform, siglip_transform)

        elif self.image_resize_strategy == "resize-crop":
            self.image_transform = DinoSigLIPImageTransform(default_dino_transform, default_siglip_transform)

        elif self.image_resize_strategy == "letterbox":
            assert isinstance(default_dino_transform, Compose), "Unexpected `default_dino_transform`!"
            assert isinstance(default_siglip_transform, Compose), "Unexpected `default_siglip_transform`!"
            assert (
                "mean" in self.dino_data_cfg and "mean" in self.siglip_data_cfg
            ), "DinoSigLIP `data_cfg` missing `mean`!"

            # Compute Padding Fill Value(s) (rescaled normalization mean if applicable)
            dino_fill = tuple([int(x * 255) for x in self.dino_data_cfg["mean"]])
            siglip_fill = tuple([int(x * 255) for x in self.siglip_data_cfg["mean"]])

            # Build New Transform
            self.image_transform = DinoSigLIPImageTransform(
                Compose([LetterboxPad(dino_fill), *default_dino_transform.transforms]),
                Compose([LetterboxPad(siglip_fill), *default_siglip_transform.transforms]),
            )

        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then both of the _entire_ featurizers."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Runs the transformed image/pixel tensors through each vision backbone, returning concatenated patches."""
        dino_patches = self.dino_featurizer(pixel_values["dino"])
        siglip_patches = self.siglip_featurizer(pixel_values["siglip"])

        return torch.cat([dino_patches, siglip_patches], dim=2)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.dino_data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.dino_featurizer.embed_dim + self.siglip_featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        assert self.dino_featurizer.patch_embed.num_patches == self.siglip_featurizer.patch_embed.num_patches
        return self.dino_featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

class DinoSigLIPViTTTFBackbone(DinoSigLIPViTBackbone):
    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        manager = self.manager
        if not manager.is_enabled():
            raise
            """Runs the transformed image/pixel tensors through each vision backbone, returning concatenated patches."""
            dino_patches = self.dino_featurizer(pixel_values["dino"])
            siglip_patches = self.siglip_featurizer(pixel_values["siglip"])

            return torch.cat([dino_patches, siglip_patches], dim=2)
        else:
            current_shallow_features = None
            # assert manager.fusion_mode == "pixel" # POST: only supported
            # Preparation for semantic/pixel difference
            img_dino, img_siglip = pixel_values["dino"], pixel_values["siglip"]
            if manager.fusion_mode == "semantic":
                raise
                # 一次性获取DINOv2和SigLIP的深、浅两种特征
                # === DINOv2 特征处理 ===
                dino_features_list = self.featurizer.get_intermediate_layers(
                    img_dino, n=[manager.semantic_shallow_layer, len(self.featurizer.blocks) - 2]
                )
                # 直接解包，不做任何切片
                new_tokens_dino, shallow_dino = dino_features_list[1], dino_features_list[0]

                # === SigLIP 特征处理 ===
                siglip_features_list = self.fused_featurizer.get_intermediate_layers(
                    img_siglip, n=[manager.semantic_shallow_layer, len(self.fused_featurizer.blocks) - 2]
                )
                # 直接解包，不做任何切片
                new_tokens_siglip, shallow_siglip = siglip_features_list[1], siglip_features_list[0]

                current_shallow_features = (shallow_dino, shallow_siglip)
                # 计算语义权重，为两种融合模式做准备
                dino_weights, siglip_weights = manager.get_semantic_fusion_weights(shallow_dino, shallow_siglip)
            elif manager.fusion_mode == "pixel":
                new_tokens_dino = self.dino_featurizer(img_dino)
                new_tokens_siglip = self.siglip_featurizer(img_siglip)
            elif manager.fusion_mode == "attention":
                # For attention mode, we always compute new features
                # The fusion will be guided by attention weights
                new_tokens_dino = self.dino_featurizer(img_dino)
                new_tokens_siglip = self.siglip_featurizer(img_siglip)
            elif manager.fusion_mode == "hybrid":
                # For hybrid mode, we always compute new features
                # The fusion will be guided by both pixel and attention weights
                new_tokens_dino = self.dino_featurizer(img_dino)
                new_tokens_siglip = self.siglip_featurizer(img_siglip)
            else:
                assert False, "Invalid fusion mode, please use 'semantic', 'pixel', 'attention', or 'hybrid'!"

            # hard fusion
            if not manager.smooth_fusion_enabled:
                if manager.is_keyframe():
                    img_dino, img_siglip = pixel_values["dino"], pixel_values["siglip"]
                    patches_main, patches_fused = self.dino_featurizer(img_dino), self.siglip_featurizer(img_siglip)
                    patches = torch.cat([patches_main, patches_fused], dim=2)
                # TOKEN REUSE CORE LOGIC
                # we first try compute the entire new frame and combine it with the cached tokens, then we try only recomputing the dynamic regions.
                else:
                    # TOKEN REUSE CORE LOGIC
                    # semantic difference
                    if manager.fusion_mode == "semantic":
                        raise
                        recompute_mask_dino = dino_weights > manager.semantic_threshold
                        recompute_mask_siglip = siglip_weights > manager.semantic_threshold
                    # pixel difference
                    elif manager.fusion_mode == "pixel":
                        recompute_mask_dino, recompute_mask_siglip = manager.get_pixel_recompute_mask(pixel_values)
                    # attention-guided difference
                    elif manager.fusion_mode == "attention":
                        recompute_mask_dino, recompute_mask_siglip = manager.get_attention_recompute_mask()
                    # hybrid difference (pixel + attention)
                    elif manager.fusion_mode == "hybrid":
                        recompute_mask_dino, recompute_mask_siglip = manager.get_hybrid_recompute_mask(pixel_values)
                    else:
                        assert False, "Invalid fusion mode, please use 'semantic', 'pixel', 'attention', or 'hybrid'!"
                    
                    num_reused_dino = (~recompute_mask_dino).sum().item()
                    num_reused_siglip = (~recompute_mask_siglip).sum().item()
                    
                    # Log attention-guided fusion info
                    if manager.fusion_mode == "attention" and manager.step_counter % 10 == 0:
                        print(f"  Vision backbone - VLA-Cache attention fusion")
                        print(f"  DINO reused: {num_reused_dino}/256, SigLIP reused: {num_reused_siglip}/256")

                    # 1. Separate the cached tokens for DINO and SigLIP
                    # DINO tokens are the first 1024 dimensions, SigLIP are the next 1152.
                    cached_tokens_dino, cached_tokens_siglip = torch.split(
                        manager.last_vision_tokens, [1024, 1152], dim=2
                    )

                    # 3. Initialize final tokens with the cached versions
                    final_tokens_dino = cached_tokens_dino.clone()
                    final_tokens_siglip = cached_tokens_siglip.clone()
                    
                    # 4. Overwrite the dynamic regions with the newly computed tokens
                    final_tokens_dino[0, recompute_mask_dino] = new_tokens_dino[0, recompute_mask_dino]
                    final_tokens_siglip[0, recompute_mask_siglip] = new_tokens_siglip[0, recompute_mask_siglip]

                    # 5. Concatenate the final DINO and SigLIP tokens
                    patches = torch.cat([final_tokens_dino, final_tokens_siglip], dim=2)
            # smooth fusion
            else:                
                raise
                if manager.is_keyframe():
                    img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
                    patches_main, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)
                    patches = torch.cat([patches_main, patches_fused], dim=2)
                # TOKEN REUSE CORE LOGIC
                # we first try compute the entire new frame and combine it with the cached tokens, then we try only recomputing the dynamic regions.
                else:
                    if manager.fusion_mode == "semantic":
                        dino_fusion_weights, siglip_fusion_weights = dino_weights, siglip_weights
                    elif manager.fusion_mode == "pixel":
                        dino_fusion_weights, siglip_fusion_weights = manager.get_fusion_weights(pixel_values)
                    elif manager.fusion_mode == "attention":
                        dino_fusion_weights, siglip_fusion_weights = manager.get_attention_fusion_weights()
                    else:
                        assert False, "Invalid fusion mode, please use 'semantic', 'pixel', or 'attention'!"
                    
                    num_reused_dino = (1 - dino_fusion_weights).sum().item()
                    num_reused_siglip = (1 - siglip_fusion_weights).sum().item()

                    # 1. Separate the cached tokens for DINO and SigLIP
                    # DINO tokens are the first 1024 dimensions, SigLIP are the next 1152.
                    cached_tokens_dino, cached_tokens_siglip = torch.split(
                        manager.last_vision_tokens, [1024, 1152], dim=2
                    )

                    # 3. Prepare weights for broadcasting
                    # Shape: (num_patches,) -> (1, num_patches, 1)
                    dino_weights = dino_fusion_weights.unsqueeze(0).unsqueeze(-1)
                    siglip_weights = siglip_fusion_weights.unsqueeze(0).unsqueeze(-1)
                    # Convert weights to the same dtype as tokens to prevent mismatch
                    dino_weights = dino_weights.to(new_tokens_dino.dtype)
                    siglip_weights = siglip_weights.to(new_tokens_siglip.dtype)

                    # 4. Perform smooth fusion using weighted average
                    final_tokens_dino = (dino_weights * new_tokens_dino) + ((1 - dino_weights) * cached_tokens_dino)
                    final_tokens_siglip = (siglip_weights * new_tokens_siglip) + ((1 - siglip_weights) * cached_tokens_siglip)

                    # 5. Concatenate the final DINO and SigLIP tokens
                    patches = torch.cat([final_tokens_dino, final_tokens_siglip], dim=2)

        if manager.is_enabled():
            manager.update_cache(pixel_values=pixel_values, vision_tokens=patches, shallow_features=current_shallow_features)

        return patches