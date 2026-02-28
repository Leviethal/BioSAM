import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry


class AdaptiveSAM(nn.Module):
    def __init__(self, sam_ckpt, medsam_ckpt, device="cuda"):
        super().__init__()

        self.device = device

        # Load SAM
        self.sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt)

        # Load MedSAM
        self.medsam = sam_model_registry["vit_b"](checkpoint=None)
        state_dict = torch.load(medsam_ckpt, map_location="cpu")
        self.medsam.load_state_dict(state_dict)

        self.sam.to(device)
        self.medsam.to(device)

        self.sam.eval()
        self.medsam.eval()

        # Freeze all weights
        for p in self.sam.parameters():
            p.requires_grad = False
        for p in self.medsam.parameters():
            p.requires_grad = False

        # Per-layer alpha (ViT-B has 12 blocks)
        num_blocks = len(self.sam.image_encoder.blocks)
        self.alpha = nn.Parameter(torch.zeros(num_blocks))

        # SAM normalization constants
        self.pixel_mean = torch.tensor(
            [123.675, 116.28, 103.53], device=device
        ).view(1, 3, 1, 1)

        self.pixel_std = torch.tensor(
            [58.395, 57.12, 57.375], device=device
        ).view(1, 3, 1, 1)

    def forward(self, image, boxes):

        # image: [B, 3, H, W]
        B, C, H, W = image.shape

        # Resize to SAM input size
        target_size = self.sam.image_encoder.img_size
        image = F.interpolate(
            image,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize
        image = (image - self.pixel_mean) / self.pixel_std

        enc_sam = self.sam.image_encoder
        enc_med = self.medsam.image_encoder

        # Patch embedding
        x_sam = enc_sam.patch_embed(image)
        x_med = enc_med.patch_embed(image)

        x_sam = x_sam + enc_sam.pos_embed
        x_med = x_med + enc_med.pos_embed

        # ---- Per-layer residual fusion ----
        for i, (block_s, block_m) in enumerate(
            zip(enc_sam.blocks, enc_med.blocks)
        ):
            out_s = block_s(x_sam)
            out_m = block_m(x_med)

            a = torch.sigmoid(self.alpha[i])

            # Residual-style interpolation (SAFE)
            x = out_s + a * (out_m - out_s)

            # Keep both streams aligned for next layer
            x_sam = x
            x_med = x

        # Neck
        x = x.permute(0, 3, 1, 2)
        x = enc_sam.neck(x)

        # Prompt encoding
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )

        # Mask decoding
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=x,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        return low_res_masks