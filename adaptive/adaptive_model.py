import torch
import torch.nn as nn
from segment_anything import sam_model_registry

class AdaptiveSAM(nn.Module):
    def __init__(self, sam_ckpt, medsam_ckpt, device="cuda"):
        super().__init__()

        self.device = device

        # Load base SAM
        self.sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt)

        # Load MedSAM
        self.medsam = sam_model_registry["vit_b"](checkpoint=None)
        state_dict = torch.load(medsam_ckpt, map_location="cpu")
        self.medsam.load_state_dict(state_dict)

        self.sam.to(device)
        self.medsam.to(device)

        self.sam.eval()
        self.medsam.eval()

        # Freeze everything
        for p in self.sam.parameters():
            p.requires_grad = False
        for p in self.medsam.parameters():
            p.requires_grad = False

        # Number of transformer blocks
        num_blocks = len(self.sam.image_encoder.blocks)

        # Learnable alpha per block
        self.alpha = nn.Parameter(torch.zeros(num_blocks))

    def forward(self, x, boxes):

        image_encoder_sam = self.sam.image_encoder
        image_encoder_med = self.medsam.image_encoder

        # Patch embedding
        x_sam = image_encoder_sam.patch_embed(x)
        x_med = image_encoder_med.patch_embed(x)

        x_sam = x_sam + image_encoder_sam.pos_embed
        x_med = x_med + image_encoder_med.pos_embed

        # Transformer blocks
        for i, (block_s, block_m) in enumerate(
            zip(image_encoder_sam.blocks, image_encoder_med.blocks)
        ):
            a = torch.sigmoid(self.alpha[i])

            out_s = block_s(x_sam)
            out_m = block_m(x_med)

            # Feature-level merge (CRITICAL FIX)
            x = a * out_s + (1 - a) * out_m

            x_sam = x
            x_med = x

        # Convert format
        x = x.permute(0, 3, 1, 2)
        x = image_encoder_sam.neck(x)

        # Prompt + decoder
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )

        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=x,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        return low_res_masks