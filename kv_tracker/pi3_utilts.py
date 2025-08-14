import torch
import numpy as np

from pi3.models.pi3 import Pi3


def load_pi3_from_pretrained(device="cuda"):
    model = Pi3.from_pretrained("yyfz233/Pi3").to(device)
    return model


def move_pi3_mlps_to_bfloat32(model):
    for i in range(len(model.decoder)):
        model.decoder[i].attn.qkv = model.decoder[i].attn.qkv.to(torch.bfloat16)
        model.decoder[i].attn.proj = model.decoder[i].attn.proj.to(torch.bfloat16)
        model.decoder[i].mlp.fc1 = model.decoder[i].mlp.fc1.to(torch.bfloat16)
        model.decoder[i].mlp.fc2 = model.decoder[i].mlp.fc2.to(torch.bfloat16)

    for i in range(len(model.encoder.blocks)):
        model.encoder.blocks[i].attn.qkv = model.encoder.blocks[i].attn.qkv.to(torch.bfloat16)
        model.encoder.blocks[i].attn.proj = model.encoder.blocks[i].attn.proj.to(torch.bfloat16)
        model.encoder.blocks[i].mlp.fc1 = model.encoder.blocks[i].mlp.fc1.to(torch.bfloat16)
        model.encoder.blocks[i].mlp.fc2 = model.encoder.blocks[i].mlp.fc2.to(torch.bfloat16)

    return model

def pi3_inference(model, images_np_list, device, cam_only=False, store_cache=False, use_cache=False, tokens_mask=None, **pi_kwargs):

    if type(images_np_list) is list or type(images_np_list) is np.ndarray:
        images_np = np.array(images_np_list)
        images = torch.tensor(images_np, device=device, dtype=torch.float32) / 255.0
    else:
        images_np = None
        images = images_np_list

    B, N, H, W, C = images.shape
    images_tensor = images.permute(0, 1, 4, 2, 3)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            results = model(images_tensor, cam_only=cam_only, store_cache=store_cache, use_cache=use_cache, tokens_mask=tokens_mask, **pi_kwargs)

    T_wc = results["camera_poses"]#.reshape(B, N, 4, 4)

    if cam_only:
        return T_wc

    conf = torch.sigmoid(results["conf"])
    pts3d = results["points"]

    if T_wc.shape[1] > 1:
        with torch.amp.autocast("cuda", dtype=torch.float64):
            origin_offset = torch.linalg.inv(T_wc[0, 0])
            T_wc = origin_offset @ T_wc
            pts3d = origin_offset[:3, :3] @ pts3d[..., None] + origin_offset[:3, 3:4]
            pts3d = pts3d.squeeze(-1)
    else:
        origin_offset = None

    return pts3d, T_wc, conf, images_np, results["local_points"], origin_offset
