import os


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    alpha_clip_weight_path = getattr(vision_tower_cfg, 'alpha_clip_weight_path', None)
    is_absolute_path_exists = os.path.exists(vision_tower)
    if alpha_clip_weight_path is not None and alpha_clip_weight_path.startswith("./checkpoints"):
        from .alpha_clip_encoder import AlphaCLIPVisionTower
        print("Loading AlphaCLIPVisionTower")
        return AlphaCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        from .clip_encoder import CLIPVisionTower
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
