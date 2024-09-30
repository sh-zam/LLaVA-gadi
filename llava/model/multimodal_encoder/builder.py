import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            model = CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
            print('Converting to float32')
            return model.to(dtype=torch.float32)
        else:
            model = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
            print('Converting to float32')
            return model.to(dtype=torch.float32)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
