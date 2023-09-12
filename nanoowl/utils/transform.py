from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
    Normalize,
    RandomResizedCrop
)


def build_owlvit_vision_transform(device, is_train: bool = False):

    if is_train:
        resize = RandomResizedCrop((768, 768), scale=(0.08, 1.3))
    else:
        resize = Resize((768, 768))
        
    transform = Compose([
        ToTensor(),
        resize.to(device),
        Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ).to(device)
    ])

    return transform

