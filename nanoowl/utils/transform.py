from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
    Normalize
)


def build_owlvit_vision_transform(device):

    transform = Compose([
        ToTensor(),
        Resize((768, 768)).to(device),
        Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ]).to(device)

    return transform