# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import glob
import PIL.Image
import os
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop
from nanoowl.utils.transform import build_owlvit_vision_transform


class ImageFolder:
    def __init__(self, root: str, transform = None, device="cpu", is_train=False):
        self.root = root
        image_paths = glob.glob(os.path.join(root, "*.jpg"))
        image_paths += glob.glob(os.path.join(root, "*.png"))
        self.image_paths = image_paths

        if transform is None:
            transform = build_owlvit_vision_transform(device, is_train=is_train)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = PIL.Image.open(self.image_paths[index]).convert("RGB")
        image = self.transform(image)
        return image