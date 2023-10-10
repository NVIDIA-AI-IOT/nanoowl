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


import time
import torch


__all__ == [
    "SyncTimer"
]


class SyncTimer():

    def __init__(self, name: str):
        self.name = name
        self.t0 = None

    def __enter__(self, *args, **kwargs):
        self.t0 = time.perf_counter_ns()

    def __exit__(self, *args, **kwargs):
        torch.cuda.current_stream().synchronize()
        t1 = time.perf_counter_ns()
        dt = (t1 - self.t0) / 1e9
        print(f"{self.name} FPS: {round(1./dt, 3)}")