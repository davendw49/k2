# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""AMR dataset."""
import pickle
from inspect import EndOfBlock
import json
import os
import gzip
import datasets
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional
from dataclasses import InitVar, dataclass, field, fields
from datasets.features.features import string_to_arrow
import pyarrow as pa
from tqdm import tqdm

logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """

There are three features:
  - src: text.
  - tgt: Linearized AMR.
"""

_TEXT = "text"

class InnerSpeechData(datasets.GeneratorBasedBuilder):
    """AMR Dataset."""

    # Version 1.0.0 expands coverage, includes ids, and removes web contents.
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {_TEXT: datasets.Value("string"),}
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        train_path = self.config.data_files["train"]
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        logger.info("generating examples from = %s", filepath[0])
        text = []
        with open(filepath[0], "r") as f:
            line = f.readline()
            while line and len(text)<500000:
                d = json.loads(line)
                t = d["text"]
                text.append(t)
                line = f.readline()
        print(f"total data num: {len(text)}")
        for idx in range(len(text)):
            yield idx, {_TEXT: text[idx]}
