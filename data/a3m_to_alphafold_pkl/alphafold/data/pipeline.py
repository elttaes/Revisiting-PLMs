# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for building the input features for the AlphaFold model."""

import os
from typing import Mapping, Sequence

import numpy as np
from absl import logging
# Internal import (7716).

from alphafold.common import residue_constants
from alphafold.data import parsers
FeatureDict = Mapping[str, np.ndarray]


def make_sequence_features(
        fasta_name: str, sequence: str, description: str, num_res: int) -> FeatureDict:
    """Constructs a feature dict of sequence features."""
    features = {}

    features['aatype'] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True)

    features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
    features['domain_name'] = np.array([description.encode('utf-8')],
                                    dtype=np.object_)
    features['residue_index'] = np.array(range(num_res), dtype=np.int32)
    features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
    features['sequence'] = np.array(
        [sequence.encode('utf-8')], dtype=np.object_)
    return features


def make_msa_features(
        msas: Sequence[Sequence[str]],
        deletion_matrices: Sequence[parsers.DeletionMatrix]) -> FeatureDict:
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError('At least one MSA must be provided.')

    int_msa = []
    deletion_matrix = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(
                f'MSA {msa_index} must contain at least one sequence.')
        for sequence_index, sequence in enumerate(msa):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append(
                [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
            deletion_matrix.append(
                deletion_matrices[msa_index][sequence_index])

    num_res = len(msas[0][0])
    num_alignments = len(int_msa)
    features = {}
    features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
    features['msa'] = np.array(int_msa, dtype=np.int32)
    features['num_alignments'] = np.array(
        [num_alignments] * num_res, dtype=np.int32)
    return features


class DataPipeline:
    """Runs the alignment tools and assembles the input features."""

    def process(self, fasta_name: str, input_a3m_path: str) -> FeatureDict:
        """Runs alignment tools on the input sequence and creates features."""
        
        with open(input_a3m_path,'r') as f:
            input_a3m_str = f.read()
        hhblits_bfd_uniclust_result = {}
        hhblits_bfd_uniclust_result['a3m'] = input_a3m_str

        bfd_msa, bfd_deletion_matrix = parsers.parse_a3m(
            hhblits_bfd_uniclust_result['a3m'])

        sequence_features = make_sequence_features(
            fasta_name=fasta_name,
            sequence=bfd_msa[0],
            description=fasta_name,
            num_res=len(bfd_msa[0])
            )

        msa_features = make_msa_features(
            msas=(bfd_msa,bfd_msa),
            deletion_matrices=(bfd_deletion_matrix,bfd_deletion_matrix))

        return {**sequence_features, **msa_features}