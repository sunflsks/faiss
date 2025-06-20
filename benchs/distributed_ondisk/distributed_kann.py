#! /usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple distributed k-ann implementation, based off of distributed_kann.py

Also relies on an abstraction for the training matrix that can be sharded over several machines.
"""

import os
import sys
import argparse

import numpy as np

import faiss

from multiprocessing.pool import ThreadPool
from faiss.contrib import rpc
from faiss.contrib.datasets import SyntheticDataset
from faiss.contrib.vecs_io import bvecs_mmap, fvecs_mmap
from faiss.contrib.clustering import DatasetAssign, kmeans

# so faiss's DatasetAssign just does brute-force search -- this is not adequate. so we reimplement
# the DatasetAssignGPU class to use a custom index that does not use brute-force search
# we will only support GPUs

class DatasetAssignGPUCustomIndex(DatasetAssign):
    def __init__(self, x, gpu_id, verbose=False):
        DatasetAssign.__init__(self, x)
        print(f"My shard contains {x.shape[0]} vectors")
        #quantizer = faiss.IndexFlatL2(x.shape[1])
        #index = faiss.IndexIVFFlat(quantizer, x.shape[1], int(np.sqrt(len(x)))) # replace with whatever index you want
        index = faiss.IndexHNSW(x.shape[1])

        if gpu_id >= 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                gpu_id, index)
        else:
            # -1 -> assign to all GPUs
            self.index = faiss.index_cpu_to_all_gpus(index)

        if not self.index.is_trained:
            self.index.train(x[:100000])  # train on a subset of the data, in case idx needs training

    def search(self, xq, k):
        """search for the closest centroids to the queries"""
        D, I = self.index.search(xq, k)
        return D, I

class DatasetAssignCustomIndex(DatasetAssign):
    def __init__(self, x, i=0, verbose=False):
        DatasetAssign.__init__(self, x)
        print(f"I am shard #{i} and I contain {x.shape[0]} vectors")
        #quantizer = faiss.IndexFlatL2(x.shape[1])
        #self.index = faiss.IndexIVFFlat(quantizer, x.shape[1], int(np.sqrt(len(x)))) # replace with whatever index you want
        self.index = faiss.IndexHNSWFlat(x.shape[1], 16)

        if not self.index.is_trained:
            self.index.train(x[:100000])  # train on a subset of the data, in case idx needs training

        self.index.add(x)

    def search(self, xq, k):
        """search for the closest centroids to the queries"""
        D, I = self.index.search(xq, k)
        return D, I

class DatasetAssignDispatch:
    """dispatches to several other DatasetAssigns and combines the
    results"""

    def __init__(self, xes, in_parallel):
        self.xes = xes
        self.d = xes[0].dim()
        if not in_parallel:
            self.imap = map
        else:
            self.pool = ThreadPool(len(self.xes))
            self.imap = self.pool.imap
        self.sizes = list(map(lambda x: x.count(), self.xes))
        self.cs = np.cumsum([0] + self.sizes)

    def count(self):
        return self.cs[-1]

    def dim(self):
        return self.d

    def get_subset(self, indices):
        res = np.zeros((len(indices), self.d), dtype='float32')
        nos = np.searchsorted(self.cs[1:], indices, side='right')

        def handle(i):
            mask = nos == i
            sub_indices = indices[mask] - self.cs[i]
            subset = self.xes[i].get_subset(sub_indices)
            res[mask] = subset

        list(self.imap(handle, range(len(self.xes))))
        return res

    def assign_to(self, centroids, weights=None):
        src = self.imap(
            lambda x: x.assign_to(centroids, weights),
            self.xes
        )
        I = []
        D = []
        sum_per_centroid = None
        for Ii, Di, sum_per_centroid_i in src:
            I.append(Ii)
            D.append(Di)
            if sum_per_centroid is None:
                sum_per_centroid = sum_per_centroid_i
            else:
                sum_per_centroid += sum_per_centroid_i
        return np.hstack(I), np.hstack(D), sum_per_centroid

    def search(self, xq, k):
        """k-ann search"""
        # dispatch to all the DatasetAssigns. we automatically split up the queries as well

        queries_per_shard = len(xq) // len(self.xes)

        def search_auto_split(xes_idx_dataset):
            xes_idx, dataset = xes_idx_dataset
            # we must split up the queries into one for each shard.
            xq_shard = xq[queries_per_shard * xes_idx : queries_per_shard * (xes_idx + 1)]
            return dataset.search(xq_shard, k)

        src = self.imap(
            search_auto_split,
            enumerate(self.xes)
        )

        # TODO - completely rewrite and figure out what is going on here

        D_allshards = []
        I_allshards = []
        for i, (Di, Ii) in enumerate(src):
            D_allshards.append(Di)
            I_allshards.append(Ii + self.cs[i]) # cs[i] tells us where indices for this shard start

        final_D = np.empty((len(xq), k), dtype='float32')
        final_I = np.empty((len(xq), k), dtype='int64')

        for q in range(len(xq)):
            candidates = []
            for D_shard, I_shard in zip(D_allshards, I_allshards):
                for j in range(k): # top k queries for this shard
                    candidates.append((D_shard[q,j], I_shard[q,j]))

            candidates.sort(key=lambda x: x[0]) # sort on distance

            for j in range(k):
                final_D[q, j] = candidates[j][0]
                final_I[q, j] = candidates[j][1]

        return final_D, final_I


class AssignServer(rpc.Server):
    """ Assign version that can be exposed via RPC """

    def __init__(self, s, assign, log_prefix=''):
        rpc.Server.__init__(self, s, log_prefix=log_prefix)
        self.assign = assign

    def __getattr__(self, f):
        return getattr(self.assign, f)


## FUNCS ##

def do_test(testdata, todo):
    k_search = 10
    if os.path.exists(testdata):
        print("Mmapping vecs")
        x = bvecs_mmap(testdata).astype("float32")
        print("Mmapping vecs done!")

        # assuming bigann sift1b, we can select 10000 vecs to use as queries.
        queries = x[np.random.choice(x.shape[0], 1_000_000, replace=False)]

        print(f"Testing over {x.shape[0]} vectors in R^{x.shape[1]}, 10000 queries")
    else:
        print("Dataset not real, exiting")
        sys.exit(1)

    if "search-cpu-flat" in todo:
        print("Testing brute force k-ANN search")
        index = faiss.IndexFlatL2(x.shape[1])
        index.add(x.astype('float32'))
        D_ref, I_ref = index.search(queries, k_search)
        print(f"Reference search complete!")

    if "search-cpu-shard" in todo:
        num_shards = os.cpu_count()
        print(f"Testing distributed-kANN over {num_shards} CPU shards")
        # by default split into $(nproc) shards, test each.

        data = DatasetAssignDispatch([
            DatasetAssignCustomIndex(x[(x.shape[0] // num_shards) * i : (x.shape[0] // num_shards) * (i + 1)], i)
            for i in range(num_shards)
        ], True)

        print("Starting search")
        D, I = data.search(queries, k_search)
        print("Distributed CPU search complete")

    if "search-gpu-shard" in todo:
        print("Testing k-ANN - gpu sharding")
        ngpus = faiss.get_num_gpus()

        if ngpus > 0:
            print(f"Sharding over {ngpus} gpus")
            data = DatasetAssignDispatch([
                DatasetAssignGPUCustomIndex(x[x.shape[0] * i // ngpus: x.shape[0] * (i + 1) // ngpus], i) for i in range(ngpus)
            ], True)

            D, I = data.search(queries, k_search)
            print("GPU sharding search complete!")
        else:
            print("No gpus available, must skip")



def main():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('general options')
    aa('--test', default='', help='perform tests (search-gpu-flat, search-gpu-shard, search-cpu-shard)')

    aa('--k', default=0, type=int, help='nb centroids')
    aa('--seed', default=1234, type=int, help='random seed')
    aa('--niter', default=20, type=int, help='nb iterations')
    aa('--gpu', default=-2, type=int, help='GPU to use (-2:none, -1: all)')

    group = parser.add_argument_group('I/O options')
    aa('--indata', default='',
       help='data file to load (supported formats fvecs, bvecs, npy')
    aa('--i0', default=0, type=int, help='first vector to keep')
    aa('--i1', default=-1, type=int, help='last vec to keep + 1')
    aa('--out', default='', help='file to store centroids')
    aa('--store_each_iteration', default=False, action='store_true',
       help='store centroid checkpoints')

    group = parser.add_argument_group('server options')
    aa('--server', action='store_true', default=False, help='run server')
    aa('--port', default=12345, type=int, help='server port')
    aa('--when_ready', default=None, help='store host:port to this file when ready')
    aa('--ipv4', default=False, action='store_true', help='force ipv4')

    group = parser.add_argument_group('client options')
    aa('--client', action='store_true', default=False, help='run client')
    aa('--servers', default='', help='list of server:port separated by spaces')

    args = parser.parse_args()

    if args.test:
        do_test(args.indata, args.test.split(','))
        return

    ## TODO: make truly distributed

if __name__ == '__main__':
    main()