#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
#Also Miguel Dominguez 2019
import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import vtk
from vtk.util import numpy_support
import pdb
import workspace as ws
import torch.utils.data as data_utils
import json
import matplotlib.pyplot as plt

def get_instance_filenames(data_source, split):
    npzfiles = []
    # pdb.set_trace()
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(class_name, instance_name + "/models/model_normalized.obj")
                if not os.path.isfile(
                    os.path.join(data_source, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning("Requested non-existent file '{}'".format(instance_filename))
                npzfiles += [instance_filename]
    return npzfiles

# def unpack_sdf_vtk_samples_repo(filename, num_points=1000):
#     #npz = np.load(filename)
#     reader = vtk.vtkOBJReader()
#     reader.SetFileName(filename)
#     reader.Update()
#     polydata = reader.GetOutput()
#     pdd = vtk.vtkImplicitPolyDataDistance()
#     pdd.SetInput(polydata)
#     sdf = []
#     closestPoints = []
#     F=3
#     V = np.random.uniform(-1,1,(num_points/2,F))
#     for i in range(num_points / 2):
#         closestPoint = np.zeros(3)
#         distance = pdd.EvaluateFunctionAndGetClosestPoint(V[i,:],closestPoint)
#         sdf.append(distance)
#         closestPoints.append(closestPoint)
    
#     closestPoints = np.stack(closestPoints)
#     sdf = np.stack(sdf)
#     sdf = np.expand_dims(np.concatenate((sdf,np.zeros(num_points/2)),axis=0),axis=1)
#     V = np.concatenate((V,closestPoints),axis=0)
#     #print(V.shape)
#     #print(sdf.shape)
#     samples = np.concatenate((V,sdf),axis=1)
#     samples = torch.from_numpy(samples).float()

#     return samples

    
def unpack_sdf_vtk_samples(filename, num_points=1000):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(filename)
    reader.Update()
    polydata = reader.GetOutput()
    pdd = vtk.vtkImplicitPolyDataDistance()
    pdd.SetInput(polydata)
    sdf = []
    closestPoints = []
    vectors = []
    F=3
    V = np.random.uniform(-1,1,(int(num_points),F))
    for i in range(int(num_points)):
        closestPoint = np.zeros(3)
        distance = pdd.EvaluateFunctionAndGetClosestPoint(V[i,:],closestPoint)
        sdf.append(distance)
        vector_gt = np.subtract(V[i,:],closestPoint)
        closestPoints.append(closestPoint)
        vectors.append(vector_gt)
    vectors = np.stack(vectors)
    vectors = np.reshape(vectors,(vectors.shape[0],-1))
    sdf = np.stack(sdf)
    sdf = np.reshape(sdf,(sdf.shape[0],-1))
    samples = np.concatenate((V,sdf,vectors),axis=1)
    samples = torch.from_numpy(samples).float()
    
    return samples

# def unpack_sdf_vtk_samples(filename, num_points=1000):
#     #npz = np.load(filename)
#     # pdb.set_trace()
#     reader = vtk.vtkOBJReader()
#     reader.SetFileName(filename)
#     reader.Update()
#     polydata = reader.GetOutput()
#     pdd = vtk.vtkImplicitPolyDataDistance()
#     pdd.SetInput(polydata)
#     sdf = []
#     closestPoints = []
#     F=3
#     V = np.random.uniform(-1,1,(num_points,F))
#     for i in range(num_points):
#         closestPoint = np.zeros(3)
#         distance = pdd.EvaluateFunction(V[i,:])
#         sdf.append(distance)
#         closestPoints.append(closestPoint)
    
#     closestPoints = np.stack(closestPoints)
#     sdf = np.stack(sdf)
#     sdf = np.reshape(sdf,(sdf.shape[0],-1))
#     #sdf = np.expand_dims(np.concatenate((sdf,np),axis=0),axis=1)
#     #V = np.concatenate((V,closestPoints),axis=0)
#     #print(V.shape)
#     #print(sdf.shape)
#     # print(V.shape)
#     # print(sdf.shape)
#     # print(np.amin(sdf))
#     # print(np.amax(sdf))
#     samples = np.concatenate((V,sdf),axis=1)
#     samples = torch.from_numpy(samples).float()
#     # print(samples.shape)
#     return samples


class SDFVTKSamples(torch.utils.data.Dataset):
    def __init__(
        self, data_source, split, num_points, print_filename=False, num_files=1000000
    ):
        self.num_points = num_points
        self.data_source = data_source
        self.files = get_instance_filenames(data_source, split)

        logging.debug(
            "using " + str(len(self.files)) + " shapes from data source " + data_source
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = os.path.join(self.data_source, self.files[idx])
        print('filename is {}'.format(filename))
        return unpack_sdf_vtk_samples(filename, self.num_points), idx




if __name__=='__main__':
    train_split_file = "examples/splits/sv2_chairs_train.json"
    pdb.set_trace()
    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    sdf_dataset = SDFVTKSamples(
        "/home/nagaharish/Downloads/deepsdf-registration", train_split, 16384)
    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    for sdf_data, indices in sdf_loader:
        print(sdf_data.shape)