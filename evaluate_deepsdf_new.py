#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import logging
import json
import numpy as np
import os
import torch
import trimesh
import deep_sdf.data_vtk as dt_vtk
import deep_sdf
import deep_sdf.workspace as ws
import networks.deep_sdf_decoder as decoder
import sys
from torch.autograd import Variable
import torch.utils.data as data_utils
import pdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# import matplotlib.pylab as plt


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def to_var(x):
    if torch.cuda.is_available():
        x=x.cuda()
    return Variable(x)

def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if not name in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())

def load_model(checkpoint_path, decoder_eval):
    checkpoint = torch.load(checkpoint_path)
    print(decoder_eval)
    return decoder_eval.load_state_dict(checkpoint['model_state_dict'])

def rotate_mat_z():
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])
    return rotation_matrix

def rotate_mat_y():
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    return rotation_matrix


def rotate_mat_x():
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]])
    return rotation_matrix

@torch.utils.hooks.unserializable_hook
def evaluate(experiment_directory, checkpoint_path):

    chamfer_results = []

    specs = ws.load_experiment_specifications(experiment_directory)
    logging.info("Experiment description: \n" + specs["Description"])

    data_source = specs["DataSource"]
    test_split_file = specs["TestSplit"]

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = False
    num_data_loader_threads =1
    # scene_per_subbatch =1
    batch_split = 1
    scene_per_subbatch = scene_per_batch // batch_split


    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)
    
    with open(test_split_file,"r") as f:
        test_split = json.load(f)

    sdf_dataset = dt_vtk.SDFVTKSamples(
        data_source, test_split, num_samp_per_scene
    )

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_subbatch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )


    decoder_eval = decoder.Decoder(0, **specs["NetworkSpecs"]).cuda()

    # for epoch in range(start_epoch, num_epochs + 1):

    #     start = time.time()

    #     logging.info("epoch {}...".format(epoch))
    # pdb.set_trace()
    checkpoint = torch.load(checkpoint_path)
    decoder_eval.load_state_dict(checkpoint['model_state_dict'])
    decoder_eval = decoder_eval.float()
    # decoder_eval.eval()
    for param in decoder_eval.parameters():
        param.requires_grad = False
    loss_l1 = torch.nn.L1Loss()
    loss_l2 = torch.nn.MSELoss()
    loss_log =[]
    # theta_x = torch.randn(1, requires_grad=True, dtype=torch.float)*3.1415
    # theta_y = torch.randn(1, requires_grad=True, dtype=torch.float)*3.1415
    # theta_z = torch.randn(1, requires_grad=True, dtype=torch.float)*3.1415
    # theta_x = theta_x.float()
    # theta_y = theta_y.float()
    # theta_z = theta_z.float()
    # theta_x.retain_grad()
    # theta_y.retain_grad()
    # theta_z.retain_grad()
    scale_one = torch.randn(1, requires_grad=True, dtype=torch.float)
    scale_two = torch.randn(1, requires_grad=True, dtype=torch.float)
    scale_three = torch.randn(1, requires_grad=True, dtype=torch.float)
    scale_one.retain_grad()
    scale_two.retain_grad()
    scale_three.retain_grad()
    # transform_matrix = torch.zeros(3,3).float().cuda()
    # transform_matrix.requires_grad_(True)
    # transform_matrix.retain_grad()
    transform_inpt = torch.randn(3,3).float().cuda()
    transform_inpt.requires_grad_(True)
    transform_inpt.retain_grad()
    bias = torch.zeros(3).float().cuda()
    bias.requires_grad_(True)
    bias.retain_grad()
    # pdb.set_trace()
    test_model = np.array(pd.read_csv("../chairs_segdata/points/1a8bbf2994788e2743e99e0cae970928.pts", header=None,sep=" ").values)
    num_epochs = 500
    learning_rate = 1e-3
    test_pts = torch.from_numpy(test_model).float()
    test_pts.requies_grad = False
    bt_size = 32
    num_batches = int(test_model.shape[0]//bt_size)
    sub = torch.Tensor([1]).cuda()
    reg = 1
    # rot_x = torch.from_numpy(rotate_mat_x()).double().cuda()
    # rot_x.requires_grad_(True)
    # rot_y = torch.from_numpy(rotate_mat_y()).double().cuda()
    # rot_y.requires_grad_(True)
    # rot_z = torch.from_numpy(rotate_mat_z()).double().cuda()
    # rot_z.requires_grad_(True)
    with torch.enable_grad():
        for j in range(num_epochs):
            # pdb.set_trace()
            # Process the input datag
            # sdf_data.requires_grad = False

            # sdf_data = (sdf_data.cuda()).reshape(
            #     num_samp_per_scene * scene_per_subbatch, 4
            # )
            
            # xyz = sdf_data[:, 0:3]
            # transform_matrix_update = torch.add(transform_matrix,bias)
            batch_loss=0
            for i in range(num_batches):
                test_torch = test_pts[i*bt_size:(i+1)*bt_size,:]
                # pdb.set_trace()
                # cosval_x = torch.cos(theta_x)
                # sinval_x = torch.sin(theta_x)
                # cosval_x.requires_grad_(True)
                # sinval_x.requires_grad_(True)
                # cosval_x.retain_grad()
                # sinval_x.retain_grad()
                # rot_x = torch.stack([torch.Tensor([1, 0, 0]),
                #             torch.cat([torch.Tensor([0]), cosval_x, -sinval_x]),
                #             torch.cat([torch.Tensor([0]), sinval_x, cosval_x])], dim=1).float().cuda()
                # rot_x.requires_grad_(True)
                # rot_x.retain_grad()
                # cosval_y = torch.cos(theta_y)
                # sinval_y = torch.sin(theta_y)
                # cosval_y.requires_grad_(True)
                # sinval_y.requires_grad_(True)
                # cosval_y.retain_grad()
                # sinval_y.retain_grad()
                # rot_y = torch.stack([torch.cat([cosval_y, torch.Tensor([0]), sinval_y]),
                #                     torch.Tensor([0, 1, 0]),
                #                     torch.cat([-sinval_y, torch.Tensor([0]), cosval_y])],dim=1).float().cuda()
                # rot_y.requires_grad_(True)
                # rot_y.retain_grad()
                # cosval_z = torch.cos(theta_z)
                # sinval_z = torch.sin(theta_z)
                # cosval_z.requires_grad_(True)
                # sinval_z.requires_grad_(True)
                # cosval_z.retain_grad()
                # sinval_z.retain_grad()
                # rot_z = torch.stack([torch.cat([cosval_z, -sinval_z, torch.Tensor([0])]),
                #                     torch.cat([sinval_z, cosval_z, torch.Tensor([0])]),
                #                     torch.Tensor([0, 0, 1])], dim=1).float().cuda()
                # rot_z.requires_grad_(True)
                # rot_z.retain_grad()
                scale_matrix = torch.cat([torch.cat([scale_one,torch.Tensor([0]),torch.Tensor([0])]),
                                          torch.cat([torch.Tensor([0]),scale_two,torch.Tensor([0])]),
                                          torch.cat([torch.Tensor([0]),torch.Tensor([0]),scale_three])]).view(3,3).float().cuda()
                # pdb.set_trace()
                scale_matrix.retain_grad()
                scale_matrix.requires_grad_(True)
                # transform_matrix = torch.matmul(torch.matmul(torch.matmul(rot_z,rot_y),rot_x),scale_matrix)
                transform_matrix = torch.matmul(transform_inpt, scale_matrix)
                transform_matrix.requires_grad_(True)
                transform_matrix.retain_grad()
                xyz = test_torch.cuda()
                xyz_transform = torch.matmul(xyz, transform_matrix)
                xyz_transform.requires_grad_(True)
                xyz_transform.retain_grad()
                transform_bias = torch.add(xyz_transform, bias).float()
                transform_bias.retain_grad()
                # diag_sum = torch.abs(torch.sum(torch.diag(transform_matrix)))
                # sdf_gt = sdf_data[:, 3].unsqueeze(1)
                pred_sdf = decoder_eval(transform_bias)
                # pred_sdf = decoder_eval(xyz_transform)
                # loss = loss_l1(pred_sdf, sdf_gt)
                target = torch.zeros(pred_sdf.shape[0],pred_sdf.shape[1]).float().cuda()
                # batch_loss += loss.item()
                # pdb.set_trace()
                diag_sum = torch.norm(torch.sub(torch.diag(scale_matrix),sub),2)
                diag_sum.retain_grad()
                diag_sum.requires_grad_(True)
                # diag_sum = torch.sum(torch.diag(transform_matrix)).cpu()
                loss1 = loss_l1(pred_sdf,target)
                loss2 = reg *diag_sum
                # loss2 = torch.abs(torch.sub(diag_sum,1))
                loss = torch.add(loss1,loss2)
                loss.backward(retain_graph=True)
                batch_loss+= loss.item()
                print('Batch Loss {:6.4f}'.format(loss.item()))
                with torch.no_grad():
                    # theta_z.data.sub_(theta_z.grad.data*learning_rate)
                    # theta_y.data.sub_(theta_y.data*learning_rate)
                    # theta_x.data.sub_(theta_x.grad.data*learning_rate)
                    bias.data.sub_(bias.grad.data*learning_rate)
                    scale_one.data.sub_(scale_one.grad.data*learning_rate)
                    scale_two.data.sub_(scale_two.grad.data*learning_rate)
                    scale_three.data.sub_(scale_three.grad.data*learning_rate)
                    transform_inpt.data.sub_(transform_inpt.grad.data*learning_rate)
                    # theta_z.grad.data.zero_()
                    # theta_y.grad.data.zero_()
                    # theta_x.grad.data.zero_()
                    bias.grad.data.zero_()
                    scale_one.grad.data.zero_()
                    scale_three.grad.data.zero_()
                    scale_two.grad.data.zero_()
                    scale_matrix.grad.data.zero_()
                    transform_bias.grad.data.zero_()
                    xyz_transform.grad.data.zero_()
                    transform_matrix.grad.data.zero_()
                    transform_inpt.grad.data.zero_()
                    diag_sum.grad.data.zero_()
                    # rot_z.grad.data.zero_()
                    # rot_x.grad.data.zero_()
                    # rot_y.grad.data.zero_()
            # pdb.set_trace()
            actual_loss = (batch_loss*bt_size)/(test_model.shape[0])
            # print("Loss after {} epoch is {:6.4f}".format(j,batch_loss))
            print("Loss after {} epoch is {:6.4f}".format(j,actual_loss))
            loss_log.append(actual_loss)
    pdb.set_trace()
    fig,ax = plt.subplots()
    ax.plot(np.arange(num_epochs),loss_log)
    ax.set(xlabel='iterations',ylabel='transformationloss')
    plt.savefig('Transformation_loss_new.png')
    torch.save(transform_matrix,'transform_matrix_new.pt')
    torch.save(bias,'bias_new.pt')
    test_pts = torch.from_numpy(pd.read_csv('test_model.pts',header=None, sep=' ').values).cuda()
    transform_pts = torch.matmul(test_pts, transform_matrix.double())
    transform_pts = torch.add(transform_pts, bias.double()).cpu().detach().numpy()
    np.savetxt('transform_points_new.pts',transform_pts)
    plot_heatmap(experiment_directory, checkpoint_path)
    # avg_loss = sum(loss_log)/len(loss_log)
    # print('Average loss is {:6.4f}'.format(avg_loss))

                
    # with open(
    #     os.path.join(
    #         ws.get_evaluation_dir(experiment_directory, checkpoint, True),
    #         "chamfer.csv",
    #     ),
    #     "w",
    # ) as f:
    #     f.write("shape, chamfer_dist\n")
    #     for result in chamfer_results:
    #         f.write("{}, {}\n".format(result[0], result[1]))

def plot_heatmap(experiment_directory, checkpoint_path):
    from matplotlib import cm
    specs = ws.load_experiment_specifications(experiment_directory)
    decoder_eval = decoder.Decoder(0, **specs["NetworkSpecs"]).cuda()
    checkpoint = torch.load(checkpoint_path)
    decoder_eval.load_state_dict(checkpoint['model_state_dict'])
    decoder_eval = decoder_eval.float().cuda()
    test_model_transformed = torch.from_numpy(np.array(pd.read_csv("transform_points_new.pts", header=None,sep=" ").values)).float().cuda()
    distances = decoder_eval(test_model_transformed)
    distances = distances.detach().cpu().numpy()
    transformed_pts = test_model_transformed.detach().cpu().numpy()
    x = transformed_pts[:,0]
    y = transformed_pts[:,1]
    z = transformed_pts[:,2]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    fig = matplotlib.pyplot.gcf()
    ax = Axes3D(fig)
    cax = ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=distances.flatten())
    # fig.colorbar(cax, shrink=0.5, aspect=10)
    cbar=plt.colorbar(cax)
    # plt.show()
    # pdb.set_trace()
    # ax = sns.heatmap(lis, linewidth=0.5)
    ax.figure.savefig('heatmap.png')

        

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Evaluate a DeepSDF autodecoder"
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment specifications in "
        + '"specs.json", and logging will be done in this directory as well.',
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint to test.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    evaluate(
        args.experiment_directory,
        args.checkpoint,
    )
    # plot_heatmap(
    #     args.experiment_directory,
    #     args.checkpoint
    #     )