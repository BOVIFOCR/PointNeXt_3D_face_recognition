import sys
import pickle
import os
import cv2
import wandb
import numpy as np
 
def save_batch_faces(data, path):
    # move tensor from GPU to CPU
    data_x = data['x'].cpu()
    data_pos = data['pos'].cpu()
    data_y = data['y'].cpu()

    path_data_x = os.path.join(path,"dataX.pkl")
    path_data_pos = os.path.join(path,"dataPOS.pkl")
    path_data_y = os.path.join(path,"dataY.pkl")

  # save tensor to disk using pickle
    with open(path_data_x, 'wb') as f:
        pickle.dump(data_x, f)
    with open(path_data_pos, 'wb') as f:
        pickle.dump(data_pos, f)
    with open(path_data_y, 'wb') as f:
        pickle.dump(data_y, f)


# Bernardo (adapted from 'https://github.com/youngLBW/HRN/blob/main/util/util_.py#L343')
def write_obj(save_path, vertices, faces=None, UVs=None, faces_uv=None, normals=None, faces_normal=None, texture_map=None, save_mtl=False, vertices_color=None):
    save_dir = os.path.dirname(save_path)
    save_name = os.path.splitext(os.path.basename(save_path))[0]

    if save_mtl or texture_map is not None:
        if texture_map is not None:
            cv2.imwrite(os.path.join(save_dir, save_name + '.jpg'), texture_map)
        with open(os.path.join(save_dir, save_name + '.mtl'), 'w') as wf:
            wf.write('# Created by HRN\n')
            wf.write('newmtl material_0\n')
            wf.write('Ka 1.000000 0.000000 0.000000\n')
            wf.write('Kd 1.000000 1.000000 1.000000\n')
            wf.write('Ks 0.000000 0.000000 0.000000\n')
            wf.write('Tr 0.000000\n')
            wf.write('illum 0\n')
            wf.write('Ns 0.000000\n')
            wf.write('map_Kd {}\n'.format(save_name + '.jpg'))

    with open(save_path, 'w') as wf:
        if save_mtl or texture_map is not None:
            wf.write("# Create by HRN\n")
            wf.write("mtllib ./{}.mtl\n".format(save_name))

        if vertices_color is not None:
            for i, v in enumerate(vertices):
                wf.write('v {} {} {} {} {} {}\n'.format(v[0], v[1], v[2], vertices_color[i][0], vertices_color[i][1], vertices_color[i][2]))
        else:
            for v in vertices:
                wf.write('v {} {} {}\n'.format(v[0], v[1], v[2]))

        if UVs is not None:
            for uv in UVs:
                wf.write('vt {} {}\n'.format(uv[0], uv[1]))

        if normals is not None:
            for vn in normals:
                wf.write('vn {} {} {}\n'.format(vn[0], vn[1], vn[2]))

        if faces is not None:
            for ind, face in enumerate(faces):
                if faces_uv is not None or faces_normal is not None:
                    if faces_uv is not None:
                        face_uv = faces_uv[ind]
                    else:
                        face_uv = face
                    if faces_normal is not None:
                        face_normal = faces_normal[ind]
                    else:
                        face_normal = face
                    row = 'f ' + ' '.join(['{}/{}/{}'.format(face[i], face_uv[i], face_normal[i]) for i in range(len(face))]) + '\n'
                else:
                    row = 'f ' + ' '.join(['{}'.format(face[i]) for i in range(len(face))]) + '\n'
                wf.write(row)


def save_batch_samples(dir_path, epoch, batch_idx, points_orig, points_sampl, pointclouds_outputs_layers=None):
    dir_sample = f'epoch={epoch}_batch={batch_idx}'
    path_dir_sample = os.path.join(dir_path, dir_sample)
    os.makedirs(path_dir_sample, exist_ok=True)

    for sample_idx in range(points_orig.shape[0]):
        path_save_points_orig_sample = os.path.join(path_dir_sample, f'batch_idx={batch_idx}_sample={sample_idx}_points_orig.obj')
        write_obj(path_save_points_orig_sample, points_orig[sample_idx])

        path_save_points_sampl_sample = os.path.join(path_dir_sample, f'batch_idx={batch_idx}_sample={sample_idx}_points_sampl.obj')
        write_obj(path_save_points_sampl_sample, points_sampl[sample_idx])

        if not pointclouds_outputs_layers is None:
            for interm_layer_idx in range(len(pointclouds_outputs_layers)):
                interm_pcs, interm_feat = pointclouds_outputs_layers[interm_layer_idx]
                path_save_interm_pc_sample = os.path.join(path_dir_sample, f'batch_idx={batch_idx}_sample={sample_idx}_interm_layer={interm_layer_idx}.obj')
                write_obj(path_save_interm_pc_sample, interm_pcs[sample_idx])


def save_gradients(dir_path, epoch, batch_idx, model, writer):
    # dir_gradients = f'epoch={epoch}_batch={batch_idx}'
    # path_dir_gradients = os.path.join(dir_path, dir_gradients)
    # os.makedirs(path_dir_gradients, exist_ok=True)

    if writer is not None:
        for p_idx, (name, param) in enumerate(model.named_parameters()):
            if not param.grad is None:
                param_grad_cpu = param.grad.squeeze().data.cpu().numpy()
                if len(param_grad_cpu.shape) == 1:
                    param_grad_cpu = np.expand_dims(param_grad_cpu, axis=1)
                if param_grad_cpu.shape[0] > param_grad_cpu.shape[1]:
                    param_grad_cpu = np.transpose(param_grad_cpu)

                # print('param_grad_cpu    name:', name, '    shape:', param_grad_cpu.shape)
                writer.add_image(f'train_grads_{name}_shape={param_grad_cpu.shape}', param_grad_cpu, epoch, dataformats='HW')

        # TODO
        # group all gradients in one single image and save it
        # writer.add_images()
