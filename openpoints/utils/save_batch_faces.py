import pickle
import os
import cv2
 
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


def save_batch_samples(dir_path, epoch, batch_idx, batch_data):
    # print('batch_data.shape:', batch_data.shape)
    dir_sample = f'epoch={epoch}_batch={batch_idx}'
    path_dir_sample = os.path.join(dir_path, dir_sample)
    os.makedirs(path_dir_sample, exist_ok=True)

    for sample_idx in range(batch_data.shape[0]):
        path_save_train_sample = os.path.join(path_dir_sample, f'batch_idx={batch_idx}_sample={sample_idx}.obj')
        write_obj(path_save_train_sample, batch_data[sample_idx])
