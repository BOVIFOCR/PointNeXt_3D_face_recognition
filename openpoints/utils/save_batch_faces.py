import pickle
import os 
 
def save_batch_faces(data,path):
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

