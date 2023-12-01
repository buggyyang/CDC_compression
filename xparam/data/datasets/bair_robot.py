import numpy as np
import torch
import torch.utils.data
import os

class RobotPushing(torch.utils.data.Dataset):
    """
    dataset class for moving-mnist dataset
    """
    def __init__(self, is_train, data_path=None):
        super(RobotPushing, self).__init__()
        if data_path is None:
            self.data_path = '/local-scratch/chenleic/Data/BAIR/robot_pushing/main_frames'
            # print()
        else:
            self.data_path = data_path


        all_vid_list = sorted(os.listdir(self.data_path))

        self.is_train = is_train
        if self.is_train:
            self.vid_list = all_vid_list[:10000]
        else:
            self.vid_list = all_vid_list[-100:]


    def __len__(self):
        return len(self.vid_list)


    def __getitem__(self, item):
        frames = np.load(os.path.join(self.data_path, self.vid_list[item]))
        frames = torch.FloatTensor(frames).permute(3, 0, 2, 1).contiguous()

        frames = frames/255

        return frames


if __name__ == '__main__':
    dataset = RobotPushing(is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset)

    import matplotlib.pyplot as plt

    for batch in dataloader:
        frame = batch[0,:,0,:,:].permute(1,2,0).contiguous()
        plt.imshow(frame)
        plt.draw()
        plt.savefig('/local-scratch/chenleic/Projects/seq_flow/seq_flow_robot_results/check_frame.jpg')
        break

        print(batch.size())


