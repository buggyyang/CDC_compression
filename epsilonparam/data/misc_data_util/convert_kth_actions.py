import os
from imageio import imwrite as imsave
from moviepy.editor import VideoFileClip
from .kth_actions_frames import kth_actions_dict

settings = ['d1', 'd2', 'd3', 'd4']
actions = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']
person_ids = {'train': ['11', '12', '13', '14', '15', '16', '17', '18'],
              'val': ['19', '20', '21', '23', '24', '25', '01', '04'],
              'test': ['22', '02', '03', '05', '06', '07', '08', '09', '10']}


def convert(data_path):
    # iterate through the data splits
    for data_split in ['train', 'val', 'test']:
        print('Converting ' + data_split)
        os.makedirs(os.path.join(data_path, data_split))
        split_person_ids = person_ids[data_split]
        # iterate through the ids, actions, and settings for this split
        for person_id in split_person_ids:
            print('     Converting person' + person_id)
            for action in kth_actions_dict['person'+person_id]:
                for setting in kth_actions_dict['person'+person_id][action]:
                    frame_nums = kth_actions_dict['person'+person_id][action][setting]
                    if len(frame_nums) > 0:
                        start_frames = [frame_pair[0] for frame_pair in frame_nums]
                        end_frames = [frame_pair[1] for frame_pair in frame_nums]
                        # load the video
                        file_name = 'person' + person_id + '_' + action + '_' + setting + '_uncomp.avi'
                        print(file_name)
                        video = VideoFileClip(os.path.join(data_path, action, file_name))
                        # write each sequence to a directory
                        sequence_frame_index = 0
                        sequence_index = 0
                        sequence_name = ''
                        in_sequence = False
                        for frame_index, frame in enumerate(video.iter_frames()):
                            if frame_index + 1 in start_frames:
                                # start a new sequence
                                in_sequence = True
                                sequence_frame_index = 0
                                sequence_name = 'person' + person_id + '_' + action + '_' + setting + '_' + str(sequence_index)
                                os.makedirs(os.path.join(data_path, data_split, sequence_name))
                            if frame_index + 1 in end_frames:
                                # end the current sequence
                                in_sequence = False
                                sequence_index += 1
                                if frame_index + 1 == max(end_frames):
                                    break
                            if in_sequence:
                                # write frame to the current sequence
                                frame = frame.astype('float32') / 255.
                                imsave(os.path.join(data_path, data_split, sequence_name, str(sequence_frame_index) + '.png'), frame)
                                sequence_frame_index += 1
                        del video.reader
                        del video
