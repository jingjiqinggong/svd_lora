import random
from torch.utils.data import Dataset
from einops import rearrange, repeat
import decord
import torch
import torchvision.transforms as T
decord.bridge.set_bridge('torch')
from .bucketing import sensible_buckets
# import imageio

# Inspired by the VideoMAE repository.
def normalize_input(
    item, 
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225],
    use_simple_norm=True
):
    if item.dtype == torch.uint8 and not use_simple_norm:
        item = rearrange(item, 'f c h w -> f h w c')
        
        item = item.float() / 255.0
        mean = torch.tensor(mean)
        std = torch.tensor(std)

        out = rearrange((item - mean) / std, 'f h w c -> f c h w')
        
        return out
    else:
        # Normalize between -1 & 1
        item = rearrange(item, 'f c h w -> f h w c')
        return  rearrange(item / 127.5 - 1.0, 'f h w c -> f c h w')

def get_prompt_ids(prompt, tokenizer):
    if tokenizer is None:
        prompt_ids = torch.tensor([0])
    else:
        prompt_ids = tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
        ).input_ids[0]
    return prompt_ids

# def process_video(vid_path, use_bucketing, w, h, get_frame_buckets, get_frame_batch):
#     if use_bucketing:
#         vr = decord.VideoReader(vid_path)
#         resize = get_frame_buckets(vr)
#         video = get_frame_batch(vr, resize=resize)

#     else:
#         vr = decord.VideoReader(vid_path)
#         video = get_frame_batch(vr)

#     return video, vr

# [Note]  my modify
# Fix len(dataset) = 1，n_sample_frames 帧
# 每次遍历dataloader都 从视频中随机取起点，
# Each time we enumerate dataloader, we take a random starting point from the original video and do sample
class SingleVideoDataset(Dataset):
    def __init__(
        self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 4,
            frame_step: int = 1,
            single_video_path: str = "",
            single_video_prompt: str = "",
            use_caption: bool = False,
            use_bucketing: bool = False,
            **kwargs
    ):
        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing
        self.frames = []
        self.index = 1      # unused

        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        assert single_video_path.endswith(self.vid_types), 'video filename suffix should in: (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")'
        assert n_sample_frames >= 1 and frame_step >= 1, "should satisfy: n_sample_frames >= 1 and frame_step >= 1 "
        
        self.n_sample_frames = n_sample_frames
        self.frame_step = frame_step

        self.single_video_path = single_video_path
        self.single_video_prompt = single_video_prompt
        self.prompt_ids = get_prompt_ids(single_video_prompt, self.tokenizer)

        self.width = width
        self.height = height
        
        self.vr = decord.VideoReader(self.single_video_path)
        self.vr_len = len(self.vr)
        self.start_idx_max = self.vr_len - 1 - (n_sample_frames-1) * frame_step     # 起点 [0, self.start_idx_max]
        assert self.start_idx_max >= 0, "should satisfy: 1 + n_sample_frames * frame_step <= total number of frames"
        
        self.resize = self.get_frame_buckets(self.vr) if self.use_bucketing else None
        
    # def create_video_chunks(self):
    #     # Create a list of frames separated by sample frames
    #     # [(1,2,3), (4,5,6), ...]
    #     vr = decord.VideoReader(self.single_video_path)
    #     vr_range = range(1, len(vr), self.frame_step)

    #     #eg. frame_step = 24, len(vr) = 360. self.frames : [(1, 25, 49, 73, 97, 121, 145), (169, 193, 217, 241, 265, 289, 313), (337,)]
    #     self.frames = list(self.chunk(vr_range, self.n_sample_frames))

    #     # Delete any list that contains an out of range index.
    #     for i, inner_frame_nums in enumerate(self.frames):
    #         for frame_num in inner_frame_nums:
    #             if frame_num > len(vr):
    #                 print(f"Removing out of range index list at position: {i}...")
    #                 del self.frames[i]

    #     return self.frames

    # def chunk(self, it, size):
    #     it = iter(it)
    #     return iter(lambda: tuple(islice(it, size)), ())


    def get_frame_buckets(self, vr):    
        h, w, _ = vr[0].shape
        width, height = sensible_buckets(self.width, self.height, h, w)
        resize = T.transforms.Resize((height, width), antialias=True)
        return resize
    
    # 随机一个起点，每间隔frame_step取一帧，一次取出 n_sample_frames 帧
    # !![Note] Now, during each training_step, n_sample_frames are seleted with 'frame_step' as the interval.
    # todo 修改采样方式? 
    def get_frame_batch_random(self, vr, resize=None):
        # x + (n_sample_frames-1) * frame_step <= self.vr_len
        start_idx = random.randint(0,self.start_idx_max)
        end_idx = start_idx + (self.n_sample_frames - 1) * self.frame_step
        indices = list(range(start_idx, end_idx+1, self.frame_step))
        frames = vr.get_batch(indices)
        video = rearrange(frames, "f h w c -> f c h w")
        if resize is not None: video = resize(video)
        
        return video
    
    # def process_video_wrapper(self, vid_path):
    #     video, vr = process_video(
    #             vid_path,
    #             self.use_bucketing,
    #             self.width, 
    #             self.height, 
    #             self.get_frame_buckets, 
    #             self.get_frame_batch
    #         )
        
    #     return video, vr 

    # def single_video_batch(self, index):
    #     train_data = self.single_video_path
    #     self.index = index

    #     if train_data.endswith(self.vid_types):
    #         video, _ = self.process_video_wrapper(train_data)

    #         prompt = self.single_video_prompt
    #         prompt_ids = get_prompt_ids(prompt, self.tokenizer)

    #         return video, prompt, prompt_ids
    #     else:
    #         raise ValueError(f"Single video is not a video type. Types: {self.vid_types}")
    
    @staticmethod
    def __getname__(): return 'single_video'

    def __len__(self):
        return 1

    def __getitem__(self, index):
        
        # 随机一个起点，每间隔frame_step取一帧，共取出 n_sample_frames 帧
        # !![Note] Now, during each training_step, n_sample_frames are seleted with 'frame_step' as the interval.
        video_chuck = self.get_frame_batch_random(self.vr, resize=self.resize) 

        example = {
            "pixel_values": normalize_input(video_chuck),
            "prompt_ids": self.prompt_ids,
            "text_prompt": self.single_video_prompt,
            'dataset': self.__getname__()
        }

        return example