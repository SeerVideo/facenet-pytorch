# There is some kind of io bottleneck preventing full gpu utilization
# It could be opening the image with pillow
# Could also be the forward pass within the source code itself

import os
from PIL import Image
import torch
from glob import glob
from facenet_pytorch import MTCNN
from tqdm import tqdm
from multiprocessing import Process, Pool, Event
import time
import argparse


def align(device_id, src_split, dst_split, event):
    os.environ["CUDA_VISIBLE_DEVICES"]='{}'.format(str(device_id))
    device = torch.device('cuda')
    mtcnn = MTCNN(image_size=img_size, margin=margin,
                  device=device, min_face_size=min_face_size,
                  keep_all=True)
    total = len(src_split)
    failed = 0

    for src, dst in zip(src_split, dst_split):
        try:
            img = Image.open(src)
        except:
            print('Failed to open image: {}'.format(src))
            failed += 1
            continue
        try:
            img_tensor = mtcnn(img, save_path=dst)
        except:
            print('Failed to align image: {}'.format(src))
            failed += 1
            continue

    print('Process running on GPU: {} finished'.format(str(device_id)))
    print('Successfully aligned: {}/{} faces'.format(total-failed,total))
    event.set()


def start_processes(args, paths):
    # paths : tuple (list, list)
    src_paths, dst_paths = paths
    partition = len(src_paths) / args.n_gpu
    processes = []
    events = []

    for i in range(args.n_gpu):
        start = int(partition * i)
        end = int(partition * (i+1))
        src_split, dst_split = src_paths[start:end], dst_paths[start:end]
        event = Event()
        events.append(event)
        p = Process(target=align, args=(i, src_split, dst_split, event,))
        p.start()
        processes.append(p)

    try:
        for e in events:
            e.wait()
    except KeyboardInterrupt:
        for i, p in enumerate(processes):
            print('Interrupted process: {}'.format(str(i)))
            p.terminate()
            print('Killed process: {}'.format(str(i)))


def generate_paths(args):
    print('Generating file paths...')
    src_root = args.src_root
    dst_root = args.dst_root
    ext = args.ext
    class_dirs = os.listdir(src_root)
    src_paths = []
    dst_paths = []

    for z, _dir in tqdm(enumerate(class_dirs), total=len(class_dirs)):
        dir_path = os.path.join(src_root, _dir)
        image_paths = glob(dir_path+'/*')

        for i, img_path in enumerate(image_paths):
            fn = img_path.split('/')[-1].split('.')[0]
            dst_dir = os.path.join(dst_root, _dir)
            if os.path.exists(dst_dir) is False:
                os.makedirs(dst_dir)
            dst_path = os.path.join(dst_root, _dir, '{}.{}'.format(fn, ext))
            if os.path.exists(dst_path) is True:
                continue
            src_paths.append(img_path)
            dst_paths.append(dst_path)

    print('Found {} images to align'.format(len(src_paths)))

    return (src_paths, dst_paths)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_root', type=str, default='/home/seercv/facenet_assets/seer')
    parser.add_argument('--dst_root', type=str, default='/home/seercv/facenet_assets/seer_a')
    parser.add_argument('--n_gpu', type=int, default=2,
                        help='number of gpus to utilize')
    parser.add_argument('--ext', type=str, default='png',
                        help='file extension to save aligned image.\
                              png allows for perfect loading from disk\
                              but jpg saves disk space')
    parser.add_argument('--img_size', type=int, default=160,
                        help='dimension of aligned image')
    parser.add_argument('--margin', type=int, default=44,
                        help='pixels to pad around face detection during alignment')
    parser.add_argument('--min_face_size', type=int, default=40,
                        help='minimum size of face to keep the aligned image')
    args = parser.parse_args()

    img_size = args.img_size
    margin = args.margin
    min_face_size = args.min_face_size

    paths = generate_paths(args)
    start_processes(args, paths)
