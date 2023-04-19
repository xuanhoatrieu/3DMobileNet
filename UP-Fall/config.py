import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Training Falling Detection on UP Dataset')
    parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--n_frames', type=int, default=16, help='number of frames to condition on')
    parser.add_argument('--n_channel', default=3, type=int)
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate init')
    parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    args = parser.parse_args()
    return args

