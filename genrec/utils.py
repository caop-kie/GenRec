from tensorboardX import SummaryWriter

class Summarizer(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)

def generate_vocabs(datasets):
    event_type_set = set()
    role_type_set = set()
    for dataset in datasets:
        event_type_set.update(dataset.event_type_set)
        role_type_set.update(dataset.role_type_set)
    
    event_type_itos = sorted(event_type_set)
    role_type_itos = sorted(role_type_set)
    
    event_type_stoi = {k: i for i, k in enumerate(event_type_itos)}
    role_type_stoi = {k: i for i, k in enumerate(role_type_itos)}
    
    return {
        'event_type_itos': event_type_itos,
        'event_type_stoi': event_type_stoi,
        'role_type_itos': role_type_itos,
        'role_type_stoi': role_type_stoi,
    }

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1

import torch
import collections
from data import GenBatch


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, collections.OrderedDict):
            # OrderedDict has attributes that needs to be preserved
            od = collections.OrderedDict((key, _apply(value)) for key, value in x.items())
            od.__dict__ = x.__dict__
            return od
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        # elif isinstance(x, GenBatch):
        #     return {_apply(x) for x in x}
        else:
            return x

    # new_sample = GenBatch()
    # for name in GenBatch._fields:
    #     old_value = getattr(sample, name)
    #     # setattr(new_sample, name, _apply(old_value))
    new_sample = GenBatch(
        input_text = _apply(getattr(sample, 'input_text')),
        target_text = _apply(getattr(sample, 'target_text')),
        enc_idxs = _apply(getattr(sample, 'enc_idxs')),
        enc_attn = _apply(getattr(sample, 'enc_attn')),
        dec_idxs = _apply(getattr(sample, 'dec_idxs')),
        dec_attn = _apply(getattr(sample, 'dec_attn')),
        lbl_idxs = _apply(getattr(sample, 'lbl_idxs')),
        raw_lbl_idxs = _apply(getattr(sample, 'raw_lbl_idxs')),
        enc_user_whole = _apply(getattr(sample, 'enc_user_whole')),
        enc_item_whole = _apply(getattr(sample, 'enc_item_whole')),
        enc_attrs = _apply(getattr(sample, 'enc_attrs')))

    # return _apply(sample)
    return new_sample


def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.to(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)

class CudaEnvironment(object):
    def __init__(self):
        cur_device = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties("cuda:{}".format(cur_device))
        self.name = prop.name
        self.major = prop.major
        self.minor = prop.minor
        self.total_memory_in_GB = prop.total_memory / 1024 / 1024 / 1024

    @staticmethod
    def pretty_print_cuda_env_list(cuda_env_list):
        """
        Given a list of CudaEnviorments, pretty print them
        """
        num_workers = len(cuda_env_list)
        center = "CUDA enviroments for all {} workers".format(num_workers)
        banner_len = 40 - len(center) // 2
        first_line = "*" * banner_len + center + "*" * banner_len
        # logger.info(first_line)
        # for r, env in enumerate(cuda_env_list):
        #     logger.info(
        #         "rank {:3d}: ".format(r)
        #         + "capabilities = {:2d}.{:<2d} ; ".format(env.major, env.minor)
        #         + "total memory = {:.3f} GB ; ".format(env.total_memory_in_GB)
        #         + "name = {:40s}".format(env.name)
        #     )
        # logger.info(first_line)