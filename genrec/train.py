import os, sys, json, logging, time, pprint, tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from model import GenerativeModel
from data import Dataset
from argparse import ArgumentParser, Namespace
import datetime
import math
from utils import *

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-pmp', '--pretrained_model_path', type=str, required=False)

args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)
config.update(args.__dict__)
config = Namespace(**config)

# fix random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# logger and summarizer
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = os.path.join(config.output_dir, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_path = os.path.join(output_dir, "train.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]',
                    handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
summarizer = Summarizer(output_dir)

# set GPU device
torch.cuda.set_device(config.gpu_device)

# output
with open(os.path.join(output_dir, 'config.json'), 'w') as fp:
    json.dump(vars(config), fp, indent=4)
best_model_path = os.path.join(output_dir, 'best_model.mdl')
dev_prediction_path = os.path.join(output_dir, 'pred.dev.txt')
test_prediction_path = os.path.join(output_dir, 'pred.test.txt')

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir, add_prefix_space=True)
special_tokens = ['<mask>']
tokenizer.add_tokens(special_tokens)

prev_best_test_scores = 0.0

# load data
if config.pretrain:
    train_set = Dataset(tokenizer,
                        config.max_length,
                        config.train_file,
                        config.max_output_length,
                        is_pretrain=True)
    dev_set = train_set
    test_set = train_set
else:
    train_set = Dataset(tokenizer, config.max_length, config.train_file, config.max_output_length)
    dev_set = Dataset(tokenizer, config.max_length, config.dev_file, config.max_output_length)
    test_set = Dataset(tokenizer, config.max_length, config.test_file, config.max_output_length)

train_batch_num = len(train_set) // config.train_batch_size + (len(train_set) % config.train_batch_size != 0)
dev_batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)

# initialize the model
model = GenerativeModel(config, tokenizer,
                        prompt_tuning=False,
                        prompt_type_method='prefix',
                        encoder_prompt_length=100,
                        encoder_prompt_projection=False,
                        encoder_embed_dim=768,
                        encoder_prompt_dim=2 * 768,
                        encoder_layers=6,
                        encoder_attention_heads=12,
                        decoder_prompt_length=100,
                        decoder_prompt_projection=False,
                        decoder_embed_dim=768,
                        decoder_prompt_dim=2 * 768,
                        decoder_layers=6,
                        decoder_attention_heads=12)

model_path = args.pretrained_model_path
if model_path is not None:
    logger.info("Loading model from {}".format(model_path))
    model.load_state_dict(torch.load(model_path, map_location=f'cuda:{config.gpu_device}'), strict=False)
    logger.info("Loaded model from {}".format(model_path))

model.cuda(device=config.gpu_device)

# optimizer
param_groups = [{'params': model.parameters(), 'lr': config.learning_rate, 'weight_decay': config.weight_decay}]
optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=train_batch_num * config.warmup_epoch,
                                           num_training_steps=train_batch_num * config.max_epoch)


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def get_full_sort_score(epoch, answers, pred_list):
    recall, ndcg = [], []
    for k in [5, 10, 15, 20]:
        recall.append(recall_at_k(answers, pred_list, k))
        ndcg.append(ndcg_k(answers, pred_list, k))
    post_fix = {
        "Epoch": epoch,
        "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
        "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
        "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
    }
    print(post_fix)
    return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)


# start training
logger.info("Start training ...")
summarizer_step = 0
best_dev_epoch = -1
best_dev_scores = {
    'ndcg@20': 0.0
}

cuda_env = CudaEnvironment()
collate_fn = train_set.pretrain_collate_fn if config.pretrain else train_set.collate_fn
for epoch in range(1, config.max_epoch + 1):
    logger.info(log_path)
    logger.info(f"Epoch {epoch}")

    # training
    progress = tqdm.tqdm(total=train_batch_num, ncols=75, desc='Train {}'.format(epoch))
    model.train()
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(
            DataLoader(train_set, batch_size=config.train_batch_size // config.accumulate_step,
                       shuffle=True, drop_last=False, collate_fn=collate_fn)):
        try:
            gpu_batch = move_to_cuda(batch)
            loss = model(gpu_batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(
                    "attempting to recover from OOM in forward/backward pass"
                )
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        if batch_idx % 1000 == 0:
            logger.info('epoch {} batch_idx {} loss {}'.format(epoch, batch_idx, loss))

        # record loss
        summarizer.scalar_summary('train/loss', loss, summarizer_step)
        summarizer_step += 1

        loss = loss * (1 / config.accumulate_step)
        loss.backward()

        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()

    progress.close()

    if epoch % 10 == 0:
        logger.info(
            'Saving epoch_{} model to {}'.format(epoch, os.path.join(output_dir, 'epoch_{}.mdl'.format(epoch))))
        torch.save(model.state_dict(), os.path.join(output_dir, 'epoch_{}.mdl'.format(epoch)))

    if config.finetuning:
        progress = tqdm.tqdm(total=dev_batch_num, ncols=75, desc='Dev {}'.format(epoch))
        model.eval()
        torch.cuda.empty_cache()
        best_dev_flag = False
        write_output = []
        my_output = []
        dev_gold_arg_num, dev_pred_arg_num, dev_match_arg_id, dev_match_arg_cls = 0, 0, 0, 0

        final_ans_list = []
        final_item20_preds = []

        for batch_idx, batch in enumerate(DataLoader(dev_set, batch_size=config.eval_batch_size,
                                                     shuffle=False, collate_fn=dev_set.collate_fn)):
            progress.update(1)
            try:
                gpu_batch = move_to_cuda(batch)
                items_pred = model.predict(gpu_batch, max_length=config.max_output_length)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(
                        "attempting to recover from OOM in forward/backward pass"
                    )
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            ans_list = []
            for t in batch.target_text:
                tmp = [int(t[-1].replace('item_', ''))]
                ans_list.append(tmp)
            final_ans_list.extend(ans_list)
            final_item20_preds.extend(items_pred)

        progress.close()

        final_ans_array = np.asarray(final_ans_list)
        final_item20_preds_array = np.asarray(final_item20_preds)
        score, post_fix = get_full_sort_score(epoch, final_ans_array, final_item20_preds_array)

        # if best dev, save model and evaluate test set
        if score[-1] > best_dev_scores['ndcg@20']:
            best_dev_scores = {
                'ndcg@20': score[-1]
            }
            best_dev_epoch = epoch

            # save best model
            logger.info('Saving best dev model to {}'.format(best_model_path))
            torch.save(model.state_dict(), best_model_path)

            logger.info('Writing dev prediction to {}'.format(dev_prediction_path))
            with open(dev_prediction_path, 'w') as fw:
                fw.writelines(post_fix + '\n')
                for a, p in zip(final_ans_list, final_item20_preds):
                    line = '{} {}\n'.format(' '.join(list(map(str, a))), ' '.join(list(map(str, p))))
                    fw.writelines(line)

            # eval test set
            logger.info('Evaluating the test dataset.')
            progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test {}'.format(epoch))
            write_output = []
            test_final_ans_list = []
            test_final_item20_preds = []
            for batch_idx, batch in enumerate(DataLoader(test_set, batch_size=config.eval_batch_size,
                                                         shuffle=False, collate_fn=test_set.collate_fn)):
                progress.update(1)
                try:
                    gpu_batch = move_to_cuda(batch)
                    items_pred = model.predict(gpu_batch, max_length=config.max_output_length)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(
                            "attempting to recover from OOM in forward/backward pass"
                        )
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

                ans_list = []
                for t in batch.target_text:
                    tmp = []
                    for tt in t:
                        tmp.append(int(tt.replace('item_', '')))
                    ans_list.append(tmp)
                test_final_ans_list.extend(ans_list)
                test_final_item20_preds.extend(items_pred)
                if len(test_final_ans_list) % 4096 == 0:
                    print('Evaluated: {}, time: {}'.format(len(test_final_ans_list), datetime.datetime.now()))
            progress.close()

            test_final_ans_array = np.asarray(test_final_ans_list)
            test_final_item20_preds_array = np.asarray(test_final_item20_preds)
            score, post_fix = get_full_sort_score(0, test_final_ans_array, test_final_item20_preds_array)

            if score[-1] > prev_best_test_scores:
                prev_best_test_scores = score[-1]
                logger.info('Writing test prediction to {}'.format(test_prediction_path))
                with open(test_prediction_path, 'w') as fw:
                    fw.writelines(post_fix + '\n')
                    for a, p in zip(test_final_ans_list, test_final_item20_preds):
                        line = '{} {}\n'.format(' '.join(list(map(str, a))), ' '.join(list(map(str, p))))
                        fw.writelines(line)
                logger.info('Current best test score: {}'.format(post_fix))

logger.info(log_path)
logger.info("Done!")
