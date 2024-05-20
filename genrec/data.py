import logging
import torch
from collections import namedtuple
import torch.nn.functional as F
import math

logger = logging.getLogger(__name__)

gen_batch_fields = ['input_text', 'target_text', 'enc_idxs', 'enc_attn', 'dec_idxs', 'dec_attn', 'lbl_idxs', 'raw_lbl_idxs', 'enc_user_whole', 'enc_item_whole', 'enc_attrs']
GenBatch = namedtuple('GenBatch', field_names=gen_batch_fields, defaults=[None] * len(gen_batch_fields))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, max_length, path, max_output_length=None, unseen_types=[], no_bos=False, dataset=None, is_pretrain=False):
        self.tokenizer = tokenizer
        self.max_length = 1024
        if max_output_length is not None:
            self.max_output_length = max_output_length
        self.path = path
        self.no_bos = no_bos  # if you use bart, then this should be False; if you use t5, then this should be True
        self.dataset = dataset
        if self.dataset is None:
            self.data = []
            if is_pretrain:
                self.load_pretrain_data()
            else:
                self.load_data(unseen_types)
        if is_pretrain:
            _lambda = 3.0
            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)
            self.p = 0.3
            self.replace_length = 1
            self.random_ratio = 0.0
        self.mask_idx = self.tokenizer.encode('<mask>', add_special_tokens=True)[1]

    def __len__(self):
        if self.dataset is None:
            return len(self.data)
        else:
            return len(self.dataset)

    def __getitem__(self, index):
        if self.dataset is None:
            item = self.data[index]
            return item
        else:
            uniq_id, user_items, ref = self.dataset[index]
            texts, attrs, user_whole, item_whole = [], [], [], []
            for t in user_items.split('\sep'):
                word, attr = t.split('\i')[0], t.split('\i')[1]
                texts.append(word)
                if 'item_' in word:
                    user_whole.append(0)
                    item_whole.append(int(word.replace('item_', '')))
                    attrs.append(attr.strip().split(', '))
                    continue
                elif 'user_' in word:
                    user_whole.append(int(word.replace('user_', '')))
                    item_whole.append(0)
                    attrs.append([0])
                    continue
                attrs.append([0])
                user_whole.append(0)
                item_whole.append(0)

            texts.append('?')
            attrs.append([0])
            user_whole.append(0)
            item_whole.append(0)

            example = {
                'text': texts,
                'attrs': attrs,
                'user_whole': user_whole,
                'item_whole': item_whole,
                'target': ref.replace('\n', '')
            }

            return example

    def load_pretrain_data(self):
        with open(self.path, 'r') as f:
            data = f.readlines()
        for line in data:
            parts = line.split('\t')
            context, target = parts[2], parts[3].replace('\n', '')
            texts, attrs, user_whole, item_whole = [], [], [], []
            user_added = False
            for t in context.split('\sep'):
                word, attr = t.split('\i')[0], t.split('\i')[1]
                if 'item_' not in word and 'user_' not in word:
                    continue
                if 'user_' in word:
                    if user_added:
                        continue
                    texts.append(word)
                    user_whole.append(int(word.replace('user_', '')))
                    item_whole.append(0)
                    attrs.append([0])
                    user_added = True
                elif 'item_' in word:
                    texts.append(word)
                    user_whole.append(0)
                    item_whole.append(int(word.replace('item_', '')))
                    attrs.append(attr.strip().split(', '))
                else:
                    texts.append(word)
                    attrs.append([0])
                    user_whole.append(0)
                    item_whole.append(0)

            if 'item_' in texts[0]:
                continue

            self.data.append({
                'text': texts,
                'attrs': attrs,
                'user_whole': user_whole,
                'item_whole': item_whole,
                'target': texts
            })

        logger.info(f'Loaded {len(self)} instances from {self.path}')

    def load_data(self, unseen_types):
        with open(self.path, 'r') as f:
            data = f.readlines()

        for line in data:
            parts = line.split('\t')
            context, target = parts[2], parts[3].replace('\n', '')
            texts, attrs, user_whole, item_whole = [], [], [], []
            user_added = False
            for t in context.split('\sep'):
                word, attr = t.split('\i')[0], t.split('\i')[1]
                if 'item_' not in word and 'user_' not in word:
                    continue
                if 'item_' in word:
                    texts.append(word)
                    user_whole.append(0)
                    item_whole.append(int(word.replace('item_', '')))
                    attrs.append(attr.strip().split(', '))
                    continue
                elif 'user_' in word:
                    if user_added:
                        continue
                    texts.append(word)
                    user_added = True
                    user_whole.append(int(word.replace('user_', '')))
                    item_whole.append(0)
                    attrs.append([0])
                    continue

            if 'item_' in texts[0]:
                continue

            self.data.append({
                'text': texts,
                'attrs': attrs,
                'user_whole': user_whole,
                'item_whole': item_whole,
                'target': target
            })

        logger.info(f'Loaded {len(self)} instances from {self.path}')

    def collate_fn(self, batch):
        input_text = [x['text'] for x in batch]
        input_item_wholes = [x['item_whole'] for x in batch]
        input_user_wholes = [x['user_whole'] for x in batch]
        # input_attrs = [x['attrs'] for x in batch]

        # encoder inputs
        tokenized_inputs = self.tokenizer(input_text,
                                          return_tensors='pt',
                                          padding=True,
                                          truncation=True,
                                          return_overflowing_tokens=True,
                                          is_split_into_words=True,
                                          max_length=self.max_length)  # max_length=512

        enc_idxs = []
        enc_attn = []
        # aligned_attrs = []
        aligned_user_wholes = []
        aligned_item_wholes = []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            item_whole = input_item_wholes[org_batch_index]
            user_whole = input_user_wholes[org_batch_index]
            # input_attr = input_attrs[org_batch_index]
            previous_word_idx = None
            user_whole_inputs = []
            item_whole_inputs = []
            attr_inputs = []
            for word_idx in word_ids:
                if word_idx is None:
                    user_whole_inputs.append(0)
                    item_whole_inputs.append(0)
                    attr_inputs.append([0])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    user_whole_inputs.append(user_whole[word_idx])
                    item_whole_inputs.append(item_whole[word_idx])
                    # attr_inputs.append(input_attr[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    user_whole_inputs.append(user_whole[word_idx])
                    item_whole_inputs.append(item_whole[word_idx])
                    # attr_inputs.append(input_attr[word_idx])
                previous_word_idx = word_idx

            tokens, attn_mask = tokenized_inputs['input_ids'][batch_index], tokenized_inputs['attention_mask'][batch_index]

            end_index = tokens.size(0) - 1
            while tokens[end_index] != 2:
                end_index -= 1

            tokens_list = list(tokens)
            tokens_list.insert(end_index, self.mask_idx)
            tokens = torch.as_tensor(tokens_list)
            user_whole_inputs.insert(end_index, 0)
            item_whole_inputs.insert(end_index, 0)
            attr_inputs.insert(end_index, [0])

            attn_mask_list = list(attn_mask)
            attn_mask_list.insert(end_index, 1)
            attn_mask = torch.as_tensor(attn_mask_list)

            aligned_user_wholes.append(user_whole_inputs)
            aligned_item_wholes.append(item_whole_inputs)
            # aligned_attrs.append(attr_inputs)
            enc_idxs.append(tokens)
            enc_attn.append(attn_mask)

        enc_idxs = torch.stack(enc_idxs, dim=0)
        enc_attn = torch.stack(enc_attn, dim=0)
        tokenized_inputs["user_wholes"] = aligned_user_wholes
        tokenized_inputs["item_wholes"] = aligned_item_wholes
        # tokenized_inputs["attrs"] = aligned_attrs

        llm_batch = tokenized_inputs
        enc_user_whole = torch.tensor(llm_batch['user_wholes'], dtype=int)
        enc_item_whole = torch.tensor(llm_batch['item_wholes'], dtype=int)

        max_attr_len = -1
        for s in llm_batch['attrs']:
            max_attr_len = max(max_attr_len, max(list(map(lambda x: len(x), s))))
        enc_attrs = torch.zeros(enc_idxs.shape[0], enc_idxs.shape[1], max_attr_len, dtype=int)
        for i, s in enumerate(llm_batch['attrs']):
            for j, ts in enumerate(s):
                if len(ts) == 1 and ts[0] == 0:
                    continue
                enc_attrs[i][j] = F.pad(torch.as_tensor(list(map(int, ts))), (0, max_attr_len - len(ts)), 'constant', 0)

        # decoder inputs
        target_text = [[x['target']] for x in batch]
        targets = self.tokenizer(target_text,
                                 return_tensors='pt',
                                 padding=True,
                                 truncation=True,
                                 return_overflowing_tokens=True,
                                 is_split_into_words=True,
                                 max_length=self.max_length)

        target_llm_batch = targets

        dec_idxs = target_llm_batch['input_ids']
        batch_size = dec_idxs.size(0)
        dec_idxs[:, 0] = self.tokenizer.eos_token_id
        dec_attn = target_llm_batch['attention_mask']

        # labels
        padding = torch.ones((batch_size, 1), dtype=torch.long)
        padding[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((dec_idxs[:, 1:], padding), dim=1)
        lbl_attn = torch.cat((dec_attn[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long)), dim=1)
        lbl_idxs = raw_lbl_idxs.masked_fill(lbl_attn == 0, 1)  # ignore padding

        return GenBatch(
            input_text=input_text,
            target_text=target_text,
            enc_idxs=enc_idxs,
            enc_attn=enc_attn,
            dec_idxs=dec_idxs,
            dec_attn=dec_attn,
            lbl_idxs=lbl_idxs,
            raw_lbl_idxs=raw_lbl_idxs,
            enc_user_whole=enc_user_whole,
            enc_item_whole=enc_item_whole,
            enc_attrs=enc_attrs)

    def pretrain_collate_fn(self, batch):
        input_text = [x['text'] for x in batch]
        input_item_wholes = [x['item_whole'] for x in batch]
        input_user_wholes = [x['user_whole'] for x in batch]
        input_attrs = [x['attrs'] for x in batch]
        target_text = []

        tokenized_inputs = self.tokenizer(input_text,
                                          return_tensors='pt',
                                          padding=True,
                                          truncation=True,
                                          return_overflowing_tokens=True,
                                          is_split_into_words=True,
                                          max_length=self.max_length)  # max_length=512

        aligned_attrs = []
        aligned_user_wholes = []
        aligned_item_wholes = []
        enc_idxs = []
        enc_attn = []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            item_whole = input_item_wholes[org_batch_index]
            user_whole = input_user_wholes[org_batch_index]
            input_attr = input_attrs[org_batch_index]
            previous_word_idx = None
            user_whole_inputs = []
            item_whole_inputs = []
            attr_inputs = []

            is_word_start = torch.zeros(len(word_ids))
            for ind, word_idx in enumerate(word_ids):
                if word_idx is None:
                    user_whole_inputs.append(0)
                    item_whole_inputs.append(0)
                    attr_inputs.append([0])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    user_whole_inputs.append(user_whole[word_idx])
                    item_whole_inputs.append(item_whole[word_idx])
                    attr_inputs.append(input_attr[word_idx])
                    is_word_start[ind] = 1
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    user_whole_inputs.append(user_whole[word_idx])
                    item_whole_inputs.append(item_whole[word_idx])
                    attr_inputs.append(input_attr[word_idx])
                previous_word_idx = word_idx

            tokens, attn_mask = tokenized_inputs['input_ids'][batch_index], tokenized_inputs['attention_mask'][batch_index]
            new_tokens, user_whole_inputs, item_whole_inputs, attr_inputs, new_attn_mask, target = self.add_bert_mask(
                tokens, user_whole_inputs, item_whole_inputs, attr_inputs, attn_mask, is_word_start, self.p)
            aligned_user_wholes.append(user_whole_inputs)
            aligned_item_wholes.append(item_whole_inputs)
            aligned_attrs.append(attr_inputs)
            enc_idxs.append(new_tokens)
            enc_attn.append(new_attn_mask)
            target_text.append([target])

        new_max_len = max(map(lambda x: len(x[x != 1]), enc_idxs))
        new_attn_max_len = max(map(lambda x: len(x[x != 0]), enc_attn))
        assert new_max_len == new_attn_max_len
        for i in range(len(batch)):
            if len(aligned_user_wholes[i]) > new_max_len:
                aligned_user_wholes[i] = aligned_user_wholes[i][:new_max_len]
            else:
                aligned_user_wholes[i].extend([0] * (new_max_len - len(aligned_user_wholes[i])))

            if len(aligned_item_wholes[i]) > new_max_len:
                aligned_item_wholes[i] = aligned_item_wholes[i][:new_max_len]
            else:
                aligned_item_wholes[i].extend([0] * (new_max_len - len(aligned_item_wholes[i])))

            if len(aligned_attrs[i]) > new_max_len:
                aligned_attrs[i] = aligned_attrs[i][:new_max_len]
            else:
                for j in range(new_max_len - len(aligned_attrs[i])):
                    aligned_attrs[i].append([0])

        enc_user_whole = torch.tensor(aligned_user_wholes, dtype=int)
        enc_item_whole = torch.tensor(aligned_item_wholes, dtype=int)

        for i, tensor in enumerate(enc_idxs):
            if len(tensor) > new_max_len:
                enc_idxs[i] = enc_idxs[i][:new_max_len]
            else:
                enc_idxs[i] = F.pad(tensor, (0, new_max_len - len(tensor)), 'constant', 1)
        enc_idxs = torch.stack(enc_idxs, dim=0)

        for i, tensor in enumerate(enc_attn):
            if len(tensor) > new_max_len:
                enc_attn[i] = enc_attn[i][:new_max_len]
            else:
                enc_attn[i] = F.pad(tensor, (0, new_max_len - len(tensor)), 'constant', 0)
        enc_attn = torch.stack(enc_attn, dim=0)

        max_attr_len = -1
        for s in aligned_attrs:
            max_attr_len = max(max_attr_len, max(list(map(lambda x: len(x), s))))

        enc_attrs = torch.zeros(enc_idxs.shape[0], enc_idxs.shape[1], max_attr_len, dtype=int)
        for i, s in enumerate(aligned_attrs):
            for j, ts in enumerate(s):
                if len(ts) == 1 and ts[0] == 0:
                    continue
                enc_attrs[i][j] = F.pad(torch.as_tensor(list(map(int, ts))), (0, max_attr_len - len(ts)), 'constant', 0)

        # decoder inputs
        targets = self.tokenizer(target_text,
                                 return_tensors='pt',
                                 padding=True,
                                 truncation=True,
                                 return_overflowing_tokens=True,
                                 is_split_into_words=True,
                                 max_length=self.max_length)

        target_llm_batch = targets

        dec_idxs = target_llm_batch['input_ids']
        batch_size = dec_idxs.size(0)
        dec_idxs[:, 0] = self.tokenizer.eos_token_id
        dec_attn = target_llm_batch['attention_mask']

        # labels
        padding = torch.ones((batch_size, 1), dtype=torch.long)
        padding[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((dec_idxs[:, 1:], padding), dim=1)
        lbl_attn = torch.cat((dec_attn[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long)), dim=1)
        lbl_idxs = raw_lbl_idxs.masked_fill(lbl_attn == 0, 1)  # ignore padding

        return GenBatch(
            input_text=input_text,
            target_text=target_text,
            enc_idxs=enc_idxs,
            enc_attn=enc_attn,
            dec_idxs=dec_idxs,
            dec_attn=dec_attn,
            lbl_idxs=lbl_idxs,
            raw_lbl_idxs=raw_lbl_idxs,
            enc_user_whole=enc_user_whole,
            enc_item_whole=enc_item_whole,
            enc_attrs=enc_attrs)

    def add_insertion_noise(self, tokens, user_wholes, item_wholes, attrs, attn_mask, p):
        if p == 0.0:
            return tokens, user_wholes, item_wholes, attrs, attn_mask

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        sorted_noise_indices = noise_indices.sort(descending=True)
        for ind in sorted_noise_indices.values:
            attrs.insert(ind, [0])
            item_wholes.insert(ind, 0)
            user_wholes.insert(ind, 0)
            attn_mask_list = list(attn_mask)
            attn_mask_list.insert(ind, 1)
            attn_mask = torch.as_tensor(attn_mask_list)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[~noise_mask] = tokens

        assert (result >= 0).all()
        assert len(result) == len(user_wholes) == len(item_wholes) == len(attrs) == len(attn_mask)
        return result, user_wholes, item_wholes, attrs, attn_mask

    def add_whole_word_mask(self, source, user_wholes, item_wholes, attrs, attn_mask, is_word_start, p):
        num_to_mask = int(math.floor(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source, user_wholes, item_wholes, attrs, attn_mask

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, user_wholes, item_wholes, attrs, attn_mask,
                                                num_inserts / source.size(0))

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx

            sorted_indices = indices.sort(descending=True)
            for ind in sorted_indices.values:
                attrs[ind] = [0]
                user_wholes[ind] = 0
                item_wholes[ind] = 0

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx

                    sorted_indices = indices.sort(descending=True)
                    for ind in sorted_indices.values:
                        attrs[ind] = [0]
                        user_wholes[ind] = 0
                        item_wholes[ind] = 0
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx

                    sorted_indices = indices.sort(descending=True)
                    for ind in sorted_indices.values:
                        attrs[ind] = [0]
                        user_wholes[ind] = 0
                        item_wholes[ind] = 0

                assert source_length - 1 not in indices

        source = source[to_keep]
        new_attrs, new_user_wholes, new_item_wholes = [], [], []
        for i, j in enumerate(to_keep):
            if j:
                new_attrs.append(attrs[i])
                new_user_wholes.append(user_wholes[i])
                new_item_wholes.append(item_wholes[i])
        attn_mask = attn_mask[to_keep]

        if num_inserts > 0:
            return self.add_insertion_noise(source, new_user_wholes, new_item_wholes, new_attrs, attn_mask, num_inserts / source.size(0))

        return source, new_user_wholes, new_item_wholes, new_attrs, attn_mask

    def add_bert_mask(self, source, user_wholes, item_wholes, attrs, attn_mask, is_word_start, p):
        mask_the_last = torch.rand(1) < 0.3
        end = len(source) - 1
        while source[end] != 2:
            end -= 1
        to_keep = torch.ones(source.size(0), dtype=torch.bool)
        target_tokens = []

        if mask_the_last[0]:
            last_word_start = end
            while is_word_start[last_word_start] != 1:
                last_word_start -= 1

            k = last_word_start
            target_tokens.append(int(source[k]))
            source[k] = self.mask_idx
            attrs[k] = [0]
            user_wholes[k] = 0
            item_wholes[k] = 0
            k += 1
            while is_word_start[k] != 1 and k != end:
                target_tokens.append(int(source[k]))
                source[k] = self.mask_idx
                attrs[k] = [0]
                user_wholes[k] = 0
                item_wholes[k] = 0
                to_keep[k] = 0
                k += 1

        else:
            num_to_mask = 1

            if num_to_mask == 0:
                return source, user_wholes, item_wholes, attrs, attn_mask

            word_starts = is_word_start.nonzero(as_tuple=False)
            indices = word_starts[
                torch.randperm(word_starts.size(0))[:num_to_mask]
            ].squeeze(1)

            source_length = source.size(0)
            assert source_length - 1 not in indices
            assert len(indices) == 1

            for ind in indices:
                target_tokens.append(int(source[ind]))
                source[ind] = self.mask_idx
                attrs[ind] = [0]
                user_wholes[ind] = 0
                item_wholes[ind] = 0

                k = ind + 1
                while is_word_start[k] != 1 and k != end:
                    target_tokens.append(int(source[k]))
                    source[k] = self.mask_idx
                    attrs[k] = [0]
                    user_wholes[k] = 0
                    item_wholes[k] = 0
                    to_keep[k] = 0
                    k += 1

        source = source[to_keep]
        target = self.tokenizer.decode(target_tokens).strip()
        new_attrs, new_user_wholes, new_item_wholes = [], [], []
        for i, j in enumerate(to_keep):
            if j:
                new_attrs.append(attrs[i])
                new_user_wholes.append(user_wholes[i])
                new_item_wholes.append(item_wholes[i])
        attn_mask = attn_mask[to_keep]

        return source, new_user_wholes, new_item_wholes, new_attrs, attn_mask, target
