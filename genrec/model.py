import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForPreTraining
import re
from bart.model import BartForConditionalGeneration


class GenerativeModel(nn.Module):
    def __init__(self, config, tokenizer,
                 prompt_tuning=False,
                 prompt_type_method='prefix',
                 encoder_prompt_length=100,
                 encoder_prompt_projection=False,
                 encoder_embed_dim=768,
                 encoder_prompt_dim=2*768,
                 encoder_layers=6,
                 encoder_attention_heads=12,
                 decoder_prompt_length=100,
                 decoder_prompt_projection=False,
                 decoder_embed_dim=768,
                 decoder_prompt_dim=2*768,
                 decoder_layers=6,
                 decoder_attention_heads=12):
        super().__init__()
        self.tokenizer = tokenizer
        # logger.info(f'Loading pre-trained model {config.model_name}')
        self.model_config = AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
        self.model = BartForConditionalGeneration.from_pretrained(config.model_name, cache_dir=config.cache_dir,
                                                                config=self.model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        if prompt_tuning:
            self.model.get_encoder().requires_grad_(False)
            self.model.get_decoder().requires_grad_(False)

            self.model.get_encoder().initialize_prompt_encoder(prompt_type_method, encoder_prompt_length, encoder_prompt_projection,
                                  encoder_embed_dim, encoder_prompt_dim, encoder_layers, encoder_attention_heads)
            self.model.get_decoder().initialize_prompt_encoder(prompt_type_method, decoder_prompt_length, decoder_prompt_projection,
                                  decoder_embed_dim, decoder_prompt_dim, decoder_layers, decoder_attention_heads)

    def forward(self, batch):
        outputs = self.model(input_ids=batch.enc_idxs,
                             attention_mask=batch.enc_attn,
                             enc_user_whole=batch.enc_user_whole,
                             enc_item_whole=batch.enc_item_whole,
                             enc_attrs=batch.enc_attrs,
                             decoder_input_ids=batch.dec_idxs,
                             decoder_attention_mask=batch.dec_attn,
                             labels=batch.lbl_idxs,
                             return_dict=True)

        loss = outputs['loss']

        return loss

    def predict(self, batch, num_beams=21, max_length=50):
        self.eval()
        with torch.no_grad():
            enc_kwargs = {
                "enc_user_whole": batch.enc_user_whole,
                "enc_item_whole": batch.enc_item_whole,
                "enc_attrs": batch.enc_attrs
            }
            outputs = self.model.generate(input_ids=batch.enc_idxs,
                                          attention_mask=batch.enc_attn,
                                          num_beams=num_beams,
                                          max_length=max_length,
                                          num_return_sequences=num_beams,
                                          do_sample=False,
                                          return_dict_in_generate=True,
                                          **enc_kwargs)

        item_preds = []
        for bid in range(len(batch.enc_idxs)):
            item20_predictions = []
            for beamId in range(21):
                cid = bid*21 + beamId
                output_sentence = self.tokenizer.decode(outputs['sequences'][cid], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)
                last_item = output_sentence.split(' ')[-1]
                tmp_item_id = last_item.replace('item', '').replace('_', '').replace('?', '').strip()
                tmp_item_id = re.sub("[^0-9]", "", tmp_item_id)
                if tmp_item_id == '':
                    item20_predictions.append(0)
                else:
                    item20_predictions.append(int(tmp_item_id))

                if len(item20_predictions) == 20:
                    break
            item_preds.append(item20_predictions)
        self.train()

        return item_preds
