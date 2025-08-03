import torch

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    # initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        # 如果decoder_input的行数等于max_len，则说明已经生成了max_len个token，跳出循环
        if decoder_input.size(1) == max_len:
            break
        
        # build tmask for the target(decoder input)。 这个是动态的，因为decoder_input的行数是动态的，他不会因为一个词一个词的不断回答而变化
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # calculate the decoder output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # get the next token
        prob = model.project(out[:, -1])
        
        next_word = torch.argmax(prob, dim=-1)
        
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).fill_(next_word.item()).type_as(source).to(device)], dim=1)
        
        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)
        
    
    

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, globl_state, writer, num_examples=2):
    model.eval()
    count = 0
    
    source_texts = []
    expected = []
    predicted = []
    
    # Size of the control window (just use a default value)
    console_width = 80
    
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device) # (bs, seq_Len)
            encoder_mask = batch['encoder_mask'].to(device) #(bs, 1,1,seq_Len)
            
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            
            # 获取预测的句子的idx
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len=max_len, device=device)

            source_text = batch['src_text'][0]     
            target_text = batch['tgt_text'][0]      
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy()) 
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print to the console
            print_msg('-' * console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')
            
            if count == num_examples:
                break
            