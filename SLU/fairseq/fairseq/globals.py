
import sys
import torch

# Make these symbols global
SOS_tag = 'SOS' #'<SOS>'
blank_token = "__" #"<blank>"
pad_token = '_' #'<pad>'
pad_char = '_'
EOS_tag = 'EOS' #'<EOS>'
unk_token = '<unk>'

start_token = '<SOT>'
end_token = '<EOT>'
seq_separator = '_SEQ_SEP_'
void_concept = 'null'
machine_semantic = 'MachineSemantic'
slu_start_concept_mark = '_SOC_'
slu_end_concept_mark = '_EOC_'
tok_separator = '|'
user_ID = 'User'
machine_ID = 'Machine'
bogus_ID = 'XXX'    # TODO: defined this way so that it is unlikely it appears in the text, but it is defined in the RoBERTa dictionary as a token
bogus_token = bogus_ID
EOD_tag = 'XXXX'    # TODO: same as above.
EOD_value = -3.141592

slu_special_symbols = ['ã', ' ', machine_semantic, slu_start_concept_mark, slu_end_concept_mark, tok_separator, user_ID, machine_ID, bogus_ID, EOD_tag, start_token, end_token, seq_separator]  # The first is in MEDIA files not correctly encoded in utf-8. The second (space) is to keep MEDIA character-level encoding.

LOSS_INIT_VALUE=999999.9
ER_INIT_VALUE=LOSS_INIT_VALUE

def extract_concepts(sequences, sos, eos, pad, eoc, move_trail=False):

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_trail: 
            assert src[-1] == eos
            dst[0] = eos
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    B, T = sequences.size() 
    concepts = []
    for i in range(B):
        concepts.append( torch.LongTensor( [sos] + [sequences[i,j-1] for j in range(T) if sequences[i,j] == eoc] + [eos] ).to(sequences) )
    lengths = [t.size(0) for t in concepts]
    max_len = max( lengths )
    res = torch.LongTensor(B, max_len).fill_(pad).to(sequences)
    for i in range(B): 
        copy_tensor(concepts[i], res[i,:concepts[i].size(0)])

    return res, torch.LongTensor( lengths ).to(sequences)

