import torch
import yaml
from model.network import create_model

run_name = 'neoag_train-0128_1730'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
config['device'] = device

model = create_model(config)
model.load_state_dict(torch.load(f'./checkpoints/{run_name}/model.pt')['model_state_dict'], strict=False)

seqs = []
with open('./data/neoag_test.csv', 'r') as f:
    for line in f.readlines():
        seq = line.strip()
        seqs.append(seq)

MAX_LEN = max(map(len, seqs))

# convert to tokens
mapping = dict(zip(
    ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','L',
    'A','G','V','E','S','I','K','R','D','T','P','N',
    'Q','F','Y','M','H','C','W'],
    range(30)
))

for i in range(len(seqs)):
    seqs[i] = [mapping[c.upper()] for c in seqs[i] if c.upper() in mapping]  # Convert to uppercase and map
    seqs[i].extend([0] * (MAX_LEN - len(seqs[i])))  # Padding to max length  # padding to max length

preds = []
with torch.inference_mode():
    for i in range(len(seqs)):
        input_ids = torch.tensor([seqs[i]]).to(device)
        attention_mask = (input_ids != 0).float()
        output = int(model(input_ids, attention_mask)[0] > 0.5)
        # print(output)
        preds.append(output)

with open('./data/output.csv', 'w') as f:
    f.write("sequence,label\n")  # Write the header
    for seq, pred in zip(seqs, preds):
        original_seq = ''.join([list(mapping.keys())[list(mapping.values()).index(i)] for i in seq if i != 0])  # Convert indices back to characters
        f.write(f"{original_seq},{pred}\n")