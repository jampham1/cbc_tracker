import torch
import sys
sys.path.append('/Users/jamespham/PycharmProjects/cbc_tracker')

from phase2_preprocessing import run_preprocessing
from phase3_part4_full_model import CBCAnomalyTCN

device = (
    torch.device('mps')  if torch.backends.mps.is_available() else
    torch.device('cuda') if torch.cuda.is_available()         else
    torch.device('cpu')
)

# Load data
data = run_preprocessing(
    csv_path   = 'data/cbc_synthetic.csv',
    output_dir = 'outputs',
)

# Load model
model = CBCAnomalyTCN(
    n_seq_features    = data['n_seq_features'],
    n_static_features = data['n_static_features'],
)
model.load_state_dict(
    torch.load('outputs/checkpoints/best_model.pt', map_location=device)
)
model = model.to(device)
model.eval()

# Now run the cancer type test
w = data['test_windows'][0]
x_seq    = torch.from_numpy(w['x_seq']).unsqueeze(0).to(device)
x_static = torch.from_numpy(w['x_static']).unsqueeze(0).to(device)

with torch.no_grad():
    score_original = model(x_seq, x_static).item()
    print(f"Original score: {score_original:.4f}")

    for cancer_enc in range(5):
        x_static_modified = x_static.clone()
        x_static_modified[0, 2] = cancer_enc
        score_modified = model(x_seq, x_static_modified).item()
        print(f"cancer_type_enc={cancer_enc}  score={score_modified:.4f}")