import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm
import pandas as pd

from models import MHNN
from datasets.utils import smi2hgraph, HData, edge_order
from train_eous25 import EOUS25Dataset


def smiles_to_hdata(smiles_list):
    data_list = []
    for smi in tqdm(smiles_list, desc="Encoding molecules"):
        atom_fvs, n_idx, e_idx, bond_fvs = smi2hgraph(smi)
        x = torch.tensor(atom_fvs, dtype=torch.long)
        edge_index0 = torch.tensor(n_idx, dtype=torch.long)
        edge_index1 = torch.tensor(e_idx, dtype=torch.long)
        edge_attr = torch.tensor(bond_fvs, dtype=torch.long)
        e_order = torch.tensor(edge_order(e_idx), dtype=torch.long)   # ✅ NEW LINE
        n_e = len(edge_index1.unique())
        data = HData(
            x=x,
            y=torch.zeros(1, 4),
            n_e=n_e,
            smi=smi,
            edge_index0=edge_index0,
            edge_index1=edge_index1,
            edge_attr=edge_attr,
            e_order=e_order,   # ✅ ADD THIS ARGUMENT TOO
        )
        data_list.append(data)
    return data_list


@torch.no_grad()
def predict(model_path, smiles_list, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dummy dataset to get feature encodings consistent with training
    dataset = EOUS25Dataset(root=args.data_dir)
    model = MHNN(4, args=args)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    data_list = smiles_to_hdata(smiles_list)
    loader = DataLoader(data_list, batch_size=1)

    sigmoid = torch.nn.Sigmoid()
    preds = []

    for batch in tqdm(loader, "Running inference"):
        try:
            batch = batch.to(device)
            out = model(batch)
            prob = sigmoid(out)
            preds.append(prob.detach().cpu())
        except:
            preds.append(torch.zeros((4,)))

    preds = torch.cat(preds, dim=0).numpy().reshape(-1, 4)
    df = pd.DataFrame(preds, columns=["F340450", "F480", "T340", "T450"])
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="best_model.pt")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--smiles_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, default="predictions.csv")
    # model args - must match training
    parser.add_argument('--method', default='mhnn', help='model type')
    parser.add_argument('--All_num_layers', default=3, type=int)
    parser.add_argument('--MLP1_num_layers', default=2, type=int)
    parser.add_argument('--MLP2_num_layers', default=2, type=int)
    parser.add_argument('--MLP3_num_layers', default=2, type=int)
    parser.add_argument('--MLP4_num_layers', default=2, type=int)
    parser.add_argument('--MLP_hidden', default=64, type=int)
    parser.add_argument('--output_num_layers', default=2, type=int)
    parser.add_argument('--output_hidden', default=64, type=int)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    parser.add_argument('--normalization', default='ln', choices=['bn', 'ln', 'None'])
    parser.add_argument('--activation', default='relu', choices=['Id', 'relu', 'prelu'])
    parser.add_argument('--dropout', default=0.0, type=float)
    args = parser.parse_args()

    # Load SMILES from text or CSV
    if args.smiles_file.endswith(".csv"):
        df = pd.read_csv(args.smiles_file)
        smiles_list = df["clean_smiles"].tolist()
    else:
        with open(args.smiles_file) as f:
            smiles_list = [line.strip() for line in f if line.strip()]

    preds_df = predict(args.model_path, smiles_list, args)
    preds_df[['T340', 'T450', 'F340450', 'F480']].to_csv(args.out_file, index=False)
    print(f"Predictions saved to {args.out_file}")
