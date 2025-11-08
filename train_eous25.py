import shutil
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset
from datasets.utils import smi2hgraph, HData, edge_order


class EOUS25Dataset(InMemoryDataset):
    """
    Molecular hypergraph dataset class for a 4-task binary classification dataset.

    The dataset should be stored in a file called 'training.csv' with columns:
        - 'clean_smiles': molecular SMILES strings
        - 'F340450', 'F480', 'T340', 'T450': binary classification targets
    """

    raw_url = None  # Not used for local datasets

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.ids = self.data.smi

    @property
    def raw_file_names(self):
        return ['training.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        """No-op: assumes 'training.csv' is already present in raw_dir."""
        pass

    def process(self):
        # Load data
        df = pd.read_csv(self.raw_paths[0])
        smiles = df['clean_smiles'].values.tolist()

        # Extract target columns
        target_cols = ['F340450', 'F480', 'T340', 'T450']
        target = torch.tensor(df[target_cols].values, dtype=torch.float)

        data_list = []
        for i, smi in enumerate(tqdm(smiles, desc="Processing molecules")):
            atom_fvs, n_idx, e_idx, bond_fvs = smi2hgraph(smi)

            x = torch.tensor(atom_fvs, dtype=torch.long)
            edge_index0 = torch.tensor(n_idx, dtype=torch.long)
            edge_index1 = torch.tensor(e_idx, dtype=torch.long)
            edge_attr = torch.tensor(bond_fvs, dtype=torch.long)
            y = target[i].unsqueeze(0)
            n_e = len(edge_index1.unique())
            e_order = torch.tensor(edge_order(e_idx), dtype=torch.long)

            data = HData(
                x=x, y=y, n_e=n_e, smi=smi,
                edge_index0=edge_index0,
                edge_index1=edge_index1,
                edge_attr=edge_attr,
                e_order=e_order,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
    
# if __name__ == "__main__":
#     ds = EOUS25Dataset(".")


import time
import argparse
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

from models import MHNN
from datasets import RandomSplitter
from utils import Logger, seed_everything


@torch.no_grad()
def evaluate(args, model, loader, num_tasks=4):
    """Compute mean ROC-AUC across all binary tasks."""
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        try:
            batch = batch.to(args.device)
            out = model(batch)  # logits, shape: [batch_size, num_tasks]
            probs = torch.sigmoid(out).cpu().numpy()
            y_pred.append(probs)
            y_true.append(batch.y.cpu().numpy())
        except:
            print("skipped batch during eval")

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0).reshape(-1, 4)

    aucs = []
    for i in range(num_tasks):
        y_t, y_p = y_true[:, i], y_pred[:, i]
        if len(np.unique(y_t)) == 2:  # only compute if both classes present
            auc = roc_auc_score(y_t, y_p)
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else 0.0


def compute_pos_weight(dataset, num_tasks=4):
    """Compute positive class weight for BCE loss for each task."""
    y = torch.cat([dataset.get(i).y for i in range(len(dataset))], dim=0)
    pos_counts = y.sum(dim=0)
    neg_counts = y.shape[0] - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-8)
    return pos_weight


if __name__ == '__main__':
    print('Task start time:')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start_time = time.time()

    parser = argparse.ArgumentParser(description='EOUS25 Binary Classification')

    # Dataset arguments
    parser.add_argument('--data_dir', type=str, required=True)

    # Training hyperparameters
    parser.add_argument('--runs', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    parser.add_argument('--log_steps', type=int, default=5)

    # Model hyperparameters
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
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Load dataset
    dataset = EOUS25Dataset(root=args.data_dir)
    num_tasks = 4  # F340450, F480, T340, T450

    # Compute class imbalance weights
    pos_weight = compute_pos_weight(dataset, num_tasks=num_tasks).to(device)
    print("Positive class weights:", pos_weight.tolist())

    # Logger
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        # Set seed
        seed = args.seed + run
        seed_everything(seed=seed, workers=True)
        print(f'\nRun No. {run + 1}:')
        print(f'Seed: {seed}\n')

        # Split dataset
        splitter = RandomSplitter()
        train_idx, valid_idx, test_idx = splitter.split(dataset, seed=seed)
        train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[valid_idx], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size, shuffle=False)

        # Model
        model = MHNN(num_tasks, args).to(device)
        print("# Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        # Loss and optimizer
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.8, patience=5, min_lr=1e-5
        )
        best_val = 0.50
        best_model_state = None

        best_val_auc = None
        for epoch in range(1, 1 + args.epochs):
            model.train()
            loss_all = 0.0
            lr = scheduler.optimizer.param_groups[0]['lr']

            for data in tqdm(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                try:
                    out = model(data)
                    loss = loss_fn(out.reshape(-1, 4), data.y)
                    loss.backward()
                    loss_all += loss.item() * data.num_graphs
                    optimizer.step()
                except:
                    print("skipped batch")

            loss_all /= len(train_loader.dataset)
            valid_auc = evaluate(args, model, valid_loader, num_tasks=num_tasks)
            scheduler.step(valid_auc)
            if valid_auc > best_val:
                best_val = valid_auc
                best_model_state = model.state_dict()

            if best_val_auc is None or valid_auc > best_val_auc:
                test_auc = evaluate(args, model, test_loader, num_tasks=num_tasks)
                best_val_auc = valid_auc

            torch.save(best_model_state, "eous25_mhnn.pt")
            print("Best model saved to eous25_mhnn.pt")
            logger.add_result(run, [loss_all, valid_auc, test_auc])

            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:03d}, '
                    f'lr: {lr:.6f}, '
                    f'Loss: {loss_all:.6f}, '
                    f'Valid AUC: {valid_auc:.6f}, '
                    f'Test AUC: {test_auc:.6f}')

        logger.print_statistics(run)

    logger.print_statistics()
    print('Task end time:')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('Total time taken: {} s.'.format(int(time.time() - start_time)))
