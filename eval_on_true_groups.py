import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import torchvision.models as models

def evaluate_model_on_ground_truth(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {model_path}")
    print(f"Using device: {device}")
    
    # Load checkpoint saved by group_DRO
    model = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(model, dict) and 'state_dict' in model:
        net = models.resnet50(weights=None)
        net.fc = nn.Linear(net.fc.in_features, 2)
        net.load_state_dict(model['state_dict'])
        model = net
    model.to(device)
    model.eval()

    # Read ground truth metadata directly for test set
    meta_path = os.path.join(os.path.dirname(__file__), 'cub', 'data', 'waterbird_complete95_forest2water2', 'metadata.csv')
    meta_df = pd.read_csv(meta_path)
    test_df = meta_df[meta_df['split'] == 2].copy()
    
    # Calculate ground truth group = y * 2 + place
    test_df['true_group'] = test_df['y'] * 2 + test_df['place']
    
    data_dir = os.path.join(os.path.dirname(__file__), 'cub', 'data', 'waterbird_complete95_forest2water2')
    
    from PIL import Image
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset, DataLoader

    class TestEvalDataset(Dataset):
        def __init__(self, df, data_dir, transform):
            self.df = df
            self.data_dir = data_dir
            self.transform = transform
            self.filenames = df['img_filename'].values
            self.labels = df['y'].values
            self.groups = df['true_group'].values

        def __len__(self):
            return len(self.filenames)

        def __getitem__(self, idx):
            img_path = os.path.join(self.data_dir, self.filenames[idx])
            img = Image.open(img_path).convert('RGB')
            x = self.transform(img)
            return x, self.labels[idx], self.groups[idx]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = TestEvalDataset(test_df, data_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    correct_per_group = {0: 0, 1: 0, 2: 0, 3: 0}
    total_per_group = {0: 0, 1: 0, 2: 0, 3: 0}

    with torch.no_grad():
        for x, y, g in test_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            preds = torch.argmax(output, dim=1)

            for pred_val, label_val, group_val in zip(preds.cpu().numpy(), y.cpu().numpy(), g.numpy()):
                group_val = int(group_val)
                total_per_group[group_val] += 1
                if pred_val == label_val:
                    correct_per_group[group_val] += 1

    group_accs = {}
    group_names = {
        0: "Landbird on Land",
        1: "Landbird on Water",
        2: "Waterbird on Land",
        3: "Waterbird on Water"
    }

    print("\n========== HELD-OUT TEST EVALUATION ON TRUE GROUPS ==========")
    total_correct = sum(correct_per_group.values())
    total_samples = sum(total_per_group.values())
    avg_acc = total_correct / total_samples

    print(f"Overall Average Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)\n")

    for g in sorted(total_per_group.keys()):
        acc = correct_per_group[g] / total_per_group[g]
        group_accs[g] = acc
        print(f"Group {g} ({group_names[g]}): {acc:.4f} ({acc*100:.2f}%)  [{correct_per_group[g]}/{total_per_group[g]}]")

    worst_group_acc = min(group_accs.values())
    worst_group_id = min(group_accs, key=group_accs.get)
    print(f"\nWORST-GROUP ACCURACY: {worst_group_acc:.4f} ({worst_group_acc*100:.2f}%) on Group {worst_group_id} ({group_names[worst_group_id]})")
    return avg_acc, group_accs, worst_group_acc

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(os.path.dirname(__file__), 'group_DRO', 'logs_text_slice', '90_model.pth')
    evaluate_model_on_ground_truth(model_path)
