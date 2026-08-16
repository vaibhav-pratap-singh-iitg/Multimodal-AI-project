import os
import pandas as pd

def report(path, name):
    if not os.path.exists(path):
        print(f"\n--- {name} --- [Log file not found: {path}]")
        return None
    df = pd.read_csv(path)
    last = df.iloc[-1]
    groups = {c: last[c] for c in df.columns if c.startswith("avg_acc_group:")}
    avg_acc = float(last["avg_acc"])
    worst_acc = float(min(groups.values()))
    
    print(f"\n--- {name} ---")
    print(f"Average Accuracy   : {avg_acc:.4f} ({avg_acc*100:.1f}%)")
    for g, a in sorted(groups.items()):
        print(f"  {g}: {float(a):.4f} ({float(a)*100:.1f}%)")
    print(f"WORST GROUP ACC    : {worst_acc:.4f} ({worst_acc*100:.1f}%)")
    return {"avg_acc": avg_acc, "worst_acc": worst_acc}

if __name__ == "__main__":
    baseline = report("group_DRO/logs_baseline/test.csv", "1. BASELINE (ERM)")
    oracle = report("group_DRO/logs_groupdro_v2/test.csv", "2. ORACLE GROUPDRO (Ground-Truth Groups)")
    random_slice = report("group_DRO/logs_randomslice/test.csv", "3. RANDOM SLICE BASELINE (Ablation)")
    text_slice = report("group_DRO/logs_text_slice/test.csv", "4. TEXT-SLICE GROUPDRO (Member C's Discovered Slices)")
    
    if baseline and oracle and text_slice:
        b_w = baseline["worst_acc"]
        o_w = oracle["worst_acc"]
        t_w = text_slice["worst_acc"]
        transfer_delta = t_w - b_w
        transfer_ratio = (t_w - b_w) / (o_w - b_w) if (o_w - b_w) != 0 else 0
        print("\n================ WEEK 3 TRANSFER RESULTS ================")
        print(f"Baseline Worst-Group Acc     : {b_w*100:.1f}%")
        print(f"Oracle DRO Worst-Group Acc   : {o_w*100:.1f}%")
        print(f"Text-Slice DRO Worst-Group Acc: {t_w*100:.1f}%")
        print(f"Absolute Gain (Delta)        : +{transfer_delta*100:.1f}%")
        print(f"Transfer Efficiency Ratio    : {transfer_ratio*100:.1f}% of Oracle gain recovered")