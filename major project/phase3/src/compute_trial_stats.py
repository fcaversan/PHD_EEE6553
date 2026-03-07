"""Compute mean +/- std across 3 training trials (T021)."""
import re, os
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts', 'results')

reports = {
    'Trial 1 (seed=42)':  os.path.join(BASE, 'trial_1_seed42',  'classification_report.txt'),
    'Trial 2 (seed=123)': os.path.join(BASE, 'trial_2_seed123', 'classification_report.txt'),
    'Trial 3 (seed=7)':   os.path.join(BASE, 'trial_3_seed7',   'classification_report.txt'),
}

data = {}
for name, path in reports.items():
    txt = open(path).read()
    acc = float(re.search(r'accuracy\s+([\d.]+)', txt).group(1))
    rows = {}
    for cls in ['benign', 'defacement', 'malware', 'phishing']:
        m = re.search(rf'{cls}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', txt)
        rows[cls] = {'p': float(m.group(1)), 'r': float(m.group(2)), 'f1': float(m.group(3))}
    data[name] = {'acc': acc, 'classes': rows}

classes = ['benign', 'defacement', 'malware', 'phishing']

print('=' * 72)
print('MULTI-TRIAL RESULTS')
print('=' * 72)
for name, d in data.items():
    acc_pct = d['acc'] * 100
    print(f"\n{name}  |  Accuracy: {acc_pct:.2f}%")
    for cls in classes:
        r = d['classes'][cls]
        print(f"  {cls:<12}  P={r['p']:.4f}  R={r['r']:.4f}  F1={r['f1']:.4f}")

print()
print('=' * 72)
print('MEAN +/- STD ACROSS 3 TRIALS')
print('=' * 72)

accs = [d['acc'] for d in data.values()]
mean_acc = np.mean(accs) * 100
std_acc  = np.std(accs) * 100
print(f"  Accuracy      : {mean_acc:.2f}% +/- {std_acc:.3f}%")

summary_lines = [
    "MALICIOUS URL DETECTION - MULTI-TRIAL SUMMARY (T021)",
    "=" * 72,
    f"  Accuracy      : {mean_acc:.2f}% +/- {std_acc:.3f}%",
]

for cls in classes:
    f1s = [d['classes'][cls]['f1'] for d in data.values()]
    ps  = [d['classes'][cls]['p']  for d in data.values()]
    rs  = [d['classes'][cls]['r']  for d in data.values()]
    line = (f"  {cls:<12}  F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}"
            f"   P: {np.mean(ps):.4f} +/- {np.std(ps):.4f}"
            f"   R: {np.mean(rs):.4f} +/- {np.std(rs):.4f}")
    print(line)
    summary_lines.append(line)

print('=' * 72)

# Save summary
out_path = os.path.join(BASE, 'multi_trial_summary.txt')
with open(out_path, 'w') as f:
    for name, d in data.items():
        acc_pct = d['acc'] * 100
        f.write(f"\n{name}  |  Accuracy: {acc_pct:.2f}%\n")
        for cls in classes:
            r = d['classes'][cls]
            f.write(f"  {cls:<12}  P={r['p']:.4f}  R={r['r']:.4f}  F1={r['f1']:.4f}\n")
    f.write('\n')
    for line in summary_lines:
        f.write(line + '\n')

print(f"\nSaved to: {out_path}")
