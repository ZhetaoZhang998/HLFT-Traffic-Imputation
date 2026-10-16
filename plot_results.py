"""Regenerate the README figures from recorded metrics and validation histories."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/verified"))
    parser.add_argument("--output", type=Path, default=Path("assets"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "summary.csv").open() as f:
        metrics = list(csv.DictReader(f))
    cities = [c for c in ["Berlin", "Guangzhou", "NewYork", "Seattle"]
              if (args.run_dir / c / "run_metadata.json").exists()]
    if not cities:
        raise ValueError("No recorded city runs were found")
    meta = {c: json.loads((args.run_dir / c / "run_metadata.json").read_text()) for c in cities}
    configs = {(m['config']['seed'], m['config']['max_epochs']) for m in meta.values()}
    settings = (f"Seed {next(iter(configs))[0]} · {next(iter(configs))[1]}-epoch cap"
                if len(configs) == 1 else "Mixed run settings; see recorded metadata")
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': '#2d3e35',
                        'axes.labelcolor': '#647068', 'xtick.color': '#647068', 'ytick.color': '#647068'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), facecolor='#faf8f3')
    for ax, city in zip(axes.ravel(), cities):
        selected = [r for r in metrics if r['city'] == city]
        vals = [float(r['rmse']) for r in selected]
        ax.set_facecolor('#faf8f3')
        ax.barh(['HLFT', 'Global mean', 'Sensor mean'], vals,
                color=['#396e60', '#d8cdb8', '#b6c6bd'], height=.55)
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.22)
        ax.set_title(city, loc='left', fontsize=15, fontweight='bold', pad=12)
        ax.set_xlabel('Test RMSE · lower is better', fontsize=9)
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * .02, i, f'{v:.3f}', va='center', fontsize=10)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#d8ded5')
        ax.tick_params(axis='y', length=0)
    for ax in axes.ravel()[len(cities):]:
        ax.set_visible(False)
    fig.suptitle('Recovering held-out traffic observations', x=.08, ha='left',
                 fontsize=21, fontweight='bold', y=.99)
    fig.text(.08, .925, 'Supplied splits · Validation-selected checkpoints · ' + settings,
             fontsize=10, color='#68736a')
    fig.text(.08, .012, 'Each city uses its supplied speed units. These are new runs, not the paper’s normalized benchmark.',
             fontsize=9, color='#68736a')
    fig.subplots_adjust(top=.84, bottom=.11, left=.12, right=.97, hspace=.65, wspace=.4)
    fig.savefig(args.output / 'model_comparison.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), facecolor='#faf8f3')
    for ax, city in zip(axes.ravel(), cities):
        with (args.run_dir / city / 'history.csv').open() as f:
            history = list(csv.DictReader(f))
        epochs = np.array([int(r['epoch']) for r in history])
        scores = np.array([float(r['valid_rmse']) for r in history])
        mask = epochs > 0
        ax.set_facecolor('#faf8f3')
        ax.plot(epochs[mask], scores[mask], color='#396e60', linewidth=2.2)
        best = meta[city]['fit']['best_epoch']
        ax.scatter([best], [scores[epochs == best][0]], color='#af825e', s=40,
                   zorder=3, label=f'Selected epoch {best}')
        ax.set_title(city, loc='left', fontweight='bold', fontsize=15)
        ax.set_xlabel('Training epoch', fontsize=9)
        ax.set_ylabel('Validation RMSE', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#d8ded5')
        ax.grid(axis='y', color='#e6e7df')
        ax.legend(frameon=False, fontsize=9)
    for ax in axes.ravel()[len(cities):]:
        ax.set_visible(False)
    fig.suptitle('Choosing the checkpoint from validation', x=.08, ha='left',
                 fontsize=21, fontweight='bold', y=.99)
    fig.text(.08, .925, 'Curves begin after the first training epoch. Test scores do not select the stopping point.',
             fontsize=10, color='#68736a')
    fig.subplots_adjust(top=.84, bottom=.10, left=.09, right=.97, hspace=.55, wspace=.3)
    fig.savefig(args.output / 'validation_curves.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
