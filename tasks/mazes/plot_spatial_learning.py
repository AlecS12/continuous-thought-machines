"""Plot recorded learning curves without rerunning training (requires matplotlib).

python -m tasks.mazes.plot_spatial_learning logs/spatial-overfit/results.json
"""

import argparse
import json
from pathlib import Path


def plot_results(results_path, output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    results_path = Path(results_path)
    results = json.loads(results_path.read_text())
    controls = list(dict.fromkeys(run['control'] for run in results['runs']))
    colors = dict(dense='#3066be', spatial='#cf6325', wave='#128575')
    fields = [('cross_entropy', 'Cross-entropy (log scale)'),
              ('success_rate', 'Collision-free route success'),
              ('collision_rate', 'Routes with a collision')]
    fig, axes = plt.subplots(len(controls), 3, figsize=(13, 3.7*len(controls)), squeeze=False)
    for row, control in enumerate(controls):
        for col, (field, title) in enumerate(fields):
            ax = axes[row, col]
            for run in results['runs']:
                if run['control'] != control:
                    continue
                curve = run['learning_curve']
                for split, style in [('train', '-'), ('id', '--')]:
                    values = [point['evaluations'][split][field] for point in curve]
                    ax.plot([point['update'] for point in curve], values, style,
                            color=colors[run['variant']], alpha=.65, linewidth=1.2)
            ax.set_title(f'{control}: {title}', fontsize=10)
            ax.set_xlabel('Optimizer updates')
            ax.grid(alpha=.15)
            if field == 'cross_entropy':
                ax.set_yscale('log')
            else:
                ax.set_ylim(-.03, 1.03)
    legend = [Line2D([0], [0], color=color, label=variant) for variant, color in colors.items()]
    legend += [Line2D([0], [0], color='black', linestyle=style, label=split)
               for split, style in [('Training set', '-'), ('Held-out ID', '--')]]
    fig.legend(handles=legend, loc='lower center', ncol=5, frameon=False)
    config = results['config']
    fig.suptitle(f"Fixed-maze learning: {config['train_examples']} training layouts; "
                 f"seeds {config['seeds']} (each line is one run)", fontsize=13)
    fig.tight_layout(rect=(0, .06, 1, .95))
    output = Path(output) if output else results_path.with_name('learning-curves.svg')
    fig.savefig(output, metadata={'Date': None})
    plt.close(fig)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results')
    parser.add_argument('--output')
    args = parser.parse_args()
    print(plot_results(args.results, args.output))
