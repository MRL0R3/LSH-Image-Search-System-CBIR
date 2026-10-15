import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List
import logging


class LSHVisualizer:

    def __init__(self, output_dir: str = "results/plots/lsh"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_performance_comparison(self, benchmark_results: Dict):
        brute_times = benchmark_results['brute_force']['times']
        lsh_times = benchmark_results['lsh']['times']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        methods = ['Brute-force', 'LSH']
        avg_times = [np.mean(brute_times), np.mean(lsh_times)]
        colors = ['#e74c3c', '#3498db']

        ax1.bar(methods, avg_times, color=colors, alpha=0.7)
        ax1.set_ylabel('Average Query Time (seconds)', fontsize=12)
        ax1.set_title('Search Method Performance', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        for i, v in enumerate(avg_times):
            ax1.text(i, v, f'{v:.6f}s', ha='center', va='bottom', fontweight='bold')

        ax2.boxplot([brute_times, lsh_times], labels=methods, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_ylabel('Query Time (seconds)', fontsize=12)
        ax2.set_title('Time Distribution', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'performance_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Performance comparison saved to {save_path}")

    def plot_speedup_chart(self, benchmark_results: Dict):
        brute_times = benchmark_results['brute_force']['times']
        lsh_times = benchmark_results['lsh']['times']

        speedups = [bt / lt if lt > 0 else 0 for bt, lt in zip(brute_times, lsh_times)]
        avg_speedup = np.mean(speedups)

        fig, ax = plt.subplots(figsize=(10, 6))

        queries = [f'Q{i + 1}' for i in range(len(speedups))]
        bars = ax.bar(queries, speedups, color='#2ecc71', alpha=0.7)
        ax.axhline(y=avg_speedup, color='red', linestyle='--',
                   label=f'Average: {avg_speedup:.2f}x', linewidth=2)

        ax.set_xlabel('Query', fontsize=12)
        ax.set_ylabel('Speedup (times faster)', fontsize=12)
        ax.set_title('LSH Speedup per Query', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{speedup:.2f}x', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'speedup_chart.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Speedup chart saved to {save_path}")

    def plot_precision_comparison(self, benchmark_results: Dict, top_k: int = 5):
        brute_results = benchmark_results['brute_force']['results']
        lsh_results = benchmark_results['lsh']['results']

        precisions = []
        query_names = []

        for i in range(len(brute_results)):
            brute_set = set([res[0] for res in brute_results[i][1][:top_k]])
            lsh_set = set([res[0] for res in lsh_results[i][1][:top_k]])

            if lsh_set:
                precision = len(lsh_set.intersection(brute_set)) / len(lsh_set)
                precisions.append(precision * 100)
                query_names.append(brute_results[i][0])

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(range(len(precisions)), precisions, color='#9b59b6', alpha=0.7)
        ax.axhline(y=np.mean(precisions), color='red', linestyle='--',
                   label=f'Average: {np.mean(precisions):.1f}%', linewidth=2)

        ax.set_xlabel('Query', fontsize=12)
        ax.set_ylabel(f'Precision@{top_k} (%)', fontsize=12)
        ax.set_title(f'LSH Precision (Top-{top_k} Results)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(query_names)))
        ax.set_xticklabels([f'Q{i + 1}' for i in range(len(query_names))], rotation=0)
        ax.set_ylim([0, 105])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        for bar, prec in zip(bars, precisions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{prec:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'precision_chart.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Precision chart saved to {save_path}")

    def create_summary_report(self, benchmark_results: Dict, top_k: int = 5):
        logging.info("Creating visualization reports...")

        self.plot_performance_comparison(benchmark_results)
        self.plot_speedup_chart(benchmark_results)
        self.plot_precision_comparison(benchmark_results, top_k)

        logging.info(f"All charts saved to {self.output_dir}")
