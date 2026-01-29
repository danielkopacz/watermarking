import matplotlib.pyplot as plt
import numpy as np


def plot_ber(results):
    algo_names = list(results.keys())
    first_algo = next(iter(results))
    attack_names = list(results[first_algo].keys())

    x = np.arange(len(attack_names))
    total_width = 0.8
    single_width = total_width / len(algo_names)

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, algo_name in enumerate(algo_names):
        ber_values = [results[algo_name].get(a, 0) for a in attack_names]
        offset = x - (total_width / 2) + (i * single_width) + (single_width / 2)

        bars = ax.bar(offset, ber_values, single_width, label=algo_name)

        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    ax.set_title("Bitowa stopa błędów (BER) dla algorytmów w zależności od zakłóceń")
    ax.set_ylabel("Bitowa stopa błędów")
    ax.set_xticks(x)
    ax.set_xticklabels(attack_names, rotation=45)
    ax.legend(title="Algorytmy")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("ber.png", dpi=150)
    plt.close()


def plot_psnr(results):
    algo_names = list(results.keys())
    values = [results[a] for a in algo_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(algo_names)))  # pyright: ignore[reportAttributeAccessIssue]
    bars = ax.bar(algo_names, values, color=colors)

    ax.set_title("Stosunek sygnału do szumu (PSNR) dla algorytmów")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)

    plt.tight_layout()
    plt.savefig("psnr.png", dpi=150)
    plt.close()


def plot_ssim(results):
    algo_names = list(results.keys())
    values = [results[a] for a in algo_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(algo_names)))  # pyright: ignore[reportAttributeAccessIssue]
    bars = ax.bar(algo_names, values, color=colors)

    ax.set_title("Średni współczynnik podobieństwa strukturalnego dla algorytmów")
    ax.set_ylabel("SSIM")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)

    plt.tight_layout()
    plt.savefig("ssim.png", dpi=150)
    plt.close()


def plot_ncc(results):
    algo_names = list(results.keys())
    first_algo = next(iter(results))
    attack_names = list(results[first_algo].keys())

    x = np.arange(len(attack_names))
    total_width = 0.8
    single_width = total_width / len(algo_names)

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, algo_name in enumerate(algo_names):
        ber_values = [results[algo_name].get(a, 0) for a in attack_names]
        offset = x - (total_width / 2) + (i * single_width) + (single_width / 2)

        bars = ax.bar(offset, ber_values, single_width, label=algo_name)

        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    ax.set_title("Normalized Cross-Correlation (NCC) dla algorytmów w zależności od zakłóceń")
    ax.set_ylabel("NCC")
    ax.set_xticks(x)
    ax.set_xticklabels(attack_names, rotation=45)
    ax.legend(title="Algorytmy")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("ncc.png", dpi=150)
    plt.close()


def plot_jpeg(results, metric_name="BER"):
    plt.figure(figsize=(14, 6))

    for method_name, attacks in results.items():
        if "jpeg" not in attacks:
            continue
        qualities, values = zip(*attacks["jpeg"])
        plt.plot(qualities, values, marker='o', label=method_name)

    plt.xlabel("Jakość JPEG")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} dla algorytmów w zależności od jakości JPEG")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metric_name.lower()}_jpeg.png", dpi=150)
    plt.close()


def plot_ng(results, metric_name="BER"):
    plt.figure(figsize=(14, 6))

    for method_name, attacks in results.items():
        if "noise_gauss" not in attacks:
            continue
        sigmas, values = zip(*attacks["noise_gauss"])
        plt.plot(sigmas, values, marker='o', label=method_name)

    plt.title(f"{metric_name} dla algorytmów w zależności od natężenia szumu Gaussowskiego")
    plt.xlabel("Sigma szumu Gaussowskiego")
    plt.ylabel(metric_name)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Algorytmy")
    plt.tight_layout()
    plt.savefig(f"{metric_name.lower()}_gaussian_noise.png", dpi=150)
    plt.close()