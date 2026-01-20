import matplotlib.pyplot as plt


def plot_ber(results):
    for algo_name, attacks in results.items():
        attack_names = list(attacks.keys())
        ber_values = [attacks[a] for a in attack_names]

        plt.figure(figsize=(12, 4))
        bars = plt.bar(attack_names, ber_values)
        plt.title(f"Bitowa stopa błędów dla {algo_name} w zależności od zakłóceń")
        plt.ylabel("Bitowa stopa błędów")
        plt.ylim(0, 0.5)
        plt.xticks(rotation=45)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        for bar, value in zip(bars, ber_values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.savefig(f"ber_{algo_name}.png", dpi=150)
        plt.close()


def plot_psnr(psnr_results):
    algo_names = list(psnr_results.keys())
    psnr_values = [psnr_results[a] for a in algo_names]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(algo_names, psnr_values)
    plt.ylabel("PSNR (dB)")
    plt.title("Stosunek sygnału do szumu dla algorytmów")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    for bar, value in zip(bars, psnr_values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.savefig("psnr.png", dpi=150)
    plt.close()


def plot_ssim(results):
    algo_names = list(results.keys())
    ssim_values = [results[a] for a in algo_names]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(algo_names, ssim_values)
    plt.ylabel("SSIM")
    plt.title("Średni współczynnik podobieństwa strukturalnego dla algorytmów")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    for bar, value in zip(bars, ssim_values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{float(value):.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.savefig("ssim.png", dpi=150)
    plt.close()


def plot_ncc(results):
    for algo_name, attacks in results.items():
        attack_names = list(attacks.keys())
        ncc_values = [attacks[a] for a in attack_names]

        plt.figure(figsize=(12, 4))
        bars = plt.bar(attack_names, ncc_values)
        plt.title(f"Normalized Cross-Correlation (NCC) watermarku dla {algo_name} w zależności od zakłóceń")
        plt.ylabel("NCC")
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        for bar, value in zip(bars, ncc_values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )


        plt.savefig(f"ncc_{algo_name}.png", dpi=150)
        plt.close()
