import matplotlib.pyplot as plt


def plot_ber(results):
    for algo_name, attacks in results.items():
        attack_names = list(attacks.keys())
        ber_values = [attacks[a] for a in attack_names]

        plt.figure(figsize=(12, 4))
        plt.bar(attack_names, ber_values)
        plt.title(f"Bitowa stopa błędów dla {algo_name} w zależności od zakłóceń")
        plt.ylabel("Bitowa stopa błędów")
        plt.ylim(0, 0.5)
        plt.xticks(rotation=45)
        plt.grid(axis="y", alpha=0.3)

        plt.savefig(f"ber_{algo_name}.png", dpi=150)
        plt.close()


def plot_psnr(psnr_results):
    algo_names = list(psnr_results.keys())
    values = [psnr_results[a] for a in algo_names]

    plt.figure(figsize=(8, 4))
    plt.bar(algo_names, values)
    plt.ylabel("PSNR (dB)")
    plt.title("Stosunek sygnału do szumu dla algorytmów")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig("psnr.png", dpi=150)
    plt.close()
