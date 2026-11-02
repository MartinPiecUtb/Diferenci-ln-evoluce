from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


D = 10
NP = 30
FES = 10000
RUNS = 10
LOWER, UPPER = -5.12, 5.12


def sphere(x):
    return np.sum(x ** 2)


def rastrigin(x):
    return 10 * D + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def de(function, variant, seed):
    rng = np.random.default_rng(seed)
    population = rng.uniform(LOWER, UPPER, (NP, D))
    fitness = np.array([function(x) for x in population])


    individual_f = np.full(NP, 0.5)
    individual_cr = np.full(NP, 0.9)

    evaluations = NP
    evaluation_counts = [evaluations]
    history = [min(fitness)]

    while evaluations < FES:
        new_population = population.copy()
        new_fitness = fitness.copy()
        best_index = np.argmin(fitness)

        for i in range(NP):
            if evaluations >= FES:
                break

            f = 0.5
            cr = 0.8
            if variant == "jDE":
                f, cr = individual_f[i], individual_cr[i]
                if rng.random() < 0.1:
                    f = rng.uniform(0.1, 0.9)
                if rng.random() < 0.1:
                    cr = rng.random()


            candidates = [k for k in range(NP) if k != i]
            if variant == "Best/1":
                candidates = [k for k in candidates if k != best_index]
                r1, r2 = rng.choice(candidates, 2, replace=False)
                mutant = population[best_index] + f * (population[r1] - population[r2])
            else:
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                mutant = population[r1] + f * (population[r2] - population[r3])

            mutant = np.clip(mutant, LOWER, UPPER)


            mask = rng.random(D) < cr
            mask[rng.integers(D)] = True
            trial = np.where(mask, mutant, population[i])


            trial_fitness = function(trial)
            evaluations += 1
            if trial_fitness <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                if variant == "jDE":
                    individual_f[i], individual_cr[i] = f, cr

        population, fitness = new_population, new_fitness
        evaluation_counts.append(evaluations)
        history.append(min(fitness))

    return np.array(evaluation_counts), np.array(history)


def plot_convergence(function, name):
    plt.figure(figsize=(10, 6))
    for variant in ("Rand/1", "Best/1", "jDE"):
        histories = []
        for run in range(RUNS):
            x, y = de(function, variant, seed=100 + run)
            histories.append(y)
        histories = np.array(histories)
        plt.semilogy(x, np.median(histories, axis=0), label=variant)
        print(f"{name:10} {variant:7} | mean: {np.mean(histories[:, -1]):.6g}")

    plt.title(f"DE convergence - {name} (D={D}, {RUNS} runs)")
    plt.xlabel("Function evaluations (FES)")
    plt.ylabel("Best fitness (median)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name(f"convergence_{name.lower()}.png"), dpi=160)
    plt.show()


if __name__ == "__main__":
    plot_convergence(sphere, "Sphere")
    plot_convergence(rastrigin, "Rastrigin")
