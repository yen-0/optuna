import logging
import time

import optuna


logging.disable(logging.CRITICAL)
N_TRIALS = 100
N_DIMENSIONS = 20


def objective(trial: optuna.Trial) -> float:
    return sum(
        (trial.suggest_float(f"x{i}", -5.0, 5.0) - 0.5) ** 2 for i in range(N_DIMENSIONS)
    )


study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(seed=123, n_startup_trials=10), direction="minimize"
)
started = time.perf_counter()
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
elapsed = time.perf_counter() - started
print({"trials": N_TRIALS, "dimensions": N_DIMENSIONS, "elapsed_s": elapsed,
       "per_trial_s": elapsed / N_TRIALS, "best_value": study.best_value})
