import random
import numpy as np


def get_temp(baseline_temp, max_randomness):
    return max(0, baseline_temp + ((random.random() - 0.5) * max_randomness))


def get_temperature_exp_decay(
    n, baseline_temperature, total_rounds, decay_constant=0.05, period=None
):
    if n < total_rounds:
        if period and period < total_rounds:
            schedule = np.array([])
            cycles = int(np.floor(total_rounds / period))

            for i in range(0, cycles):
                x = np.arange(0, period)
                schedule = np.hstack(
                    (schedule, baseline_temperature * np.exp(-decay_constant * x))
                )
        else:
            x = np.arange(0, total_rounds)
            schedule = baseline_temperature * np.exp(-decay_constant * x)

        return schedule[n]

    else:
        print("conversation round exceeds total rounds")
        return
