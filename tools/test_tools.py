import numpy
import random
import tools.task_tools


# Taken with inspiration from this Stack overflow answer:
# https://stackoverflow.com/a/3590105
def sum_constraint_random(amount, total) -> list[int]:
    dividers = sorted(random.sample(range(1, total), amount - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]


def create_test_tasks(
    length: int,
    no_tasks_per_block: int,
    sec_windows: int = 0,
    acc_windows: int = 0,
    units: int = 4,
    timing_slack: int = 90,
) -> list[tools.task_tools.task]:
    borders: dict[int, list[int]] = {}

    tasks_per_unit = sum_constraint_random(units, no_tasks_per_block)
    sec_positions = sum_constraint_random(sec_windows + 1, length)

    task_id = 1
    tasks: list[tools.task_tools.task] = []

    for current_length in sec_positions:
        security_task_unit = random.randint(0, units - 1)

        for u in range(units):
            borders[u] = [
                border
                for border in sum_constraint_random(tasks_per_unit[u], current_length)
            ]

            for d in range(tasks_per_unit[u]):

                name = (
                    "task-"
                    + str(sec_positions.index(current_length))
                    + "-"
                    + str(u)
                    + "-"
                    + str(d)
                )

                # If security windows are activated and current task is the last one on the given unit
                if (
                    sec_windows != 0
                    and u == security_task_unit
                    and d == tasks_per_unit[u] - 1
                ):
                    used_border = int(borders[u][d] * (timing_slack / 100))
                    current_task = tools.task_tools.task(
                        name, task_id, used_border, "/", length, exclusive_windows=[[0, 1]]
                    )
                else:
                    used_border = int(borders[u][d] * (timing_slack / 100)) - 2
                    current_task = tools.task_tools.task(
                        name, task_id, used_border, "/"
                    )

                task_id += 1
                tasks.append(current_task)

    acc_window_tasks = random.sample(range(len(tasks)), acc_windows)
    already_chosen_starts: list[int] = [0, 1]
    retries = 0
    for acc_window_task in acc_window_tasks:
        start_candidate = (
            random.randint(0, length) % tasks[acc_window_task].get_exec_time() - 1
        )
        while start_candidate in already_chosen_starts:
            retries += 1
            start_candidate = (
                random.randint(0, length) % tasks[acc_window_task].get_exec_time()
            )
            if retries > 100:
                break
        already_chosen_starts.append(start_candidate)

        tasks[acc_window_task].access_windows_ = [
            (start_candidate, start_candidate + 1)
        ]

    return tasks
