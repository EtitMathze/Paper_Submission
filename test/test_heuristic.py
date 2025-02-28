from collections import defaultdict
from math import ceil
import numpy
import random
import platform_exports.task_tools
from platform_exports.unit_tools import unit

ALTERNATIVE_LENGTH_FACTOR = 1.5
MAX_WINDOW_LENGTH = 0.2

# Taken with inspiration from this Stack overflow answer:
# https://stackoverflow.com/a/3590105
def sum_constraint_random(amount, total) -> list[int]:
    dividers = sorted(random.sample(range(1, total), amount - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]


def get_task_borders(amount: int, max: int, exclude: list[int] = []) -> list[int]:
    all_borders = random.sample(range(1, max - 1), amount - 1)

    for exclude_elem in exclude:
        all_borders.remove(exclude_elem)

    while len(all_borders) < amount - 1:
        to_add = random.randint(1, max - 1)
        if to_add not in exclude:
            all_borders.append(to_add)

    all_borders.extend([0, max])
    all_borders.sort()
    return all_borders


def adapt_tasks_to_fixed_aer(task_list : list[platform_exports.task_tools.task], total_resource_number : int, exec_units : list[unit], do_one_shared_resource : bool, do_phase_order : bool) -> list[platform_exports.task_tools.task]:
    """
    Function to turn the task list into AER based phased execution models with the access windows at the start and end as well as only one shared resource.
    """
    for current_task in task_list:
        for exec_unit in exec_units:
            if current_task.access_windows_[exec_unit]:
                if do_one_shared_resource:
                    access_windows = current_task.access_windows_[exec_unit].copy()
                    new_access_windows = []
                    for current_access_window in access_windows:
                        for copy_to_window_idx in range(1, total_resource_number + 1):
                            if copy_to_window_idx == current_access_window.get_resource():
                                continue
                
                            new_window = platform_exports.task_tools.execution_window(current_access_window.get_start(),current_access_window.get_stop(), copy_to_window_idx)
                            new_access_windows.append(new_window)

                    current_task.access_windows_[exec_unit].extend(new_access_windows)
                    
                if do_phase_order:
                    kept_access_windows = current_task.access_windows_[exec_unit].copy()
                    current_task.access_windows_[exec_unit].clear()

                    for current_resource in range(1, total_resource_number + 1):
                        total_length = 0
                        for current_access_windows in kept_access_windows:
                            total_length += current_access_windows.get_stop() - current_access_windows.get_start()
                        
                        new_access_window_front = platform_exports.task_tools.execution_window(0, int(total_length / 2), current_resource)
                        new_access_window_back = platform_exports.task_tools.execution_window(current_task.get_exec_time(exec_unit.get_type()) - int(total_length / 2), current_task.get_exec_time(exec_unit.get_type()), current_resource)

                        current_task.access_windows_[exec_unit].extend([new_access_window_front, new_access_window_back])

    return task_list
    
def degeneralize_task_relations(tasks : list[platform_exports.task_tools.task], length : int):
    for current_task in tasks:
        for relating_task in tasks:
            relation = current_task.get_relation(relating_task)
            if relation is not None:
                relation.window_start_ = 0
                relation.window_end_ = int(length / 2)

    return tasks


def create_test_tasks(
    schedule_length: int,
    units : list[unit],
    no_tasks: int,
    acc_windows: int = 0,
    acc_window_resources: int = 1,
    timing_slack: int = 10,
    chance_exec_time_alternatives = 0,
    chance_task_relation = 0
) -> list[platform_exports.task_tools.task]:
    """
    Main function for generating task test cases for generating schedule.
    This function generates random task sets given the arguments.
    """
    # Incorporate slack into schedule length
    slack_schedule_length = int(schedule_length * (100 - timing_slack * 100) / 100)
    no_units = len(units)

    # Generate time slot starts of access windows and shared resource
    acc_positions = []
    acc_units = []
    if acc_windows != 0:
        acc_positions = random.sample(range(0, slack_schedule_length), acc_windows)
        acc_units = [i % acc_window_resources + 1 for i in range(acc_windows)]

    # Create the task borders for every unit
    task_borders: dict[unit, list[int]] = {}
    for current_unit in units:
        task_borders[current_unit] = get_task_borders(no_tasks, slack_schedule_length)
        task_borders[current_unit].sort()

    # Create the tasks
    task_id = 0
    tasks: list[platform_exports.task_tools.task] = []

    # Determine which unit (and with that task) holds the acc_windows
    acc_window_units: list[int] = []
    for i in range(len(acc_positions)):
        acc_window_units.append(random.randint(0, no_units))

    # Determine, if units of multiple types exist
    for current_unit in units:
        for t in range(len(task_borders[current_unit]) - 1):
            current_start_time = task_borders[current_unit][t]
            current_end_time = task_borders[current_unit][t + 1]
            true_exec_time = current_end_time - current_start_time
            exec_time_dict = {current_unit.get_type() : true_exec_time}

            # Create acc windows, if they are executed on this unit during this time
            current_acc_windows: dict[int, list[platform_exports.task_tools.execution_window]] = defaultdict(list)
            for acc_idx in range(len(acc_positions)):
                current_acc_win = acc_positions[acc_idx]
                if (
                    current_start_time <= current_acc_win
                    and current_acc_win <= current_end_time
                    and acc_window_units[acc_idx]
                ):

                    acc_window_max_size = acc_positions[acc_idx + 1] - acc_positions[acc_idx] if acc_idx != len(acc_positions) - 1 else schedule_length - acc_positions[acc_idx]
                    acc_window_max_size = int(min(MAX_WINDOW_LENGTH * true_exec_time,acc_window_max_size))
                    actual_acc_window_end = min(current_start_time + acc_window_max_size, current_end_time)

                    current_acc_window = platform_exports.task_tools.execution_window(current_acc_win - current_start_time, actual_acc_window_end - current_start_time,acc_units[acc_idx])
                    for unit_same_type in units:
                        if unit_same_type.get_type() == current_unit.get_type():
                            current_acc_windows[unit_same_type].append(current_acc_window)

            # If there is the possibility to get multiple access length to a unit, do that to make it more difficult
            possible_alternative_units = [current_u.get_type() for current_u in units if current_u.get_type() != current_unit.get_type]

            if possible_alternative_units and (random.random() < chance_exec_time_alternatives):
                alt_unit = random.choice(possible_alternative_units)
                alt_exec_time = int(ALTERNATIVE_LENGTH_FACTOR * true_exec_time)

                exec_time_dict[alt_unit] = alt_exec_time

                # If there is an alternative access window.
                if current_acc_windows:
                    alternative_acc_window = platform_exports.task_tools.execution_window(int(0.2 * alt_exec_time), int(0.2 * alt_exec_time + 1), acc_units[acc_idx])
                    current_acc_windows[alt_unit].append(alternative_acc_window)

            # Create name based on ID, unit, index and start time
            current_name = "task-ID" + str(task_id) + "-u" + str(units.index(current_unit)) + "-i" + str(t) + "-st" + str(current_start_time)

            new_task = platform_exports.task_tools.task(current_name,task_id,exec_time_dict,"/test/",schedule_length,current_acc_windows)

            if t != 0 and random.random() < chance_task_relation:
                old_task = tasks[-1]
                old_runtime = old_task.get_exec_time(current_unit.get_type())
                previous_start_min = int(0.5 * old_runtime)
                previous_start_max = int(2.5 * old_runtime)
                new_task_relation = platform_exports.task_tools.task_relation(previous_start_min,previous_start_max)
                old_task.task_relations_ = {new_task : new_task_relation}

            tasks.append(new_task)

            task_id += 1

    return tasks