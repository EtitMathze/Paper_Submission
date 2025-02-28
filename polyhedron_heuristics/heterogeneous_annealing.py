import math
import sys
import time
from typing import Any
import numpy
from numpy.core.multiarray import array as array
import scipy.optimize
import platform_exports
import platform_exports.platform_tools
from platform_exports.task_tools import task, task_relation
import platform_exports.task_tools
import platform_exports.unit_tools
import polyhedron_heuristics.schedule_tools
import scipy

# Epsilon value to allow for a predefined unit in the optimization
EPS = sys.float_info.epsilon

# Definitions penalties
TASK_OVERLAP_STR = "task_overlap"
TASK_OVERLAP_PENALTY = 500

ACCESS_WINDOW_STR = "access_windows"
ACCESS_WINDOW_PENALTY = 500

TASK_RELATION_STR = "task_relations"
TASK_RELATION_PENALTY = 500

TASK_OVERRUN_STR = "task_overrun"
TASK_OVERRUN_PENALTY = 15000

DEADLINE_MISS_STR = "deadline_miss"
DEADLINE_MISS_PENALTY = 500

MULTIPLE_EXEC_STR = "multiple_exec"
MULTIPLE_EXEC_PENALTY = 25

NO_EXEC_STR = "no_execution"
NO_EXEC_SEVERITY = 500000

# Definitions bonusses
UNIT_SAVINGS_STR = "unit_saving"
UNIT_SAVINGS_BONUS = 10

# General definitions
UNUSED_START = -1
PLATFORM_MINIMUM_RES = 100

# Simulated Annealing definitions
MAX_ITERATIONS = 3000
VISIT_VALUE = 3.9
ACCEPT_VALUE = -10


class basic_cost_function:
    """
    Cost function class that gets called to determine the penaltys and costs of a proposed schedule.

    Members:
        tasks_: list[task] - A list of all tasks used for indexing
        unit_lists_: int - The list of (heterogeneous) execution units
        max_runtime_: int = math.inf - The maximum allowed runtime
        max_exex_number_: int = 1 - The number of allowed executions per task in this TDMA schedule table

    Definition of functions is based on optimizing the vector x : numpy.array, which is build like this:
    [t_00, t_00, ..., t_10, t_11, ..., u_00, u_01, ..., u_10, u_11, ...]
    At first, the time of every execution is presented.
    Afterward, execution unit for each starting point is presented.

    If a start is not used, this is determined by the UNUSED_START definition.
    """

    def __init__(
        self,
        tasks: list[task],
        unit_list: list[platform_exports.unit_tools.unit],
        extra_execution_slack=0.2,
        negative_features: list[str] = [
            TASK_OVERLAP_STR,
            ACCESS_WINDOW_STR,
            TASK_RELATION_STR,
            TASK_OVERRUN_STR,
            DEADLINE_MISS_STR,
            MULTIPLE_EXEC_STR,
            NO_EXEC_STR,
        ],
        positive_features: list[str] = [UNIT_SAVINGS_STR],
    ) -> None:
        self.tasks_ = tasks
        self.units_ = unit_list
        self.max_runtime_ = platform_exports.task_tools.get_common_period(tasks)
        self.assigned_vector_slots_ = self._calculate_vector_assignment(tasks, extra_execution_slack)
        self.total_exec_slots_ = max([task_indeces[-1] for task_indeces in self.assigned_vector_slots_.values()]) + 1
        self.task_unit_dict: dict[task, dict[int, int]] = {}
        self.included_features_ = negative_features + positive_features

        self.task_unit_dict = polyhedron_heuristics.schedule_tools.create_local_to_global_dict(tasks, unit_list)

        print(
            "Cost function: for "
            + str(len(self.tasks_))
            + " tasks, "
            + str(len(self.units_))
            + " units.\nSchedule period: "
            + str(self.max_runtime_)
            + " with max "
            + str(self.assigned_vector_slots_.values())
            + " executions per task."
        )

    def _calculate_vector_assignment(self, tasks: list[task], extra_percentage) -> dict[task, list[int]]:
        """
        Calculate which parts of the vector are assigned to which task. The result is saved in a dictionary.
        """
        task_index_dict: dict[task, list[int]] = {}
        previous_end = 0

        for t in tasks:
            if t.get_periodic_deadline() is not None and t.get_periodic_deadline() != -1:
                no_execs = math.ceil((self.max_runtime_ / t.get_periodic_deadline()) * (1 + extra_percentage))
            else:
                no_execs = math.ceil(1 + extra_percentage)

            task_index_dict[t] = list(range(previous_end, previous_end + no_execs))
            previous_end += no_execs

        return task_index_dict

    def _get_real_unit_from_task_unit(self, t: task, u_index: int) -> int:
        """
        This function translates the local task unit to the "real" global unit. This is necessary because of the constraints placed on the tasks.
        """
        return self.task_unit_dict[t][u_index]

    def _sorted_overlapping_time(self, start1: int, start2: int, end1: int, end2: int) -> int:
        """
        Return the overlap between sorted time slots.
        """
        return len(range(max(start1, start2), min(end1, end2)))

    def _overlapping_time(self, start_time1: int, start_time2: int, end_time1: int, end_time2: int) -> int:
        """
        Calculates the overlapping time slots between two (1 and 2) time periods and account for possible rollover.
        """
        # Check, if execution rolls around
        rollover1 = start_time1 >= end_time1
        rollover2 = start_time2 >= end_time2

        # Generate the blocked time for t1
        if rollover1:
            time_slots1 = set(range(0, end_time1)) if end_time1 != 0 else set()
            time_slots1.update(range(start_time1, self.max_runtime_))
        else:
            time_slots1 = set(range(start_time1, end_time1))

        # Generate the blocked time for t2
        if rollover2:
            time_slots2 = set(range(0, end_time2)) if end_time2 != 0 else set()
            time_slots2.update(set(range(start_time2, self.max_runtime_)))
        else:
            time_slots2 = set(range(start_time2, end_time2))

        # Generate the overlapped slots
        overlapped_slots = time_slots1.intersection(time_slots2)

        return len(overlapped_slots)

    def _get_task_starts(self, x: numpy.array, t: task) -> list[int]:
        """
        Returns ORDERED set of all starts inside the schedules, the list size is not fixed.
        """
        number_task_starts = len(self.assigned_vector_slots_[t])
        first_start = x[self.tasks_.index(t)]

        starts = [
            round((first_start + i * t.get_periodic_deadline()) % self.max_runtime_) for i in range(number_task_starts)
        ]
        starts.sort()

        return starts

    def _get_task_units_indeces(self, x: numpy.array, t: task) -> list[int]:
        """
        Returns units for the first, second, ... starting point in ascending order.
        """
        units_indices = [round(x[len(self.tasks_) + u_index]) for u_index in self.assigned_vector_slots_[t]]

        if units_indices == []:
            print("ERROR: Unit list empty")

        return units_indices

    def _get_task_start_n(self, x: numpy.array, t: task, i: int) -> int:
        """
        Returns the ith starting point of task t.
        """
        return self._get_task_starts(x, t)[i]

    def _get_task_unit_n(self, x: numpy.array, t: task, i: int) -> platform_exports.unit_tools.unit:
        """
        Returns the GLOBAL unit of the ith starting point of task t.
        """
        # Get the unit indices
        units = self._get_task_units_indeces(x, t)

        # Important! Get the GLOBAL unit
        unit_index = self.task_unit_dict[t][units[i]]
        return self.units_[unit_index]

    def _get_unit_type(self, local_unit_index: int) -> int:
        """
        Returns the unit type of the unit index.
        """
        return self.units_[local_unit_index].get_type()

    def _get_number_of_executions(self, x: numpy.array, t: task) -> int:
        """
        Returns the number of executions per scheduling table

        Arguments:
            x : numpy.array - current optimization value
            t1 : task - first task

        Returns:
            int - Number of executions.
        """
        start_times = self._get_task_starts(x, t)
        return len(start_times)

    def _get_potential_overlaps(self, x: numpy.array, start1: int, end1: int, unit: int) -> int:
        """
        Return the sum of occupied time during start and end on the given unit.
        """
        overlap = 0
        # Iterate over all tasks
        for t in self.tasks_:

            # Get task starts
            starts = self._get_task_starts(x, t)

            # Iterate over task starts and check for potential overlaps
            for index in range(len(starts)):

                unit2 = self._get_task_unit_n(x, t, index)
                if unit2 == unit:
                    start2 = starts[index]
                    end2 = (start2 + t.get_exec_time(unit2.get_type())) % self.max_runtime_

                    overlap += self._overlapping_time(start1, start2, end1, end2)

        return overlap

    def _get_expected_executions(self, t: task) -> int:
        """
        Returns the number of expected executions of this task
        """
        if t.get_periodic_deadline() < 1:
            expected_executions = 1
        else:
            expected_executions = math.ceil(self.max_runtime_ / t.get_periodic_deadline())
        return expected_executions

    def _get_no_execution_units(self, x: numpy.array) -> int:
        """
        Returns the number of used execution units.
        """
        # Create empty set of execution units
        units: set[platform_exports.unit_tools.unit] = set()

        # Add task wise the units to the set
        for t in self.tasks_:
            for i in range(self._get_number_of_executions(x, t)):
                units.add(self._get_task_unit_n(x, t, i))

        # Return the length of this set
        return len(units)

    def get_exec_unit_bonus(self, x: numpy.array) -> int:
        """
        Returns a bonus depending on the amount of not used execution units.
        """
        return len(self.units_) - self._get_no_execution_units(x)

    def get_task_exec_overlap(self, x: numpy.array, t1: task, t2: task) -> int:
        """
        Calculates the total overlap of all windows of task t1 and task t2.
        Note: There might be a rollover at the end of the scheduling period into the next one. This needs to be checked and managed accordingly.

        Arguments:
            x : numpy.array - current optimization value
            t1 : task - first task
            t2 : task - second task

        Returns:
            int, Number of overlaping time slots.
        """
        overlap = 0  # Initialize overlap

        # Start time of tasks
        starts_task1 = self._get_task_starts(x, t1)
        starts_task2 = self._get_task_starts(x, t2)

        for idx1 in range(len(starts_task1)):
            start_time1 = starts_task1[idx1]
            unit1 = self._get_task_unit_n(x, t1, idx1)
            end_time1 = (start_time1 + t1.get_exec_time(unit1.get_type())) % self.max_runtime_

            for idx2 in range(len(starts_task2)):
                # Only skip, if the task and the execution start is identical
                if t1 == t2 and idx1 == idx2:
                    continue

                # If the windows are not on the same unit, skip this combo
                unit2 = self._get_task_unit_n(x, t2, idx2)

                if unit1 != unit2:
                    continue

                start_time2 = starts_task2[idx2]
                end_time2 = (start_time2 + t2.get_exec_time(unit2.get_type())) % self.max_runtime_

                overlap += self._overlapping_time(start_time1, start_time2, end_time1, end_time2)

        return overlap

    def get_access_window_overlap(self, x: numpy.array, t1: task, t2: task) -> int:
        """
        Check for every access window of tasks t1 and t1, if they overlap.
        Note: There might be a rollover at the end of the scheduling period into the next one. This needs to be checked and managed accordingly.

        Arguments:
            x : numpy.array - current optimization value
            t1 : task - first task
            t2 : task - second task

        Returns:
            int, Number of overlaping time slots for every access window.
        """
        if t1 == t2:
            return 0

        # Check if access windows are present
        if t1.get_access_windows() is None or t2.get_access_windows() is None:
            return 0

        # Initialize overlap
        overlap = 0

        # Start time of tasks
        task_starts1 = self._get_task_starts(x, t1)
        task_starts2 = self._get_task_starts(x, t2)

        # Iterate over first task start points
        for idx1 in range(len(task_starts1)):
            unit1 = self._get_task_unit_n(x, t1, idx1)
            acc_windows1 = t1.get_access_windows()[unit1.get_type()]
            start_time_task1 = task_starts1[idx1]

            # Iterate over second task start points
            for idx2 in range(len(task_starts2)):
                unit2 = self._get_task_unit_n(x, t2, idx2)
                acc_windows2 = t2.get_access_windows()[unit2.get_type()]
                start_time_task2 = task_starts2[idx2]

                # Iterate over first window
                for acc_window1 in acc_windows1:
                    abs_start1 = (start_time_task1 + acc_window1.get_start()) % self.max_runtime_
                    abs_end1 = (start_time_task1 + acc_window1.get_stop()) % self.max_runtime_

                    # Iterate over second window
                    for acc_window2 in acc_windows2:

                        if acc_window1.get_resource() != acc_window2.get_resource():
                            continue

                        abs_start2 = (start_time_task2 + acc_window2.get_start()) % self.max_runtime_
                        abs_end2 = (start_time_task2 + acc_window2.get_stop()) % self.max_runtime_

                        # Calculate overlapping time slots. Rollover gets taken car of by function
                        overlap += self._overlapping_time(abs_start1, abs_start2, abs_end1, abs_end2)

        return overlap

    def get_deadline_misses(self, x: numpy.array, t: task) -> int:
        """
        Returns a metric for missed deadlines between two executions.

        Arguments:
            x : numpy.array - current optimization value
            t1 : task - first task

        Returns:
            int - Number of time slots violating the deadline between tasks.
        """
        # If the task is not a critical task, no deadline is provided and the function returns with 0
        if not t.is_critical():
            return 0

        # Initialize penalty
        penalty = 0

        task_starts = self._get_task_starts(x, t)

        for idx in range(len(task_starts)):
            deadline_gap = 0

            # If last point is reached, keep wraping in mind
            if idx == len(task_starts) - 1:
                deadline_gap += self.max_runtime_ - task_starts[idx]
                deadline_gap += task_starts[0]

            # If last point is not reached, the gap is just the gap between both points
            else:
                deadline_gap += task_starts[idx + 1] - task_starts[idx]

            penalty += max(0, deadline_gap - t.get_periodic_deadline())

        return penalty

    def get_diference_from_optimum_executions(self, x: numpy.array, t: task) -> int:
        """
        Returns the difference from the optimal number of executions for every task in conjunction with the maximum amount of executions and the periodic deadlines.
        Has a higher penalty for 0 executions.

        Arguments:
            x : numpy.array - current optimization vector
            t: task - the task

        Returns:
            int - The difference between the optimum and the current execution value.
        """
        exec_amount = self._get_number_of_executions(x, t)
        if exec_amount == 0:
            return NO_EXEC_SEVERITY

        expected_executions = self._get_expected_executions(t)

        difference = abs(exec_amount - expected_executions)
        return difference

    def get_task_overrun(self, x: numpy.array, t: task) -> int:
        """
        Returns a penalty if the task can not reasonabely be executed on the designated unit.
        Has a higher penalty for 0 executions.

        Arguments:
            x : numpy.array - current optimization vector
            t: task - the task

        Returns:
            int - The difference between the optimum and the current execution value.
        """
        penalty = 0
        for idx in range(len(self._get_task_starts(x, t))):
            current_unit = self._get_task_unit_n(x, t, idx)

            penalty += max(0, t.get_exec_time(current_unit.get_type()) - self.max_runtime_)

        return penalty

    def get_task_relation_difference(self, x: numpy.array, t1: task, t2: task) -> int:
        """
        Returns the difference between two tasks and their defined starting relation.

        Arguments:
            x : numpy.array - current optimization vector
            t1: task - the first task
            t2: task - the first task

        Returns:
            int - The difference between the maximum allowed start differnce and the true start difference
        """
        # Get the task relations between the two tasks
        relation1 = t1.get_relation(t2)
        relation2 = t2.get_relation(t1)

        # If no relation exists for both tasks, return 0
        if relation1 is None and relation2 is None:
            return 0

        # If only task 1 has a defined relation to task 2, use task 1 as reference
        if relation1 is not None and relation2 is None:
            used_relation = relation1
            source_task = t1
            dest_task = t2

        # If only task 2 has a defined relation to task 1, use task 2 as reference
        elif relation1 is None and relation2 is not None:
            used_relation = relation2
            source_task = t2
            dest_task = t1

        # If both tasks have a relation, use task 1 and the minimum of both relations
        else:
            time_slots1 = set(range(relation1.get_window_start(), relation1.get_window_end()))
            time_slots2 = set(range(-relation2.get_window_end(), -relation2.get_window_start()))
            combined_slots = time_slots1.intersection(time_slots2)

            used_relation = task_relation(min(combined_slots), max(combined_slots))
            source_task = t1
            dest_task = t2

        penalty = 0

        if self._get_expected_executions(t1) != self._get_expected_executions(t2):
            raise RuntimeError(
                "Difficult to calculate start difference for task "
                + t1.get_id()
                + " and "
                + t2.get_id()
                + " because of differint execution expectations"
            )

        # Get all starts of both tasks
        starts_task1 = self._get_task_starts(x, source_task)
        starts_task2 = self._get_task_starts(x, dest_task)

        rel_best_start = (used_relation.get_window_start() + used_relation.get_window_end()) / 2

        for start_time1 in starts_task1:
            abs_best_start = int((start_time1 + rel_best_start) % self.max_runtime_)
            earliest_allowed_start = start_time1 + used_relation.get_window_start()
            latest_allowed_start = start_time1 + used_relation.get_window_end()

            # Create list of allowed starts (should wrap around)
            allowed_starts = [
                possible_start % self.max_runtime_
                for possible_start in range(earliest_allowed_start, latest_allowed_start)
            ]

            # If there is a fitting start of task 2 in the allowed starts, continue
            if set(starts_task2).intersection(allowed_starts):
                continue

            # Otherwise calculate the distance to the nearest task
            min_distance = 2 * self.max_runtime_
            for start_time2 in starts_task2:
                distance = abs(abs_best_start - start_time2)
                if distance < min_distance:
                    min_distance = distance

            penalty += min_distance

        return penalty

    def check_for_no_execution(self, x: numpy.array, t: task) -> int:
        """
        Checks if a task is executed at all or not.

        Arguments:
            x : numpy.array - current optimization vector
            t: task - the task

        Returns:
            int - 1, if it is not executed 0 if it is executed.
        """
        not_executed = self._get_number_of_executions(x, t) == 0
        if not_executed:
            return 1
        else:
            return 0

    def __call__(self, x: numpy.array, *args: Any) -> Any:
        """
        Many entry to cost calculation of optimizers. Takes in x as numpy array. Other arguments are not used.

        Arguments:
            x : numpy.array - vector that is optimized by optimization problem

        Returns:
            Any - the cost of the schedule
        """
        # --- Calculate penalties --- #
        penalty = 0
        for t1 in range(len(self.tasks_)):
            for t2 in range(t1, len(self.tasks_)):
                # Check if task execution overlap
                if TASK_OVERLAP_STR in self.included_features_:
                    penalty += TASK_OVERLAP_PENALTY * self.get_task_exec_overlap(x, self.tasks_[t1], self.tasks_[t2])

                # Check if task access windows overlap
                if ACCESS_WINDOW_STR in self.included_features_:
                    penalty += ACCESS_WINDOW_PENALTY * self.get_access_window_overlap(
                        x, self.tasks_[t1], self.tasks_[t2]
                    )

                # Check if the task relation is upholded
                if TASK_RELATION_STR in self.included_features_:
                    penalty += TASK_RELATION_PENALTY * self.get_task_relation_difference(
                        x, self.tasks_[t1], self.tasks_[t2]
                    )

            # Check if the task execution on the current unit is longer than the scheduling time
            if TASK_OVERRUN_STR in self.included_features_:
                penalty += TASK_OVERRUN_PENALTY * self.get_task_overrun(x, self.tasks_[t1])

            # Check, if the deadline is reached, done for multiple execution windows inside one TDMA schedule table
            if DEADLINE_MISS_STR in self.included_features_:
                penalty += DEADLINE_MISS_PENALTY * self.get_deadline_misses(x, self.tasks_[t1])

            # Add a small cost for more execuitons per schedule table
            if MULTIPLE_EXEC_STR in self.included_features_:
                penalty += MULTIPLE_EXEC_PENALTY * self.get_diference_from_optimum_executions(x, self.tasks_[t1])

            # Add a large cost, if a task is not scheduled at all
            if NO_EXEC_STR in self.included_features_:
                penalty += NO_EXEC_SEVERITY * self.check_for_no_execution(x, self.tasks_[t1])

        # --- Calculate bonusses --- #
        bonus = 0

        # Calculate a bonus for not used execution units
        if UNIT_SAVINGS_STR in self.included_features_:
            bonus += UNIT_SAVINGS_BONUS * self.get_exec_unit_bonus(x)
        return penalty - bonus


def convert_output_to_schedule(
    x: numpy.array, units: list[platform_exports.unit_tools.unit], all_tasks: list[task], cost_function
) -> polyhedron_heuristics.schedule_tools.abstract_schedule:
    """
    Convert a given result to an abstract_schedule class.

    Arguments:
        x : numpy.array - The optimizer result
        no_units : int - The number of used execution units
        all_tasks : list[task] - A list of all used tasks

    Returns:
        abstract_schedule : Generated abstract_schedule object
    """
    # Get amount of shared resources
    unique_share_resources: set[int] = set()
    for t in all_tasks:
        if t.get_access_windows() is not None:
            unique_share_resources.update(t.get_access_windows().keys())
    number_shared_resources = len(unique_share_resources)

    # Create empty schedule
    current_schedule = polyhedron_heuristics.schedule_tools.abstract_schedule(
        all_tasks[0].get_resolution(),
        len(units),
        number_shared_resources,  # Processor is handled in constructor
        periodic_deadline=cost_function.max_runtime_,
    )

    # Cycle through every task
    for current_task in all_tasks:
        starts = cost_function._get_task_starts(x, current_task)

        # Cycle through every starting index of that task
        for current_start_idx in range(len(starts)):
            current_start = starts[current_start_idx]
            unit = cost_function._get_task_unit_n(x, current_task, current_start_idx)

            used_access_windows = (
                None
                if current_task.get_access_windows() is None
                else current_task.get_access_windows()[unit.get_type()]
            )

            # Create the next polyhedron and enter it into the schedule
            current_polyhedron = platform_exports.task_tools.polyhedron(
                len(units),
                units.index(unit),
                current_task.get_exec_time(unit.get_type()),
                current_task.get_id(),
                used_access_windows,
                number_shared_resources,
            )
            current_schedule.add_polyhedron_to_schedule(current_polyhedron, current_start)

    return current_schedule


class annealing_schedule_creator(polyhedron_heuristics.schedule_tools.abstract_schedule_creator):
    """
    Scheduling creator based on simulated / dual annealing. The most work is done in the cost function class.
    """

    def __init__(
        self,
        finishing_score: int = 0,
        start_with_zero: int = False,
        iterative: bool = False,
        negative_features: list[str] = [
            TASK_OVERLAP_STR,
            ACCESS_WINDOW_STR,
            TASK_RELATION_STR,
            TASK_OVERRUN_STR,
            DEADLINE_MISS_STR,
            MULTIPLE_EXEC_STR,
            NO_EXEC_STR,
        ],
        positive_features: list[str] = [UNIT_SAVINGS_STR],
    ) -> None:
        self.timing_list_ = []
        self.start_with_zero_ = start_with_zero
        self.finishing_score_ = finishing_score
        self.do_iteratively_ = iterative
        self.positive_features_ = positive_features
        self.negative_features_ = negative_features
        super().__init__()

    def callback(self, x, tasks: list[task], e=None, context=None) -> bool:
        """
        Callback function to call from the optimizer. Only prints the result and triggers exit, if self.finishing_score_ is reached is reached.
        """
        if e is None:
            e = self.cost_function(x)

        print(str(x) + " with cost value: " + str(e))
        for t in tasks:
            all_task_starts = self.cost_function._get_task_starts(x, t)
            for current_start in all_task_starts:
                current_unit = self.cost_function._get_task_unit_n(x, t, all_task_starts.index(current_start))

                print(
                    "Task : "
                    + str(t.get_id())
                    + " Start: "
                    + str((current_start) % self.cost_function.max_runtime_)
                    + " End: "
                    + str((current_start + t.get_exec_time(current_unit.get_type())) % self.cost_function.max_runtime_)
                    + " Unit: "
                    + str(self.cost_function.units_.index(current_unit))
                )

                if t.get_access_windows() is not None and t.get_access_windows()[current_unit.get_type()]:
                    windows = t.get_access_windows()[current_unit.get_type()]

                    for access_window in windows:
                        print(
                            "Access window position: "
                            + str((current_start + access_window.get_start()) % self.cost_function.max_runtime_)
                            + " until: "
                            + str((current_start + access_window.get_stop()) % self.cost_function.max_runtime_)
                        )

        total_runtime = time.time() - self.start_time_
        timing_tuple = [total_runtime, e]
        self.timing_list_.append(timing_tuple)
        return e <= self.finishing_score_

    def get_timing_tuple(self):
        """
        Function to return the timing of the annealing process.
        """
        return self.timing_list_

    def create_upper_bounds_units(
        self, tasks: list[task], unit_dict: dict[task, dict[int, int]], to_filled_lentgh: int
    ) -> numpy.array:
        """
        Creates the upper bounds array for the task decision variables for execution. Because of the different execution units,
        different upper bounds are needed.

        Arguments:
            tasks : list[task] - The list of all tasks to be scheduled
            exec_max : int - The maximum amount a task can be executed

        Returns:
            numpy.array - The upper bounds
        """
        # Create the array with 0s
        ub_array = numpy.zeros(to_filled_lentgh)

        for t in tasks:
            for i in self.cost_function.assigned_vector_slots_[t]:
                # Fill the range of the array with the maximum local key (the maximum number available)
                ub_array[i] = max(unit_dict[t].keys())

        return ub_array

    def create_starting_exec_points(self, sched_tasks: list[task]) -> numpy.array:
        """
        Creates a starting schedule so that the simulated annealing does not need to move to far.
        """
        x = numpy.zeros(len(sched_tasks))
        next_start = 0

        # Set the first start of every unit to start after the completion of the previous task
        for t in sched_tasks:

            # Get the shortest runtime for the unit
            best_runtime = t.get_periodic_deadline()
            for index in t.get_supported_exec_unit_indeces():
                this_runtime = t.get_exec_time(index)
                if this_runtime < best_runtime:
                    best_runtime = this_runtime

            x[sched_tasks.index(t)] = next_start
            next_start = (next_start + best_runtime) % self.cost_function.max_runtime_

        return x

    def create_starting_units(self, sched_tasks: list[task]) -> numpy.array:
        """
        Creates the starting unit assignments.
        """
        # Create shell numpy array
        x = numpy.zeros(max([self.cost_function.assigned_vector_slots_[t][-1] for t in sched_tasks]) + 1)
        next_unit = 0

        for t in sched_tasks:
            # Create starting and ending index as well as the number of local units
            no_units = len(self.cost_function.task_unit_dict[t])

            # Assign the unit in a round robin style
            for task_unit_index in self.cost_function.assigned_vector_slots_[t]:
                x[task_unit_index] = next_unit % no_units

            next_unit += 1

        return x

    def create_schedule(
        self, sched_tasks: list[task], units: list[platform_exports.unit_tools.unit]
    ) -> polyhedron_heuristics.schedule_tools.abstract_schedule:

        self.start_time_ = time.time()
        self.cost_function = basic_cost_function(
            sched_tasks,
            units,
            extra_execution_slack=0,
            negative_features=self.negative_features_,
            positive_features=self.positive_features_,
        )

        # Set same timing resolution for all tasks
        timing_res = platform_exports.task_tools.get_timing_resolution(
            sched_tasks, platform_minimum=PLATFORM_MINIMUM_RES
        )
        for t in sched_tasks:
            t.fit_to_resolution(timing_res)

        # Get the max number of executions from the cost function.
        number_unit_slots = max([self.cost_function.assigned_vector_slots_[t][-1] for t in sched_tasks]) + 1
        number_tasks = len(sched_tasks)

        # x is build as [t_start00, t_start_01, ... , t_startN_D0, t_startN_D1, ... , u_00, u_01, ...]
        x = numpy.zeros(number_unit_slots + number_tasks)

        lb = numpy.full(x.shape, 0)  # set lower bound for all to 0

        ub = numpy.full(x.shape, self.cost_function.max_runtime_)  # set upper bound for tasks to schedule table length
        ub[number_tasks::] = self.create_upper_bounds_units(
            sched_tasks, self.cost_function.task_unit_dict, number_unit_slots
        )  # set upper bound for units to no_units
        ub = ub + EPS
        bounds = scipy.optimize.Bounds(lb, ub)

        # Create an x vector start
        if self.start_with_zero_ is False:
            # If no one is provided, try a rotating assignment
            x[number_tasks::] = self.create_starting_units(sched_tasks)
            x[:number_tasks:] = self.create_starting_exec_points(sched_tasks)

        if self.do_iteratively_:
            tmp_x = x
            used_optimizations = []
            all_included_features = self.cost_function.included_features_
            for optimizations in all_included_features:
                used_optimizations.append(optimizations)
                self.cost_function.included_features_ = used_optimizations
                print("Used optim: " + str(used_optimizations))

                # If the cost function is already 0, skip this part and do the next one.
                if tmp_x is not None and self.cost_function(tmp_x) == 0:
                    print("Cost function already 0, thus skip and continue with next constraint set.")
                    continue

                # Do the annealing process
                res = scipy.optimize.dual_annealing(
                    self.cost_function,
                    bounds,
                    maxiter=MAX_ITERATIONS,
                    visit=VISIT_VALUE,
                    accept=ACCEPT_VALUE,
                    callback=(lambda x, e, context: self.callback(x, sched_tasks, e, context)),
                    x0=tmp_x,
                )
                tmp_x = res["x"]

        else:
            res = scipy.optimize.dual_annealing(
                self.cost_function,
                bounds,
                maxiter=MAX_ITERATIONS,
                visit=VISIT_VALUE,
                accept=ACCEPT_VALUE,
                callback=(lambda x, e, context: self.callback(x, sched_tasks, e, context)),
                x0=x,
            )

        # Print the resulting stats
        print(str(res["x"]) + " with cost value: " + str(res["fun"]))

        # Convert the annealing result to a schedule
        schedule = convert_output_to_schedule(res["x"], units, sched_tasks, self.cost_function)
        schedule.set_valid()
        return schedule
