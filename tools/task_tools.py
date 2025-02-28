import io
from math import inf
import math
import pathlib
import csv
import numpy


class task:
    def __init__(
        self,
        name_str: str,
        id: int,
        exec_time: int,
        file_path: str,
        deadline: int = -1,
        access_windows: list[tuple[int, int]] | None = None,
        exclusive_windows: list[tuple[int, int]] | None = None,
        resulution: int = 1,
    ) -> None:
        self.name_ = name_str
        self.id_ = id
        self.exec_time_ = exec_time
        self.path_ = pathlib.Path(file_path)
        # If no deadline exists, the task is no critical task
        self.criticality_ = deadline > 0
        self.periodic_deadline_ = deadline
        self.access_windows_ = access_windows
        self.exclusive_windows_ = exclusive_windows
        self.resolution_ = resulution

    def get_id(self) -> int:
        return self.id_

    def get_name(self) -> str:
        return self.name_

    def get_exec_time(self) -> int:
        return self.exec_time_

    def get_path(self) -> str:
        return self.path_

    def is_critical(self) -> bool:
        return self.criticality_

    def get_periodic_deadline(self) -> int:
        return self.periodic_deadline_

    def has_exclusive_window(self) -> bool:
        return not self.exclusive_windows_ is None

    def get_exclusive_windows(self) -> list[tuple[int, int]] | None:
        return self.exclusive_windows_

    def has_access_window(self) -> bool:
        return not self.access_windows_ is None

    def get_access_windows(self) -> list[tuple[int, int]] | None:
        return self.access_windows_

    def get_resolution(self) -> int:
        return self.resolution_

    def fit_to_resolution(self, new_resolution: int) -> None:
        correction_factor = self.resolution_ / new_resolution
        self.exec_time_ = int(self.exec_time_ * correction_factor)
        self.periodic_deadline_ = int(self.periodic_deadline_ * correction_factor)
        if self.exclusive_windows_ != None:
            new_windows: list[tuple[int, int]] = []
            for exclusive_window in self.exclusive_windows_:
                new_windows.append(
                    [
                        int(exclusive_window[0] * correction_factor),
                        int(exclusive_window[1] * correction_factor),
                    ]
                )
            self.exclusive_windows_ = new_windows

        if self.access_windows_ != None:
            new_windows: list[tuple[int, int]] = []
            for acc_window in self.access_windows_:
                new_windows.append(
                    [
                        int(acc_window[0] * correction_factor),
                        int(acc_window[1] * correction_factor),
                    ]
                )
            self.access_windows_ = new_windows

        self.resolution_ = new_resolution


class task_builder:
    def get_tasks(self) -> list[task]:
        raise NotImplementedError


class file_task_builder(task_builder):
    del_ = ";"
    quote_ = "\n"
    name_str = "Name"
    exec_time_str = "ExTime"
    path_str = "Path"
    crit_str = "Crit"
    deadline_str = "Deadline"
    excl_windows_str = "ExclWindow"
    excl_window_del_str = ","
    excl_split_str = "/"
    acc_windows_str = "AccWindow"
    acc_window_del_str = ","
    acc_split_str = "/"

    def __init__(self, file_str: str) -> None:
        self.file_str = file_str

    def from_file_str(self, file_str: str) -> list[task]:
        with open(file_str, "r") as f:
            task_list = self.from_file(f)
        return task_list

    def _get_windows(
        self, line_str: str, delimiter_str: str, split_str: str
    ) -> list[tuple[int, int]]:
        splitted = line_str.strip().split(delimiter_str)
        windows: list[tuple[int, int]] = []
        for pair in splitted:
            if pair == "":
                return None

            pair_split = pair.split(split_str)
            windows.append([int(pair_split[0]), int(pair_split[1])])
        return windows

    def get_excl_windows(self, line_str: str) -> list[tuple[int, int]] | None:
        if line_str == None:
            return None

        return self._get_windows(
            line_str,
            file_task_builder.excl_window_del_str,
            file_task_builder.excl_split_str,
        )

    def get_acc_windows(self, line_str: str) -> list[tuple[int, int]] | None:
        if line_str == None:
            return None

        return self._get_windows(
            line_str,
            file_task_builder.acc_window_del_str,
            file_task_builder.acc_split_str,
        )

    def from_file(self, file: io.TextIOWrapper) -> list[task]:
        csv_reader = csv.DictReader(
            file, delimiter=file_task_builder.del_, quotechar=file_task_builder.quote_
        )
        task_list = []
        task_id = 1
        for line in csv_reader:
            excl_windows = self.get_excl_windows(
                line[file_task_builder.excl_windows_str]
            )
            acc_windows = self.get_acc_windows(line[file_task_builder.acc_windows_str])

            current_task = task(
                line[file_task_builder.name_str],
                task_id,
                int(line[file_task_builder.exec_time_str]),
                line[file_task_builder.path_str],
                int(line[file_task_builder.deadline_str]),
                acc_windows,
                excl_windows,
            )
            task_list.append(current_task)
            task_id = task_id + 1
        return task_list

    def get_tasks(self) -> list[task]:
        return self.from_file_str(self.file_str)


def get_window_lenth(tasks: list[task]) -> int:
    deadline: int = inf
    for task in tasks:
        if task.is_critical() and task.get_periodic_deadline() < deadline:
            deadline = task.get_periodic_deadline()

    return deadline


def get_timing_resolution(tasks: list[task], platform_minimum: int = 1) -> int:
    exec_times: list[int] = []

    for task in tasks:
        round_up = math.ceil(task.get_exec_time() / platform_minimum) * platform_minimum
        exec_times.append(round_up)
        if task.get_exclusive_windows() != None:
            for exclusive_windows in task.get_exclusive_windows():
                exec_times.extend(exclusive_windows)

    exec_time_gcd = math.gcd(*exec_times)

    return exec_time_gcd


class polyhedron:
    def __init__(
        self,
        no_units: int,
        exec_unit: int,
        length: int,
        task_id: int,
        acc_windows: list[tuple[int, int]] | None = None,
        sec_windows: list[tuple[int, int]] | None = None,
    ) -> None:
        # Book keeping
        self.no_units_ = no_units
        self.exec_unit_ = exec_unit
        self.length_ = length
        self.task_id_ = task_id
        self.acc_windows_ = acc_windows
        self.sec_windows_ = sec_windows

        # Scheduling tensor
        # Execution frame
        self.tensor_ = numpy.zeros(shape=[2, self.no_units_, self.length_])
        for t in range(0, self.length_):
            self.tensor_[0][self.exec_unit_][t] = self.task_id_

        # Access window
        if self.acc_windows_ is not None:
            for u in range(0, self.no_units_):
                for timing_window in self.acc_windows_:
                    for t in range(timing_window[0], timing_window[1]):
                        self.tensor_[1][u][t] = self.task_id_

        # Security window
        if self.sec_windows_ is not None:
            for u in range(0, self.no_units_):
                for sec_windows in self.sec_windows_:
                    for t in range(sec_windows[0], sec_windows[1]):
                        self.tensor_[0][u][t] = self.task_id_
                        self.tensor_[1][u][t] = self.task_id_

    def get_task_id(self) -> int:
        return self.task_id_

    def get_length(self) -> int:
        return self.length_

    def get_tensor(self) -> numpy.array:
        return self.tensor_

    def __str__(self) -> str:
        string = "Polyhedron: T: "
        string.append(self.task_id_ + " ")
        string.append("U: ")
        string.append(str(self.exec_unit_) + "/" + str(self.no_units_) + "\n")
        string.append("Shape Execution:\n")

        for t in range(0, self.length_):
            for u in range(0, self.no_units_):
                string.append(str(self.tensor_[0][u][t]) + " ")
            string.append("\n")

        string.append("Shape Access:\n")

        for t in range(0, self.length_):
            for u in range(0, self.no_units_):
                string.append(str(self.tensor_[1][u][t]) + " ")
            string.append("\n")

        return string


def create_task_polyhedrons(input_task: task, no_units: int) -> list[polyhedron]:
    polyhedron_list: list[polyhedron] = []

    for u in range(0, no_units):
        new_polyhedron = polyhedron(
            no_units,
            u,
            input_task.get_exec_time(),
            input_task.get_id(),
            input_task.get_access_windows(),
            input_task.get_exclusive_windows(),
        )
        polyhedron_list.append(new_polyhedron)

    return polyhedron_list
