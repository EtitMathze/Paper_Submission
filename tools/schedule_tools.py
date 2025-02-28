import io
import math
import sys
import tools.task_tools
import tools.unit_tools
import numpy
import scipy.optimize

empty_symbol = None
max_duration_setup = 4000

t_axis = 2
u_axis = 1
acc_axis = 0


class abstract_schedule:
    def __init__(
        self,
        timing_resolution: int,
        no_units: int,
        S_matrix: numpy.ndarray | None = None,
        periodic_deadline: int | None = None,
    ) -> None:
        self.timing_resolution_ = timing_resolution
        self.no_units_ = no_units
        self.periodic_deadline_ = periodic_deadline

        if S_matrix is None:
            self.valid_ = False
            if periodic_deadline is None:
                self.S_matrix_ = numpy.zeros(
                    shape=[2, self.no_units_, max_duration_setup], dtype=numpy.int16
                )
            else:
                self.S_matrix_ = numpy.zeros(
                    shape=[2, self.no_units_, self.periodic_deadline_],
                    dtype=numpy.int16,
                )
        else:
            self.valid_ = True
            self.S_matrix_ = S_matrix

    def get_timing_resolution(self) -> int:
        return self.timing_resolution_

    def get_length(self) -> int:
        return self.S_matrix_.shape[t_axis]

    def get_no_units(self) -> int:
        return self.S_matrix_.shape[u_axis]

    def get_scheduled_tasks(self) -> list[int]:
        return numpy.unique(self.S_matrix_[self.S_matrix_ != 0])

    def get_first_free_t(self, polyhedron: tools.task_tools.polyhedron) -> int:
        t_x = polyhedron.tensor_.shape[t_axis]
        length_diff = self.S_matrix_.shape[t_axis] - t_x
        for t in range(0, length_diff):
            used_submatrix = self.S_matrix_[:, :, t : t + t_x]
            mult = numpy.multiply(used_submatrix, polyhedron.tensor_)
            sum = numpy.sum(mult)

            if sum == 0:
                return t

        return self.S_matrix_.shape[t_axis]

    def add_polyhedron_to_schedule(
        self, polyhedron: tools.task_tools.polyhedron, t_start: int
    ):
        self.S_matrix_ = register_polyhedron(self, t_start, polyhedron)

    def get_task_at(self, unit: int, time: int) -> int:
        return self.S_matrix_[0, unit, time]

    def get_latest_index(self) -> list[int]:
        latest: list[int] = []
        for u in range(0, self.S_matrix_.shape[u_axis]):
            reversed_matrix = self.S_matrix_[0, u, ::-1]
            latest.append(len(reversed_matrix) - numpy.argmax(reversed_matrix != 0) - 1)

        return latest

    def clear_task(self, unit_id: int) -> None:
        self.S_matrix_[self.S_matrix_ == unit_id] = 0

    def get_swapping_tasks(
        self,
        polyhedron: tools.task_tools.polyhedron,
        all_tasks: list[tools.task_tools.task],
    ) -> tuple[int, list[int]]:
        best_cost = math.inf
        best_list: list[int] = []
        polyhedron_mask = tools.task_tools.polyhedron(
            polyhedron.no_units_,
            polyhedron.exec_unit_,
            polyhedron.get_length(),
            1,
            polyhedron.acc_windows_,
            polyhedron.sec_windows_,
        )

        t_x = polyhedron_mask.tensor_.shape[t_axis]
        length_diff = self.S_matrix_.shape[t_axis] - t_x
        for t in range(0, length_diff):
            used_submatrix = self.S_matrix_[:, :, t : t + t_x]
            mult = numpy.multiply(used_submatrix, polyhedron_mask.tensor_)
            unique_tasks_idx = numpy.unique(mult)
            unique_tasks_idx = unique_tasks_idx[unique_tasks_idx != 0]
            for index in unique_tasks_idx:
                clipped = mult.clip(0, 1)
                cost = numpy.sum(clipped)

                if cost < best_cost:
                    best_cost = cost
                    best_list = unique_tasks_idx

        if best_cost == math.inf:
            print("No swapping found!")

        return best_cost, [d.get_id() for d in all_tasks if d.get_id() in best_list]


def register_polyhedron(
    schedule: abstract_schedule, t_start: int, polyhedron: tools.task_tools.polyhedron
) -> numpy.ndarray:
    t_x = polyhedron.tensor_.shape[t_axis]
    test_submatrix = schedule.S_matrix_[:, :, t_start : t_start + t_x]
    mult = numpy.multiply(test_submatrix, polyhedron.tensor_)
    sum = numpy.sum(mult)

    if sum != 0:
        raise RuntimeError("Time slot not free!")

    inserted_array = numpy.add(polyhedron.tensor_, test_submatrix)

    deleted_array = numpy.delete(
        schedule.S_matrix_, range(t_start, t_x + t_start), t_axis
    )
    finished_array = numpy.insert(
        deleted_array, obj=[t_start], values=inserted_array, axis=t_axis
    )

    return finished_array


def print_abstract_schedule_table(
    table: abstract_schedule, output: io.TextIOWrapper = sys.stdout
) -> None:
    # Execution
    for u in range(0, table.no_units_):
        output.write("Unit :" + str(u) + " |\t")
        for t in range(table.S_matrix_.shape[t_axis]):
            output.write("{:1X}".format(table.S_matrix_[0][u][t]))

        output.write("\n")

    # Access window
    for u in range(0, table.no_units_):
        output.write("Mem :" + str(u) + " |\t")
        for t in range(table.S_matrix_.shape[t_axis]):
            output.write("{:1X}".format(table.S_matrix_[1][u][t]))

        output.write("\n")


def get_last_first_empty(S_matrix: numpy.array) -> list[int]:
    last_used_list: list[int] = []
    for u in range(0, S_matrix.shape[u_axis]):
        beta = S_matrix[0][u][:]
        index = numpy.where(beta != 0)
        if numpy.size(index) == 0:
            last_used_list.append(0)
        else:
            last_used_list.append(numpy.max(index) + 1)

    return last_used_list


class abstract_schedule_creator:
    def __init__(self) -> None:
        pass

    def create_schedule(
        self,
        sched_tasks: list[tools.task_tools.task],
        no_units: int,
        start_schedule: abstract_schedule | None = None,
        all_tasks: list[tools.task_tools.task] | None = None,
    ) -> abstract_schedule:
        raise NotImplemented("Abstract Schedule Creator not implemented!")


class block_step(abstract_schedule_creator):

    def has_parallel_execution(self, poly: tools.task_tools.polyhedron) -> bool:
        scheduled_on_unit: list[bool] = []
        for u in range(poly.tensor_.shape[u_axis]):
            scheduled_on_unit.append(numpy.sum(poly.tensor_[0, u, :]) != 0)

        return sum(bool(x) for x in scheduled_on_unit) != 1

    def get_schedule_cost(
        self, schedule: abstract_schedule, polyhedron: tools.task_tools.polyhedron
    ) -> tuple[int, int]:
        t_temp = schedule.get_first_free_t(polyhedron)

        if t_temp == schedule.S_matrix_.shape[t_axis]:
            return math.inf, math.inf

        # t_max_before = get_last_first_empty(schedule.S_matrix_)
        test_matrix = register_polyhedron(schedule, t_temp, polyhedron)
        t_max_after = get_last_first_empty(test_matrix)

        if self.has_parallel_execution(polyhedron):
            cost = 0
            for u in range(0, test_matrix.shape[u_axis]):
                test_slice = test_matrix[0, u, 0 : t_max_after[u]]
                cost += t_max_after[u] - numpy.count_nonzero(test_slice)

        else:
            nonzeros = 0
            for u in range(0, test_matrix.shape[u_axis]):
                test_slice = test_matrix[0, u, 0 : max(t_max_after)]
                nonzeros += numpy.count_nonzero(test_slice)

            cost = 3 * max(t_max_after) - nonzeros

        return cost, t_temp

    def place_next_polyhedron(
        self,
        schedule: abstract_schedule,
        polyhedrons: list[tools.task_tools.polyhedron],
        tasks: list[tools.task_tools.task],
    ) -> tools.task_tools.polyhedron:
        cost: int = math.inf
        t_used = 0
        for polyh in polyhedrons:
            current_cost, t_tmp = self.get_schedule_cost(schedule, polyh)
            if current_cost < cost:
                cost = current_cost
                used_polyhedron = polyh
                t_used = t_tmp
            elif current_cost == cost and tasks[polyh.get_task_id() - 1].is_critical():
                used_polyhedron = polyh
                t_used = t_tmp

        if cost == math.inf:
            raise RuntimeError("Infinite cost for all polyhedrons!")

        schedule.add_polyhedron_to_schedule(used_polyhedron, t_used)
        return used_polyhedron

    def create_schedule(
        self,
        sched_tasks: list[tools.task_tools.task],
        no_units: int,
        start_schedule: abstract_schedule | None = None,
        all_tasks: list[tools.task_tools.task] | None = None,
    ) -> abstract_schedule:

        if all_tasks is None:
            all_tasks = sched_tasks

        if start_schedule is None:
            current_schedule = abstract_schedule(all_tasks[0].get_resolution(), no_units)
        else:
            current_schedule = start_schedule


        polyhedron_dict: dict[
            tools.task_tools.task, list[tools.task_tools.polyhedron]
        ] = {
            current_task: tools.task_tools.create_task_polyhedrons(
                current_task, no_units
            )
            for current_task in sched_tasks
        }
        marked_scheduled: dict[tools.task_tools.task, bool] = {
            current_task: False for current_task in sched_tasks
        }

        while False in marked_scheduled.values():
            considered_polyhedrons: list[tools.task_tools.polyhedron] = []
            for current_task in sched_tasks:
                if marked_scheduled[current_task] == False:
                    considered_polyhedrons.extend(polyhedron_dict[current_task])

            try:
                next_polyhedron = self.place_next_polyhedron(
                    current_schedule, considered_polyhedrons, all_tasks
                )

            except RuntimeError as e:
                print(e)
                return current_schedule

            marked_scheduled[all_tasks[next_polyhedron.get_task_id() - 1]] = True

        return current_schedule


    def iterative_improve(self,
        schedule: abstract_schedule, all_tasks: list[tools.task_tools.task]
    ) -> tuple[abstract_schedule, int]:
        # Get violating task
        no_tasks = int(numpy.unique(schedule.S_matrix_).size)
        latest = schedule.get_latest_index()
        latest_unit = latest.index(max(latest))
        latest_task_idx = schedule.get_task_at(latest_unit, max(latest))
        schedule.clear_task(latest_task_idx)

        possible_polyhedrons = tools.task_tools.create_task_polyhedrons(
            all_tasks[latest_task_idx - 1], schedule.get_no_units()
        )
        min_cost = math.inf
        swapped_task_idx: list[int] = []

        for u in range(0, schedule.get_no_units()):
            current_cost, swap_task_idxs = schedule.get_swapping_tasks(
                possible_polyhedrons[u], all_tasks
            )
            if current_cost < min_cost:
                min_cost = current_cost
                swapped_task_idx = swap_task_idxs

        for task_to_clear in swapped_task_idx:
            schedule.clear_task(task_to_clear)

        new_sched_1 = self.create_schedule(
            [all_tasks[latest_task_idx - 1]], schedule.get_no_units(), schedule, all_tasks
        )

        non_scheduled_tasks: list[tools.task_tools.task] = [
            d
            for d in all_tasks
            if d.get_id() not in schedule.get_scheduled_tasks()
        ]
        new_sched_2 = self.create_schedule(
            non_scheduled_tasks, new_sched_1.get_no_units(), new_sched_1, all_tasks
        )
        no_tasks_end = int(numpy.unique(schedule.S_matrix_).size)
        
        print("Lost tasks: " + str(no_tasks - no_tasks_end))

        return new_sched_2, no_tasks - no_tasks_end


class ilp_schedule_creator(abstract_schedule_creator):

    def calculate_continuity(self, input_array : numpy.array) -> int:
        
        where_array = numpy.argwhere(input_array == 1)
        if where_array.size == 0:
            return 0
        first = where_array.min()
        last = where_array.max()

        in_between = numpy.argwhere(input_array[first:last] == 0)
        if in_between.size != 0:
            return 0       
        return 1
        

    def create_schedule(
        self,
        sched_tasks: list[tools.task_tools.task],
        no_units: int,
        start_schedule: abstract_schedule | None = None,
        all_tasks: list[tools.task_tools.task] | None = None,
    ) -> abstract_schedule:
        # Checks
        if start_schedule is not None:
            raise RuntimeError("Start schedule not supported.")
        
        if all_tasks is None:
            all_tasks = sched_tasks

        # Adapt all tasks to the timing resolution
        self.timing_resolution_ = tools.task_tools.get_timing_resolution(all_tasks)
        for task in all_tasks:
            task.fit_to_resolution(self.timing_resolution_)

        # Get values
        self.t_p = min([d.get_periodic_deadline() for d in all_tasks if d.get_periodic_deadline() != -1])
        self.D = sched_tasks

        # Create vectors
        self.N_d = len(sched_tasks)
        self.N_u = no_units
        self.p_k = numpy.zeros(shape=[no_units, self.N_d])
        self.S_matrix_ = numpy.zeros(shape=[2, no_units, self.t_p])
        self.t_k = numpy.zeros(shape=[self.N_d, self.t_p])

        self.s = numpy.zeros(shape=[self.t_p])
        self.a = numpy.zeros(shape=[self.t_p])

        # Create X vector for ILP
        self.x = numpy.empty(shape=[0])
        for d in range(self.N_d):
            self.x = numpy.append(self.x, self.p_k[:,d])
            self.x = numpy.append(self.x, self.t_k[d,:])

        # Define constraints
        # Assignment Constraint
        one_unit_assingment_constraints : list[scipy.optimize.LinearConstraint] = []
        total_runtime_constraints : list[scipy.optimize.LinearConstraint] = []
        continuity_constraints : list[scipy.optimize.NonlinearConstraint] = []

        for d in range(self.N_d):
            sum_matrix_assign = numpy.zeros(shape=[self.N_u, self.N_u])
            sum_matrix_assign[:,0] = 1
            sum_vector_assign = numpy.zeros(shape=[self.N_u])
            sum_vector_assign[0] = 1
            one_unit_assingment_constraint = scipy.optimize.LinearConstraint(A=sum_matrix_assign, lb=sum_vector_assign, ub=sum_vector_assign)
            one_unit_assingment_constraints.append(one_unit_assingment_constraint)

            sum_matrix_runtime = numpy.zeros(shape=[self.t_p, self.t_p])
            sum_matrix_runtime[:,0] = 1
            sum_vector_runtime = numpy.zeros(shape=[self.t_p])
            sum_vector_runtime[0] = all_tasks[d].exec_time_
            total_runtime_constraint = scipy.optimize.LinearConstraint(A=sum_matrix_runtime,lb=sum_vector_runtime)
            total_runtime_constraints.append(total_runtime_constraint)

            continuity_constraint_vector = numpy.ones(shape=[self.N_d])
            continuity_constraint = scipy.optimize.NonlinearConstraint(self.calculate_continuity,continuity_constraint_vector,continuity_constraint_vector)
            

        self.all_tasks = all_tasks