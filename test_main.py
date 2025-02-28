import polyhedron_heuristics.heterogeneous_annealing
import test
import polyhedron_heuristics.schedule_tools
import platform_exports.unit_tools
import test.test_heuristic
import time
import csv

# CSV field names
TASK_SLACK_STR = "Slack"
TASK_NO_STR = "Tasks"
UNIT_NO_STR = "Units"
UNIT_TYPES_STR = "UnitTypes"
TASK_RELATION_CHANCE_STR = "TaskRelations"
SHARED_RESOURCE_NO_STR = "SharedResourceNo"
ACC_WINDOW_NO_STR = "AccWindows"
EMULATE_AER_STR = "AER"
TASK_ALTERNATIVE_STR = "AltChances"
RUNTIME_MIN_STR = "Min"
RUNTIME_MAX_STR = "Max"
RUNTIME_AVG_STR = "Avg"
REP_NR_STR = "Rep"
STEP_STR = "Step"
COST_STR = "Cost"

fieldnames_runtime = [
    TASK_SLACK_STR,
    TASK_NO_STR,
    UNIT_NO_STR,
    UNIT_TYPES_STR,
    TASK_RELATION_CHANCE_STR,
    SHARED_RESOURCE_NO_STR,
    ACC_WINDOW_NO_STR,
    EMULATE_AER_STR,
    TASK_ALTERNATIVE_STR,
    RUNTIME_MIN_STR,
    RUNTIME_MAX_STR,
    RUNTIME_AVG_STR,
]

fieldnames_quality = [
    TASK_SLACK_STR,
    TASK_NO_STR,
    UNIT_NO_STR,
    UNIT_TYPES_STR,
    TASK_RELATION_CHANCE_STR,
    SHARED_RESOURCE_NO_STR,
    ACC_WINDOW_NO_STR,
    EMULATE_AER_STR,
    REP_NR_STR,
    TASK_ALTERNATIVE_STR,
    STEP_STR,
    COST_STR,
]


def test_one(
    length: int,
    timing_slack: int,
    no_tasks: int,
    no_units: int,
    no_unit_types: int,
    no_acc_windows: int,
    no_shared_resources: int,
    ratio_task_rel: float,
    move_to_old_phased_resource: bool,
    move_to_old_phased_order: bool,
    move_to_easy_relations : bool,
    alternative_chance : float
) -> tuple[float, bool, list[tuple[float, int]]]:
    if no_units < no_unit_types:
        raise RuntimeError(
            "Number of units can't be smaller than the number of unit types! "
            + str(no_units)
            + " vs. "
            + str(no_unit_types)
        )

    units = []
    type_index = 0
    for u_index in range(no_units):
        new_unit = platform_exports.unit_tools.unit("Unit" + str(u_index), type_index)
        type_index = (type_index + 1) % no_unit_types
        units.append(new_unit)

    tasks = test.test_heuristic.create_test_tasks(
        length,
        units,
        no_tasks=no_tasks,
        acc_windows=no_acc_windows,
        acc_window_resources=no_shared_resources,
        timing_slack=timing_slack,
        chance_exec_time_alternatives=alternative_chance,
        chance_task_relation=ratio_task_rel,
    )

    tasks = test.test_heuristic.adapt_tasks_to_fixed_aer(
        tasks, no_shared_resources, units, move_to_old_phased_resource, move_to_old_phased_order
    )

    if move_to_easy_relations:
        tasks = test.test_heuristic.degeneralize_task_relations(tasks, length)

    schedule_creator = polyhedron_heuristics.heterogeneous_annealing.annealing_schedule_creator()

    success_value = True
    start_time = time.time()

    try:
        schedule = schedule_creator.create_schedule(tasks, units)
    except Exception as e:
        print("Can't find schedule.")
        success_value = False
    
    timing_list = schedule_creator.get_timing_tuple()
    end_time = time.time()

    return end_time - start_time, success_value, timing_list


def main():
    """
    Iterate through the variables and test runtime.
    """
    # slack_tests = [50,40,30,20,10,0]
    slack_tests = [40,30]

    #task_no_tests = range(10, 21, 2)
    task_no_tests = [10]

    #unit_no_tests = [6]
    unit_no_tests = [6]

    # unit_types_tests = [1,2,3]
    unit_types_tests = [2]

    # task_relation_chances = [0,10,20,30]
    task_relation_chances_tests = [0,5,10,15,20,25,30,35,40,45,50,55,60]

    # task_no_shared_resources_tests = [1, 2, 3, 4]
    task_no_shared_resources_tests = [1]

    task_no_access_windows_tests = [0]
    # task_no_access_windows_tests = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26,28,30,32,34,36,38,40,42,44,46,48,50]

    # tasks_emulate_aer_tests = [True, False]
    tasks_emulate_aer_tests = [False]

    tasks_emulate_easy_relations = [False]

    #task_multiple_version_chances = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    task_multiple_version_chances = [0]

    repetitions = 3

    quality_data_name = "name_for_experiment.csv"
    runtimes_data_name = "name_for_experiment_runtime.csv"

    with open(runtimes_data_name, "w") as f:
        csv_writer = csv.DictWriter(f, delimiter=";", fieldnames=fieldnames_runtime)
        csv_writer.writeheader()

    with open(quality_data_name, "w") as f:
        csv_writer = csv.DictWriter(f, delimiter=";", fieldnames=fieldnames_quality)
        csv_writer.writeheader()

    for slack in slack_tests:
        for no_tasks in task_no_tests:
            for no_units in unit_no_tests:
                for no_unit_types in unit_types_tests:
                    for no_relation_task_rel in task_relation_chances_tests:
                        for emulate_aer in tasks_emulate_aer_tests:
                            for no_shared_resources in task_no_shared_resources_tests:
                                for no_access_windows in task_no_access_windows_tests:
                                    for task_alternatives_chances in task_multiple_version_chances:
                                        run_times = []
                                        successes = False
                                        for rep in range(repetitions):
                                            run_time, success, timing_result = test_one(
                                                1000,
                                                slack / 100,
                                                no_tasks,
                                                no_units,
                                                no_unit_types,
                                                no_access_windows,
                                                no_shared_resources,
                                                no_relation_task_rel / 100,
                                                emulate_aer,
                                                emulate_aer,
                                                tasks_emulate_easy_relations[0],
                                                task_alternatives_chances / 100
                                            )

                                            successes = successes or success
                                            run_times.append(run_time)

                                            if success:
                                                with open(quality_data_name, "a") as f:
                                                    csv_writer = csv.DictWriter(
                                                        f, delimiter=";", fieldnames=fieldnames_quality
                                                    )
                                                    for quality_tuple in timing_result:
                                                        result_dict = {
                                                            TASK_SLACK_STR: slack,
                                                            TASK_NO_STR: no_tasks,
                                                            UNIT_NO_STR: no_units,
                                                            UNIT_TYPES_STR: no_unit_types,
                                                            TASK_RELATION_CHANCE_STR: no_relation_task_rel,
                                                            EMULATE_AER_STR: emulate_aer,
                                                            SHARED_RESOURCE_NO_STR: no_shared_resources,
                                                            ACC_WINDOW_NO_STR: no_access_windows,
                                                            REP_NR_STR: rep,
                                                            TASK_ALTERNATIVE_STR: task_alternatives_chances,
                                                            STEP_STR: quality_tuple[0],
                                                            COST_STR: quality_tuple[1],
                                                        }
                                                        csv_writer.writerow(result_dict)

                                        if not successes:
                                            run_times.append(-1)

                                        result_dict = {
                                            TASK_SLACK_STR: slack,
                                            TASK_NO_STR: no_tasks,
                                            UNIT_NO_STR: no_units,
                                            UNIT_TYPES_STR: no_unit_types,
                                            TASK_RELATION_CHANCE_STR: no_relation_task_rel,
                                            EMULATE_AER_STR: emulate_aer,
                                            SHARED_RESOURCE_NO_STR: no_shared_resources,
                                            ACC_WINDOW_NO_STR: no_access_windows,
                                            TASK_ALTERNATIVE_STR: task_alternatives_chances,
                                            RUNTIME_MIN_STR: min(run_times),
                                            RUNTIME_MAX_STR: max(run_times),
                                            RUNTIME_AVG_STR: sum(run_times) / len(run_times),
                                        }

                                        print(result_dict)

                                        with open(runtimes_data_name, "a") as f:
                                            csv_writer = csv.DictWriter(f, delimiter=";", fieldnames=fieldnames_runtime)
                                            csv_writer.writerow(result_dict)


if __name__ == "__main__":
    main()
