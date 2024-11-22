import polyhedron_heuristics.heterogeneous_annealing
import test
import polyhedron_heuristics.schedule_tools
import platform_exports.unit_tools
import test.test_heuristic
import time

CHANCE_EXEC_TIME_ALTERNATIVES = 0.1

def test_one(length : int, timing_slack : int, no_tasks : int, no_units : int, no_unit_types : int, no_acc_windows : int, no_shared_resources : int, ratio_task_rel : float) -> tuple[float, bool]:
    if no_units < no_unit_types:
        raise RuntimeError("Number of units can't be smaller than the number of unit types! " + str(no_units) + " vs. " + str(no_unit_types))

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
        chance_exec_time_alternatives=CHANCE_EXEC_TIME_ALTERNATIVES,
        chance_task_relation=ratio_task_rel
    )

    schedule_creator = polyhedron_heuristics.heterogeneous_annealing.annealing_schedule_creator()
    
    success_value = True
    start_time = time.time()

    try:
        schedule = schedule_creator.create_schedule(tasks, units)
    except Exception as e:
        print("Can't find schedule.")
        success_value = False

    end_time = time.time()

    return end_time - start_time, success_value

def main():
    """
    Iterate through the variables and test runtime.
    """
    #slack_tests = [50,40,30,20,10,0]
    slack_tests = [0]

    task_no_tests = [6,7,8,9,10]
    #task_no_tests = [6]

    unit_no_tests = [4,5,6]
    #unit_no_tests = [4]

    unit_types_tests = [1,2,3]
    #unit_types_tests = [2]

    task_relation_chances = [0,10,20,30]
    #task_relation_chances = [10]

    repetitions = 5

    data_name = "test/runtime_results.txt"
    for slack in slack_tests:
        for no_tasks in task_no_tests:
            for no_units in unit_no_tests:
                for no_unit_types in unit_types_tests:
                    for ration_task_rel in task_relation_chances:
                        run_times = []
                        successes = False
                        for rep in range(repetitions):
                            config_str = "Slack: " + str(slack) + ", Tasks: " + str(no_tasks) + ", Units: " + str(no_units) + ", Unit Types: " + str(no_unit_types) + " Task Relation Ratio: " + str(ration_task_rel / 10)
                            print("Next: " + config_str + "\n")
                            run_time, success = test_one(1000,slack/100,no_tasks,no_units,no_unit_types,2,2,ration_task_rel / 100)
                            
                            successes = successes or success
                            run_times.append(run_time)

                        with open("slack" + str(slack) + "_" + data_name,"a") as f:
                            f.write(config_str + " | ")
                            if successes:
                                f.write("Runtime : " + str(sum(run_times) / len(run_times)) + ", Max: " + str(max(run_times)) + ", Min: " + str(min(run_times)) + "\n")
                            else:
                                f.write("FAILURE!\n")


if __name__ == "__main__":
    main()
