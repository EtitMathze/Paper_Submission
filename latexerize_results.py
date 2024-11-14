
class results_line:
    def __init__(self, runtime_min: float, runtime_mean : float, runtime_max : float, no_tasks : int, no_units : int, no_types : int, slack : int, relation_ratio : float):
        self.runtime_min_ = runtime_min
        self.runtime_max_ = runtime_max
        self.runtime_mean_ = runtime_mean
        self.no_tasks_ = no_tasks
        self.no_units_ = no_units
        self.no_types_ = no_types
        self.slack_ = slack
        self.relation_ratio_ = relation_ratio

def string_to_resultsline(input_string : str) -> results_line:
    split_str = input_string.split("|")
    definition_strings = split_str[0].split(",")

    slack = int(definition_strings[0].split(":")[1])
    tasks = int(definition_strings[1].split(":")[1])
    units = int(definition_strings[2].split(":")[1])
    inbetween_split = definition_strings[3].split(":")
    splitted_at_space = inbetween_split[1].split(" ")
    types = int(splitted_at_space[1])
    relation_ratio = float(inbetween_split[2])

    if "FAILURE" in split_str:
        return results_line(20000,20000,20000,tasks,units,types,slack,relation_ratio)

    runtime_strings = split_str[1].split(",")
    runtime_mean = float(runtime_strings[0].split(":")[1])
    runtime_max = float(runtime_strings[1].split(":")[1])
    runtime_min = float(runtime_strings[2].split(":")[1])

    new_results_line = results_line(runtime_min,runtime_mean,runtime_max,tasks,units,types,slack,relation_ratio)

    return new_results_line

def turn_group_into_latex_str(input_results : list[results_line]) -> str:
    exec_units_4_list : list[results_line] = []
    exec_units_5_list : list[results_line] = []
    exec_units_6_list : list[results_line]= []

    result_string = ""

    for result in input_results:
        if result.no_units_ == 4:
            exec_units_4_list.append(result)
        elif result.no_units_ == 5:
            exec_units_5_list.append(result)
        elif result.no_units_ == 6:
            exec_units_6_list.append(result)

    result_string += "Results for: Types: " + str(result.no_types_) + ", relation ratio: " + str(result.relation_ratio_) + "\n"
    result_string += "4 exec units\n"
    for current_result in exec_units_4_list:
        result_string += "(" + str(float(current_result.no_tasks_) - 0.15) + "," + str(current_result.runtime_mean_) + ") "
        result_string += " +- (" + str(current_result.runtime_mean_ - current_result.runtime_min_) + "," + str(current_result.runtime_max_ - current_result.runtime_mean_) + ")\n"
    result_string += "\n"

    result_string += "5 exec units\n"
    for current_result in exec_units_5_list:
        result_string += "(" + str(float(current_result.no_tasks_)) + "," + str(current_result.runtime_mean_) + ") "
        result_string += " +- (" + str(current_result.runtime_mean_ - current_result.runtime_min_) + "," + str(current_result.runtime_max_ - current_result.runtime_mean_) + ")\n"
    result_string += "\n"

    result_string += "6 exec units\n"
    for current_result in exec_units_6_list:
        result_string += "(" + str(float(current_result.no_tasks_) + 0.15) + "," + str(current_result.runtime_mean_) + ") "
        result_string += " +- (" + str(current_result.runtime_mean_ - current_result.runtime_min_) + "," + str(current_result.runtime_max_ - current_result.runtime_mean_) + ")\n"
    result_string += "\n"
    
    return result_string
    

def turn_results_into_latex_string(input_results : list[results_line]):
    relation_30_list = []
    relation_20_list = []
    relation_10_list = []
    relation_0_list = []


    for line in input_results:
        # Choose one number of execution unit types
        if line.no_types_ == 2:

            # Sort the lines into respective groups
            if line.relation_ratio_ == 3.0:
                relation_30_list.append(line)
            elif line.relation_ratio_ == 2.0:
                relation_20_list.append(line)
            elif line.relation_ratio_ == 1.0:
                relation_10_list.append(line)
            elif line.relation_ratio_ == 0.0:
                relation_0_list.append(line)

    all_string = "2 Unit types:\n"
    all_string += "Relation 30:\n"
    all_string += turn_group_into_latex_str(relation_30_list)
    all_string += "Relation 20:\n"
    all_string += turn_group_into_latex_str(relation_20_list)
    all_string += "Relation 10:\n"
    all_string += turn_group_into_latex_str(relation_10_list)
    all_string += "Relation 0:\n"
    all_string += turn_group_into_latex_str(relation_0_list)

    relation_30_list = []
    relation_20_list = []
    relation_10_list = []
    relation_0_list = []

    for line in input_results:
        # Choose one number of execution unit types
        if line.no_types_ == 3:

            # Sort the lines into respective groups
            if line.relation_ratio_ == 3.0:
                relation_30_list.append(line)
            elif line.relation_ratio_ == 2.0:
                relation_20_list.append(line)
            elif line.relation_ratio_ == 1.0:
                relation_10_list.append(line)
            elif line.relation_ratio_ == 0.0:
                relation_0_list.append(line)

    all_string += "3 Unit types:\n"
    all_string += "Relation 30:\n"
    all_string += turn_group_into_latex_str(relation_30_list)
    all_string += "Relation 20:\n"
    all_string += turn_group_into_latex_str(relation_20_list)
    all_string += "Relation 10:\n"
    all_string += turn_group_into_latex_str(relation_10_list)
    all_string += "Relation 0:\n"
    all_string += turn_group_into_latex_str(relation_0_list)

    return all_string


def main():
    in_filename = "test/slack50_runtime_results.txt"
    out_filename = "latex/slack50_runtime_results_latex.txt"
    result_list = []

    with open(in_filename,"r") as f:
        for line in f:
            result_list.append(string_to_resultsline(line))

    out_string = turn_results_into_latex_string(result_list)
    with open(out_filename, "w") as f:
        f.write(out_string)

if __name__ == "__main__":
    main()