import csv
import test_main

input_file = "input_csv_name.csv"
output_file = "input_latex_name.txt"

def filter_func(line: dict[str,str]) -> bool:
    rep = int(line[test_main.REP_NR_STR])
    unit_types = int(line[test_main.UNIT_TYPES_STR])
    no_tasks = int(line[test_main.TASK_NO_STR])
    slack = int(line[test_main.TASK_SLACK_STR])

    return unit_types == 2 and no_tasks == 20 and slack == 30 and rep == 1


def main():
    output_tuples : list[tuple[float,int]]= []
    with open(input_file, "r") as f:
        csv_reader = csv.DictReader(f, delimiter=";")

        for line_dict in csv_reader:
            if filter_func(line_dict):
                this_tuple = (line_dict[test_main.STEP_STR], line_dict[test_main.COST_STR])
                output_tuples.append(this_tuple)

    with open(output_file, "a") as f:
        for o_tuple in output_tuples:
            f.write("(" + str(o_tuple[0]) + "," + str(o_tuple[1]) + ")\n")
        f.write("\n")

if __name__ == "__main__":
    main()