import csv
import test_main

input_file = "input_csv_name.csv"
output_file = "input_latex_name.txt"

def main():
    output_tuples_avg : list[tuple[float,int]]= []
    output_tuples_min : list[tuple[float,int]]= []
    output_tuples_max : list[tuple[float,int]]= []

    with open(input_file, "r") as f:
        csv_reader = csv.DictReader(f, delimiter=";")

        for line_dict in csv_reader:
            if line_dict[test_main.TASK_SLACK_STR] == "30" and line_dict[test_main.UNIT_TYPES_STR] == "3":
                output_tuples_avg.append((line_dict[test_main.TASK_RELATION_CHANCE_STR], line_dict[test_main.RUNTIME_AVG_STR]))
                output_tuples_min.append((line_dict[test_main.TASK_RELATION_CHANCE_STR], line_dict[test_main.RUNTIME_MIN_STR]))
                output_tuples_max.append((line_dict[test_main.TASK_RELATION_CHANCE_STR], line_dict[test_main.RUNTIME_MAX_STR]))
    
    """
    with open(output_file, "a") as f:
        for index in range(len(output_tuples_min)):
            avg_value = float(output_tuples_avg[index][1])
            min_value = float(output_tuples_min[index][1])
            max_value = float(output_tuples_max[index][1])

            f.write("(" + str(output_tuples_avg[index][0]) + "," + str(avg_value) + ") +-(" + str(avg_value - min_value) + "," + str(max_value - avg_value) + ")\n")
        
        f.write("\n")

    with open(output_file, "a") as f:
        for o_tuple in output_tuples_min:
            f.write("(" + str(o_tuple[0]) + "," + str(o_tuple[1]) + ")\n")
        f.write("\n")

        for o_tuple in output_tuples_avg:
            f.write("(" + str(o_tuple[0]) + "," + str(o_tuple[1]) + ")\n")
        f.write("\n")

        for o_tuple in output_tuples_max:
            f.write("(" + str(o_tuple[0]) + "," + str(o_tuple[1]) + ")\n")
        f.write("\n")"""
    

    with open(output_file, "a") as f:
        for o_tuple in output_tuples_min:
            f.write("(" + str(o_tuple[0]) + "," + str(o_tuple[1]) + ")\n")
        f.write("\n")



if __name__ == "__main__":
    main()