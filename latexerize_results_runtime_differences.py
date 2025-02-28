import csv
import test_main

input_file_1 = "input_csv_name_1.csv"
input_file_2 = "input_csv_name_2.csv"
output_file = "input_latex_name.txt"

def main():
    output_tuples_avg_true : list[tuple[float,int]]= []
    output_tuples_min_true : list[tuple[float,int]]= []
    output_tuples_max_true : list[tuple[float,int]]= []

    output_tuples_avg_false : list[tuple[float,int]]= []
    output_tuples_min_false : list[tuple[float,int]]= []
    output_tuples_max_false : list[tuple[float,int]]= []

    differences : list[float] = []

    with open(input_file_1, "r") as f:
        csv_reader = csv.DictReader(f, delimiter=";")

        for line_dict in csv_reader:
            if line_dict[test_main.UNIT_TYPES_STR] == "3" and line_dict[test_main.TASK_SLACK_STR] == "30":
                output_tuples_avg_false.append((line_dict[test_main.TASK_RELATION_CHANCE_STR], line_dict[test_main.RUNTIME_AVG_STR]))
                output_tuples_min_false.append((line_dict[test_main.TASK_RELATION_CHANCE_STR], line_dict[test_main.RUNTIME_MIN_STR]))
                output_tuples_max_false.append((line_dict[test_main.TASK_RELATION_CHANCE_STR], line_dict[test_main.RUNTIME_MAX_STR]))


    with open(input_file_2, "r") as f:
        csv_reader = csv.DictReader(f, delimiter=";")
        for line_dict in csv_reader:
            if line_dict[test_main.UNIT_TYPES_STR] == "3" and line_dict[test_main.TASK_SLACK_STR] == "30":
                output_tuples_avg_true.append((line_dict[test_main.TASK_RELATION_CHANCE_STR], line_dict[test_main.RUNTIME_AVG_STR]))
                output_tuples_min_true.append((line_dict[test_main.TASK_RELATION_CHANCE_STR], line_dict[test_main.RUNTIME_MIN_STR]))
                output_tuples_max_true.append((line_dict[test_main.TASK_RELATION_CHANCE_STR], line_dict[test_main.RUNTIME_MAX_STR]))


       
    with open(output_file, "a") as f:
        for index in range(min(len(output_tuples_min_true),len(output_tuples_min_false))):
            avg_value = float(output_tuples_avg_true[index][1]) - float(output_tuples_avg_false[index][1])
            min_value = abs(float(output_tuples_min_true[index][1]) - float(output_tuples_min_false[index][1]))
            max_value = abs(float(output_tuples_max_true[index][1]) - float(output_tuples_max_false[index][1]))

            f.write("(" + str(output_tuples_avg_true[index][0]) + "," + str(avg_value) + ") +-(" + str(avg_value - min_value) + "," + str(max_value - avg_value) + ")\n")
        
        f.write("\n")
    
       
    """for index in range(min(len(output_tuples_min_true),len(output_tuples_min_false))):
        avg_value = float(output_tuples_avg_true[index][1]) - float(output_tuples_avg_false[index][1])
        min_value = abs(float(output_tuples_min_true[index][1]) - float(output_tuples_min_false[index][1]))
        max_value = abs(float(output_tuples_max_true[index][1]) - float(output_tuples_max_false[index][1]))

        differences.append(avg_value)

    print(sum(differences) / len(differences))"""
    

    
    """
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
    

if __name__ == "__main__":
    main()