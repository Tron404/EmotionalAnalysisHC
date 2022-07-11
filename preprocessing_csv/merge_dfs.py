import pandas as pd
import time

path= "C:\\Users\\Eduard\\Desktop\\project_preprocess\\Processed\\"

file_name = "COVID_FAKES_trimmed"

fileCSV = pd.read_csv(path + f"{file_name}_{1}.csv", sep=",", index_col=False, skipinitialspace=True, skip_blank_lines=True, dtype=str)

for i in range(2,32):
    aux = pd.read_csv(path + f"{file_name}_{i}.csv", sep=",", index_col=False, skipinitialspace=True, skip_blank_lines=True, dtype=str)
    fileCSV = pd.concat([fileCSV, aux], ignore_index=True)
    print(aux.head())
    print(f"Merged with file {i}")
    print("--------------------------------------------------------------------------------------")

fileCSV.to_csv(f"Processed\\{file_name}_processed.csv", index=False)
