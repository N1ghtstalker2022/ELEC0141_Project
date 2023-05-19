# Open the file for reading
with open("/scratch/zczqyc4/ELEC0141-DataSet/english_subset.txt", "r") as file:
    # Read the first ten lines
    for i in range(10):
        line = file.readline()
        if not line:
            break
        print(line.strip())  # Print each line without trailing spaces

# Open the file for reading
with open("/scratch/zczqyc4/ELEC0141-DataSet/french_subset.txt", "r") as file:
    # Read the first ten lines
    for i in range(10):
        line = file.readline()
        if not line:
            break
        print(line.strip())  # Print each line without trailing spaces
