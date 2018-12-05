def import_raw_data(filename):
    data = []
    with open(filename, "r") as f:
        line = f.readline().strip()
        while line != "":
            row = line.split("\t")
            if row[2] != "NA":
                row[2] = float(row[2])
            if row[3] == "on-target":
                row[3] = 1
            else:
                row[3] = 0
            data.append(row)
            line = f.readline().strip()
    return data

def import_formatted_data(filename):
    data = []
    with open(filename, "r") as f:
        line = f.readline().strip()
        while line != "":
            row = line.split("\t")
            if row[2] == "NA":
                line = f.readline().strip()
                continue
            row[2] = float(row[2])
            data.append(row)
            line = f.readline().strip()
    return data

def write_data(data, filename):
    with open(filename, "w") as f:
        for row in data:
            row = [str(x) for x in row]
            f.write("\t".join(row) + "\n")
