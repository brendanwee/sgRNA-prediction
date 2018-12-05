from io_med import import_data


#name	seq	score	type

def check_data():
    filename = "raw_data.tab"
    data = import_data(filename)
    name = data[0][0]
    i = 0
    p = 0
    while data[i][0] == name:
        p += float(data[i][2])
        i += 1
    print p

check_data()