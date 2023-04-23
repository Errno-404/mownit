from matplotlib import pyplot as plt
import csv

def get_data():
    l = []
    with open("data/errors.csv") as file:
        reader_obj = csv.reader(file)
        for row in reader_obj:
            if row[0] != 'm':

                l.append(row)
            # print(row)
    return l

def sort_by(data, idx):
    data.sort(key=lambda x: int(x[idx]))
    return data


def plot():
    srt = sort_by(get_data(), 0)
    y = []




if __name__ == "__main__":
    get_data()
    srt = sort_by(get_data(), 0)
    for row in srt:
        print(row)

    plot()