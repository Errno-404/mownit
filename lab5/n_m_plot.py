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


def select(data, x):
    sl = []
    for row in data:
        if int(row[1]) == x:
            sl.append(row)

    return sl


def draw(x, y):
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":

    s = select(get_data(), 55)
    srt = sort_by(s, 0)

    x, y = [], []
    for row in srt:
        x.append(int(row[0]))
        y.append(float(row[2]))
    draw(x, y)
