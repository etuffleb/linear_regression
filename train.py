
import os
import argparse
import matplotlib.pyplot as plt
from math import sqrt


class LinearRegresion:
    def __init__(self, xs, ys, loss_type):
        self.xs = xs
        self.ys = ys
        self.th0 = 0.0
        self.th1 = 0.0
        self.loss_type = loss_type
        self.ers = [] 
        self.th0s = []
        self.th1s = []
        

    def loss(self):
        if not self.loss_type == "loss":
            return self.distance_loss()
        loss = 0.0
        for i in range(len(self.ys)):
            loss += self.ys[i] - (self.th1 * self.xs[i] + self.th0)
        return loss / len(self.ys)


    def distance_loss(self):
        loss = 0.0
        t0 = self.th0
        t1 = self.th1                
        for i in range(len(self.ys)):
            x0 = self.xs[i]
            y0 = self.ys[i]
            diff_x = (x0 + t1*y0 - t1*t0) / (t1*t1 + 1) - x0
            diff_y = t1 * (x0 + t1*y0 - t1*t0) / (t1*t1 + 1) + t0 - y0
            distance = diff_x * diff_x + diff_y * diff_y
            if self.loss_type == "squared":
                distance = sqrt(distance)
            loss += distance            
        return loss / len(self.ys)


    def fit(self, lr):
        x = self.xs
        y = self.ys
        self.xs = [(i - min(self.xs)) / (max(self.xs) - min(self.xs)) for i in self.xs]
        self.ys = [(i - min(self.ys)) / (max(self.ys) - min(self.ys)) for i in self.ys]
        m = len(self.xs)
        sum_dif = 0
        iter_limit = 500000
        eps = 0.00001
        iter = 0
        while iter < 2 or abs(self.ers[-1] - self.ers[-2]) > eps:
            tmp_th0 = 0
            tmp_th1 = 0
            for i in range(len(self.ys)):
                predict_y = self.th0 + self.th1 * self.xs[i]
                dif = predict_y - self.ys[i]
                tmp_th0 += dif
                tmp_th1 += dif * self.xs[i]
                sum_dif += abs(dif)
            self.th0 -= lr * tmp_th0 / m
            self.th1 -= lr * tmp_th1 / m
            self.ers.append(self.loss())
            self.th0s.append(self.th0)
            self.th1s.append(self.th1)            
            iter += 1
            if (iter == iter_limit):
                print(str(iter_limit) + " iterations have passed, and the model hasn't yet learned. \nI advise you to change the input parameters or the iteration limit")
                exit()      

        self.th0 = self.th0 * (max(y) - min(y)) + min(y) + (self.th1 * min(x) * (min(y) - max(y))) / (max(x) - min(x))
        self.th1 = self.th1 * (max(y) - min(y)) / (max(x) - min(x))

        print("Model trained in " + str(len(self.ers)) + " iterations")
        print("th0 = " + str(self.th0))
        print("th1 = " + str(self.th1))


    def visualize(self, xs, ys):
        plt.figure(1)
        plt.scatter(xs, ys)
        x_min = min(xs)
        x_max = max(xs)
        plt.plot([x_min, x_max], [self.predict(x_min), self.predict(x_max)] , color = 'red')
        plt.xlabel('mileage')
        plt.ylabel('price')
        plt.figure(2)
        plt.plot(self.ers, color = 'red')
        plt.xlabel('iterations')
        plt.ylabel('loss: ' + self.loss_type)
        plt.figure(3)
        plt.plot(self.th0s, color = 'green')
        plt.xlabel('iterations')
        plt.ylabel('th0')
        plt.figure(4)
        plt.plot(self.th1s, color = 'black')
        plt.xlabel('iterations')
        plt.ylabel('th1')
        plt.show()


    def predict(self, x):
        return self.th0 + (x * self.th1)


    def write_koefs(self, file):
        with open(file, 'w') as f:
            f.write(str(self.th0) + ',' + str(self.th1))


def read_file(file):
    try:
        with open(file, 'r') as f:
            content = f.readlines()
    except IOError:
        print("File not found")
        exit()
    except Exception:
        print("Cannot read from file")
        exit()
    return content


def parse_and_validate_data(file):
    content = read_file(file)
    xs = []
    ys = []
    header_exist = False
    for line in content:
        if line.startswith("#") or line == '\n':
            continue
        line = line.split('#')[0].strip(' \t\n\r')
        split_line = line.split(',')
        if len(split_line) != 2:
            print("There are only two parametrs should be in line")
            exit()

        x = split_line[0].strip(' \t\n\r')
        y = split_line[1].strip(' \t\n\r')

        if x.isalpha() and y.isalpha():
            if not header_exist:
                header_exist = True
                continue;
            else:
                print("There are only one header should exist")
                exit()
        elif x.isdigit() and y.isdigit():
            xs.append(int(x))
            ys.append(int(y))
        else:
            print("A number and a string cannot be on the same line")
            exit()
    if not header_exist:
        print("Header should exist")
        exit()
    return xs, ys


def parse_args(parser):
    parser.add_argument('-f', dest="file_name", help='file name', required=True)
    parser.add_argument('-v', dest="visualize", help='visualize', action='store_true')
    parser.add_argument('-d', dest="distance_loss", help='distance loss', action='store_true')
    parser.add_argument('-s', dest="sqared_distance_loss", help='sqared distance loss', action='store_true')
    return parser.parse_args()


def main():
    parser = argparse.ArgumentParser(add_help=True, conflict_handler='resolve')
    args = parse_args(parser)
    xs, ys = parse_and_validate_data(args.file_name)
    loss_type = "distance" if args.distance_loss else "squared" if args.sqared_distance_loss else "loss"
    linear_regresion = LinearRegresion(xs, ys, "distance")
    learning_rate = 0.2
    print()
    linear_regresion.fit(learning_rate)
    linear_regresion.write_koefs("train_result.csv")
    if (args.visualize):
        linear_regresion.visualize(xs, ys)
    




if __name__ == '__main__':
    main()
    