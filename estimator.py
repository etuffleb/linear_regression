from train import read_file


def predict(th0, th1, x):
    return th0 + (x * th1)


def validate_estimator(content):
    for line in content:
        if line.startswith("#") or line == '\n':
            continue
        try:
            th0 = float(line.split(',')[0])
            th1 = float(line.split(',')[1])
        except Exception:
            print('Wrong format of \'train_result.csv\'')
            exit()
    return th0, th1


def main():
    content = read_file("train_result.csv")
    th0, th1 = validate_estimator(content)
    mileage = float(input("\nPlease enter mileage: "))
    try:
        price = predict(th0, th1, mileage)
    except OverflowError:
        print('The number is too big')
        exit()
    print('Estimated price: {0}'.format(price))
    


if __name__ == '__main__':
    main()

