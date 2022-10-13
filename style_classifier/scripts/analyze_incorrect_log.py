def read_log(filename='incorrect_log.txt'):
    with open(filename) as f:
        lines = f.readlines()

    lines = lines[1:-4]
    lines = list(filter(lambda a: a.strip() != '', lines))
    return lines

def analyze_log():
    lines = read_log()
    longer = 0

    for arg1, arg2, label in zip(lines[0::3], lines[1::3], lines[2::3]):
        arg1_len = len(arg1.replace('Argument 1: ', ''))
        arg2_len = len(arg2.replace('Argument 2: ', ''))
        pred = label.split(',')[0][-1]
        actual = label.split(',')[1][-2]


        if (arg1_len > arg2_len and int(pred) == 1) or (arg2_len > arg1_len and int(pred) == 2):
            longer += 1
            
        # print(arg1_len, arg2_len, label, '\n')
    print(longer/(len(lines)/3))

analyze_log()