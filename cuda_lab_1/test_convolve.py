import subprocess
import os

import numpy as np

# little hack: http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
class bcolors:
    HEADER      = '\033[95m'
    OKBLUE      = '\033[94m'
    OKGREEN     = '\033[92m'
    WARNING     = '\033[93m'
    FAIL        = '\033[91m'
    ENDC        = '\033[0m'
    BOLD        = '\033[1m'
    UNDERLINE   = '\033[4m'


BIN_DIR     = 'bin/'
BIN_NAME    = 'lab_1'
BIN_PATH    = os.path.join(BIN_DIR, BIN_NAME)

INPUT_FNAME  = "input.txt"
OUTPUT_FNAME = "output.txt"

sequences = [
            (np.ones(shape=(1024, 1024)), np.ones(shape=(3, 3)), 'Ones, N=1024, M=3'),
            (np.ones(shape=(1024, 1024)), np.ones(shape=(9, 9)), 'Ones, N=1024, M=9'),
            (np.ones(shape=(1, 1)),       np.ones(shape=(9, 9)), 'Ones, N=1,    M=9'),
            (np.ones(shape=(31, 31)),     np.ones(shape=(9, 9)), 'Ones, N=31,   M=9'),
            (np.ones(shape=(1023, 1023)), np.ones(shape=(9, 9)), 'Ones, N=1023, M=9'),

            (np.random.normal(size=(16, 16), loc=1., scale=0.1), np.random.normal(size=(4, 4)), 'Random normal, N=16, M=4'),
            (np.random.normal(size=(256, 256), loc=5., scale=1.), np.random.normal(size=(4, 4)), 'Random normal, N=256, M=4'),
            (np.random.normal(size=(512, 512), loc=10., scale=2.5), np.random.normal(size=(9, 9)), 'Random normal, N=512, M=9'),
            (np.random.normal(size=(1024, 1024), loc=25, scale=5.), np.random.normal(size=(16, 16)), 'Random normal, N=1024, M=16'),
            ]


def write_data(A, B, fname):
    with open(fname, 'w') as out_file:
        out_file.write('{} {}\n'.format(A.shape[1], B.shape[1]))

        for row in A:
            out_file.write(' '.join([str(round(v, 3)) for v in row]) + '\n')

        for row in B:
            out_file.write(' '.join([str(round(v, 3)) for v in row]) + '\n')


def read_data(fname):
    res = list()

    with open(fname) as in_file:
        for line in in_file:
            nums = list(map(float, line.split()))
            res.append(nums)

    return np.array(res)

def run_test():
    for A, B, test_name in sequences:
        write_data(A, B, INPUT_FNAME)

        popen = subprocess.Popen([BIN_PATH], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        output, err = popen.communicate()
        output = output.strip()

        answer = read_data(OUTPUT_FNAME)

        print('Test name: {} {} {}'.format(bcolors.OKBLUE, test_name, bcolors.ENDC))
        if output == b'Ok!':
            print('Answer: {}*** CORRECT ***{}'.format(bcolors.OKGREEN, bcolors.ENDC))
        else:
            print('Answer: {}*** CORRECT ***{}'.format(bcolors.FAIL, bcolors.ENDC))

        print('-'*20)


if __name__ == '__main__':
    run_test()