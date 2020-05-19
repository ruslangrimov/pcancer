import sys
import os
import time
import argparse
from subprocess import Popen


def main(exp_subfolder, gpu):
    while True:
        experiments = os.listdir(os.path.join('experiments_configs',
                                              exp_subfolder))
        experiments = ['.'.join(e.split('.')[:-1]) for e in experiments
                       if not e.startswith('_')]

        for experiment in experiments:
            print(f"Experiment {experiment}")
            result_path = os.path.join('experiments_results', exp_subfolder,
                                       experiment)
            if os.path.isdir(result_path):
                print(f"Experiment is processed already or in process")
            else:
                print(f"Launching experimen {experiment} on gpu {gpu}")
                p = Popen([sys.executable, "-u",
                           "train/train_wsi_on_1x1_pretrained.py",
                           "-g", str(gpu),
                           "-e", f"{exp_subfolder}.{experiment}"],
                          # stdout=subprocess.PIPE,
                          # stderr=subprocess.STDOUT,
                          shell=False, env={"PYTHONPATH": "./"})
                p.wait()
                break

        time.sleep(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment.')

    parser.add_argument('-p', dest='path', type=str,
                        help='experiments subfolder', required=True)

    parser.add_argument('-g', dest='gpu', type=int,
                        help='gpu number', default=0)

    args = parser.parse_args()

    main(args.path, args.gpu)
