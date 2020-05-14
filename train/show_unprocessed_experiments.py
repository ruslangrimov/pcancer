import os
import argparse


def main(exp_subfolder, only_unprocessed):
    experiments = os.listdir(os.path.join('experiments_configs',
                                          exp_subfolder))
    experiments = ['.'.join(e.split('.')[:-1]) for e in experiments
                   if not e.startswith('_')]

    for experiment in experiments:
        result_path = os.path.join('experiments_results', exp_subfolder,
                                   experiment)
        processed = os.path.isdir(result_path)
        if only_unprocessed:
            if not processed:
                print(f"Experiment {experiment}")
        else:
            print(f"Experiment {experiment}")
            if processed:
                print(f"Processed")
            else:
                print(f"Unprocessed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment.')

    parser.add_argument('-p', dest='path', type=str,
                        help='experiments subfolder', required=True)

    parser.add_argument('-u', action='store_true')

    args = parser.parse_args()

    main(args.path, args.u)
