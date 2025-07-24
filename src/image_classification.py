import common
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method_name",
                        type=str,
                        default=None,
                        help="Which method to use.")
    parser.add_argument("--model_name",
                        type=str,
                        default=None,
                        help="Which model to attack.")
    parser.add_argument("--dataset_name",
                        type=str,
                        default=None,
                        help="Which dataset to attack.")
    parser.add_argument("--epsilon",
                        type=float,
                        default=-1,
                        help="What epsilon to use.")

    args = parser.parse_args()
    epsilons = [-1]
    if args.epsilon != -1:
        epsilons = [args.epsilon]
    # print(args.method_name, args.model_name, args.dataset_name, type(epsilons))
    common.attacks(task=0, method_name=args.method_name, model_name=args.model_name, dataset_name=args.dataset_name,
                   save_path='../adv-img/', epsilons=epsilons)


if __name__ == '__main__':
    main()

