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

    args = parser.parse_args()
    # print(args.method_name, args.model_name, args.dataset_name)
    common.attacks(task=1, method_name=args.method_name, model_name=args.model_name, dataset_name=args.dataset_name,
                   save_path='../adv-img/', save_boxes_path='../results-img/object-detection/', epsilons=[-1])


if __name__ == '__main__':
    main()
