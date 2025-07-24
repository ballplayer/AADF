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
                        required=True,
                        help="What epsilon to use.")

    args = parser.parse_args()
    epsilons = [args.epsilon]
    # print(args.method_name, args.model_name, args.dataset_name, epsilons)
    common.attacks(task=2, method_name=args.method_name, model_name=args.model_name, dataset_name=args.dataset_name,
                   save_path='../adv-img/', save_boxes_path='../results-img/semantic-segmentation/', epsilons=epsilons)


if __name__ == '__main__':
    main()
