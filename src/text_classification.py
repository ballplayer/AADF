import os
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def text_attack(dataset_name, model_name, save_path):
    save_path = save_path + dataset_name + '/' + model_name

    if dataset_name == 'ag':
        nclasses = 4
    if model_name == 'wordLSTM':
        command = 'python attack_text.py ' \
                  '--dataset_path ../data/' + dataset_name + ' --nclasses ' + str(nclasses) + ' ' \
                  '--target_model wordLSTM --batch_size 128 ' \
                  '--target_model_path ../model-weights/text/wordLSTM_' + dataset_name + ' ' \
                  '--word_embeddings_path ./TextFooler/embeddings/glove.6B/glove.6B.200d.txt ' \
                  '--counter_fitting_embeddings_path ./TextFooler/embeddings/counter-fitted-vectors.txt ' \
                  '--counter_fitting_cos_sim_path ./TextFooler/cos_sim_counter_fitting.npy ' \
                  '--USE_cache_path ./tf_cache ' \
                  '--output_dir ' + save_path + ' '
    elif model_name == 'wordCNN':
        command = 'python attack_text.py ' \
                  '--dataset_path ../data/' + dataset_name + ' --nclasses ' + str(nclasses) + ' ' \
                  '--target_model wordCNN --batch_size 1 ' \
                  '--target_model_path ../model-weights/text/wordCNN_' + dataset_name + ' ' \
                  '--word_embeddings_path ./TextFooler/embeddings/glove.6B/glove.6B.200d.txt ' \
                  '--counter_fitting_embeddings_path ./TextFooler/embeddings/counter-fitted-vectors.txt ' \
                  '--counter_fitting_cos_sim_path ./TextFooler/cos_sim_counter_fitting.npy ' \
                  '--USE_cache_path ./tf_cache ' \
                  '--output_dir ' + save_path + ' '
    else:
        command = 'python attack_text.py ' \
                  '--dataset_path ../data/' + dataset_name + ' --nclasses ' + str(nclasses) + ' ' \
                  '--target_model bert ' \
                  '--target_model_path ../model-weights/text/bert_' + dataset_name + ' ' \
                  '--max_seq_length 256 --batch_size 128 ' \
                  '--counter_fitting_embeddings_path ./TextFooler/embeddings/counter-fitted-vectors.txt ' \
                  '--counter_fitting_cos_sim_path ./TextFooler/cos_sim_counter_fitting.npy ' \
                  '--USE_cache_path ./tf_cache ' \
                  '--output_dir ' + save_path + ' '
    os.system(command)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method_name",
                        type=str,
                        default="TextFooler",
                        help="Which method to use.")
    parser.add_argument("--model_name",
                        type=str,
                        default=None,
                        help="Which model to attack.")
    parser.add_argument("--dataset_name",
                        type=str,
                        default=None,
                        help="What epsilon to use.")

    args = parser.parse_args()
    # print(args.method_name, args.model_name, args.dataset_name)
    text_attack(args.dataset_name, args.model_name, '../adv-text/')


if __name__ == '__main__':
    main()
