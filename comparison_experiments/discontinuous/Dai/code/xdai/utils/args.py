import argparse, logging


logger = logging.getLogger(__name__)


'''Update date: 2019-Nov-5'''
def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    ## Data
    parser.add_argument("--train_filepath", default=r'/home/administrator/PycharmProjects/Expressway_extract/controlled_experiment/acl2020-transition-discontinuous-ner-master/data/CADEC/train.txt', type=str)
    parser.add_argument("--num_train_instances", default=None, type=int)
    parser.add_argument("--dev_filepath", default=r'/home/administrator/PycharmProjects/Expressway_extract/controlled_experiment/acl2020-transition-discontinuous-ner-master/data/CADEC/dev.txt', type=str)
    parser.add_argument("--num_dev_instances", default=None, type=int)
    parser.add_argument("--test_filepath", default=r'/home/administrator/PycharmProjects/Expressway_extract/controlled_experiment/acl2020-transition-discontinuous-ner-master/data/CADEC/test.txt', type=str)
    parser.add_argument("--cache_dir", default=r'/home/administrator/PycharmProjects/Expressway_extract/controlled_experiment/acl2020-transition-discontinuous-ner-master/cahe', type=str)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--output_dir", default=r'/home/administrator/PycharmProjects/Expressway_extract/controlled_experiment/acl2020-transition-discontinuous-ner-master/output', type=str)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--log_filepath", default=r'/home/administrator/PycharmProjects/Expressway_extract/controlled_experiment/acl2020-transition-discontinuous-ner-master/log/log.txt', type=str)
    parser.add_argument("--summary_json", default=None, type=str)
    parser.add_argument("--encoding", default="utf-8", type=str)

    ## Train
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--train_batch_size_per_gpu", default=8, type=int)
    parser.add_argument("--num_train_epochs", default=20, type=int)
    parser.add_argument("--max_steps", default=5, type=int, help="If > 0, override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--logging_steps", default=50, type=int)
    parser.add_argument("--save_steps", default=0, type=int)
    parser.add_argument("--max_save_checkpoints", default=0, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--max_grad_norm", default=None, type=float)
    parser.add_argument("--grad_clipping", default=5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--weight_decay", default=0.0001, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of update steps to accumulate before performing a backward/update pass")
    parser.add_argument("--seed", default=52, type=int)
    parser.add_argument("--cuda_device", default="0", type=str, help="a list cuda devices, splitted by ,")

    ## Evaluation
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--eval_batch_size_per_gpu", default=8, type=int)
    parser.add_argument("--eval_metric", default='f1-overall', type=str)
    parser.add_argument("--eval_all_checkpoints", action="store_true", help="Evaluate all checkpoints.")
    parser.add_argument("--eval_during_training", action="store_true",
                        help="Evaluate during training at each save step.")

    ## Model
    parser.add_argument("--model_type", default=None, type=str)
    parser.add_argument("--pretrained_model_dir", default=None, type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--labels", default="0,1", type=str)
    parser.add_argument("--label_filepath", default=None, type=str)
    parser.add_argument("--tag_schema", default="B,I")
    parser.add_argument("--do_lower_case", action="store_true")

    args, _ = parser.parse_known_args()

    return args