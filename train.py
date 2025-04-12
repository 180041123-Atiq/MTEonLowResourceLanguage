import argparse
import sys

from data import generateDataLoaders
from models.llama2 import tokenizer, train, evaluate

class Tee:
    def __init__(self, filename, mode="w"):
        self.file = open(filename, mode, encoding="utf-8")
        self.stdout = sys.stdout  # original terminal output

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)  # also write to terminal

    def flush(self):
        self.file.flush()
        self.stdout.flush()

if __name__ == '__main__':

    # print('Initializing Argument Parser')
    parser = argparse.ArgumentParser(description="Argument For Machine Translation Evaluation Tasks")

    parser.add_argument('--model', type=str, required=True, help='name of the model to use')
    parser.add_argument('--prompt', type=str, required=True, help='type of prompting')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--batch', type=int, default=2, help='batch size')
    parser.add_argument('--train-path', type=str, default="train_comet_da_scaled.csv", help='path to csv that contains training data')
    parser.add_argument('--test-path', type=str, default="test_comet_da_scaled.csv", help='path to csv that contains testing data')
    parser.add_argument('--log-path', type=str, default="logs/log.txt", help='path to log')

    args = parser.parse_args()

    original_stdout = sys.stdout
    sys.stdout = Tee(args.log_path)

    print(f"argumenets are {args}")

    train_loader,test_loader = generateDataLoaders(
        train_data_path=args.train_path,
        test_data_path=args.test_path,
        tokenizer=tokenizer,
        type=args.prompt
    )

    if args.model == 'llama2':
        train(train_loader=train_loader, epochs=args.epochs)
        evaluate(test_loader=test_loader)

    sys.stdout = original_stdout