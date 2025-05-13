import argparse
import sys
import os

from allinone import main

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
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--train-path', type=str, default="train.csv", help='path to csv that contains training data')
    parser.add_argument('--val-path', type=str, default='val.csv', help='path to csv that contains validation data')
    parser.add_argument('--test-path', type=str, default="test.csv", help='path to csv that contains testing data')
    parser.add_argument('--output-path', type=str, default="output", help='path to trained models')
    parser.add_argument('--log-path', type=str, default="logs/log.txt", help='path to log')
    parser.add_argument('--only-test', action='store_true', help='boolean value denoting only test will be performed or not')
    parser.add_argument('--quantized', action='store_true', help='If true will make the model quantized')
    parser.add_argument('--cusTok', action='store_true', help='If true will add custom tokenizer')
    args = parser.parse_args()

    original_stdout = sys.stdout
    sys.stdout = Tee(args.log_path)

    print(f"argumenets are {args}")

    output_path = os.path.join(args.output_path, f'{args.model}_{args.prompt}_zero{args.only_test}_cus{args.cusTok}.pth')
      
    main(
      model_type = args.model,
      prompt = args.prompt,
      epochs = args.epochs,
      batch_size = args.batch,
      lr = args.lr,
      train_path = args.train_path,
      val_path = args.val_path,
      test_path = args.test_path,
      output_path = output_path,
      only_test = args.only_test,
      quantized = args.quantized,
      cusTok = args.cusTok
    )

    sys.stdout = original_stdout
