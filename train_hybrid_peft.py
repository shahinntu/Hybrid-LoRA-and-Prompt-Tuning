import sys

from src import parse_train_args, MLPipeline

args = parse_train_args(sys.argv[1:])

train_pipeline = MLPipeline.for_training(args)
train_pipeline.run()
