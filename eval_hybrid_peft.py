import sys

from src import parse_eval_args, MLPipeline

args = parse_eval_args(sys.argv[1:])

eval_pipeline = MLPipeline.for_evaluation(args)
eval_pipeline.run()
