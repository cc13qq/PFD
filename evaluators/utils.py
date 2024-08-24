# from openood.evaluators.mos_evaluator import MOSEvaluator
from utils.config import Config

from .base_evaluator import BaseEvaluator
from . SP_evaluator import SP_Evaluator

def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'SP': SP_Evaluator,
    }
    return evaluators[config.evaluator.name](config)
