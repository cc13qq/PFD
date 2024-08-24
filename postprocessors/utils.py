from utils.config import Config

from .base_postprocessor import BasePostprocessor
from .SP_postprocessor import SP_Postprocessor

def get_postprocessor(config: Config):
    postprocessors = {
        'msp': BasePostprocessor,
        'SP': SP_Postprocessor
    }

    return postprocessors[config.postprocessor.name](config)
