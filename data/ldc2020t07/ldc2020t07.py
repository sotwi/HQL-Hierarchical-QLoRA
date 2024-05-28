# coding=utf-8

# Lint as: python3
'''ldc2020t07'''

import datasets
from os.path import basename, normpath

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = '''ldc2020t07'''

_FILE_PATHS = {
    'graph_amr':{
        'test':'data/graph_amr/graph_amr-test.txt',
    },
    'deu_Latn':{
        'test':'data/deu_Latn/deu_Latn-test.txt',
    },
    'eng_Latn':{
        'test':'data/eng_Latn/eng_Latn-test.txt',
    },
    'spa_Latn':{
        'test':'data/spa_Latn/spa_Latn-test.txt',
    },
    'ita_Latn':{
        'test':'data/ita_Latn/ita_Latn-test.txt',
    }
}

class ldc2020t07Config(datasets.BuilderConfig):
    '''BuilderConfig for ldc2020t07'''

    def __init__(self, language, **kwargs):
        # Version history:
        # 0.0.1: Initial version.
        super().__init__(
            name=language,
            version='0.0.1',
            **kwargs,
        )
        self.language = language

class ldc2020t07(datasets.GeneratorBasedBuilder):
    '''Sldc2020t07.'''

    VERSION = datasets.Version('0.0.1')

    BUILDER_CONFIG_CLASS = ldc2020t07Config
    BUILDER_CONFIGS = [
        ldc2020t07Config(
            language='deu_Latn'
        ),
        ldc2020t07Config(
            language='eng_Latn'
        ),
        ldc2020t07Config(
            language='spa_Latn'
        ),
        ldc2020t07Config(
            language='ita_Latn'
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'lang': datasets.Value('string'),
                    'text': datasets.Value('string'),
                    'graph': datasets.Value('string'),
                }
            )
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        '''This function generates the splits'''

        graph_dir = dl_manager.download(_FILE_PATHS['graph_amr'])
        data_dir = dl_manager.download(_FILE_PATHS[self.config.name])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={'filepath': data_dir['test'], 'graphpath': graph_dir['test']},
            )
        ]

    def _generate_examples(self, filepath, graphpath):
        '''This function returns the examples.'''
        logger.info('generating examples from = %s', filepath)
        lang, split = basename(normpath(filepath)).split('-')
        with open(filepath, 'r') as file1:
            with open(graphpath, 'r') as file2:
                for id_, (text, graph) in enumerate(zip(file1, file2)):
                    yield id_, {
                        'lang':lang,
                        'text': text.strip(),
                        'graph': graph.strip(),
                    }
