# coding=utf-8

# Lint as: python3
'''The goldAMRsilverText dataset contains a colelction of human-made AMR graphs
paired with their equivalent machine translation in other languages.

The data is pulled from AMR3.0, Bio, and Little Prince corpora.

The amr NL text is generated with nllb-200-distilled-1.3B from the English sentence.'''

import datasets
import os
import itertools

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = '''The goldAMRsilverText dataset contains a colelction of human-made AMR graphs
paired with their equivalent sentences on other languages.

The data is pulled from AMR3.0, Bio, and Little Prince corpora.

The amr NL text is generated with nllb-200-distilled-1.3B from the English sentence.'''

_LANGUAGES = [
    'ast_Latn',
    'deu_Latn',
    'fra_Latn',
    'hat_Latn',
    'ita_Latn',
    'lim_Latn',
    'ltz_Latn',
    'nld_Latn',
    'scn_Latn',
    'spa_Latn',
    'tpi_Latn',
]

_DATASET_FILES = {
    'amr3':['train', 'validation', 'test'],
}

_SPLIT_NAMES = {
    'validation':datasets.Split.VALIDATION,
    'test':datasets.Split.TEST,
    'train':datasets.Split.TRAIN,
}

_FILE_PATHS = {
    dataset:{
        language:{
            split:os.path.join('data', dataset, language, f'{dataset}_{language}_{split}.tsv') for split in _DATASET_FILES[dataset]
        }
        for language in _LANGUAGES
    }
    for dataset in _DATASET_FILES.keys()
}

class GoldAMR_SilverTextConfig(datasets.BuilderConfig):
    '''BuilderConfig for  goldAMRsilverText.'''

    def __init__(self, dataset, lang, version='0.0.1', **kwargs):
        # Version history:
        # 0.0.1: Initial version.
        super().__init__(
            name=f'{dataset}.{lang}',
            version=version,
            **kwargs,
        )
        self.dataset = dataset
        self.lang = lang

class GoldAMR_SilverText(datasets.GeneratorBasedBuilder):
    ''' goldAMRsilverText dataset.'''

    VERSION = datasets.Version('0.0.1')

    BUILDER_CONFIG_CLASS = GoldAMR_SilverTextConfig
    BUILDER_CONFIGS = [
        GoldAMR_SilverTextConfig(dataset, lang)
        for dataset in _DATASET_FILES.keys()
        for lang in _LANGUAGES
    ]
    BUILDER_CONFIGS += [GoldAMR_SilverTextConfig('all', lang) for lang in _LANGUAGES]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'source': datasets.Value('string'),
                    'source_lang': datasets.Value('string'),
                    'source_quality': datasets.Value('string'),
                    'target': datasets.Value('string'),
                    'target_lang': datasets.Value('string'),
                    'target_quality': datasets.Value('string'),
                }
            )
        )
    
    def _split_generators(self, dl_manager: datasets.DownloadManager):
        '''This function generates the splits'''
        
        dataset, lang = self.config.name.split('.')
        
        if dataset != 'all':    
            lang_dirs = [dl_manager.download(_FILE_PATHS[dataset][lang])]
            
        else:
            lang_dirs = [dl_manager.download(_FILE_PATHS[dataset][lang]) for dataset in _DATASET_FILES.keys()]
            
        lang_split_files = {}
        
        for dataset in lang_dirs:
            for split, path in dataset.items():
                if split not in lang_split_files:
                    lang_split_files[split] = []
                lang_split_files[split].append(path)  
        
        return [
            datasets.SplitGenerator(
                name=_SPLIT_NAMES[split],
                gen_kwargs={'lang_paths': lang_split_files[split]},
            )
            for split in lang_split_files.keys()
        ]
        
    def _generate_examples(self, lang_paths):
        '''This function returns the examples.'''
        
        dataset, lang = self.config.name.split('.')
        
        counter = -1
        for lang_path in lang_paths:
            logger.info('generating examples from = %s', lang_path)
            with open(lang_path, 'r') as lang_file:
                for id_, line in enumerate(lang_file):
                    amr_text, lang_text = line.split('\t')
                    counter += 1
                    yield counter, {
                        'source':amr_text,
                        'source_lang':'graph_amr',
                        'source_quality':'gold',
                        'target':lang_text.strip(),
                        'target_lang':lang,
                        'target_quality':'silver',
                    }
