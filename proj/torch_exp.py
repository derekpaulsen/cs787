from utils import  get_index_name,  Timer, get_logger, read_gold, write_parquet, stream_write_dataframe_to_parquet
from utils import   init_spark, Labeler, persisted
from es.index import IndexConfig, LuceneIndex
from es.query_generator import QuerySpec
from experiment import THRESHOLD_VAL, RECALL_AT_THRESHOLD, CSSR_AT_THRESHOLD
from experiment import RAW_THRESHOLD_VAL, RECALL_AT_RAW_THRESHOLD, CSSR_AT_RAW_THRESHOLD
import pyspark.sql.functions as F
from ml.active_learning import EntropyActiveLearner
from ml.features import TFIDFFeature, JaccardFeature, OverlapCoeffFeature, ExactMatchFeature
from ml.tokenizer import AlphaNumericTokenizer, NGramTokenizer, NumericTokenizer
from ml.fv_generator import FVGenerator
from ml.ml_model import  SKLearnModel
from xgboost import XGBClassifier
from pyspark.sql import SparkSession
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
from copy import deepcopy
import sys
sys.path.append('.')
from es.search import Searcher
from es.index_optimizer import IndexOptimizer, AUCQueryScorer, LinProgOptimizer, TorchOptimizer
from data_sources import MongoDataSource
import os
import config
from config import  LUCENE_INDEX_DIR
from experiment import AbstractExperiment
from config import DATASETS


OUTPUT_DIR = config.OUTPUT_DIR / 'es'
log = get_logger(__name__)


AUC = 'auc'
LINPROG = 'linprog'
AUTO = 'auto'
DEFAULT = 'default'
CONCAT = 'concat'
BUILD_METHODS = [
        AUC,
        DEFAULT,
        CONCAT,
        LINPROG
]

class ESExperiment(AbstractExperiment):

    def __init__(self, dataset, columns,
                k=250,
                default_analyzer='standard',
                copy_mapping=None,
                is_concat=False,
                build_method=AUC,
                recall_only=False,
                mongo_out=None
                ):
        
        if build_method not in BUILD_METHODS:
            raise ValueError(f'unknown build_method ({build_method}), please select from {BUILD_METHODS}')
        
        self.k = k
        self.default_analyzer = default_analyzer
        self.build_method = build_method
        self.recall_only = recall_only
        # default es / lucene sim
        self._auc_sim = {
            'type' : 'BM25',
            'b' : .75,
            'k1' : 1.2
        }
            
        # output for the search job
        f = f'{dataset.dataset_name}_{"_".join(columns)}.parquet'
        
        out_sub_dir = ESExperiment.get_out_sub_dir(build_method, default_analyzer)

        output_dir = OUTPUT_DIR / out_sub_dir
        output_file = output_dir / dataset.dataset_name / f
        self.index_name = get_index_name(dataset.dataset_name, dataset.datatype, out_sub_dir)

        self.copy_mapping = copy_mapping
        self.is_concat = is_concat
        
        self._mongo_out = mongo_out
        if self._mongo_out is not None:
            if isinstance(self._mongo_out, dict):
                self._mongo_out = MongoDataSource.format_mongo_uri(**self._mongo_out)

        super().__init__(dataset=dataset, columns=columns, output_file=output_file)

        self.run_meta_data  = {
                'columns' : self.columns,
                'build_method' : self.build_method,
                'topk' : self.k
        }
    

    @staticmethod
    def get_out_sub_dir(build_method, default_analyzer):
        if build_method == AUTO:
            if default_analyzer is None:
                return build_method
            else:
                return default_analyzer
        else:
            if default_analyzer is None:
                return build_method
            else:
                return build_method + '_' + default_analyzer



    @staticmethod 
    def generate_from_dataset(dataset, default_analyzer=None, **kwargs):
        
        if kwargs.get('build_method', '') in {AUC, LINPROG}:
            column_sets = dataset.opt_column_sets
        else:
            column_sets = dataset.column_sets
        exps = []
        for cs in column_sets:
            exps.append( ESExperiment(
                dataset,
                cs,
                default_analyzer=default_analyzer,
                **kwargs
            ) )

        return exps

    
    def run(self, overwrite):
        init_spark()
        if os.path.exists(self.output_file) and not overwrite:
            log.info(f'running {self.dataset.dataset_name} : {self.columns} - {self.default_analyzer} ({self.k}) ## SKIPPED ##')
            return False
        log.info(f'running {self.dataset.dataset_name} : {self.columns} - {self.default_analyzer} ({self.k})')

        exp_ran = self._run_opt(overwrite)

        return exp_ran
        
    
    def _get_new_index(self):
        index = LuceneIndex(LUCENE_INDEX_DIR / self.index_name)
        return index

    
    def _run_opt(self, overwrite):
        # deepcopy is necessary to avoid pickling error
        columns = self.columns if len(self.columns) != 0 else deepcopy(self.dataset.table_a.read_spark().columns)
        columns = [c for c in columns if c != '_id']

        index = self._build_test_index_auc(columns)

        search_df = self.dataset.table_b.read_spark()

        query_scorer = AUCQueryScorer()
        #query_scorer = TFIDFQueryScorer(analyze_standard, search_df.set_index('_id'))
        #query_scorer = RankQueryScorer(threshold=.1, k=250)

        if not os.path.exists(self.output_file):
            index_optimizer = IndexOptimizer(index, query_scorer)
            if self.build_method == AUC:
                spec = index_optimizer.optimize(search_df)
            else:
                raise RuntimeError(self.build_method)

            print(spec)

            self._search(index, overwrite, spec)
        

        lin_prog_input = pd.read_parquet(self.output_file)



        lin_prog_search_df = search_df.filter(F.col('_id').isin(lin_prog_input['id2'].tolist()))\
                                        .toPandas()

        # create an index spec based on the predictions of the ML model
        opt = TorchOptimizer(index)
        spec = opt.optimize(lin_prog_search_df, lin_prog_input)



        self.output_file = Path('./out.parquet')
        self._search(index, True, spec)


        return True
    
    def _print_search_res(self, meta):
        hist = np.array(meta['histogram'], dtype=np.int64)
        n_gold_matches = self.dataset.gold.size()
        recall_at_k = hist.cumsum() / n_gold_matches
        
        TP = meta['true_positives']
        
        print(f'\ntrue postives = {TP}')
        print(f'recall = {meta["recall"]}')

        for k in [1,3,5,10,15,25,50,75,100]:
            if k > len(hist):
                break
            print(f'recall @ {k} : {recall_at_k[k-1]}')
        print()
        for r in [.95, .96, .97, .98, .99]:
            print(f'k for recall = {r} : {np.argmax(recall_at_k >= r)+1}')



    def _create_search_res_meta(self, searcher, res_df, gold, id_col, query_spec):
        meta = {}

        if gold is not None:
            res_df = res_df.toPandas()
            meta.update(searcher.compute_stats(res_df, gold, self.dataset.dedupe))

        if query_spec is None:
            query_spec = searcher.get_full_query_spec()

        meta['query_spec'] = query_spec
        meta['search_cols'] = list(query_spec.keys())
        meta.update(self.run_meta_data)

        return meta, res_df
        
    def _remove_records_with_no_matches(self, table_b):
        gold = self.dataset.gold.read()
        gold.columns = ['id1', 'id2']
        ids = np.unique(gold['id2'].values)
        if self.dataset.dedupe:
            ids = np.union1d(ids, gold['id1'].values)

        table_b = table_b.set_index('_id')\
                            .loc[ids]\
                            .reset_index()
        return table_b

    def _search(self, index, overwrite, query_spec, *, output_file=None):
        output_file = output_file if output_file is not None else self.output_file
        timer = Timer()
        if os.path.exists(output_file) and not overwrite:
            return False

        # add deps to spark, this isn't done else where
        # because starting a spark context causes a error in joblib pickle
        init_spark()
        # TODO add logging
        log.info(f'running search for {self.dataset.dataset_name} [{self.dataset.datatype}]: {query_spec}')
        searcher = self.get_searcher(index)
        df = self.dataset.table_b.read_spark()
        if self.recall_only:
            df = self._remove_records_with_no_matches(df)
            
        self.run_meta_data['query_spec'] = query_spec.to_dict()
        with persisted(searcher.search(df, query_spec)) as res_df:
            # trigger search and time it
            res_df.count()
            self.run_meta_data['search_time'] = timer.get_interval()
            
            if self._mongo_out is not None:
                # write out to mongo
                res_df.write\
                        .mode('overwrite')\
                        .format('mongo')\
                        .option('uri', self._mongo_out)\
                        .save()

            else:
                gold = read_gold(self.dataset.gold, self.dataset.dedupe) if self.dataset.gold.exists() else None

                meta, res_df = self._create_search_res_meta(searcher, res_df, gold, '_id', query_spec)

                if gold is not None:
                    self._print_search_res(meta)

                if isinstance(res_df, pd.DataFrame):
                    log.info('writing as pandas dataframe')
                    write_parquet(res_df, output_file, meta)
                else:
                    log.info('stream writing spark dataframe')
                    stream_write_dataframe_to_parquet(res_df, output_file, meta)

        return True

    def get_searcher(self, index, k=None):
        k = k if k is not None else self.k
        return Searcher(index, limit=k)

    def _build_test_index_auc(self, columns):
        analyzers = [
                'standard',
                '3gram',
                #'standard36edgegram'
        ]
        index_config = IndexConfig()

        for c in columns:
            if c == '_id':
                continue
            index_config.add_field(c, analyzers)
        
        cols = [c for c in columns if c != '_id']
        index_config.add_concat_field('concat_all', cols, analyzers)
        index_config.sim = deepcopy(self._auc_sim)

        return self._build(index_config)
    
    def _build(self, index_config):
        timer = Timer()
        stream = self.dataset.table_a.read_stream(10000, index_config.get_analyzed_columns())
        index = self._get_new_index()
        index.build(stream, index_config)
        self.run_meta_data['build_time'] = timer.get_interval()
        self.run_meta_data['index_config'] = index_config.to_dict()
        return index

    def _apply_thresholding(self, df):
        scores = pd.Series(np.concatenate(df['scores'].values))
        matches = {}
        
        dkey_t = frozenset if self.dataset.dedupe else tuple

        recall_df = df.loc[df['is_match'].apply(np.any)]
        for c in ['id1_list', 'scores']:
            recall_df[c] = recall_df.apply(lambda x : x[c][x['is_match']], axis=1)

        keys = map(dkey_t, recall_df[['id1_list', 'id2']].explode('id1_list').values)

        for k, v in zip(keys, np.concatenate(recall_df['scores'].values)):
            matches[k] = max(matches.get(k, -1), v)

        recall_scores = pd.Series(list(matches.values()))
        
                         
        binned_recall = recall_scores.value_counts()\
                                            .sort_index(ascending=False)\
                                            .cumsum()

        binned_sizes = scores.value_counts()\
                                .sort_index(ascending=False)\
                                .cumsum()\
                                .loc[binned_recall.index]

        recall_at_threshold = binned_recall.values / self.dataset.gold.size()
        cssr_at_threshold = binned_sizes.values / (self.dataset.table_a.size() * self.dataset.table_b.size())
        threshold_val = binned_recall.index.values

        return recall_at_threshold, cssr_at_threshold, threshold_val


    def _compute_thresholding_results(self, df, meta):

        recall_at_threshold = None
        cssr_at_threshold = None
        threshold_val = None

        recall_at_raw_threshold = None
        cssr_at_raw_threshold = None
        raw_threshold_val = None

        if df is not None or meta is not None:
            df = df.dropna(subset=['scores'])
            df = df.loc[df['scores'].apply(len) != 0]
            # norm score div currently not working fill with 1
            df['norm_score_div'].fillna(1.0, inplace=True)
            norm_df = df.assign(scores=df.apply(lambda x : np.around(x['scores'] / x['norm_score_div'], 3), axis=1))
                                        
            recall_at_threshold, cssr_at_threshold, threshold_val = self._apply_thresholding(norm_df)

            # scores are assumed to be sorted in decreasing order 
            # normalize the scores between 0 and 1
            df = df.assign(scores=df['scores'].apply(lambda x : np.around(x / x[0], 3)))
            recall_at_raw_threshold, cssr_at_raw_threshold, raw_threshold_val = self._apply_thresholding(df)

        return {
                RECALL_AT_THRESHOLD : recall_at_threshold,
                CSSR_AT_THRESHOLD : cssr_at_threshold,
                THRESHOLD_VAL : threshold_val,
                RECALL_AT_RAW_THRESHOLD : recall_at_raw_threshold,
                CSSR_AT_RAW_THRESHOLD : cssr_at_raw_threshold,
                RAW_THRESHOLD_VAL : raw_threshold_val,
        }


    def read_result_as_row(self, when_missing='warn'):
        try: 
            df, meta = self.read_result()
        except Exception as e:
            if when_missing == 'warn':
                warnings.warn(UserWarning(f'{e} unable to read results from {self.output_file} returning null'))
                df, meta = None, None
            else:
                raise e

        row = self._raw_data_to_row(df, meta)

        row.update(self._compute_thresholding_results(df, meta))


        return row

        
def main():
    ds = DATASETS[0]
    exp = ESExperiment( ds, [])

    exp.run(True)


if __name__ == '__main__':
    main()
