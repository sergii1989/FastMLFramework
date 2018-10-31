import os
import gc
import pandas as pd

from pandas import testing as pdt
from builtins import filter
from future.utils import iteritems, itervalues
from modeling.prediction import Predictor


class Ensembler(object):

    @staticmethod
    def _verify_oof_input_data_completeness(oof_input_files):
        for solution in itervalues(oof_input_files):
            assert len(solution['files']) >= 2, (
                    "There should be at least two input files: one containing 'train_OOF' in the file name and"
                    " the other 'test'. Instead got: %s" % solution['files'])
            assert len(list(filter(lambda x: 'train_OOF' in x, solution['files']))) >= 1, (
                "There should be at least one input file containing train out-of-fold predictions for each "
                "provided path (i.e. file containing 'train_OOF' in the name)")
            assert len(list(filter(lambda x: 'test' in x, solution['files']))) >= 1, (
                "There should be at least one input file containing test predictions for each "
                "provided path (i.e. file containing 'test' in the name)")

    @staticmethod
    def _verify_consistency_of_input_data(meta_data, model_data, target_column, index_column, filename):
        assert meta_data.shape[0] == model_data.shape[0]
        if 'train_OOF' in filename:
            assert target_column in meta_data, ('Please add {0} column to the {1}'.format(target_column, filename))
            pdt.assert_series_equal(meta_data[target_column], model_data[target_column])
        if Predictor.verify_index_column_is_defined(index_column):
            assert index_column in meta_data, ('Please add {0} column to the {1}'.format(index_column, filename))
            pdt.assert_series_equal(meta_data[index_column], model_data[index_column])

    @staticmethod
    def _join_oof_results(main_df, index_column, target_column, list_preds_df, target_decimals):
        df = pd.concat(list_preds_df, axis=1).round(target_decimals)

        # Convert to int if target rounding precision is 0 decimals
        if target_decimals == 0:
            df = df.astype(int)

        # Add index column to the beginning of DF if index column is valid
        if Predictor.verify_index_column_is_defined(index_column):
            index_values = main_df[index_column].values
            df.insert(loc=0, column=index_column, value=index_values)

        # Add column with real target values to OOF dataframe
        if target_column in main_df and main_df[target_column].notnull().all():
            df[target_column] = main_df[target_column].values

        return df

    def load_oof_target_and_test_data(self, oof_input_files, stack_bagged_results, train_df, test_df,
                                      target_column, index_column, target_decimals, project_location):

        self._verify_oof_input_data_completeness(oof_input_files)

        list_train_oof_df = []
        list_test_preds_df = []
        for results_suffix, solution_details in iteritems(oof_input_files):
            if stack_bagged_results:
                oof_files = list(filter(lambda x: 'bagged' in x, solution_details['files']))
            else:
                oof_files = list(filter(lambda x: 'bagged' not in x, solution_details['files']))

            for filename in oof_files:
                # Composing absolute path to a single OOF results file
                if project_location in solution_details['path']:
                    full_path = os.path.normpath(os.path.join(solution_details['path'], filename))
                else:
                    full_path = os.path.normpath(os.path.join(project_location, solution_details['path'], filename))

                # Reading OOF predictions
                if 'train_OOF' in filename:
                    df = pd.read_csv(full_path)
                    self._verify_consistency_of_input_data(df, train_df, target_column, index_column, filename)
                    df = df.loc[:, ~df.columns.isin([index_column, target_column])]
                    df.columns = ['_'.join([results_suffix, col]) for col in df.columns]
                    list_train_oof_df.append(df)
                elif 'test' in filename:
                    df = pd.read_csv(full_path)
                    self._verify_consistency_of_input_data(df, test_df, target_column, index_column, filename)
                    df = df.loc[:, df.columns != index_column]
                    df.columns = ['_'.join([results_suffix, col]) for col in df.columns]
                    list_test_preds_df.append(df)
                else:
                    raise ValueError("File '%s' does not contains 'train_OOF' or 'test' in name." % filename)

        oof_train_df = self._join_oof_results(train_df, index_column, target_column, list_train_oof_df, target_decimals)
        oof_test_df = self._join_oof_results(test_df, index_column, target_column, list_test_preds_df, target_decimals)

        del list_train_oof_df, list_test_preds_df; gc.collect()
        return oof_train_df, oof_test_df
