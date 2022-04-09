import pandas as pd
from typing import List
from modules.helmet_assignment.helmet_assignment.score import NFLAssignmentScorer


def prepare_submission(baseline_helmets_all: List, sample_submission, output_video_root, data_labels):
    """
    Prepare a submission and evaluate the result
    :param baseline_helmets_all:
    :param sample_submission:
    :param output_video_root:
    :param data_labels:
    :return:
    """
    concat = pd.concat(baseline_helmets_all)
    submission = concat[sample_submission.columns]
    submission = submission[submission['label'] != 'unknown']
    submission.to_csv(f'{output_video_root}/submission.csv', index=False)
    data_labels = data_labels[data_labels['playID'].isin(concat.playID.unique())].copy()
    scorer = NFLAssignmentScorer(data_labels)
    score = scorer.score(submission)
    return score, submission, scorer.sub_labels
