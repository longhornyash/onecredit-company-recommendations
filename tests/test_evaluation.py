import pytest

from evaluation.run import ndcg


def test_ndcg_rewards_correct_order_and_handles_no_results():
    assert ndcg([2, 1, 0, 0], [2, 1, 0, 0]) == pytest.approx(1)
    assert ndcg([1, 2, 0, 0], [2, 1, 0, 0]) < 1
    assert ndcg([], [2]) == 0
    assert ndcg([0], [0]) == 0
