import pytest

import megatron.core.parallel_state as ps


def _groups_stay_within_local_blocks(groups, local_world_size):
    return all(len({rank // local_world_size for rank in ranks}) == 1 for ranks in groups)


def test_rank_generator_preserves_legacy_positional_order_argument():
    rank_generator = ps.RankGenerator(2, 1, 1, 2, 1, "tp-cp-ep-pp-dp")

    assert rank_generator.order == "tp-cp-ep-pp-dp"
    assert rank_generator.get_ranks("pp") == [[0, 2], [1, 3]]


def test_cdc_size_one_does_not_rewrite_order_unless_explicit():
    rank_generator = ps.RankGenerator(
        tp=2, ep=1, dp=1, pp=2, cp=1, order="tp-cp-ep-dp-pp", cdc=1
    )
    explicit_cdc_generator = ps.RankGenerator(
        tp=2, ep=1, dp=1, pp=2, cp=1, order="tp-cp-ep-dp-pp-cdc", cdc=1
    )

    assert rank_generator.order == "tp-cp-ep-dp-pp"
    assert explicit_cdc_generator.order == "tp-cp-ep-dp-pp-cdc"
    assert explicit_cdc_generator.get_ranks("cdc") == [[0], [1], [2], [3]]


def test_cdc_rank_generator_adds_cdc_as_outer_dimension():
    rank_generator = ps.RankGenerator(
        tp=2, ep=1, dp=1, pp=2, cp=1, cdc=2, order="tp-cp-ep-pp-dp"
    )

    assert rank_generator.order == "tp-cp-ep-pp-dp-cdc"
    assert rank_generator.get_ranks("pp") == [[0, 2], [1, 3], [4, 6], [5, 7]]
    assert rank_generator.get_ranks("cdc") == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert rank_generator.get_ranks("dp-cdc") == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_dense_and_expert_pp_groups_match_with_ep_and_pp():
    dense_generator = ps.RankGenerator(
        tp=2, ep=1, dp=1, pp=2, cp=1, cdc=2, order="tp-cp-ep-pp-dp"
    )
    expert_generator = ps.RankGenerator(
        tp=1, ep=2, dp=1, pp=2, cp=1, cdc=2, order="tp-cp-ep-pp-dp"
    )

    assert dense_generator.get_ranks("pp") == expert_generator.get_ranks("pp")
    assert expert_generator.get_ranks("ep") == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_cdc_rank_generator_keeps_cp_and_ep_inside_each_dc():
    local_world_size = 8
    dense_generator = ps.RankGenerator(
        tp=2, ep=1, dp=1, pp=2, cp=2, cdc=3, order="tp-cp-ep-pp-dp"
    )
    expert_generator = ps.RankGenerator(
        tp=2, ep=2, dp=1, pp=2, cp=1, cdc=3, order="tp-cp-ep-pp-dp"
    )

    assert dense_generator.get_ranks("pp") == expert_generator.get_ranks("pp")
    assert dense_generator.get_ranks("cdc") == [[rank, rank + 8, rank + 16] for rank in range(8)]
    assert _groups_stay_within_local_blocks(dense_generator.get_ranks("cp"), local_world_size)
    assert _groups_stay_within_local_blocks(expert_generator.get_ranks("ep"), local_world_size)


def test_cdc_dimension_must_be_last_if_explicitly_specified():
    with pytest.raises(RuntimeError, match="CDC rank dimension must be the last"):
        ps.RankGenerator(tp=2, ep=1, dp=1, pp=2, cp=1, cdc=2, order="tp-cdc-ep-pp-dp")
