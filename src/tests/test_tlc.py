from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets mnist" \
                       " --network LeNet --num-tasks 3 --seed 1 --batch-size 32" \
                       " --nepochs 3" \
                       " --num-workers 0" \
                       " --approach tlc"


def test_tlc_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    run_main_and_assert(args_line)


def test_tlc_exemplars_lambda():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --lamb 1"
    run_main_and_assert(args_line)


def test_tlc_exemplars_per_class():
    args_line = FAST_LOCAL_TEST_ARGS
    run_main_and_assert(args_line)


def test_tlc_with_warmup():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --warmup-nepochs 5"
    args_line += " --warmup-lr-factor 0.5"
    run_main_and_assert(args_line)
