import logging

from lm_eval.tasks.ruler.common_utils import (
    DEFAULT_SEQ_LENGTHS,
    build_cached_dataset,
    get_tokenizer,
)
from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack


TEMPLATE = """Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?"""
eval_logger = logging.getLogger(__name__)


def niah_single_1(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    tokenizer_name = kwargs.get("tokenizer")
    pretrained = kwargs.get("pretrained")
    tokenizer = get_tokenizer(**kwargs)
    return build_cached_dataset(
        task_name="niah_single_1",
        seq_lengths=seq_lengths,
        tokenizer_name=tokenizer_name,
        pretrained=pretrained,
        generator_fn=lambda seq: generate_samples(
            get_haystack(type_haystack="repeat"),
            max_seq_length=seq,
            template=TEMPLATE,
            type_haystack="repeat",
            type_needle_k="words",
            type_needle_v="numbers",
            num_samples=500,
            TOKENIZER=tokenizer,
        ),
    )


def niah_single_2(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    tokenizer_name = kwargs.get("tokenizer")
    pretrained = kwargs.get("pretrained")
    tokenizer = get_tokenizer(**kwargs)
    return build_cached_dataset(
        task_name="niah_single_2",
        seq_lengths=seq_lengths,
        tokenizer_name=tokenizer_name,
        pretrained=pretrained,
        generator_fn=lambda seq: generate_samples(
            get_haystack(type_haystack="essay"),
            max_seq_length=seq,
            template=TEMPLATE,
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="numbers",
            num_samples=500,
            TOKENIZER=tokenizer,
        ),
    )


def niah_single_3(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    tokenizer_name = kwargs.get("tokenizer")
    pretrained = kwargs.get("pretrained")
    tokenizer = get_tokenizer(**kwargs)
    return build_cached_dataset(
        task_name="niah_single_3",
        seq_lengths=seq_lengths,
        tokenizer_name=tokenizer_name,
        pretrained=pretrained,
        generator_fn=lambda seq: generate_samples(
            get_haystack(type_haystack="essay"),
            max_seq_length=seq,
            template=TEMPLATE,
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="uuids",
            num_samples=500,
            TOKENIZER=tokenizer,
        ),
    )


def niah_multikey_1(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    tokenizer_name = kwargs.get("tokenizer")
    pretrained = kwargs.get("pretrained")
    tokenizer = get_tokenizer(**kwargs)
    return build_cached_dataset(
        task_name="niah_multikey_1",
        seq_lengths=seq_lengths,
        tokenizer_name=tokenizer_name,
        pretrained=pretrained,
        generator_fn=lambda seq: generate_samples(
            get_haystack(type_haystack="essay"),
            max_seq_length=seq,
            template=TEMPLATE,
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="numbers",
            num_needle_k=4,
            num_samples=500,
            TOKENIZER=tokenizer,
        ),
    )


def niah_multikey_2(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    tokenizer_name = kwargs.get("tokenizer")
    pretrained = kwargs.get("pretrained")
    tokenizer = get_tokenizer(**kwargs)
    return build_cached_dataset(
        task_name="niah_multikey_2",
        seq_lengths=seq_lengths,
        tokenizer_name=tokenizer_name,
        pretrained=pretrained,
        generator_fn=lambda seq: generate_samples(
            get_haystack(type_haystack="needle"),
            max_seq_length=seq,
            template=TEMPLATE,
            type_haystack="needle",
            type_needle_k="words",
            type_needle_v="numbers",
            num_samples=500,
            TOKENIZER=tokenizer,
        ),
    )


def niah_multikey_3(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    tokenizer_name = kwargs.get("tokenizer")
    pretrained = kwargs.get("pretrained")
    tokenizer = get_tokenizer(**kwargs)
    return build_cached_dataset(
        task_name="niah_multikey_3",
        seq_lengths=seq_lengths,
        tokenizer_name=tokenizer_name,
        pretrained=pretrained,
        generator_fn=lambda seq: generate_samples(
            get_haystack(type_haystack="needle"),
            max_seq_length=seq,
            template=TEMPLATE,
            type_haystack="needle",
            type_needle_k="uuids",
            type_needle_v="uuids",
            num_samples=500,
            TOKENIZER=tokenizer,
        ),
    )


def niah_multivalue(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    tokenizer_name = kwargs.get("tokenizer")
    pretrained = kwargs.get("pretrained")
    tokenizer = get_tokenizer(**kwargs)
    return build_cached_dataset(
        task_name="niah_multivalue",
        seq_lengths=seq_lengths,
        tokenizer_name=tokenizer_name,
        pretrained=pretrained,
        generator_fn=lambda seq: generate_samples(
            get_haystack(type_haystack="essay"),
            max_seq_length=seq,
            template=TEMPLATE,
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="numbers",
            num_needle_v=4,
            num_samples=500,
            TOKENIZER=tokenizer,
        ),
    )


def niah_multiquery(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    tokenizer_name = kwargs.get("tokenizer")
    pretrained = kwargs.get("pretrained")
    tokenizer = get_tokenizer(**kwargs)
    return build_cached_dataset(
        task_name="niah_multiquery",
        seq_lengths=seq_lengths,
        tokenizer_name=tokenizer_name,
        pretrained=pretrained,
        generator_fn=lambda seq: generate_samples(
            get_haystack(type_haystack="essay"),
            max_seq_length=seq,
            template=TEMPLATE,
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="numbers",
            num_needle_q=4,
            num_samples=500,
            TOKENIZER=tokenizer,
        ),
    )
