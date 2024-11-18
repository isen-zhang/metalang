from seqthetic import DatasetSpec, Vocabulary, Zipf
from seqthetic.dataset_spec import AddLossMask, WithDomainDelimiter
from seqthetic.domains import Deduplicate

import numpy as np
from seqthetic.domains.delete import Delete
from seqthetic.range import Range


# domain = Deduplicate(
#     sequence_length=20,
#     unique_token_ratio=Range(min=0.2, max=0.5),
#     repeat_token_ratio=Range(min=0.6, max=0.8),
#     vocab=Vocabulary(num_vocab=10000, prob=Zipf()),
# )

# res = domain.make_sequences(num_token=40)

# print(res)
# def verify_deduplicate_sequence(sequence):
#     """
#     验证deduplicate序列是否符合要求

#     Args:
#         sequence: numpy array，包含输入序列和期望输出(-1作为分隔符)

#     Returns:
#         bool: 是否符合要求
#         str: 错误信息（如果有）
#     """
#     # 检查是否存在分隔符-1
#     if -1 not in sequence:
#         return False, "No delimiter (-1) found"

#     # 根据分隔符分割序列
#     delimiter_idx = np.where(sequence == -1)[0][0]
#     input_seq = sequence[:delimiter_idx]
#     output_seq = sequence[delimiter_idx + 1 :]

#     # 检查基本要求
#     if len(input_seq) == 0 or len(output_seq) == 0:
#         return False, "Empty input or output sequence"

#     # 检查输入序列是否有重复
#     if len(input_seq) <= len(set(input_seq)):
#         return False, "Input sequence has no duplicates"

#     # 检查输出序列是否是输入序列的去重结果
#     input_unique = np.unique(input_seq)
#     if not np.array_equal(np.sort(input_unique), np.sort(output_seq)):
#         return False, "Output sequence is not equal to deduplicated input sequence"

#     return True, "Sequence is valid"

# for seq in res:
#     print(verify_deduplicate_sequence(seq))
# # 不重复 token 有多少，repeat_ratio 里多少 token 被重复 * 重复次数
# # repeat_ratio * repeat_time = seqlen / (uniq_tok) - 1
# # seqlen = repeat_ratio * repeat_time * unique_tok + 1 + unique_tok


# spec = DatasetSpec(
#     num_token=60,
#     domains=[
#         WithDomainDelimiter(
#             domains=[
#                 Deduplicate(
#                     sequence_length=20,
#                     unique_token_ratio=Range(min=0.2, max=0.5),
#                     repeat_token_ratio=Range(min=0.6, max=0.8),
#                     vocab=Vocabulary(num_vocab=10000, prob=Zipf()),
#                 ),
#                 Deduplicate(
#                     sequence_length=20,
#                     unique_token_ratio=Range(min=0.2, max=0.5),
#                     repeat_token_ratio=Range(min=0.6, max=0.8),
#                     vocab=Vocabulary(num_vocab=10000, prob=Zipf()),
#                 ),
#                 Deduplicate(
#                     sequence_length=20,
#                     unique_token_ratio=Range(min=0.2, max=0.5),
#                     repeat_token_ratio=Range(min=0.6, max=0.8),
#                     vocab=Vocabulary(num_vocab=10000, prob=Zipf()),
#                 ),
#             ]
#         )
#     ],
#     postprocessors=[
#         AddLossMask(),
#     ],
# )

# spec.make_dataset(path='./', single_thread=True)


spec = DatasetSpec(
    num_token=60,
    domains=[
        Delete(
            sequence_length=20,
            should_delete_ratio=1,
            seq_to_delete_length=Range(min=2, max=4),
            vocab=Vocabulary(num_vocab=20000),
        )
    ],
    postprocessors=[
        AddLossMask(),
    ],
)

item = spec.make_dataset(single_thread=True)
print(item)