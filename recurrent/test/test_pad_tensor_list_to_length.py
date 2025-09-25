import pytest
import torch
from typing import List
# 假设被测试的函数在module.py中
from recurrent.utils import pad_tensor_list_to_length
parametrize = ("lengths, pad_token_id, max_length", [
        ([2, 1, 3], -1, None),
        ([2, 1, 3], -1, 4),
        ([2, 2, 2], -1, None),
        ([2, 2, 2], -1, 4),
])

@pytest.mark.parametrize(
    *parametrize
)
def test_left_pad(lengths, pad_token_id, max_length):
    # 构造输入：[torch.arange(l) for l in lengths]
    response = [torch.arange(l) for l in lengths]

    if max_length is None:
        max_length = max(lengths)

    # left
    padded_response = pad_tensor_list_to_length(response, pad_token_id, max_length=max_length, left_pad=True)
    # 检查输出的形状
    if max_length is None:
        max_length = max(lengths)
    assert padded_response.shape == (len(lengths), max_length), f"Expected shape ({len(lengths)}, {max_length}), got {padded_response.shape}"
    # 检查填充的元素是否正确
    for i, l in enumerate(lengths):
        assert padded_response[i, -l:].tolist() == response[i].tolist(), f"Expected {response[i].tolist()}, got {padded_response[i, -l:]}"
        assert padded_response[i, :max_length-l].tolist() == [pad_token_id] * (max_length-l), f"Expected {[pad_token_id] * (max_length-l)}, got {padded_response[i, :max_length-l]}"
@pytest.mark.parametrize(
    *parametrize
)
def test_right_pad(lengths, pad_token_id, max_length):
    # 构造输入：[torch.arange(l) for l in lengths]
    response = [torch.arange(l) for l in lengths]

    if max_length is None:
        max_length = max(lengths)

    # right
    padded_response = pad_tensor_list_to_length(response, pad_token_id, max_length=max_length, left_pad=False)
    # 检查输出
    assert padded_response.shape == (len(lengths), max_length), f"Expected shape ({len(lengths)}, {max_length}), got {padded_response.shape}"
    # 检查填充的元素是否正确
    for i, l in enumerate(lengths):
        assert padded_response[i, :l].tolist() == response[i].tolist(), f"Expected {response[i].tolist()}, got {padded_response[i, :l]}"
        assert padded_response[i, l:].tolist() == [pad_token_id] * (max_length-l), f"Expected {[pad_token_id] * (max_length-l)}, got {padded_response[i, l:]}"

@pytest.mark.parametrize(
    *parametrize
)
def test_left_mask_pad(lengths, pad_token_id, max_length):
    # 构造输入：[torch.arange(l) for l in lengths]
    response = [torch.arange(l) for l in lengths]

    if max_length is None:
        max_length = max(lengths)
    # left + mask
    padded_response, attention_mask = pad_tensor_list_to_length(response, pad_token_id, max_length=max_length, left_pad=True, return_mask=True)
    # 检查输出
    assert padded_response.shape == (len(lengths), max_length), f"Expected shape ({len(lengths)}, {max_length}), got {padded_response.shape}"


    # 检查attention_mask
    assert attention_mask.shape == (len(lengths), max_length), f"Expected shape ({len(lengths)}, {max_length}), got {attention_mask.shape}"
    assert (attention_mask == (padded_response != pad_token_id)).all(), f"Expected {padded_response!= pad_token_id}, got {padded_response == pad_token_id}"


@pytest.mark.parametrize(
    *parametrize
)
def test_right_mask_pad(lengths, pad_token_id, max_length):
    # 构造输入：[torch.arange(l) for l in lengths]
    response = [torch.arange(l) for l in lengths]

    if max_length is None:
        max_length = max(lengths)
    # right + mask
    padded_response, attention_mask = pad_tensor_list_to_length(response, pad_token_id, max_length=max_length, left_pad=False, return_mask=True)
    # 检查输出
    assert padded_response.shape == (len(lengths), max_length), f"Expected shape ({len(lengths)}, {max_length}), got {padded_response.shape}"
    # 检查填充的元素是否正确
    for i, l in enumerate(lengths):
        assert padded_response[i, :l].tolist() == response[i].tolist(), f"Expected {response[i].tolist()}, got {padded_response[i, :l]}"
        assert padded_response[i, l:].tolist() == [pad_token_id] * (max_length-l), f"Expected {[pad_token_id] * (max_length-l)}, got {padded_response[i, l:]}"

    # 检查attention_mask
    assert attention_mask.shape == (len(lengths), max_length), f"Expected shape ({len(lengths)}, {max_length}), got {attention_mask.shape}"
    assert (attention_mask == (padded_response != pad_token_id)).all(), f"Expected {padded_response!= pad_token_id}, got {padded_response == pad_token_id}"