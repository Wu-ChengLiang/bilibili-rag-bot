#!/usr/bin/env python3
"""Debug GTE embedding input validation"""

from src.rag.embeddings import GTEEmbedding
import traceback

print("=" * 80)
print("GTE Embedding Input Debug Test")
print("=" * 80)

embedding = GTEEmbedding(model_name="thenlper/gte-base-zh")

# Test cases
test_cases = [
    ("单个字符串", "何老师上过什么课"),
    ("带问号的查询", "何老师上过什么课？"),
    ("列表输入", ["何老师上过什么课", "今天天气怎么样"]),
    ("空字符串", ""),
    ("只有空格", "   "),
    ("None值", None),
    ("整数", 123),
    ("字节串", b"hello"),
    ("特殊字符", "@#$%^&*()"),
    ("很长的文本", "这是一个很长的文本" * 100),
    ("混合列表", ["有效文本", "", None, "另一个有效文本"]),
]

for test_name, test_input in test_cases:
    print(f"\n📝 测试: {test_name}")
    print(f"   输入类型: {type(test_input)}")
    if isinstance(test_input, (str, list)):
        print(f"   输入值: {repr(test_input)[:100]}")

    try:
        result = embedding.encode(test_input)
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                print(f"   ✓ 成功! 返回 {len(result)} 个向量，维度 {len(result[0])}")
            else:
                print(f"   ✓ 成功! 返回单个向量，维度 {len(result)}")
        else:
            print(f"   ✓ 成功! 返回结果类型: {type(result)}")
    except Exception as e:
        print(f"   ❌ 失败!")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        print(f"   追踪:")
        for line in traceback.format_exc().split('\n')[-4:-1]:
            print(f"     {line}")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
