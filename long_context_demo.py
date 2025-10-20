"""演示：LLM 长上下文能力测试 (Arize-style Needle In Haystack)

这个测试与 RAG 向量检索测试不同：
- RAG测试：测试向量检索 + Rerank 的准确性
- 长上下文测试：测试 LLM 从长文档中找信息的能力

用途：
- 评估 LLM 的上下文窗口利用能力
- 测试不同文档长度和 needle 位置对性能的影响
- 类似 Arize 的 Needle In Haystack 测试
"""

import os
from src.rag.document_loader import DocumentLoader
from src.rag.llm_client import LLMClient
from src.rag.long_context_test import LongContextTest


def main():
    print("=" * 80)
    print("LLM 长上下文能力测试 (Arize-style)")
    print("=" * 80)

    # 步骤 1: 加载文档作为 haystack
    print("\n[步骤 1] 加载文档...")
    loader = DocumentLoader()

    # 加载所有文档
    data_files = [
        "data/fire2.txt",
        "data/life3.txt",
        "data/life4.txt"
    ]

    all_chunks = []
    for file_path in data_files:
        file_chunks = loader.load_file(
            file_path,
            strategy="smart",
            chunk_size=300
        )
        all_chunks.extend(file_chunks)
        print(f"   - {file_path}: {len(file_chunks)} 个块")

    print(f"   总计: {len(all_chunks)} 个文档块")

    # 步骤 2: 初始化 LLM 客户端
    print("\n[步骤 2] 初始化 LLM 客户端...")
    KIMI_API_KEY = os.getenv("MOONSHOT_API_KEY", "sk-QMVGFIxphgo70al8We9W76woZIhz2dER0VyfZb0DSRwHPrlO")
    llm_client = LLMClient(
        api_key=KIMI_API_KEY,
        model="kimi-k2-0905-preview"  # 使用 K2 模型
    )
    tester = LongContextTest(llm_client)

    # 步骤 3: 使用真实 needle（关于梦醒公司）
    print("\n[步骤 3] 使用真实 Needle...")
    needle = "后来，在某个城市漫步的我，进入了一个初创公司，名叫梦醒，开始了旅程"
    query = "我在漫步的时候加入了哪家公司？"
    expected_answer = "梦醒"
    print(f"   Needle: {needle}")
    print(f"   查询: {query}")
    print(f"   期望答案: {expected_answer}")

    # 步骤 4: 运行单次测试
    print("\n[步骤 4] 运行单次测试...")
    print(f"   使用前 30 个文档块作为上下文")

    result = tester.run_test(
        documents=all_chunks[:30],
        needle=needle,
        query=query,
        expected_answer=expected_answer,
        needle_position=15  # 插入在中间位置
    )

    print("\n" + "=" * 80)
    print("单次测试结果")
    print("=" * 80)
    print(f"成功: {result['success']}")
    print(f"期望答案: {result['expected_answer']}")
    print(f"LLM 回答: {result['llm_answer']}")
    print(f"Needle 位置: {result['needle_position']}/{result['haystack_size']} " +
          f"({result['needle_depth_percent']:.1f}%)")
    print(f"上下文长度: {result['context_length_chars']} 字符 " +
          f"(~{result['context_length_tokens']} tokens)")

    # 步骤 5: 运行多次测试（可选）
    run_multiple = input("\n是否运行多次测试以生成热力图数据？(y/n): ").strip().lower()

    if run_multiple == 'y':
        print("\n[步骤 5] 运行多次测试...")
        print("   测试配置:")
        print("   - 上下文大小: [10, 30, 50, 70, 98]")
        print("   - Needle 深度: [0%, 25%, 50%, 75%, 100%]")
        print("   - 每配置试验次数: 2")
        print("   - 使用相同 needle: 梦醒公司")

        results = tester.run_multiple_tests(
            documents=all_chunks,
            needle=needle,
            query=query,
            expected_answer=expected_answer,
            context_sizes=[10, 30, 50, 70, 98],
            depth_percentages=[0, 25, 50, 75, 100],
            trials_per_config=2
        )

        # 打印结果摘要
        tester.print_results_summary(results)

        # 保存详细结果
        import json
        with open("long_context_test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存到: long_context_test_results.json")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

    # 说明
    print("\n【测试说明】")
    print("- 这个测试评估 LLM 从长文档中检索信息的能力")
    print("- 与 RAG 向量检索测试不同，这里直接把所有文档给 LLM")
    print("- 类似 Arize 的 Needle In Haystack 测试方法")
    print("- 可用于评估不同 LLM 的长上下文能力")


if __name__ == "__main__":
    main()
