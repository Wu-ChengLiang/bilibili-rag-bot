"""Tests for Needle In a Haystack framework"""

import pytest
from src.rag.client import RAGClient
from src.rag.needle_test import NeedleTest


@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary database path"""
    return str(tmp_path / "needle_test_db")


@pytest.fixture
def needle_tester(test_db_path):
    """Create a needle tester instance"""
    client = RAGClient(persist_directory=test_db_path, collection_name="needle_test")
    return NeedleTest(client)


class TestNeedleGeneration:
    """Test haystack and needle generation"""

    def test_generate_haystack(self, needle_tester):
        """Test generating haystack documents"""
        haystack = needle_tester.generate_haystack(size=10)
        assert len(haystack) == 10
        assert all(isinstance(doc, str) for doc in haystack)

    def test_generate_custom_haystack(self, needle_tester):
        """Test generating haystack with custom template"""
        haystack = needle_tester.generate_haystack(
            size=5,
            template="这是第{}号文档"
        )
        assert len(haystack) == 5
        assert "这是第0号文档" in haystack

    def test_insert_needle(self, needle_tester):
        """Test inserting needle into haystack"""
        haystack = needle_tester.generate_haystack(size=10)
        needle = "这是重要信息：秘密密码是BANANA"

        docs, position = needle_tester.insert_needle(haystack, needle, position=5)

        assert len(docs) == 11  # Original 10 + 1 needle
        assert docs[5] == needle
        assert position == 5

    def test_insert_needle_random_position(self, needle_tester):
        """Test inserting needle at random position"""
        haystack = needle_tester.generate_haystack(size=10)
        needle = "重要信息"

        docs, position = needle_tester.insert_needle(haystack, needle)

        assert len(docs) == 11
        assert needle in docs
        assert 0 <= position <= 10


class TestNeedleSearch:
    """Test needle retrieval"""

    def test_run_simple_needle_test(self, needle_tester):
        """Test running a simple needle test"""
        needle = "在2025年，阿良曾遇见了一个叫雅薇的女孩"

        result = needle_tester.run_test(
            needle=needle,
            haystack_size=20,
            query="阿良遇见了谁"
        )

        assert result is not None
        assert "needle" in result
        assert "needle_found" in result
        assert "needle_rank" in result
        assert "success" in result
        assert result["haystack_size"] == 20

    def test_needle_found_in_small_haystack(self, needle_tester):
        """Test that needle is found in small haystack"""
        needle = "秘密答案是：42"

        result = needle_tester.run_test(
            needle=needle,
            haystack_size=10,
            query="秘密答案"
        )

        assert result["needle_found"] is True
        assert result["needle_rank"] is not None
        assert result["needle_rank"] >= 1

    def test_needle_retrieval_accuracy(self, needle_tester):
        """Test needle retrieval accuracy"""
        needle = "福贵是一头勤劳的老牛"

        result = needle_tester.run_test(
            needle=needle,
            haystack_size=50,
            query="福贵"
        )

        # Needle should be found
        assert result["needle_found"] is True

        # Ideally ranked first, but at least in top 5
        assert result["needle_rank"] <= 5


class TestMultipleNeedleTests:
    """Test running multiple needle tests"""

    def test_run_multiple_tests(self, needle_tester):
        """Test running multiple tests with different configurations"""
        needle = "测试文档"

        results = needle_tester.run_multiple_tests(
            needle=needle,
            haystack_sizes=[10, 20, 30],
            trials_per_size=2
        )

        assert len(results) == 6  # 3 sizes * 2 trials
        assert all("needle_found" in r for r in results)
        assert all("haystack_size" in r for r in results)

    def test_success_rate_calculation(self, needle_tester):
        """Test calculating success rate across multiple tests"""
        needle = "特定信息"

        results = needle_tester.run_multiple_tests(
            needle=needle,
            haystack_sizes=[5, 10],
            trials_per_size=3
        )

        success_count = sum(1 for r in results if r["needle_found"])
        success_rate = success_count / len(results)

        # Should have reasonable success rate
        assert success_rate > 0


class TestNeedleTestWithRealData:
    """Test with more realistic scenarios"""

    def test_chinese_text_needle(self, needle_tester):
        """Test with Chinese text needle"""
        needle = "中国的首都是北京，这是一个历史悠久的城市"

        result = needle_tester.run_test(
            needle=needle,
            haystack_size=100,
            query="中国首都在哪里"
        )

        assert result["needle_found"] is True

    def test_specific_query_vs_needle_content(self, needle_tester):
        """Test that specific query finds exact needle"""
        needle = "密码是：XYZ123"

        result = needle_tester.run_test(
            needle=needle,
            haystack_size=50,
            query="密码"
        )

        assert result["needle_found"] is True
        assert "密码" in result["needle"]
