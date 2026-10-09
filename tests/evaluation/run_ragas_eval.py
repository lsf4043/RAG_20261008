"""
RAGAS 评估脚本（升级版新增，P0 优化）
对 RAG 系统进行自动化质量评估，量化检索召回率、回答忠实度、相关性等核心指标。

使用方式：
    python -m tests.evaluation.run_ragas_eval

评估指标：
- faithfulness        忠实度：答案中的每个陈述是否被检索内容支撑（0-1，越高越好）
- answer_relevancy    答案相关性：答案是否直接回答了问题（0-1）
- context_precision   上下文精确率：检索内容中有多少是相关的（0-1）
- context_recall      上下文召回率：需要的信息是否都被检索到（0-1）

输出：
- 控制台打印各指标得分
- 可选：导出评估报告到 JSON 文件
"""
import os
import sys
import json
import asyncio
from typing import List, Dict, Any

# 将项目根目录加入 sys.path，确保能导入 app 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.config import settings
from app.graph.state import initial_state
from app.graph.graph import get_graph
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    加载评测数据集

    Args:
        dataset_path: 数据集 JSON 文件路径

    Returns:
        问题列表
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_rag_single(question: str) -> Dict[str, Any]:
    """
    调用 RAG 系统获取答案和检索上下文

    Args:
        question: 问题

    Returns:
        包含 answer 和 contexts 的字典
    """
    state = initial_state(question)
    result = get_graph().invoke(state)

    answer = result.get("answer", "")
    contexts = [
        s.get("text", "")
        for s in result.get("sources", [])
    ]
    sources = result.get("sources", [])

    return {
        "answer": answer,
        "contexts": contexts,
        "sources": sources,
    }


def evaluate_with_ragas(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    使用 RAGAS 计算评估指标

    Args:
        samples: 样本列表，每项含 question/answer/contexts/ground_truth

    Returns:
        评估结果
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError as e:
        logger.error(f"缺少 RAGAS 依赖，请先安装: pip install ragas datasets")
        raise

    # 构造 RAGAS 数据格式
    data = {
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
        "ground_truth": [s["ground_truth"] for s in samples],
    }
    dataset = Dataset.from_dict(data)

    logger.info(f"开始 RAGAS 评估，共 {len(samples)} 个样本")

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    logger.info("RAGAS 评估完成")
    return result


def main():
    """评估主流程"""
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_dataset.json")
    logger.info(f"加载评测数据集: {dataset_path}")

    dataset = load_dataset(dataset_path)
    logger.info(f"共 {len(dataset)} 个评测问题")

    # 1. 构建状态机（确保已初始化）
    logger.info("构建 LangGraph 状态机...")
    get_graph()

    # 2. 对每个问题调用 RAG 系统
    samples = []
    for i, item in enumerate(dataset):
        question = item["question"]
        logger.info(f"[{i + 1}/{len(dataset)}] 评估问题: {question[:40]}")

        rag_result = run_rag_single(question)
        samples.append({
            "question": question,
            "answer": rag_result["answer"],
            "contexts": rag_result["contexts"],
            "ground_truth": item["ground_truth"],
        })

    # 3. RAGAS 评估
    logger.info("开始计算 RAGAS 指标...")
    result = evaluate_with_ragas(samples)

    # 4. 打印结果
    print("\n" + "=" * 60)
    print("RAGAS 评估结果")
    print("=" * 60)
    print(result)
    print("=" * 60)

    # 5. 导出报告
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": result._scores if hasattr(result, "_scores") else {},
                "samples": samples,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"评估报告已导出: {report_path}")


if __name__ == "__main__":
    main()