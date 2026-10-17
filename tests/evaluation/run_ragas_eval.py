"""
RAGAS 评估脚本（支持消融对比）

相对旧版的关键修复：
1. **真正接上项目自己的模型**：旧脚本调用 ``evaluate()`` 时没有传 llm / embeddings，
   RAGAS 会默认 `from openai import OpenAI()` 并要求 OPENAI_API_KEY —— 项目里根本
   没有这个 Key，所以脚本从来没有跑通过。现在用 DeepSeek(ChatModel) + DashScope
   (Embeddings) 的 LangChain 包装器。
2. **报告不再为空**：旧代码读 ``result._scores``（新版本已改名 ``_scores_dict``），
   ``hasattr`` 判空后把 metrics 静默写成 ``{}``。现在用 ``to_pandas()`` 输出完整明细。
3. **可复现**：先检查知识库是否可连；支持 ``--ingest`` 先把 ``--corpus`` 目录灌入向量库。
4. **支持消融实验**：依次跑「纯向量 / 混合检索 / +重排 / 全量」，输出对比表，
   这样"优化有效"才是被证明的，而不是被声称的。

用法：
    python -m tests.evaluation.run_ragas_eval
    python -m tests.evaluation.run_ragas_eval --variants hybrid
    python -m tests.evaluation.run_ragas_eval --ingest --corpus ./docs
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.config import settings  # noqa: E402
from app.graph.graph import get_graph  # noqa: E402
from app.graph.state import initial_state  # noqa: E402
from app.utils.logger import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(EVAL_DIR, "eval_dataset.json")
REPORT_PATH = os.path.join(EVAL_DIR, "eval_report.json")


@dataclass
class Variant:
    """一组配置覆盖，用于消融实验"""

    name: str
    description: str
    overrides: Dict[str, Any] = field(default_factory=dict)


VARIANTS: Dict[str, Variant] = {
    "vector_only": Variant(
        name="vector_only",
        description="纯向量检索（无 BM25 / 无重排 / 无改写 / 无 CRAG）",
        overrides={
            "BM25_WEIGHT": 0.0,
            "RERANK_ENABLED": False,
            "CRAG_ENABLED": False,
            "QUERY_REWRITE_ENABLED": False,
            "MULTI_QUERY_ENABLED": False,
        },
    ),
    "hybrid": Variant(
        name="hybrid",
        description="混合检索（向量 + BM25 + 加权 RRF），无重排/改写/CRAG",
        overrides={
            "RERANK_ENABLED": False,
            "CRAG_ENABLED": False,
            "QUERY_REWRITE_ENABLED": False,
            "MULTI_QUERY_ENABLED": False,
        },
    ),
    "hybrid_rerank": Variant(
        name="hybrid_rerank",
        description="混合检索 + Cross-Encoder 重排",
        overrides={
            "RERANK_ENABLED": True,
            "CRAG_ENABLED": False,
            "QUERY_REWRITE_ENABLED": False,
            "MULTI_QUERY_ENABLED": False,
        },
    ),
    "full": Variant(
        name="full",
        description="全量：混合检索 + 重排 + Multi-Query + CRAG",
        overrides={
            "RERANK_ENABLED": True,
            "CRAG_ENABLED": True,
            "QUERY_REWRITE_ENABLED": True,
            "MULTI_QUERY_ENABLED": True,
        },
    ),
}

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


@contextlib.contextmanager
def apply_overrides(overrides: Dict[str, Any]) -> Iterable[None]:
    """临时覆盖 settings（评估结束后恢复）"""
    original = {key: getattr(settings, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(settings, key, value)
        yield
    finally:
        for key, value in original.items():
            setattr(settings, key, value)


def load_dataset(path: str = DATASET_PATH) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    logger.info("加载评测数据集 %s：%s 条", path, len(data))
    return data


def run_rag_single(question: str) -> Dict[str, Any]:
    """调用 RAG 状态机，返回答案与检索上下文"""
    result = get_graph().invoke(initial_state(question), {"recursion_limit": 50})
    contexts = [source.get("text", "") for source in result.get("sources", [])]
    return {
        "answer": result.get("answer", ""),
        "contexts": contexts,
        "used_fallback": bool(result.get("used_fallback")),
        "degraded": bool(result.get("degraded")),
        "iterations": int(result.get("iteration", 0) or 0),
    }


def build_ragas_components():
    """把项目自己的 LLM / Embedding 包装成 RAGAS 组件"""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from app.core.embeddings import get_embedding_service
    from app.core.llm import get_llm_service

    llm = LangchainLLMWrapper(get_llm_service().get_llm())
    embeddings = LangchainEmbeddingsWrapper(get_embedding_service().embeddings)
    return llm, embeddings


def evaluate_variant(variant: Variant, dataset, metrics, llm, embeddings) -> Dict[str, Any]:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate

    samples = []
    logger.info("=== 变体 %s：%s ===", variant.name, variant.description)

    with apply_overrides(variant.overrides):
        for index, item in enumerate(dataset, start=1):
            question = item["question"]
            logger.info("[%s/%s] %s", index, len(dataset), question[:40])
            rag = run_rag_single(question)
            if rag["used_fallback"]:
                logger.warning("问题「%s」没有检索到上下文，指标会被拉低", question[:30])
            samples.append(
                SingleTurnSample(
                    user_input=question,
                    response=rag["answer"],
                    retrieved_contexts=rag["contexts"] or [""],
                    reference=item.get("ground_truth", ""),
                )
            )

    result = evaluate(
        EvaluationDataset(samples=samples), metrics=metrics, llm=llm, embeddings=embeddings
    )
    frame = result.to_pandas()
    scores = {
        name: (float(frame[name].mean()) if name in frame.columns else None)
        for name in METRIC_NAMES
    }
    return {"scores": scores, "rows": frame.to_dict(orient="records")}


def print_comparison(results: Dict[str, Dict[str, Any]]) -> None:
    header = f"{'variant':<16}" + "".join(f"{name:>20}" for name in METRIC_NAMES)
    print("\n" + "=" * len(header))
    print("RAGAS 消融对比（越高越好）")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, payload in results.items():
        row = f"{name:<16}"
        for metric in METRIC_NAMES:
            value = payload["scores"].get(metric)
            row += f"{(f'{value:.4f}' if value is not None else 'n/a'):>20}"
        print(row)
    print("=" * len(header) + "\n")


def ingest_corpus(corpus_dir: str) -> None:
    """把语料灌入向量库（让评估可复现）"""
    from app.core.thread_pool import thread_pool
    from app.core.vector_store import get_vector_store_service
    from app.services.document_service import get_document_service

    if not get_vector_store_service().connect():
        raise SystemExit("无法连接 Milvus，请先启动 Milvus 或检查 MILVUS_HOST/MILVUS_PORT")

    if not os.path.isdir(corpus_dir):
        raise SystemExit(f"语料目录不存在: {corpus_dir}")

    files = sorted(
        os.path.join(corpus_dir, name)
        for name in os.listdir(corpus_dir)
        if os.path.isfile(os.path.join(corpus_dir, name))
    )
    if not files:
        raise SystemExit(f"语料目录为空: {corpus_dir}")

    service = get_document_service()

    async def _ingest() -> None:
        for path in files:
            with open(path, "rb") as handle:
                content = handle.read()
            result = await service.process_document(content, os.path.basename(path))
            logger.info("已入库 %s（%s chunks）", path, result["total_chunks"])

    asyncio.run(_ingest())
    thread_pool.shutdown(wait=False)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RAGAS 评估（支持消融对比）")
    parser.add_argument("--dataset", default=DATASET_PATH, help="评测集 JSON 路径")
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS.keys()),
        help=f"要跑的变体，逗号分隔。可选: {', '.join(VARIANTS.keys())}",
    )
    parser.add_argument("--ingest", action="store_true", help="评估前先把语料灌入知识库")
    parser.add_argument("--corpus", default="./docs", help="语料目录（配合 --ingest）")
    parser.add_argument("--report", default=REPORT_PATH, help="报告输出路径")
    args = parser.parse_args(argv)

    setup_logging()

    if not settings.DEEPSEEK_API_KEY or not settings.DASHSCOPE_API_KEY:
        print("缺少 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY，评估需要它们作为裁判模型。")
        return 2

    if args.ingest:
        ingest_corpus(args.corpus)

    dataset = load_dataset(args.dataset)
    selected = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in selected if name not in VARIANTS]
    if unknown:
        print(f"未知变体: {unknown}，可选: {list(VARIANTS.keys())}")
        return 2

    from app.core.vector_store import get_vector_store_service

    if not get_vector_store_service().connect():
        print("无法连接 Milvus，评估中止。")
        return 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        llm, embeddings = build_ragas_components()

        results: Dict[str, Dict[str, Any]] = {}
        for name in selected:
            results[name] = evaluate_variant(VARIANTS[name], dataset, metrics, llm, embeddings)

    print_comparison(results)

    payload = {
        "dataset": args.dataset,
        "model": settings.DEEPSEEK_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "variants": {
            name: {
                "description": VARIANTS[name].description,
                "overrides": VARIANTS[name].overrides,
                "scores": results[name]["scores"],
                "rows": results[name]["rows"],
            }
            for name in selected
        },
    }
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    logger.info("评估报告已导出: %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
