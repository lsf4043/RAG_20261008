"""
当前时间上下文

大模型**没有时钟**：它不知道今天是几号，也不知道“最近/今年/上个月”具体指哪一段。
如果不把当前时间放进提示词，模型只能凭训练数据里的日期瞎猜（或者回答“我无法获取实时时间”）。
所以每次请求都要把“现在”注入 system prompt。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

_WEEKDAYS = "一二三四五六日"


def resolve_timezone(name: Optional[str]):
    """解析时区；无效时退回 UTC+8，保证永远不抛异常"""
    from zoneinfo import ZoneInfo

    if name:
        try:
            return ZoneInfo(name), name
        except Exception:  # noqa: BLE001
            pass
    return timezone(timedelta(hours=8)), "UTC+8"


def now_text(tz_name: Optional[str] = None) -> str:
    """返回如 “2026-10-20 18:03 星期二（Asia/Shanghai）”"""
    if tz_name is None:
        try:
            from app.config import settings

            tz_name = settings.APP_TIMEZONE
        except Exception:  # noqa: BLE001
            tz_name = None

    tz, label = resolve_timezone(tz_name)
    now = datetime.now(tz)
    return f"{now:%Y-%m-%d %H:%M} 星期{_WEEKDAYS[now.weekday()]}（{label}）"


def time_context(tz_name: Optional[str] = None) -> str:
    """拼进 system prompt 的时间块"""
    return (
        "## 当前时间\n"
        f"现在是 {now_text(tz_name)}。\n"
        "用户问“今天/现在/最近/今年/上个月”等相对时间时，一律以此为准；"
        "你具备获取当前时间的能力，不要回答“我无法获取实时时间/我没有时间信息”。"
    )
