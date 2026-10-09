"""
API v1 路由聚合
"""
from fastapi import APIRouter

from app.api.v1 import documents, chat, health

router = APIRouter(prefix="/api/v1")

# 注册子路由
router.include_router(health.router)
router.include_router(documents.router)
router.include_router(chat.router)