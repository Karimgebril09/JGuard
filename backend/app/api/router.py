from fastapi import APIRouter

from backend.app.api.routes.chat import router as chat_router
from backend.app.api.routes.eval import router as eval_router
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.redteam import router as redteam_router
from backend.app.api.routes.system import router as system_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(system_router)
api_router.include_router(chat_router)
api_router.include_router(redteam_router)
api_router.include_router(eval_router)
