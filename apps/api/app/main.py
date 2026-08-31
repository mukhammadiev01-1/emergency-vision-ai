"""FastAPI Application Entry Point."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apps.api.app.config import settings
from apps.api.app.api.routes import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifespan context."""
    # Startup initialization
    yield
    # Graceful shutdown cleanup
    from apps.api.app.api.dependencies import get_stream_service
    get_stream_service().cleanup_all()


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    debug=settings.API_DEBUG,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register aggregated routes
app.include_router(api_router, prefix=settings.API_PREFIX)

# Also expose top-level health and events endpoints for client convenience
from apps.api.app.api.routes.health import router as health_router
from apps.api.app.api.routes.events import router as events_router
app.include_router(health_router)
app.include_router(events_router, prefix="/events", tags=["Events"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "apps.api.app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_DEBUG,
    )
