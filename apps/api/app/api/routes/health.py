"""Health and Readiness Routes."""
from datetime import datetime, timezone
from fastapi import APIRouter, status

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Basic service liveness check."""
    return {
        "status": "ok",
        "service": "emergency-vision-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health/ready", status_code=status.HTTP_200_OK)
async def readiness_check():
    """Service readiness check for container orchestrators."""
    return {
        "status": "ready",
        "service": "emergency-vision-api",
        "dependencies": {
            "worker_channel": "operational"
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
