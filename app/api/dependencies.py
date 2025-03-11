from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from app.config import settings
from app.utils.logging import logger

# 
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_api_key(api_key: str = Depends(oauth2_scheme)):
    """Verify API key for secured endpoints."""
    if not api_key:
        logger.warning("API request missing API key")
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Missing API key",
                    "type": "auth_error",
                    "code": "missing_api_key"
                }
            }
        )
    
    # In development mode, allow default key
    if settings.DEBUG and api_key == "debug":
        return api_key
        
    # Production validation
    if api_key != settings.API_KEY:
        logger.warning("Invalid API key used")
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key",
                    "type": "auth_error",
                    "code": "invalid_api_key"
                }
            }
        )
        
    return api_key