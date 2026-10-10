from app.core.config import get_settings

settings = get_settings()
print(f"Starting {settings.app_name} in {settings.app_env} environment.")