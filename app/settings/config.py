from pathlib import Path

from dynaconf import Dynaconf

settings_dir = Path(__file__).parent
config = Dynaconf(settings_files=[settings_dir / 'settings.yml'],
                  load_dotenv=True,
                  lowercase_read=False)
