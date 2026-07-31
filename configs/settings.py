"""
Configuration Loader

Author: Ishita Dutta
Project: UIFAF
"""

from pathlib import Path
import yaml

from .schema import (
    ProjectConfig,
    ImageConfig,
    SettingsConfig,
)


class Settings:

    _instance = None

    def __new__(cls):

        if cls._instance is None:

            cls._instance = super().__new__(cls)

            cls._instance._load()

        return cls._instance

    def _load(self):

        config_path = (
            Path(__file__).parent / "config.yaml"
        )

        with open(config_path, "r", encoding="utf-8") as file:

            config = yaml.safe_load(file)

        self.settings = SettingsConfig(

            project=ProjectConfig(
                **config["project"]
            ),

            image=ImageConfig(
                **config["image"]
            ),

            bit_depth_mapping=config["bit_depth_mapping"],
        )

    def __getattr__(self, item):
        return getattr(self.settings, item)


settings = Settings()