"""
Configuration schema for UIFAF.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: str


@dataclass(frozen=True)
class ImageConfig:
    max_file_size_mb: int
    supported_extensions: list[str]
    supported_modes: list[str]


@dataclass(frozen=True)
class SettingsConfig:
    project: ProjectConfig
    image: ImageConfig
    bit_depth_mapping: dict[str, int]