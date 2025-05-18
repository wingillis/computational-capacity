import logging
import datetime
import polars as pl
from pathlib import Path

logger = logging.getLogger(__name__)


class SavingBuffer:

    def __init__(self, buffer_size: int = 2000, folder_path: str | None = None, file_base_name: str | None = None):
        self.buffer = []
        self.buffer_size = buffer_size
        self.folder_path = folder_path
        self.index = 0
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_base_name = f"{file_base_name}_{now}" if file_base_name else f"run_{now}"
        Path(folder_path).mkdir(parents=True, exist_ok=True)

    def add(self, data: dict):
        if len(self.buffer) >= self.buffer_size:
            save_df(self.buffer, self.folder_path, index=self.index, file_base_name=self.file_base_name)
            self.index += 1
            self.buffer.clear()
        self.buffer.append(data)

    def __del__(self):
        if len(self.buffer) > 0:
            save_df(self.buffer, self.folder_path, index=self.index, file_base_name=self.file_base_name)


def dict_to_df(data: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(data)


def save_df(
    data: list[dict] | pl.DataFrame,
    folder_path: str | None = None,
    index: int | None = None,
    file_base_name: str | None = None,
):
    if folder_path is None:
        folder_path = Path.cwd()
    else:
        folder_path = Path(folder_path)

    if file_base_name is None:
        name = "computational_capacity_run"
    else:
        name = file_base_name

    if isinstance(data, (list, tuple)):
        data = dict_to_df(data)
    if index is None:
        file_path = folder_path / f"{name}.parquet"
    else:
        file_path = folder_path / f"{name}_{index:05d}.parquet"
    try:
        data.write_parquet(file_path, compression_level=5)
    except pl.exceptions.ComputeError as e:
        print(data.glimpse())
    logger.info(f"Saved data to {file_path}")
