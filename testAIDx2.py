from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import tiffile as tiff


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
      builder=self,
      features=tfds.features.FeaturesDict({
        'hr': tfds.features.Image(shape=[600, 600, 3]),
        'lr': tfds.features.Image(shape=[300, 300, 3])
      }),
      supervised_keys=("lr", "hr")
      # If there's a common (input, target) tuple from the
      # features, specify them here. They'll be used if
      # `as_supervised=True` in `builder.as_dataset`.
      #supervised_keys=('image', 'label'),  # Set to `None` to disable
      #homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    extracted_path = dl_manager.manual_dir
    return {"test": self._generate_examples(hr_path=extracted_path/'test_AID_tiff', lr_path=extracted_path/'test_AID_x2_tiff')}

  def _generate_examples(self, lr_path, hr_path):
    """Yields examples."""
    for root, _, files in tf.io.gfile.walk(lr_path):
      for file_path in files:
        # Select only tif files.
        if file_path.endswith(".tiff"):
          yield file_path, {
              "lr": tiff.imread(Path(lr_path)/file_path),
              "hr": tiff.imread(Path(hr_path)/file_path),
          }
