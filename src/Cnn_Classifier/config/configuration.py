from src.Cnn_Classifier.constant import *

from Cnn_Classifier.entity.config_entity import DataIngestionConfig , EvaluationConfig
from Cnn_Classifier.utils.common import read_yaml , create_directories , save_json

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            train_data_dir=config.train_data_dir,
            val_data_dir=config.val_data_dir
        )
        return data_ingestion_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/my_model_83%.h5",
            training_data="artifacts/data_ingestion/bone_marrow_cell_dataset",
            mlflow_uri="https://dagshub.com/samarmubarakofficial/Bone-Marrow-Cells-Classification-CNN.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config
