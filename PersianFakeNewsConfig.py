class PersianDatasetConfig:
    def __init__(self):
        self.AllData = "./dataset_fakenews/dataset_all.csv"
        self.TrainDataPath = "./dataset_fakenews/train_set.csv"
        self.DevDataPath = "./dataset_fakenews/dev_set.csv"
        self.TestDataPath = "./dataset_fakenews/test_set.csv"
        self.DataPath = {
            "All": self.AllData,
            "Train": self.TrainDataPath,
            "Dev": self.DevDataPath,
            "Test": self.TestDataPath
        }

class PersianFakeNewsDetectionConfig:
    def __init__(self, modelName):
        if modelName == "ParsBert":
            self.ModelName = "HooshvareLab/bert-base-parsbert-uncased"
        if modelName == "ParsBert_Sentiment_Digikla":
            self.ModelName = "HooshvareLab/bert-fa-base-uncased-sentiment-digikala"
        if modelName == "Bert":
            # https://huggingface.co/google-bert/bert-base-multilingual-cased
            self.ModelName = "google-bert/bert-base-multilingual-cased"
        if modelName == "XLM-Base":
            # https://huggingface.co/FacebookAI/xlm-roberta-base
            self.ModelName = "FacebookAI/xlm-roberta-base"