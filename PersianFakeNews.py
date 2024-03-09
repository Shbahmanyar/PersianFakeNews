import pandas as pd
import tensorflow as tf
from hazm import Normalizer
from transformers import AutoTokenizer
from PersianFakeNewsConfig import PersianDatasetConfig, PersianFakeNewsDetectionConfig
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class PersianFakeNewsUtility:
    def __init__(self,setting):
        self.setting = setting
        self.dataConfig = PersianDatasetConfig()
        self.modelConfig = PersianFakeNewsDetectionConfig(self.setting["EmbeddingModel"])
        self.normalizer = Normalizer().normalize
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelConfig.ModelName)

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df["CleanText"] = df["Text"].apply(self.normalizer)
        return df

    def create_dataset(self,dataset_type, df, labels:list, batch_size):
        x_train = self.tokenizer(
            df["CleanText"].tolist(),
            padding='max_length',
            truncation=True,
            return_tensors='tf',
            max_length=280
        )

        train_text_data = {
            'input_ids': x_train['input_ids'],
            'attention_masks': x_train['attention_mask']
        }
        
        Labels = {}
        for label in labels:
            Labels[label] = df[label].values
        
        data = tf.data.Dataset.from_tensor_slices((train_text_data, Labels))
        if dataset_type == "Train":
            data = data.shuffle(buffer_size=batch_size*4).batch(batch_size)
        else:
            data = data.batch(batch_size)
        return data
    
    def get_dataset(self, dataset_type, labels, batch_size = 8):
        df = self.load_data(self.dataConfig.DataPath[dataset_type])
        dataset = self.create_dataset(dataset_type, df, labels, batch_size)
        return dataset

    def plot_history(self, history):
        labels = self.setting["Labels"]

        fig1 = plt.figure(figsize=(7, 7))
        if len(labels) > 1:
            for label in labels[:-1]:
                plt.plot(history.history[f"{label}_accuracy"], label=f'{label} Accuracy')
            plt.plot(history.history[f"{labels[-1]}_accuracy"], label='Fake Accuracy')
        else:
            plt.plot(history.history[f"{labels[0]}_accuracy"], label='Fake Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{self.setting["Number"]} {self.setting["Title"]} - Training Accuracy')
        plt.legend()
        plt.show()
        accuracy_path = f"results/{self.setting['Number']}_{self.setting['TaskName']}_{self.setting['Model']}_{self.setting['EmbeddingModel']}_Epochs{self.setting['Epochs']}_Batchs{self.setting['BatchSize']}_Accuracy.png"
        fig1.savefig(accuracy_path, transparent=True)

        # Define figure size
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

        # Plot training accuracy
        if len(labels) > 1:
            for label in labels[:-1]:
                ax1.plot(history.history[f"{label}_accuracy"], label=f'{label} Accuracy')
            ax1.plot(history.history[f"{labels[-1]}_accuracy"], label='Fake Accuracy')
        else:
            ax1.plot(history.history[f"{labels[0]}_accuracy"], label='Fake Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training Accuracy')
        ax1.legend()

        # Plot training loss
        if len(labels) > 1:
            for label in labels[:-1]:
                ax2.plot(history.history[f"{label}_loss"], label=f'{label} Loss')
            ax2.plot(history.history[f"{labels[-1]}_loss"], label='Fake Loss')
        else:
            ax2.plot(history.history[f"{labels[0]}_loss"], label='Fake Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.legend()

        # Plot validation accuracy
        if len(labels) > 1:
            for label in labels[:-1]:
                ax3.plot(history.history[f"{label}_accuracy"], label=f'{label} Accuracy')
            ax3.plot(history.history[f"{labels[-1]}_accuracy"], label='Fake Accuracy')
        else:
            ax3.plot(history.history[f"{labels[0]}_accuracy"], label='Fake Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Validation Accuracy')
        ax3.legend()

        # Plot validation loss
        if len(labels) > 1:
            for label in labels[:-1]:
                ax4.plot(history.history[f"{label}_loss"], label=f'{label} Loss')
            ax4.plot(history.history[f"{labels[-1]}_loss"], label='Fake Loss')
        else:
            ax4.plot(history.history[f"{labels[0]}_loss"], label='Fake Loss')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('Validation Loss')
        ax4.legend()

        # Adjust layout
        fig.tight_layout(pad=4.0)
        fig.suptitle(f'{self.setting["Number"]} {self.setting["Title"]} - Accuracy and Loss')

        # Show plot
        plt.show()

        # Save plot
        accuracy_loss_path = f"results/{self.setting['Number']}_{self.setting['TaskName']}_{self.setting['Model']}_{self.setting['EmbeddingModel']}_Epochs{self.setting['Epochs']}_Batchs{self.setting['BatchSize']}_Accuracy_Loss.png"
        fig.savefig(accuracy_loss_path, transparent=True)

    def generate_predictions(self, model):
        labels = self.setting["Labels"]
        batch_size = self.setting["BatchSize"]
        test_dataset = self.get_dataset("Test", labels, batch_size)
        predictions = model.predict(test_dataset)
        y_pred_fake = np.argmax(predictions[-1], axis=1)
        y_true_fake = []
        for _, labels_batch in self.get_dataset("Test", labels, batch_size):
            y_true_fake.extend(labels_batch["FAKE_label"].numpy())

        classification_rep = classification_report(y_true_fake, y_pred_fake, target_names=["Real", "Fake"])
        print(classification_rep)
        classification_rep_path = f"results/{self.setting['Number']}_{self.setting['TaskName']}_{self.setting['Model']}_{self.setting['EmbeddingModel']}_Epochs{self.setting['Epochs']}_Batchs{self.setting['BatchSize']}_Classification.txt"
        with open(classification_rep_path, "w") as file:
            file.write(classification_rep)
        conf_matrix = confusion_matrix(y_true_fake, y_pred_fake)

        plt.figure(figsize=(7, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='g', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f"{self.setting['Number']} Confusion Matrix {self.setting['EmbeddingModel']}")
        confusion_matrix_path = f"results/{self.setting['Number']}_{self.setting['TaskName']}_{self.setting['Model']}_{self.setting['EmbeddingModel']}_Epochs{self.setting['Epochs']}_Batchs{self.setting['BatchSize']}_ConfusionMatrix.png"
        plt.savefig(confusion_matrix_path, transparent=True)

        accuracy = accuracy_score(y_true_fake, y_pred_fake)
        precision = precision_score(y_true_fake, y_pred_fake)
        recall = recall_score(y_true_fake, y_pred_fake)
        f1 = f1_score(y_true_fake, y_pred_fake)

        # Plot performance metrics
        metrics = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
        values = [accuracy, f1, recall, precision]

        # Define colors suitable for a scientific article
        colors = plt.cm.tab10.colors

        # Plot performance metrics with values annotated on top
        plt.figure(figsize=(7, 7))
        bars = plt.bar(metrics, values, color=colors)
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title(f'Performance Metrics for {self.setting["Title"]}')
        plt.ylim(0, 1)

        # Annotate values on top of bars
        performance_metrics_path = f"results/{self.setting['Number']}_{self.setting['TaskName']}_{self.setting['Model']}_{self.setting['EmbeddingModel']}_Epochs{self.setting['Epochs']}_Batchs{self.setting['BatchSize']}_PerformanceMetrics.png"
        plt.savefig(performance_metrics_path, transparent=True, dpi=300)
    
    def get_cpu_info(self):
        try:
            import platform
            if platform.system() == "Windows":
                return platform.processor()
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f.readlines():
                        if "model name" in line:
                            return line.split(":")[1].strip()
            elif platform.system() == "Darwin":
                return platform.processor()
            else:
                return "Unknown platform"
        except Exception as e:
            return f"Error occurred: {e}"
        
    def get_ram_info(self):
        try:
            import psutil
            ram = psutil.virtual_memory()
            return f"Total RAM: {ram.total / (1024 ** 3):.2f} GB"
        except:
            return "No RAM information available (psutil module not installed)"

    def get_gpu_info(self):
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_info = ""
            for gpu in gpus:
                gpu_info += f"GPU: {gpu.name}, Memory: {gpu.memoryTotal}MB\n"
            return gpu_info.strip()
        except ImportError:
            return "No GPU information available (GPUtil module not installed)"
        
    def get_summary_system(self):
        print("CPU Model:", self.get_cpu_info())
        print("RAM:", self.get_ram_info())
        print("GPUs:")
        print(self.get_gpu_info())