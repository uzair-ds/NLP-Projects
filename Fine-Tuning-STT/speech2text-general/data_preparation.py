import pandas as pd
from datasets import Dataset, Audio, Value

df_train = pd.read_csv("/home/ubuntu/uzair/fine_tuning_stt/train_df.csv")
df_test = pd.read_csv("/home/ubuntu/uzair/fine_tuning_stt/test_df.csv")


# Prepare Training Data 

csv_file_path = "/home/ubuntu/uzair/fine_tuning_stt/train_df.csv"
output_data_dir = "/home/ubuntu/uzair/fine_tuning_stt/train_data_dir"
train_dir = "/home/ubuntu/uzair/fine_tuning_stt/dataset/recordings/train"
test_dir  = "/home/ubuntu/uzair/fine_tuning_stt/dataset/recordings/test"

df = pd.read_csv(csv_file_path)

audio_paths = [f"{train_dir}/{filename}" for filename in df['filename']]
sentences = df['transcription'].tolist()

audio_dataset = Dataset.from_dict({"audio": audio_paths, "sentence": sentences})
audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16_000))
audio_dataset = audio_dataset.cast_column("sentence", Value("string"))
audio_dataset.save_to_disk(output_data_dir)
print('Data preparation done')

# Prepare Testing Data

csv_file_path = "/home/ubuntu/uzair/fine_tuning_stt/test_df.csv"
output_data_dir = "/home/ubuntu/uzair/fine_tuning_stt/test_data_dir"
train_dir = "/home/ubuntu/uzair/fine_tuning_stt/dataset/recordings/train"
test_dir  = "/home/ubuntu/uzair/fine_tuning_stt/dataset/recordings/test"

df = pd.read_csv(csv_file_path)

audio_paths = [f"{test_dir}/{filename}" for filename in df['filename']]
sentences = df['transcription'].tolist()

audio_dataset = Dataset.from_dict({"audio": audio_paths, "sentence": sentences})
audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16_000))
audio_dataset = audio_dataset.cast_column("sentence", Value("string"))
audio_dataset.save_to_disk(output_data_dir)
print('Data preparation done')

