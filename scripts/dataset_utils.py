import re
import torch as t
from datasets import Dataset

#Multiple instance learning
class MiSonGynyDataset(Dataset):
    def __init__(self, songs, labels, ids, tokenizer, tokenizer_params: dict):
        self.songs = songs
        self.labels = labels
        self.tokenizer = tokenizer
        self.ids = ids
        self.tokenizer_params = tokenizer_params

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        versos = self.songs[idx]
        label = self.labels[idx]

        inputs = self.tokenizer(
            versos,
            return_tensors="pt",
            **self.tokenizer_params
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "label": t.tensor(label, dtype=t.float),
            "id": self.ids[idx]
        }

def split_songs_into_verses(song_list, chunk_size=4):
    """
    1. Get all sentences per song
    2. Clean each sentence by removing 'as low as', sentences with a single token, parenthesis, and squared brackets 
    3. Remove repeated sentences (chorus, and so on)
    4. Split them by chunks (according to chunk size)
    5. Get only first 20 chunks per song maximum
    """
    songs = []

    for idx, song in enumerate(song_list):
        sentences = re.sub(r'(?<=[a-z])(?=[A-Z])', '\n', song)
        sentences = sentences.split("\n")

        sentences = [ clean_sentence(s) for s in sentences ]
        sentences = [ s for s in sentences if 'as low as $' not in s and ' ' in s and len(s.strip()) >= 1 ]
        sentences = list(dict.fromkeys(sentences))
        
        verses = [ " ".join(sentences[i:i + chunk_size]).strip() for i in range(0, len(sentences), chunk_size)]
        verses = [ clean_sentence(v) for v in verses if len(v) > 0 ]
        songs.append(verses[:20])

    return songs

def clean_sentence(sentence):
    """
    Remove brackets, parenthesis and extra spaces
    """
    sentence = re.sub(r'\[.*\]', '', sentence)
    sentence = re.sub(r'\(.*\)', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence.strip()