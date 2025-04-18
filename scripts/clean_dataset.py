import re
import torch as t
from datasets import Dataset

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
    
def clean_verse(verse):
    sentences = verse.split('\n')
    new_verse = []
    new_verse = " ".join([ sentence for sentence in sentences if sentence not in new_verse ])
    new_verse = re.sub(r'\[.*\]', '', new_verse)
    new_verse = re.sub(r'\(.*\)', '', new_verse)
    new_verse = re.sub(r'\s+', ' ', new_verse)
    return new_verse.strip()

def split_songs_into_verses(song_list):
    songs = []
    for idx, song in enumerate(song_list):
        verses = song.split('\n\n')
        verses = [ clean_verse(v) for v in verses ]
        verses = [ v for v in verses if len(v) != 0 and v != ' ' ]
        songs.append(verses)
    return songs