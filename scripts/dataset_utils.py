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

import re 

def split_songs_into_verses(song_list, verse_size=1, num_verses=20, sentence_split_token=" "):
    """
    1. Get all sentences per song
    2. Clean each sentence by removing 'as low as', sentences with a single token, parenthesis, and squared brackets 
    3. Remove repeated sentences (chorus, and so on)
    4. Split them by chunks (according to chunk size)
    5. Get only first k chunks per song maximum

    if verse_size is greater than the actual number of sentences, it will be ignored
    """
    songs = []

    for idx, song in enumerate(song_list):
        sentences = song.split("\n")

        sentences = [ clean_sentence(s) for s in sentences ]
        sentences = [ s for s in sentences if 'as low as $' not in s and ' ' in s and len(s.strip()) >= 1 ]
        
        # Get only the unique sentences preserving the order of appareance
        sentences = get_most_repeated_sentences(sentences)
        sentences = [ s[0] for s in sentences ]
        
        if len(sentences) > verse_size:
            verses = [ sentence_split_token.join(sentences[i:i + verse_size]).strip() for i in range(0, len(sentences), verse_size)]
            verses = [ clean_sentence(v) for v in verses if len(clean_sentence(v)) > 0 ]
            songs.append(verses[:num_verses])
        else:
            verses = [ sentence.strip() for sentence in sentences ]
            songs.append(verses[:num_verses])

    return songs

def get_most_repeated_sentences(sentences_list):
    results = {}
    for sentence in sentences_list:
        if sentence not in results:
            results[sentence] = 0
        results[sentence] += 1

    sorted_results = sorted(results.items(), key=lambda x:x[1], reverse=True)

    return sorted_results

def clean_sentence(sentence):
    """
    Remove brackets, parenthesis and extra spaces
    """
    sentence = re.sub(r'\[.*\]', '', sentence)
    sentence = re.sub(r'\(.*\)', '', sentence)
    # Delete camel case
    sentence = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', sentence)
    sentence = re.sub(r'\,', ', ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence.strip()