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

def split_songs_into_verses(song_list, window_size=4, num_verses=20, offset=2, sentence_split_token=" "):
    """
    1. Get all sentences per song
    2. Clean each sentence by removing 'as low as', sentences with a single token, parenthesis, and squared brackets 
    3. Remove repeated sentences (chorus, and so on)
    4. Split them by chunks (according to chunk size)
    5. Get only first k chunks per song maximum

    if window_size is greater than the actual number of sentences, it will be ignored
    """
    songs = []

    con_dict = get_contractions_dict(song_list)

    for song in song_list:
        sentences = song.split("\n")

        sentences = [ clean_sentence(s) for s in sentences ]
        sentences = [ s for s in sentences if 'as low as $' not in s and len(s.strip().split(' ')) > 1 ]
        sentences = [ clean_contractions(s, con_dict) for s in sentences ]

        verses = [ sentence_split_token.join(sentences[i:i + window_size]).strip() for i in range(0, len(sentences), offset) ]
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
    sentence = re.sub(r' \,', ',', sentence)
    sentence = re.sub(r"(P|p)a\'l", r"\1ara el ", sentence)
    sentence = re.sub(r"(P|p)a’", r"\1ara ", sentence)
    sentence = re.sub(r"(P|p)a\'", r"\1ara ", sentence)
    sentence = re.sub(r"(P|p)\'", r"\1ara ", sentence)
    sentence = re.sub(r"(N|n)a\'", r"\1ada ", sentence)  
    sentence = re.sub(r"\'(T|t)amo", r"estamos ", sentence)
    sentence = re.sub(r"(D)i\'que", r"\1isque ", sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence.strip()

def get_contractions(song_list):
    contractions = []
    con_num = {}

    for song in song_list:
        sentences = song.split("\n")

        sentences = [ clean_sentence(s) for s in sentences ]
        sentences = [ s for s in sentences if 'as low as $' not in s and len(s.strip().split(' ')) > 1 ]
        for sentence in sentences:
            contractions.extend( [ s for s in sentence.split(" ") if "'" in s ] )
    
    for c in contractions:
        if c not in con_num:
            con_num[c] = 0
        con_num[c] += 1
        
    return list(set(contractions)), con_num

def get_contractions_dict(sentences):
    contractions, _ = get_contractions(sentences)
    print(contractions)
    contractions_dict = {}

    for c in contractions:     
        if (len(c) > 1 and c[-2] == "'" and not c[-1].isalnum()) or c[-1] == "'":
            contractions_dict[c] = c.replace("'", "s")
        elif "a'o" in c:
            contractions_dict[c] = c.replace("a'o", "ado")
        elif "i'o" in c:
            contractions_dict[c] = c.replace("i'o", "ido")
        elif "í'o" in c:
            contractions_dict[c] = c.replace("í'o", "ído")

    return { **contractions_dict, **additional_contractions }

def clean_contractions(sentence, con_dict):
    words = sentence.split(" ")
    return " ".join([ con_dict[word] if word in con_dict else word for word in words ])

additional_contractions = {
    "'e": 'de',
    "vo'a": 'voy a',
    "'tá": 'está',
    "to'a": 'toda',
    "'toy": 'estoy',
    "'tán": 'están',
    "to's": 'todos',
    "to'as": 'todas',
    "'Toy": 'Estoy',
    "'Tá": 'Está',
    "'el": 'del',
    "To'as": 'Todas',
    "¿oí'te,": '¿oíste',
    "'Tán": 'Están',
    "'onde": 'donde',
    "'tás": 'estás',
    "vamo'a": 'vamos a',
    "Vo'a": 'Voy a',
    "'taba": 'estaba',
    "Vamo'a": 'Vamos a',
    "To'a": 'Toda',
    "e'toy": 'estoy',
    "e'ta": 'esta',
    "de'o": 'dedo',
    "'Tás": 'Estás',
    "'tar": 'estar',
    "'tate": 'estate',
    "ere'—Tú": 'eres tú',
    "escondi'as": 'escondidas',
    "co'quillita": 'cosquillita',
    "prendí'a": 'prendida',
    "ha'ta": 'hasta',
    "di'que": 'disque',
    "nue'tro": 'nuestro',
    "'ta": 'esta',
    "e'te": 'este',
    "prendi'a": 'prendida',
    "mi'mo": 'mismo',
    "E'to": 'Esto',
    "To's": 'Todos',
    "to'a,": 'toda,',
    "cr'eme": 'créeme',
    "'Toy-'Toy-'Toy": 'Estoy',
    "supie'n": 'supieron',
    "vamo'alla,": 'vamos allá',
    "to'os": 'todos',
    "gu'ta": 'gusta',
    "'tén": 'estén',
    "'apá": 'papá',
    "Que'l": 'Que él',
    "¿'tás": '¿estás',
    "'trás": 'detrás',
    "escondi'a": 'escondida',
    "'Taba": 'Estaba',
    "m'importa": 'me importa',
    "'esbarato": '',
    "'tan": 'estan',
    "pega'ito": 'pegadito',
    "'tas": 'estas',
    "e'tá": 'está',
    "bu'cando": 'buscando',
    "Ha'ta": 'Hasta',
    "comi'a": 'comida',
    "'taban": 'estaban',
    "no'más": 'nada más',
    "¿oí'te": '¿oíste',
    "pelu'os": 'peludos',
    "Vamono'a": 'Vamonos a',
    "comprometí'a": 'comprometida',
    "comprometí'a,": 'comprometida,',
    "ve'-eh-eh": 'ves',
    "ve'-eh-eh,": 'ves,',
    "'Tuve": 'Estuve',
    "Comi'a": 'Comida',
    "Aparesi'te": 'Aparesiste',
    "diji'te": 'dijiste',
    "'Tábamos": 'Estábamos',
    "Loco'-loco'-locos": 'locos',
    "so'aba": 'sobaba',
    "camara'a!": 'camarada!',
    "parti'a,": 'partida',
    "de'de": 'desde',
    "fundi'a": 'fundida',
    "u'ted": 'usted',
    "mete'lo,": 'metelo',
    "ve'lo": 'velo',
    "oí'te,": 'oíste',
    "salie'n": 'salieron',
    "pal'abrigo": 'para el abrigo',
    "corre'le": 'correle',
    "co'quillita,": 'cosquillita',
    "'Tan": 'Estan',
    "e'una": 'es una',
    "de'pués": 'despues',
    "ca'ile": 'caele',
    "dese'perado": 'desesperado',
    "hiju'eputa": 'hijo de puta',
    "escondí'a": 'escondida',
    "E'cuchando": 'Escuchando',
    "tooo's": 'todos',
    "'tá,": 'está',
    "queda'n": 'quedan',
    "'To-'Toy": 'Estoy',
    "'Tective": 'Detective',
    "fundie'n": 'fundieron',
    "'Ca-ca-cause": 'Because',
    "pue'to": 'puesto',
    "'té": 'esté',
    "hace'lo": 'hacedlo',
    "que'a": 'queda',
    "pue'en": 'pueden',
    "To-To-To'a": 'Toda',
    "¿'ta": '¿esta',
    'Bacatrane\'",': '',
    "'tabas": 'estabas',
    "'Tate": 'Estate',
    "homb'e!": 'hombre!',
    "a'í": 'así',
    "pue'o": 'puedo',
    'demente\'",': 'dementes',
    "mojaí'ta,": 'mojadita',
    "'guama": 'caguama',
    "tiene'?,": 'tienes?',
    "cliente'...": 'clientes',
    "'tando": 'estando',
    "'Tonces,": 'Entonces',
    "'ca": 'aca',
    "'ebajo": 'debajo',
    "oi'te": 'oiste',
    "'Ámonos": 'Vámonos',
    "¡'Ámonos,": '¡Vámonos',
    "lamo'-Bailamo'-Ba—,": 'Bailamos',
    "Bailamo'-Bailamo'-Bailamo'-Ba—,": 'Bailamos',
    "pare'co": 'parezco',
    "trai'te": 'trajiste',
    "to'itos": 'todos',
    "tri'te": 'triste',
    "actitu'-tu’": 'actitud',
    "tembla'era": 'tembladera',
    "fre'cura": 'frescura',
    "desnu'itos": 'desnudos',
    "prendí'a,": 'prendida',
    "rendi'a": 'rendida',
    "¿oí'te?": '¿oíste?',
    "calla'íta,": 'calladita',
    "vestí'a": 'vestida',
    "corri'ó": 'corrido',
    "so'amos": 'entonces vamos',
    'perdío\'?"': 'perdidos?',
    'hacemo\'?"': 'hacemos',
    "Para'lante,": 'para adelante',
    "to'ito": 'todo',
    "calla'ito": 'calladito',
    "'l": 'el',
    "'Taban": 'estaban',
    'vo\'?",': 'vos?',
    "vo'?,": 'vos?,',
    "re'pirá": 'respira',
    "de'pedí": 'despedí',
    "pue'a": 'pueda',
    "compra'te": 'compraste',
    "to'itas": 'todas',
    "'Pérate": 'espérate',
    "vi'ta": 'vista',
    "resi'tirme": 'resistirme',
    "mi'mo?": 'mismo',
    "e'quina": 'esquina',
    "pega'íto": 'pegadito',
    "'entro": 'dentro',
    "encendio's": 'encendidos',
    'cruzamo\'?"': 'cruzamos?',
    "callaí'ta": 'callada',
    "¿oi'te": '¿oíste',
    "mi'jo!": 'mi hijo!',
    "mi'jo": 'mi hijo',
    "mama'te": 'mamaste',
    '"Adió\'",': 'Adiós,'}