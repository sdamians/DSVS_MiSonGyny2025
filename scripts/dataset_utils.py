import re
import torch as t
import unicodedata
from datasets import Dataset


def split_songs_into_verses(song_list, window_size=4, num_verses=20, offset=2, sentence_split_token=" "):
    
    songs = []

    for song in song_list:
        # 1. Separar oraciones
        #    Si cada canción solo tiene una oración, separarla por cada que se encuentre una mayúscula
        sentences = split_into_sentences(song)

        # 2. Eliminar oraciones 
        #    Que tienen [], (), Verse, BREAK, verso, break, as low as $, o vacías
        sentences = clean_sentences(sentences)

        # 3. Limpieza de caracteres
        sentences = [ clean_characters(sentence) for sentence in sentences]
        #   Limpieza a nivel palabra
        sentences = [ clean_words(sentence) for sentence in sentences ]
        #   Se añaden oraciones únicas
        songs.append(list(dict.fromkeys(sentences)))

    # 4. Limpieza de contracciones
    contractions = get_contractions_dict(songs)
    new_songs = []

    for sentences in songs:
        new_sentences = []
        for sentence in sentences:
            new_sentence = clean_contractions(sentence, contractions)
            new_sentence = remove_sentences_with_contractions(new_sentence)
            if new_sentence != "":
                new_sentences.append(new_sentence)

        verses = [ sentence_split_token.join(new_sentences[i:i + window_size]).strip() 
                  for i in range(0, len(new_sentences), offset) 
                  if i != len(new_sentences) - 1]
        new_songs.append(verses[:num_verses])
        
    return new_songs


def split_into_sentences(song):
    # Camel case split
    song = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', song)
    
    sentences = song.split("\n")
    if len(sentences) < 5:
        new_sentences = []
        for sentence in sentences:
            sentence = re.split(r'(?=[A-Z])', sentence)
            new_sentences.extend(sentence)
        return new_sentences
    
    #print(f"split_into_sentences: {sentences}")
    return sentences

def clean_sentences(sentences):
    pattern = r'as low as \$|\bBREAK|\bVerse|\bBridge|\bCORO|\bVerso'

    sentences = [sentence for sentence in sentences 
                 if not re.search(pattern=pattern, string=sentence, flags=re.IGNORECASE)
                 and ' ' in sentence]

    foreign_char = r"[âãäçêëûüāœ]"

    sentences = [sentence for sentence in sentences
                 if not re.search(pattern=foreign_char, string=sentence)]    
    
    pattern = r"\[.*\]|\(.*\)|{.*}|{.*]"

    sentences = [re.sub(pattern, "", sentence) for sentence in sentences]
    sentences = [ sentence.strip() for sentence in sentences if len(sentence.strip()) > 0 ]
    #print(f"clean_sentences: {sentences}")
    return sentences

def clean_characters(sentence):
    sentence = re.sub(r'[\]\)\*“”\"#$%/@\[\(°=&:;]', '', sentence)
    sentence = re.sub(r'\.+', " ", sentence)
    sentence = re.sub(r"''", "", sentence)
    sentence = re.sub(r"\xad|\x81|…|_|\u200b", " ", sentence)
    sentence = re.sub(r'\,', ', ', sentence)
    sentence = re.sub(r' \,', ',', sentence)
    
    sentence = re.sub(r"б|Ã¡|à", "á", sentence)
    sentence = re.sub(r"Ã©|è", "é", sentence)
    sentence = re.sub(r"н|Ã|Ã ­|�|ì", "í", sentence)
    sentence = re.sub(r"у|Ã³|í³|ò", "ó", sentence)
    sentence = re.sub(r"ъ|Ãº|ù", "ú", sentence)

    sentence = re.sub(r"Ã|À", "Á", sentence)
    sentence = re.sub(r"Ã‰|È", "É", sentence)
    sentence = re.sub(r"Ã|Ì", "Í", sentence)
    sentence = re.sub(r"Ã“|Ò", "Ó", sentence)
    sentence = re.sub(r"Ãš|Ù", "Ú", sentence)
    
    sentence = re.sub(r"Ã‘", "Ñ", sentence)
    sentence = re.sub(r"с|Ã±|a±|í±", "ñ", sentence)
    sentence = re.sub(r"е", "e", sentence)
    sentence = re.sub(r"Â¿", "¿", sentence)
    sentence = re.sub(r"éÂ¼", "üe", sentence)
    sentence = re.sub(r"`|‘|’|´", "'", sentence)
    sentence = re.sub(r"ss", "", sentence)
    #sentence = re.sub(f"([0-9])+\'|\'([0-9]+)|[0-9\.,]+", "número", sentence)

    sentence = re.sub(r"(P|p)a\'l", r"\1ara el ", sentence)
    sentence = re.sub(r"(P|p)a\'", r"\1ara ", sentence)
    sentence = re.sub(r"(P|p)\'", r"\1ara ", sentence)
    sentence = re.sub(r"(N|n)a\'", r"\1ada ", sentence)  
    sentence = re.sub(r"\'(T|t)amo", r"estamos ", sentence)
    sentence = re.sub(r"(D)i\'que", r"\1isque ", sentence)
    sentence = re.sub(r'\s+', ' ', sentence)

    #print(f"clean_characters: {sentence.strip()}")
    return unicodedata.normalize('NFC', sentence.strip())

def clean_words(sentence):
    odd_characters = r"[-—–]|ja(ja)+|\bEy,?|\bWoh,?|\boh,?|\bYeah,?|m(m)+|la( la)+|\bPrr,?|\bEh,?|\bah,?|\buh,?|\bei,?|\bie,?|\beah,?"
    words = sentence.split(" ")
    res = [words[0]]
    for word in words[1:]:
        if re.sub(r"\W+", "", word.lower()) != re.sub(r"\W+", "", res[-1].lower()):
            res.append(word)
    
    sentence = " ".join([ word for word in res if re.search(odd_characters, word, flags=re.IGNORECASE) is None ])
    sentence = re.sub(r'\s+', ' ', sentence)

    #print(f"clean_words: {sentence}")
    return sentence

def get_contractions(songs):
    contractions = {}

    all_words = [ word for sentences in songs 
                     for sentence in sentences 
                     for word in sentence.split(" ")]

    for word in all_words:
        if "'" in word:
            if word not in contractions:
                contractions[word] = 0
            contractions[word] += 1

    return contractions

def get_contractions_dict(sentences):
    contractions = get_contractions(sentences)
    contractions_dict = {}

    for c in list(contractions.keys()):     
        if (len(c) > 1 and c[-2] == "'" and not c[-1].isalnum()) or c[-1] == "'":
            contractions_dict[c] = c.replace("'", "s")
        elif "a'o" in c:
            contractions_dict[c] = c.replace("a'o", "ado")
        elif "i'o" in c:
            contractions_dict[c] = c.replace("i'o", "ido")
        elif "í'o" in c:
            contractions_dict[c] = c.replace("í'o", "ído")
        elif "e'n" in c:
            contractions_dict[c] = c.replace("e'n", "eron")

    return { **contractions_dict, **additional_contractions }

def clean_contractions(sentence, con_dict):
    words = sentence.split(" ")
    return " ".join([ con_dict[word] if word in con_dict else word for word in words ])

def remove_sentences_with_contractions(sentence):
    if "'" in sentence:
        print(f"DELETED: {sentence}")
        return ""
    
    return sentence


additional_contractions = {
    "to'": "todos",
    "mu'": "muy",
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
    '"Adió\'",': 'Adiós,',
    "to'el": "todo el",
    "enca'quillo,": "encasquillo,",
    "mata'n": "matan",
    "ma'i": "mami",
    "ex's": "ex",
    "de'os": "dedos",
    "foto'l": "foto del",
    "adi's": "adiós",
    "m's": "más",
    "qu'": "qué",
    "segu'as": "seguías",
    "s'lo": "solo",
    "as'y": "así y",
    "cacha's": "cachas",
    "'ón": "dónde",
    "po'l": "por el",
    "Shannan's": "Shannan",
    "'pa": "para",
    "mc's": "mc",
    "MC's": "MC",
    "Mc's": "Mc",
    "ma'i": "mami",
    "'esboque": "desboque",
    "Llega'n": "Llegaron",
    "'se": "ese",
    "drage'o": "dragueo",
    "CD's": "CDs",
    "No'mas": "Nada mas",
    "'lante": "adelante",
    "'tras": "atrás",
    "'cer": "hacer",
    "'migos,": "amigos",
    "saca'me": "sacarme",
    "daña'n": "dañan",
    "Kellogg's": "Kellogs",
    "coraçao": "corazón"}