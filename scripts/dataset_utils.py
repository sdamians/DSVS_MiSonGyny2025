import re
import unicodedata
from datasets import Dataset


def df_to_dataset(df, columns=["lyrics", "id", "label"], text_column="lyrics", contractions_dict=None):
    dataset_dict = { column: df[column].to_numpy() for column in columns if column != text_column }

    df['song'] = df[text_column].apply(clean_characters)
    
    if contractions_dict is None:
        contractions = get_contractions(df['song'])
        contractions = sorted(contractions.items(), key=lambda x: x[1], reverse=True)
        contractions_dict = get_contractions_dict([ x[0] for x in contractions ])

    songs = [ song.split("\n") for song in df['song'].to_numpy() ]

    new_songs = []
    for sentences in songs:
        new_sentences = [ clean_words(sentence, contractions_dict) for sentence in sentences ]
        new_songs.append(new_sentences)

    dataset_dict['songs'] = [ clean_sentences(sentences) for sentences in new_songs ]

    return Dataset.from_dict(dataset_dict), contractions_dict
    
def clean_words(sentence, contractions_dict):
    odd_characters = r"[-—–]|ja(ja)+|\bEy|\bWoh|\boh|\bYeah|m(m)+|la( la)+|\bPrr|\bEh|\bah|\buh|\bei|\bie|\beah|\bYeh|\bO+h|\bUah|\bOah|\baoh|\bParte [0-9]"
    words = sentence.strip().split(" ")
    
    res = []
    for word in words:
        
        if word.lower() in contractions_dict and contractions_dict[word.lower()] == "[LIMPIAR]":
            new_word = ""
        elif word.lower() in contractions_dict and contractions_dict[word.lower()] != '':
            new_word = contractions_dict[word.lower()]
            if word[0].isupper():
                new_word = new_word.capitalize()
        else:
            new_word = word
        
        if len(res) == 0 and new_word != "":
            res.append(new_word)

        elif len(res) > 0 and new_word.lower() != res[-1].lower():
            res.append(new_word)
    
    sentence = " ".join([ word for word in res if re.search(odd_characters, word, flags=re.IGNORECASE) is None ])
    
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    sentence = re.sub(r"\[.*\]|\(.*\)|{.*}|{.*\]|\[.*\)", "", sentence)
    
    sentence = re.sub(r"\s+([,!\?])", r"\1", sentence)
    sentence = re.sub(r"([¿¡])\s+", r"\1", sentence)
    
    sentence = re.sub(r"^,+ ", "", sentence)
    sentence = re.sub(r",+", ",", sentence)
    sentence = re.sub(r"[\(\)\[\]{}]", "", sentence)

    #print(f"clean_words: {sentence}")
    return sentence.strip()

def smart_capitalize(text):
    # Busca la primera letra alfabética (incluyendo letras acentuadas y ñ)
    match = re.search(r'([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ])', text)
    if not match:
        return text  # No hay letra alfabética

    idx = match.start()
    return text[:idx] + text[idx].upper() + text[idx+1:]

def clean_sentences(sentences):
    pattern = r'as low as \$|as low as|\bBREAK|\bVerse|\bBridge|\bCORO|\bVerso|Todos los Derechos Reservados'

    sentences = [sentence for sentence in sentences 
                 if not re.search(pattern=pattern, string=sentence, flags=re.IGNORECASE)
                 and ' ' in sentence.strip()]

    sentences = [ re.sub(pattern, "", sentence) for sentence in sentences ]
    sentences = [ smart_capitalize(sentence) for sentence in sentences if re.search(r"[A-Za-z]", sentence) is not None ]
    
    #print(f"clean_sentences: {sentences}")
    return sentences

def clean_characters(text):
    text = re.sub('\n+', '\n', text)
    text = re.sub(r"`|‘|’|´", "'", text)
    text = re.sub(r"''", "", text)
    text = re.sub(r"\xad|\x81|…|_|\u200b", " ", text)
    text = re.sub("[-—–]+", "-", text) 

    text = re.sub(r'([,¡!¿\?\[\]\(\)])', r' \1 ', text)
    
    text = re.sub(r"б|Ã¡|à", "á", text)
    text = re.sub(r"Ã©|è|й", "é", text)
    text = re.sub(r"н|Ã|Ã ­|�|ì", "í", text)
    text = re.sub(r"у|Ã³|í³|ò", "ó", text)
    text = re.sub(r"ъ|Ãº|ù", "ú", text)

    text = re.sub(r"Ã|À", "Á", text)
    text = re.sub(r"Ã‰|È", "É", text)
    text = re.sub(r"Ã|Ì", "Í", text)
    text = re.sub(r"Ã“|Ò", "Ó", text)
    text = re.sub(r"Ãš|Ù", "Ú", text)
    
    text = re.sub(r"Ã‘", "Ñ", text)
    text = re.sub(r"с|Ã±|a±|í±", "ñ", text)
    text = re.sub(r"е", "e", text)
    text = re.sub(r"Â¿|Ї", "¿", text)
    text = re.sub(r"éÂ¼", "üe", text)
    text = re.sub(r"ss ", " ", text)  

    text = re.sub(r"[^A-Za-zÁÉÍÓÚáéíóúüÑñ¿\?!¡,\[\]\(\)\n'0-9- ]", '', text)
    text = re.sub(" +", " ", text)

    return unicodedata.normalize('NFC', text.strip())

def get_contractions(songs):
    vocab = {}
    for song in songs:
        sentences = song.split("\n")
        for sentence in sentences:
            for word in sentence.split(" "):
                if "'" in word:
                    if word.lower() not in vocab:
                        vocab[word.lower()] = 0
                    vocab[word.lower()] += 1    
    return vocab

def get_contractions_dict(contractions):
    contractions_dict = { k: "" for k in contractions }

    contraction_patterns = { "ao'": "ado", "eo'": "edo", "íos'": "idos", "ío'": "ido", "io'": "ido",
                             "a'o": "ado", "i'o": "ido", "í'o": "ído", "e'n": "eron", "i'a": "ida",
                             "a'n": "aron", "ías'": "didas", "in'": "ing", "íta'": "da", "íto'": "do", "pa'": "para ",
                              "p'": "para " }
    
    for cont in contractions_dict:
        for patt in contraction_patterns:
            if patt in cont:
                contractions_dict[cont] = cont.replace(patt, contraction_patterns[patt])

        if cont[-1] == "'":
            contractions_dict[cont] = cont.replace("'", "s")

    return { **contractions_dict, **additional_contractions, **additional_contractions_en }

additional_contractions = {
    "so'amos": "soñamos",
    "intil": "inútil",
    "s": "sé",
    "corazn": "corazón",
    "podr": "podré", 
    "amia'": "la mía",
    "pa'l": "para el",
    "'tá": "está",
    "'toy": "estoy",
    "to'a": "toda",
    "'tán": "están",
    "to'as": "todas",
    "to's": "todos",
    "pa'ca'": "para acá",
    "pa'llá": "para allá",
    "pa'lante": "para adelante",
    "pa'arriba": "para arriba",
    "'tate": "estate",
    "ha'ta": "hasta",
    "e'toy": "estoy", 
    "e'ta": "esta", 
    "de'o": "dedo", 
    "co'quillita": "cosquillas", 
    "prendí'a": "prendida", 
    "'tar": "estar", 
    "ere'-tú": "eres tú", 
    "p'arriba": "para arriba", 
    "e'te": "este", 
    "e'to": "esto", 
    "pa'-pa'-pa'-pa'-pa'-pa'-pa'-": "para", 
    "pa'ti": "para ti", 
    "comprometí'a": "comprometida", 
    "ve'-eh-eh": "ver", 
    "nue'tro": "nuestro", 
    "'tan": "están", 
    "oi'te": "oiste", 
    "pa'ca": "para acá", 
    "'pa": "para", 
    "pa'ra": "para", 
    "pa'dentro": "para adentro", 
    "m's": "mas", 
    "'ámonos": "vámonos", 
    "'taban": "estaban", 
    "pa'desearte": "para desearte", 
    "cr'eme": "créeme", 
    "to'a-ah": "toda", 
    "pa'encima": "para encima", 
    "vamo'alla": "vamos alla", 
    "to'os": "todos", 
    "gu'ta": "gusta", 
    "'tén": "estpen", 
    "'apá": "papá", 
    "que'l": "que el", 
    "ma'i": "mami", 
    "'trás": "atrás", 
    "'a": "'a", 
    "m'importa": "me importa", 
    "'esbarato": "desbarato", 
    "do-do-do-don't": "don't", 
    "na'que": "nada que", 
    "pega'ito": "pegado",
    "'tas": "estas", 
    "pa'fuera": "para fuera", 
    "e'tá": "está", 
    "bu'cando": "buscando", 
    "'tando": "estando", 
    "ponga'rompe": "ponga rompe", 
    "'la": "la", 
    "no'más": "nada más", 
    "to'itos": "todos", 
    "pelu'os": "peludos", 
    "vamono'a": "vámonos a", 
    "'tuve": "estuve", 
    "aparesi'te": "apareciste", 
    "diji'te": "dijiste", 
    "'tábamos": "estábamos", 
    "mi'jo": "mi hijo", 
    "loco'-loco'-locos": "locos", 
    "tie's": "tienes", 
    "so'aba": "soñaba", 
    "camara'a": "camarada",  
    "d's": "d's", 
    "de'de": "desde", 
    "pa'roer": "para roer", 
    "to'el": "todo el", 
    "u'ted": "usted", 
    "mete'lo": "metelo", 
    "ve'lo": "velo", 
    "pal'abrigo": "para el abrigo", 
    "tevo'adejar": "te voy a dejar", 
    "corre'le": "correle", 
    "enca'quillo": "encasquillo",
    "vamo'aversi": "vamos a ver si", 
    "p'al": "para el", 
    "'perate": "ésperate", 
    "e'una": "es una", 
    "so'ar": "soñar", 
    "de'pués": "después", 
    "ca'ile": "caele", 
    "'lin": "lin", 
    "hace'que": "hace que", 
    "dese'perado": "desesperado", 
    "pa'delante": "para adelante", 
    "allarga't": "lárgate", 
    "hiju'eputa": "hijo de puta", 
    "escondí'a": "escondidas", 
    "e'cuchando": "escuchado",
    "tooo's": "todos", 
    "ere'otra": "eres otra", 
    "'to-'toy": "estoy", 
    "ex's": "ex's", 
    "'tective": "detective", 
    "'esbocar": "desbocar", 
    "break'esito": "descanso", 
    "pue'to": "puesto", 
    "'té": "esté", 
    "s'an": "se han", 
    "hace'lo": "hacerlo", 
    "que'a": "queda", 
    "pue'en": "pueden", 
    "to-to-to'a": "toda", 
    "pa'ó": "pasó", 
    "'tabas": "estabamos", 
    "pa'matar": "para matar", 
    "na'y": "nada y", 
    "na'ya": "nada ya", 
    "na'si": "nada si", 
    "voypa''onde": "voy para donde", 
    "carro'enla": "carros en la", 
    "pa'i": "para mi", 
    "homb'e": "hombre", 
    "pa'loque": "para lo que", 
    "a'í": "así", 
    "pue'o": "puedo", 
    "de'os": "dedos", 
    "mojaí'ta": "mojadita", 
    "'guama": "caguama", 
    "foto'l": "foto el", 
    "grande'l": "grande el", 
    "sie'por": "si es por", 
    "adi's": "adiós", 
    "segu'as": "seguías", 
    "'por": "por", 
    "as'y": "así y", 
    "s'lo": "solo", 
    "cacha's": "cachas",
    "'tos": "estos", 
    "'ón": "dónde", 
    "'tonces": "entonces",
    "'inche": "pinche", 
    "po'l": "por el", 
    "'tamo": "estamos",
    "vamo'a": "vamos a",
    "'el": "del",
    "oí'te": "oíste",
    "'tamos": "estamos",
    "'tás": "estás",
    "'taba": "estaba",
    "pa'que": "para que",
    "pa'tras": "para atrás",
    "mi'mo": "mismo",
    "pa'quererte": "para quererte",
    "oa'dónde": "para dónde",
    "'onde": "dónde",
    "'ta": "esta",
    "di'que": "disque",
    "yo'": "[LIMPIA]",
    "voca'": "[LIMPIA]",
    "-ema'": "[LIMPIA]",
    "gua'": "[LIMPIA]",
    "ya'": "[LIMPIA]",
    "ey-yo'": "[LIMPIA]",
    "ra-pa-pa-pa-pai'": "[LIMPIA]",
    "je'": "[LIMPIA]",
    "-da'": "[LIMPIA]",
    "acho'": "[LIMPIA]",
    "pa'": "para",
    "to'": "todo",
    "na'": "nada",
    "vo'": "voy",
    "mu'": "muy",
    "tiguere'": "tigres",
    "toa'": "todas",
    "lao'": "lado",
    "'tamo'": "estamos",
    "chivirika'": "mujeres coquetas",
    "verda'": "verdad",
    "lu'": "luz",
    "mai'": "mami",
    "pal'": "para el",
    "to'-to'-to'-to'": "todo",
    "uste'": "usted",
    "sei'": "sed",
    "pa'tra'": "para atrás",
    "pa'trá'": "para atrás",
    "blone'": "puros",
    "po'": "por",
    "callaíta": "callada",
    "bo'": "bobo",
    "se'": "ser",
    "t'": "te",
    "ticke'": "ticket",
    "chavó'": "chavo",
    "ta'": "estás",
    "frontee'": "afrontes",
    "chabe": "sabes",
    "timide'": "timidez",
    "cu'": "culo",
    "pegaíto'": "pegado",
    "ca'": "cada",
    "killin'": "krillin",
    "verdá": "verdad",
    "pendejá": "pendeja",
    "tamo'": "estamos",
    "feli": "feliz",
    "nomá'": "nada más",
    "o'ite": "oiste",
    "dímele'": "diles",
    "tos'": "todos",
    "guta'": "gusta",
    "bregamo'": "luchamos",
    "cuernu": "cuernos",
    "avestru": "avestruz",
    "'trá'": "atrás",
    "tá'": "está",
    "mela'": "mela",
    "de'pué'": "después",
    "reggaetone'": "reggaetoneros",
    "pali'": "pali",
    "pienna'": "piernas",
    "depre'": "depresión",
    "auri'": "auriculares",
    "a'": "a",
    "cangri'": "cangri",
    "pikete": "piquetes",
    "pode'": "poder",
    "tranqui'": "tranquilo",
    "cora'": "corazón",
    "actitu'": "actitud",
    "actitú": "actitud",
    "herma'": "hermano",
    "o'": "o",
    "mamaíta'": "mamacita",
    "vo'a": "voy a",
    "va-va-va-vamo'": "vamos",
    "to'a'": "todas",
    "est'": "esté",
    "toy'": "estoy",
    "hach'": "hachís",
    "trá'": "atrás",
    "polq'": "porque",
    "iraquí'": "iraquíes",
    "ánge": "ángel",
    "jo-o-oda'": "jodas",
    "que-pa'": "que para",
    "'esnu'": "desnudo",
    "to-toda'": "todas",
    "ju'to'": "justos",
    "realida'": "realidad",
    "photosho'": "photoshop",
    "per'": "puros",
    "alante'": "delante",
    "mykono'": "mykono",
    "lo'tarjetero'": "los tarjeteros",
    "teestreses'": "te estreses",
    "e'tamo'": "estamos",
    "mi'ma'": "mismas",
    "hijuepu": "hijo de puta",
    "madri'": "madrid",
    "q'": "que",
    "'mpezamo'": "empezamos",
    "enamor'": "enamora",
    "llor'": "llora",
    "brind'": "brindo",
    "qu'": "que",
    "as'": "así",
    "s'": "sé",
    "lapi'": "lápiz",
    "relo'": "reloj",
    "manife'temo'": "manifestemos",
    "jue'": "juez",
    "'e": "de",
    "'amos": "vamos", 
    "'e-": "de puta", 
    "erapa'jugar": "era para jugar", 
    "podía'enamorar": "podías enamorar", 
    "pa'elante": "para adelante", 
    "p'delante": "para adelante", 
    "'ca": "acá", 
    "lo'greti": "los greti", 
    "pa''onde": "para dónde", 
    "tiene'unculo": "tiene un culo", 
    "no'-no'-noriel": "Noriel", 
    "mo'-mo'-mosty": "mosty", 
    "'ebajo": "debajo", 
    "pa'gozar": "para gozar", 
    "lamo'-bailamo'-ba-": "bailamos", 
    "bailamo'-bailamo'-bailamo'-bai-": "bailamos", 
    "bailamo'-bailamo'-bailamo'-ba-": "bailamos", 
    "pa'tras": "para atrás", 
    "pare'co": "parezco",
    "trai'te": "trajiste", 
    "tri'te": "triste", 
    "pa'recordar": "para recorda", 
    "pa'cobrar": "para cobrar", 
    "pa'decir": "para decir", 
    "pa'gritarte": "para gritarte", 
    "'cucha": "escucha", 
    "tembla'era": "tembladera", 
    "fre'cura": "frescura", 
    "p'a": "para",
    "desnu'itos": "desnudos", 
    "'esboque": "desboque", 
    "mone'a": "monedas", 
    "calla'íta": "callada", 
    "vestí'a": "vestida", 
    "corri'ó": "corrido", 
    "p'algo": "para algo", 
    "party-pa'l": "fiesta para el", 
    "so'amos": "soñamos", 
    "maicera'que": "maiceras que", 
    "para'lante": "para adelante", 
    "to'ito": "todo", 
    "calla'ito": "callado", 
    "'l": "el", 
    "o'frécome": "[LIMPIAR]", 
    "re'pirá": "respirá", 
    "de'pedí": "despedí", 
    "'fano": "fano", 
    "'se": "ese", 
    "pue'a": "pueda", 
    "compra'te": "compraste", 
    "m'ijo": "mi hijo", 
    "to'itas": "todas", 
    "drage'o": "dragueo", 
    "'pérate": "espérate", 
    "vi'ta": "vista", 
    "resi'tirme": "resistirme", 
    "e'quina": "esquina", 
    "to'l": "todo el", 
    "pega'íto": "pegado",
    "'entro": "dentro", 
    "pa'bajo": "para abajo", 
    "paga'me": "pagarme", 
    "callaí'ta": "callada", 
    "no'mas": "nada más", 
    "'lante": "adelante",
    "'tras": "atrás", 
    "'cer": "hacer", 
    "'migos": "amigos", 
    "'tádemasiadocaliente": "está demasiado caliente", 
    "reggaeton'a": "reggaeton", 
    "gana'te": "ganaste", 
    "mama'te": "mamaste", 
    "saca'me": "sacarme", 
    "rasgo'de": "rasgos de", 
    "pa'ta": "pata",
    "pam'": "[LIMPIAR]",
    "la-pa'": "la para",
    "pa'-pa'-pa'-pa'-pa'-pa'-pa'": "para",
    "ipa'": "iPad",
    "yeru'": "Jesús",
    "que'": "Qué",
    "sinfo'": "Sinfo",
    "maidita'": "malditas",
    "ah-ah-ah-ah-ante'": "Antes",
    "soado": "soñado"
}

additional_contractions_en = {
    "hol'": "hold",
    "bro'": "bro",
    "phillie'": "phillie'",
    "partie'": "parties",
    "gon'": "going to",
    "weekene'": "weekends",
    "leggo'": "leggos",
    "lil'": "little",
    "shot'": "shots",
    "ya'": "you",
    "motherfucka": "motherfucker",
    "trapper'": "trapper",
    "feka'": "fake",
    "go'": "go",
    "n'": "and",
    "lef'": "left",
    "off'": "off",
    "lu'-lu'-lunay": "Lunay",
    "cat'n'bucket": "can't bucket",
    "'ca-ca-cause": "because", 
    "o-o-o-only's": "only's",
    "aingt": "ain't"
}