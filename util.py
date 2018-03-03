def get_lookup():
  """
  Note that we are maping both Persian and Farsi to the farsilanguage code..
  """
  return {
    "Afar": "aar-Latn",
    "Amharic": "amh-Ethi",
    "Bengali": "ben-Beng",
    "Catalan": "cat-Latn",
    "Cebuano": "ceb-Latn",
    "Mandarin-Simplified": "cmn-Hans",
    "Mandarin-Traditional": "cmn-Hant",
    "Sorani": "ckb-Arab",
    "German": "deu-Latn",
    "English": "eng-Latn",
    "Farsi": "fas-Arab",
    "Persian": "fas-Arab",
    "French": "fra-Latn",
    "Hausa": "hau-Latn",
    "Hindi": "hin-Deva",
    "Hungarian": "hun-Latn",
    "Ilocano": "ilo-Latn",
    "Indonesian": "ind-Latn",
    "Italian": "ita-Latn",
    "Javanese": "jav-Latn",
    "Kazakh-Cyrillic": "kaz-Cyrl",
    "Kazakh-Latin": "kaz-Latn",
    "Kinyarwanda": "kin-Latn",
    "Kyrgyz-Perso-Arabic": "kir-Arab",
    "Kyrgyz-Cyrillic": "kir-Cyrl",
    "Kyrgyz-Latin": "kir-Latn",
    "Kurmanji": "kmr-Latn",
    "Lao": "lao-Laoo",
    "Marathi": "mar-Deva",
    "Burmese": "mya-Mymr",
    "Malay": "msa-Latn",
    "Dutch": "nld-Latn",
    "Chichewa": "nya-Latn",
    "Oromo": "orm-Latn",
    "Punjabi": "pan-Guru",
    "Polish": "pol-Latn",
    "Portuguese": "por-Latn",
    "Russian": "rus-Cyrl",
    "Shona": "sna-Latn",
    "Somali": "som-Latn",
    "Spanish": "spa-Latn",
    "Swahili": "swa-Latn",
    "Swedish": "swe-Latn",
    "Tamil": "tam-Taml",
    "Telugu": "tel-Telu",
    "Tajik": "tgk-Cyrl",
    "Tagalog": "tgl-Latn",
    "Thai": "tha-Thai",
    "Tigrinya": "tir-Ethi",
    "Ukrainian": "ukr-Cyrl",
    "Uyghur-Perso-Arabic)": "uig-Arab",
    "Uzbek-Cyrillic)": "uzb-Cyrl",
    "Uzbek-Latin": "uzb-Latn",
    "Vietnamese": "vie-Latn",
    "Xhosa": "xho-Latn",
    "Yoruba": "yor-Latn",
    "Zulu": "zul-Latn",
  }


def lang2ISO(lang):
  """
  Convert a given language to its ISO code for lookup in epitran.
  """
  # Most of the languages in epitran, per their README
  lookup =  {k.lower(): v for k, v in get_lookup().items()}
  return(lookup[lang.lower()])
