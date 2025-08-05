import spacy

# Lade deutsches Sprachmodell
nlp = spacy.load("de_core_news_sm")  # ggf. vorher installieren mit: python -m spacy download de_core_news_sm

def extract_key_entities(query):
    doc = nlp(query)

    # Listen für mögliche relevante Entitäten
    keywords = []

    for token in doc:
        # Nomen und wirtschaftsrelevante Begriffe wie "Umsatz"
        if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"] and token.is_stop == False:
            keywords.append(token.text)

    # Duplikate entfernen, falls nötig
    keywords = ' '.join(keywords)

    return keywords

def delete_stopwords(query):
    doc = nlp(query)
    keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
    keywords = ' '.join(keywords)
    return keywords