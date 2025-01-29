from flask import Flask, request, jsonify, render_template
from bs4 import BeautifulSoup, NavigableString
import random
import re
from typing import List, Tuple
import spacy
from spacy.language import Language
from spacy.tokens import Doc
import time

app = Flask(__name__)

# URL backlink tunggal
BACKLINK_URL = "https://depokwebsite.com/"

# Load model multilingual
try:
    nlp = spacy.load('xx_ent_wiki_sm')
    # Tambahkan sentencizer ke pipeline
    if 'sentencizer' not in nlp.pipe_names:
        nlp.add_pipe('sentencizer')
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    raise

def get_valid_anchor_texts(text: str, skip_words: set) -> List[str]:
    """
    Mendapatkan kata-kata yang valid untuk anchor text dari teks biasa
    """
    # Split teks menjadi kata-kata
    words = text.split()
    
    # Filter kata-kata yang valid
    valid_words = [
        word for word in words 
        if len(word) > 4 
        and word.lower() not in skip_words 
        and not any(char in word for char in '.,!?:;()')
        and not word.isnumeric()  # Skip angka
    ]
    
    # Dapatkan frasa 2-3 kata yang bermakna
    phrases = []
    for i in range(len(words) - 2):
        # Frasa 2 kata
        phrase2 = ' '.join(words[i:i+2])
        if (len(phrase2) > 8 and 
            not any(w.lower() in skip_words for w in words[i:i+2]) and
            not any(w.isnumeric() for w in words[i:i+2])):
            phrases.append(phrase2)
        
        # Frasa 3 kata
        phrase3 = ' '.join(words[i:i+3])
        if (len(phrase3) > 12 and 
            not any(w.lower() in skip_words for w in words[i:i+3]) and
            not any(w.isnumeric() for w in words[i:i+3])):
            phrases.append(phrase3)
    
    return list(set(valid_words + phrases))

def is_good_context(text: str) -> bool:
    """
    Memeriksa apakah konteks logis untuk backlink
    """
    # Skip kalimat pendek
    if len(text.split()) < 5:
        return False
        
    # Skip kalimat tanya atau seruan
    if any(char in text for char in '?!'):
        return False
        
    # Harus memiliki kata-kata yang menunjukkan definisi/penjelasan
    context_words = {
        'adalah', 'merupakan', 'memiliki', 'terdiri', 'menghasilkan',
        'berfungsi', 'digunakan', 'menyediakan', 'menghadirkan',
        'menciptakan', 'mengembangkan', 'menawarkan', 'terdapat',
        'tersedia', 'sebagai', 'termasuk', 'menjadi', 'dapat', 'bisa'
    }
    return True  # Longgarkan validasi konteks

def get_best_sentence_for_anchor(text: str, anchor: str) -> str:
    """
    Mendapatkan kalimat terbaik untuk penempatan anchor text
    """
    sentences = text.split('.')
    best_sentence = None
    best_score = 0
    
    for sentence in sentences:
        if anchor.lower() in sentence.lower():
            score = 0
            # Panjang kalimat ideal (15-30 kata)
            words = sentence.split()
            if 15 <= len(words) <= 30:
                score += 2
            
            # Bonus untuk kalimat informatif
            informative_words = {
                'adalah', 'merupakan', 'memiliki', 'memberikan',
                'menghasilkan', 'menyediakan', 'terdiri', 'terdapat'
            }
            if any(word in sentence.lower() for word in informative_words):
                score += 1
            
            # Bonus untuk kalimat dengan kata kunci penting
            important_keywords = {
                'penting', 'utama', 'kunci', 'hasil', 'manfaat',
                'solusi', 'metode', 'sistem', 'teknologi', 'proses'
            }
            if any(word in sentence.lower() for word in important_keywords):
                score += 1
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
    
    return best_sentence

def get_paragraph_score(text: str) -> float:
    """
    Menghitung skor paragraf berdasarkan beberapa faktor
    """
    score = 1.0
    
    # Panjang paragraf (lebih panjang = lebih penting)
    length = len(text.split())
    if length > 100:
        score *= 1.3
    elif length > 50:
        score *= 1.2
    elif length < 20:  # Penalti untuk paragraf terlalu pendek
        score *= 0.5
    
    # Penalti untuk paragraf yang tidak ideal
    if '?' in text:  # Kalimat tanya
        score *= 0.7
    if len(text.split('.')) < 2:  # Terlalu sedikit kalimat
        score *= 0.8
    
    # Bonus untuk paragraf dengan kata kunci penting
    important_keywords = {
        'penting', 'utama', 'kunci', 'hasil', 'manfaat', 'solusi', 
        'metode', 'sistem', 'teknologi', 'proses', 'aplikasi', 
        'layanan', 'fitur', 'fungsi', 'kualitas'
    }
    if any(keyword in text.lower() for keyword in important_keywords):
        score *= 1.2
    
    return score

def distribute_backlinks_strategically(text_nodes: list, max_backlinks: int) -> List[int]:
    """
    Mendistribusikan backlink secara strategis dan stabil
    """
    # Minimal paragraf yang dibutuhkan
    min_paragraphs = 3
    if len(text_nodes) < min_paragraphs:
        return [1] * len(text_nodes)
    
    # Hitung skor untuk setiap text node
    scores = [get_paragraph_score(node.string) for node in text_nodes]
    total_score = sum(scores)
    
    # Distribusi awal
    distribution = [0] * len(text_nodes)
    remaining_links = max_backlinks
    
    # Jamin distribusi di bagian penting artikel
    key_positions = {
        0: 0.2,  # Awal artikel (20% dari total backlink)
        len(text_nodes) // 2: 0.3,  # Tengah artikel (30%)
        -1: 0.2  # Akhir artikel (20%)
    }
    
    # Alokasi untuk posisi kunci
    for pos, percentage in key_positions.items():
        links = int(max_backlinks * percentage)
        if remaining_links > 0 and links > 0:
            distribution[pos] = min(2, links)  # Maksimal 2 link per node
            remaining_links -= distribution[pos]
    
    # Distribusi sisanya berdasarkan skor
    if remaining_links > 0:
        for i in range(len(text_nodes)):
            if i not in key_positions and remaining_links > 0:
                score_ratio = scores[i] / total_score
                links = min(2, int(score_ratio * remaining_links))
                distribution[i] = links
                remaining_links -= links
    
    # Distribusi sisa link jika masih ada
    if remaining_links > 0:
        for i in range(len(distribution)):
            if distribution[i] < 2 and remaining_links > 0:
                distribution[i] += 1
                remaining_links -= 1
    
    return distribution

def add_backlink_to_text(text: str, backlink_url: str, skip_words: set, used_anchors: set) -> Tuple[str, bool]:
    """
    Menambahkan backlink dengan pemilihan konteks yang lebih natural
    """
    # Split teks menjadi kalimat
    sentences = text.split('.')
    valid_sentences = [s for s in sentences if is_good_context(s)]
    
    if not valid_sentences:
        return text, False
    
    # Pilih kalimat terbaik
    best_sentence = max(valid_sentences, key=len)
    words = best_sentence.split()
    
    # Filter kata-kata yang valid
    valid_words = [
        word for word in words 
        if len(word) > 4 
        and word.lower() not in skip_words 
        and not any(char in word for char in '.,!?:;()')
        and word.lower() not in used_anchors
    ]
    
    # Dapatkan frasa 2 kata yang valid
    phrases = []
    for i in range(len(words) - 1):
        phrase = ' '.join(words[i:i+2])
        if (len(phrase) > 8 
            and not any(w.lower() in skip_words for w in words[i:i+2])
            and phrase.lower() not in used_anchors):
            phrases.append(phrase)
    
    # Kombinasikan kandidat
    candidates = phrases + valid_words
    if not candidates:
        return text, False
    
    # Pilih anchor text dari kalimat terbaik
    anchor_text = random.choice(candidates)
    used_anchors.add(anchor_text.lower())
    
    # Buat link dengan format yang konsisten
    link_html = f'''<a href="{backlink_url}" target="_blank" style="text-decoration: none !important;">{anchor_text}</a>'''
    
    # Ganti teks dengan link
    pattern = re.compile(f'\\b{re.escape(anchor_text)}\\b', re.IGNORECASE)
    modified_text = pattern.sub(link_html, text, count=1)
    
    return modified_text, True

def preserve_html_tags(soup):
    """
    Memastikan tag HTML penting tetap terjaga
    """
    # Daftar tag yang harus dipreservasi
    preserved_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
                     'ul', 'ol', 'li', 'strong', 'em', 'b', 'i',
                     'table', 'tr', 'td', 'th']
    
    # Pastikan setiap tag memiliki wrapper yang benar
    for tag in preserved_tags:
        for element in soup.find_all(tag):
            # Jika tag adalah list item, pastikan parent-nya adalah ul/ol
            if tag == 'li':
                if not element.parent or element.parent.name not in ['ul', 'ol']:
                    new_ul = soup.new_tag('ul')
                    element.wrap(new_ul)
            # Pertahankan struktur tag lainnya
            element.name = tag

    return soup

def get_tag_weight(tag_name: str) -> float:
    """
    Memberikan bobot berdasarkan hierarki tag
    """
    weights = {
        'h1': 1.5,
        'h2': 1.3,
        'h3': 1.2,
        'h4': 1.1,
        'h5': 1.0,
        'h6': 0.9,
        'p': 1.0,
        'li': 1.1,
        'td': 0.9,
        'th': 1.0
    }
    return weights.get(tag_name, 1.0)

def should_process_node(node) -> bool:
    """
    Memeriksa apakah sebuah node sebaiknya diproses untuk backlink
    """
    if not isinstance(node, NavigableString):
        return False
        
    # Skip jika parent adalah tag yang tidak seharusnya memiliki link
    if node.parent.name in ['a', 'script', 'style', 'code', 'pre']:
        return False
        
    # Skip jika teks terlalu pendek
    if len(node.string.strip()) < 20:
        return False
        
    return True

def is_good_phrase(phrase: str) -> bool:
    """
    Memeriksa apakah frasa layak untuk backlink
    """
    # Skip frasa pendek
    if len(phrase.split()) < 2:
        return False
    
    # Skip frasa dengan kata-kata yang tidak diinginkan
    bad_words = {'dalam', 'adalah', 'dengan', 'untuk', 'yang', 'dari', 'pada', 
                'akan', 'jika', 'bila', 'ketika', 'karena', 'sehingga', 'namun',
                'tetapi', 'maka', 'lalu', 'kemudian'}
    
    if any(word.lower() in bad_words for word in phrase.split()):
        return False
    
    return True

def get_phrase_score(phrase: str) -> float:
    """
    Menghitung skor untuk sebuah frasa berdasarkan konteks dan makna
    """
    score = 1.0
    words = phrase.split()
    
    # Bonus untuk panjang frasa yang ideal
    if 3 <= len(words) <= 8:  # Perluas range kata
        score *= (1.0 + (len(words) * 0.1))
    
    # Bonus untuk frasa yang memiliki subjek dan predikat
    doc = nlp(phrase)
    has_subject = False
    has_predicate = False
    
    for token in doc:
        if token.dep_ in {'nsubj', 'nsubjpass'}:
            has_subject = True
        if token.pos_ == 'VERB':
            has_predicate = True
    
    if has_subject and has_predicate:
        score *= 1.5
    
    # Bonus untuk frasa yang memiliki kata benda penting
    important_nouns = {
        'sistem', 'teknologi', 'aplikasi', 'metode', 'strategi', 'proses',
        'pengembangan', 'peningkatan', 'manajemen', 'solusi', 'implementasi',
        'optimasi', 'efisiensi', 'produktivitas', 'kualitas', 'performa'
    }
    
    if any(word.lower() in important_nouns for word in words):
        score *= 1.3
    
    return score

def is_meaningful_phrase(phrase: str, sentence: str) -> bool:
    """
    Memeriksa apakah frasa memiliki makna yang logis dalam konteks kalimat
    """
    # Analisis sintaksis
    doc = nlp(sentence)
    phrase_doc = nlp(phrase)
    
    # Periksa koherensi gramatikal
    if len(phrase_doc) > 1:
        has_valid_structure = False
        for token in phrase_doc:
            if token.dep_ in {'ROOT', 'nsubj', 'dobj', 'pobj'}:
                has_valid_structure = True
                break
        if not has_valid_structure:
            return False
    
    # Periksa konteks semantik
    main_verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']
    main_nouns = [token.lemma_ for token in doc if token.pos_ == 'NOUN']
    
    phrase_verbs = [token.lemma_ for token in phrase_doc if token.pos_ == 'VERB']
    phrase_nouns = [token.lemma_ for token in phrase_doc if token.pos_ == 'NOUN']
    
    # Harus ada hubungan kata kerja atau kata benda dengan kalimat utama
    if not (set(phrase_verbs) & set(main_verbs) or set(phrase_nouns) & set(main_nouns)):
        return True  # Longgarkan untuk frasa yang independen tapi bermakna
    
    return True

def is_good_sentence(text: str) -> bool:
    """
    Memeriksa apakah kalimat cukup panjang dan berkualitas untuk backlink
    """
    # Minimal 80 karakter (sedikit dilonggarkan)
    if len(text) < 80:
        return False
    
    # Minimal 7 kata
    if len(text.split()) < 7:
        return False
    
    # Skip kalimat tanya atau seruan
    if any(char in text for char in '?!'):
        return False
    
    # Harus memiliki kata-kata informatif (diperluas)
    informative_words = {
        'merupakan', 'memiliki', 'menghasilkan', 'berfungsi', 'digunakan',
        'menyediakan', 'menghadirkan', 'menciptakan', 'mengembangkan',
        'menawarkan', 'terdapat', 'tersedia', 'termasuk', 'menjadi',
        'dapat', 'bisa', 'mampu', 'memberikan', 'menggunakan', 'melakukan',
        'sistem', 'proses', 'teknologi', 'aplikasi', 'layanan', 'produk',
        'solusi', 'metode', 'cara', 'strategi'
    }
    
    return True  # Longgarkan validasi kata informatif

def add_backlinks(article: str, backlink_url: str, target_backlinks: int, track_backlink) -> str:
    try:
        if not article.strip():
            return article
            
        # Parse HTML dengan parser yang lebih ketat
        soup = BeautifulSoup(article, 'html.parser', multi_valued_attributes=None)
        
        # Dapatkan text nodes dengan mempertahankan struktur asli
        text_nodes = []
        valid_tags = {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'div', 'span'}
        
        for element in soup.descendants:
            if (isinstance(element, NavigableString) 
                and element.parent 
                and element.parent.name in valid_tags
                and element.parent.name not in ['a', 'script', 'style', 'code', 'pre']
                and len(element.strip()) > 80):
                text_nodes.append(element)
        
        if not text_nodes:
            return article
            
        used_anchors = set()
        added_links = 0
        attempts = 0
        max_attempts = len(text_nodes) * 3
        
        while added_links < target_backlinks and attempts < max_attempts:
            for node_index, node in enumerate(text_nodes):
                if added_links >= target_backlinks:
                    break
                    
                if not node.parent:
                    continue
                    
                # Pastikan parent tag tetap terjaga
                parent_tag = node.parent
                parent_attrs = dict(parent_tag.attrs)
                
                text = node.string.strip()
                doc = nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]
                long_sentences = [s for s in sentences if is_good_sentence(s)]
                
                if not long_sentences:
                    continue
                
                candidates = []
                for sentence in long_sentences:
                    words = sentence.split()
                    
                    for length in range(8, 1, -1):
                        if len(candidates) >= 5:
                            break
                        for i in range(len(words) - (length - 1)):
                            phrase = ' '.join(words[i:i+length])
                            if (len(phrase) > 7
                                and phrase.lower() not in used_anchors
                                and not any(char in phrase for char in '.,!?:;()')
                                and is_good_phrase(phrase)):
                                candidates.append((phrase, sentence))
                
                if candidates and node.parent:
                    phrase, sentence = max(candidates, key=lambda x: len(x[0]))
                    used_anchors.add(phrase.lower())
                    
                    link_html = f'<a href="{backlink_url}" target="_blank" style="color: #2563eb !important; text-decoration: none !important;">{phrase}</a>'
                    
                    try:
                        pattern = re.compile(f'\\b{re.escape(phrase)}\\b')
                        new_sentence = pattern.sub(link_html, sentence, count=1)
                        new_text = text.replace(sentence, new_sentence)
                        
                        if new_text != text and node.parent:
                            # Buat fragment dengan mempertahankan tag parent
                            fragment = BeautifulSoup(new_text, 'html.parser')
                            
                            # Jika parent bukan paragraph, wrap konten dalam tag parent yang sama
                            if parent_tag.name != 'p':
                                new_parent = soup.new_tag(parent_tag.name, **parent_attrs)
                                for content in fragment.contents:
                                    new_parent.append(content)
                                node.replace_with(new_parent)
                            else:
                                node.replace_with(*fragment.contents)
                                
                            added_links += 1
                            print(f"Added link in sentence: {phrase}")
                            track_backlink(phrase)
                    except Exception as e:
                        print(f"Error replacing text: {e}")
                        continue
            
            attempts += 1
            if attempts % len(text_nodes) == 0:
                used_anchors.clear()
        
        # Return HTML dengan format asli yang terjaga
        return str(soup)
        
    except Exception as e:
        print(f"Error in add_backlinks: {e}")
        return article

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_article():
    try:
        article = request.form.get('article', "")
        target_backlinks = max(50, int(request.form.get('num_backlinks', 50)))
        
        # Track backlinks yang ditambahkan
        backlinks_added = []
        
        def track_backlink(phrase):
            backlinks_added.append(phrase)
        
        # Proses artikel dengan tracking
        processed_article = add_backlinks(article, BACKLINK_URL, target_backlinks, track_backlink)
        
        return jsonify({
            'processed_article': processed_article,
            'total_backlinks': len(backlinks_added),
            'target_backlinks': target_backlinks,
            'backlinks': backlinks_added,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'processed_article': article
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
