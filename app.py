from flask import Flask, render_template, request, jsonify
import PyPDF2
import nltk
import spacy
import math
import re
import urllib.request
from nltk.stem import WordNetLemmatizer
import bs4 as bs

# Initialize Flask app
app = Flask(__name__)

# Initialize NLP tools
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()

# Download NLTK data if not present
nltk.download('wordnet')

def file_text(filepath):
    with open(filepath, 'r') as f:
        return f.read().replace("\n", '')

def pdfReader(pdf_path):
    with open(pdf_path, 'rb') as pdfFileObject:
        pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
        count = pdfReader.numPages
        start_page = 0
        end_page = count - 1
        return_text = ''
        for i in range(start_page, end_page + 1):
            page = pdfReader.getPage(i)
            return_text += page.extractText()
        return return_text

def wiki_text(url):
    scrap_data = urllib.request.urlopen(url)
    article = scrap_data.read()
    parsed_article = bs.BeautifulSoup(article, 'lxml')
    paragraphs = parsed_article.find_all('p')
    article_text = ""
    for p in paragraphs:
        article_text += p.text
    return re.sub(r'\[[0-9]*\]', '', article_text)

def frequency_matrix(sentences):
    freq_matrix = {}
    stopWords = nlp.Defaults.stop_words
    for sent in sentences:
        freq_table = {}
        words = [word.text.lower() for word in sent if word.text.isalnum()]
        for word in words:
            word = lemmatizer.lemmatize(word)
            if word not in stopWords:
                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1
        freq_matrix[sent[:15]] = freq_table
    return freq_matrix

def compute_tf_matrix(freq_matrix):
    tf_matrix = {}
    for sent, freq_table in freq_matrix.items():
        tf_table = {}
        total_words_in_sentence = len(freq_table)
        for word, count in freq_table.items():
            tf_table[word] = count / total_words_in_sentence
        tf_matrix[sent] = tf_table
    return tf_matrix

def sentences_per_words(freq_matrix):
    sent_per_words = {}
    for sent, f_table in freq_matrix.items():
        for word in f_table.keys():
            if word in sent_per_words:
                sent_per_words[word] += 1
            else:
                sent_per_words[word] = 1
    return sent_per_words

def idf_matrix(freq_matrix, sent_per_words, total_sentences):
    idf_matrix = {}
    for sent, f_table in freq_matrix.items():
        idf_table = {}
        for word in f_table.keys():
            idf_table[word] = math.log10(total_sentences / float(sent_per_words[word]))
        idf_matrix[sent] = idf_table
    return idf_matrix

def tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, tf_value), (word2, idf_value) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(tf_value * idf_value)
        tf_idf_matrix[sent1] = tf_idf_table
    return tf_idf_matrix

def score_sentences(tf_idf_matrix):
    sentenceScore = {}
    for sent, f_table in tf_idf_matrix.items():
        total_tfidf_score_per_sentence = 0
        total_words_in_sentence = len(f_table)
        for word, tf_idf_score in f_table.items():
            total_tfidf_score_per_sentence += tf_idf_score
        if total_words_in_sentence != 0:
            sentenceScore[sent] = total_tfidf_score_per_sentence / total_words_in_sentence
    return sentenceScore

def average_score(sentence_score):
    total_score = 0
    for sent in sentence_score:
        total_score += sentence_score[sent]
    return total_score / len(sentence_score)

def create_summary(sentences, sentence_score, threshold):
    summary = ''
    for sentence in sentences:
        if sentence[:15] in sentence_score and sentence_score[sentence[:15]] >= threshold:
            summary += " " + sentence.text
    return summary

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    input_text_type = request.form.get('input_text_type')
    text = ''
    
    if input_text_type == '1':
        text = request.form.get('text_input')
    elif input_text_type == '2':
        file = request.files.get('file_input')
        if file:
            text = file.read().decode('utf-8')
    elif input_text_type == '3':
        file = request.files.get('file_input')
        if file:
            file.save('temp.pdf')
            text = pdfReader('temp.pdf')
    elif input_text_type == '4':
        wiki_url = request.form.get('wiki_url')
        if wiki_url:
            text = wiki_text(wiki_url)
    
    if text:
        text = nlp(text)
        sentences = list(text.sents)
        num_words_in_original_text = len(text.text.split())
        freq_matrix = frequency_matrix(sentences)
        tf_matrix_result = compute_tf_matrix(freq_matrix)
        num_sent_per_words = sentences_per_words(freq_matrix)
        idf_matrix_result = idf_matrix(freq_matrix, num_sent_per_words, len(sentences))
        tf_idf_matrix_result = tf_idf_matrix(tf_matrix_result, idf_matrix_result)
        sentence_scores = score_sentences(tf_idf_matrix_result)
        threshold = average_score(sentence_scores)
        summary = create_summary(sentences, sentence_scores, 1.3 * threshold)
        num_words_in_summary = len(summary.split())
        
        return jsonify({
            'summary': summary,
            'num_words_in_original_text': num_words_in_original_text,
            'num_words_in_summary': num_words_in_summary
        })
    
    return jsonify({'summary': 'No text provided'})

if __name__ == '__main__':
    app.run(debug=True)
