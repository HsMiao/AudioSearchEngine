import os
import json
import numpy as np
import logging
import faiss
from sentence_transformers import SentenceTransformer

from flask import Flask, render_template, request

from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import Term
from whoosh.analysis import StemmingAnalyzer

from together import Together

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_search_index(index_dir):
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)

    schema = Schema(
        title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        playlist=TEXT(stored=True),
        name=TEXT(stored=True),
        link=ID(stored=True),
        score=TEXT(stored=True),
        path=ID(stored=True, unique=True),
        id=ID(stored=True, unique=True)
    )
    ix = create_in(index_dir, schema)
    writer = ix.writer()

    for item in data:
        id = item.get('id', '') - 1
        title = item.get('subtitle', 'No title')
        if title is None:
            title = 'No title'
        content = item.get('joke_text', '')
        playlist = item.get('playlist', '')
        name = item.get('audio_name', '')
        link = item.get('url', '')
        score = item.get('score', '')
        path = item.get('clip_file', '')
        writer.add_document(
            id=str(id),
            title=title,
            content=content,
            playlist=playlist,
            name=name,
            link=link,
            path=path,
            score=str(score)
        )
    writer.commit()
    print(f"Total documents indexed: {ix.doc_count()}")

def extract_data(data_list):
    return [(data['id'], data['subtitle'], data['joke_text'], data['playlist']) for data in data_list]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('search.html')

@app.route('/search', methods=['GET'])
def search():
    query_text = request.args.get('query', '')
    category = request.args.get('category', 'all')
    filter = request.args.get('filter', 'all') 
    try:
        page = int(request.args.get('page', 1))
    except ValueError:
        page = 1 

    results_per_page = 10

    if query_text == '':
        results_list = sorted(data, key=lambda x: x["score"], reverse=True)
    else:
        ix = open_dir("indexdir")
        if category == 'content':
            q_vec = model.encode(query_text, normalize_embeddings=True).astype('float32')
            D, I = vec_index.search(np.array([q_vec]), k=50)
            ids = I[0]
        else:
            q_vec = model.encode(query_text, normalize_embeddings=True).astype('float32')
            D, I = vec_index.search(np.array([q_vec]), k=100)
            with ix.searcher() as searcher:
                if category == 'all':
                    qp = MultifieldParser(["title", "content", "playlist"], schema=ix.schema)
                    query = qp.parse(query_text)
                elif category == 'link':
                    query = Term("link", query_text)
                else:
                    qp = QueryParser(category, schema=ix.schema)
                    query = qp.parse(query_text)
                
                results = searcher.search(query, limit=100)
                ids = []
                for hit in results:
                    ids.append(hit['id'])
            if category == 'all':
                ids = [int(i) for i in ids if int(i) in I[0]]
            else:
                ids = [int(i) for i in ids]
            if filter == 'Top-tier humor':
                results_list = [data[i] for i in ids if data[i]['score'] == '3']
            elif filter == 'Genuinely funny':
                results_list = [data[i] for i in ids if data[i]['score'] >= '2']
            elif filter == 'Slightly amusing':
                results_list = [data[i] for i in ids if data[i]['score'] >= '1']
            else:
                results_list = [data[i] for i in ids]

    total_pages = int(min(np.ceil(len(results_list) / results_per_page), 10))
    results_list = extract_data(results_list)[(page - 1) * results_per_page : page * results_per_page]
        
    return render_template('results.html', query=query_text, results=results_list, page=page, total_pages=total_pages, category=category, filter=filter)

@app.route('/similar', methods=['GET'])
def similar():
    id = request.args.get('id', '')
    vec = vec_index.reconstruct(int(id))
    D, I = vec_index.search(np.array([vec]), k=10)
    results = [data[i] for i in I[0]]
    results = extract_data(results)
    return render_template('results.html', query='', results=results, page=1, total_pages=1, category='all', filter='all')

@app.route('/expansion', methods=['GET'])
def expansion():
    query_text = request.args.get('query', '')
    client = Together()

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {
                "role": "user",
                "content": f"""I am using a search engine. Here is my query: {query_text}. 
                What do you suggest that I add to my query to get better results? Only return the query expansion."""
            }
        ]
    )
    return response.choices[0].message.content

@app.route('/view/<id>', methods=['GET'])
def view_document(id):
    query = request.args.get('query', '')
    category = request.args.get('category', 'all')
    filter = request.args.get('filter', 'all')
    ix = open_dir("indexdir")
    with ix.searcher() as searcher:
        doc = searcher.document(id=str(id))
        print(doc['link'])
        print(doc['path'])
        if doc:
            return render_template('document.html', 
                                   id=doc['id'],
                                   title=doc['title'], 
                                   content=doc['content'],
                                   playlist=doc['playlist'],
                                   name=doc['name'],
                                   link=doc['link'],
                                   path=doc['path'],
                                   query=query,
                                   category=category,
                                   filter=filter,)
        else:
            return 'Document not found', 404

if __name__ == '__main__':
    with open("jokes_metadata.json", 'r') as f:
        data = json.load(f)
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    dim = model.get_sentence_embedding_dimension()

    ## Example vector database search
    vec_index = faiss.read_index("jokes_index.faiss")
    try: 
        open_dir("indexdir")
    except Exception:
        logging.info("Index not found. Creating index...")
        create_search_index("indexdir")  
    app.run(debug=True)
