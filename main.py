import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os
import re
import hashlib
from collections import Counter
import pickle
import atexit
from tempfile import mkdtemp
from shutil import rmtree
from deepmultilingualpunctuation import PunctuationModel
import openai
from nltk.tokenize import PunktSentenceTokenizer
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import time
import tiktoken
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import nltk
nltk.download('punkt', quiet=True)
# Ensure that 'punkt' tokenizer data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)




# Initialize the punctuation model
punctuation_model = PunctuationModel()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Using a lightweight model for embeddings


# OpenAI API 
openai.api_key = "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"
# Adjusted text splitter for chunking large text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)




# Create a temporary directory to store cache files
temp_dir = mkdtemp()
# In-memory cache for URL data
url_cache = {}
# Define the FAISS cache as a global variable
faiss_cache = {} 




# Register a cleanup function to delete temporary files when the program exits
def cleanup_temp_dir():
    #print("Running cleanup_temp_dir")
    if os.path.exists(temp_dir):
        rmtree(temp_dir)
    print(f"Temporary directory {temp_dir} has been cleaned up.")
atexit.register(cleanup_temp_dir)




# Helper function to create a unique file path for a given URL
def get_cache_file_path(url):
    #print("Running get_cache_file_path")
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    return os.path.join(temp_dir, f"{url_hash}.pkl")




# Function to check if URL data is already cached in memory or file system
def is_url_cached(url):
    #print("Running is_url_cached")
    if url in url_cache:
        return True
    cache_file_path = get_cache_file_path(url)
    return os.path.exists(cache_file_path)




# Modified function to prevent loading empty or invalid cached data
def load_cached_data(url):
    #print("Running load_cached_data")
    if url in url_cache:
        data = url_cache[url]
        # Ensure cached data is not empty
        if data.get("content", "").strip():
            return data
        else:
            print(f"Cached data for {url} is invalid (empty).")
            return None
    
    cache_file_path = get_cache_file_path(url)
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "rb") as f:
            data = pickle.load(f)
            # Ensure loaded cached data is valid
            if data.get("content", "").strip():
                url_cache[url] = data
                return data
            else:
                print(f"Cached file for {url} contains invalid (empty) content.")
                return None
    return None




# Modified cache_data function to avoid caching empty content
def cache_data(url, data):
    #print("Running cache_data")
    
    # Check if the data is not empty before caching
    if data and data.get("content", "").strip():
        url_cache[url] = data
        cache_file_path = get_cache_file_path(url)
        with open(cache_file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Cached content for {url}")
    else:
        print(f"Skipping cache for {url} as the content is empty or invalid.")




# Scraping function (modified to use caching)
def scrape_path(url, paths):
    #print("Running scrape_path")
    
    # Check if URL data is already cached
    if is_url_cached(url):
        print(f"Loading cached data for {url}")
        cached_data = load_cached_data(url)
        if cached_data:
            return cached_data["content"]
    print(f"Start Processing {url}")
    base_content = ""
    seen_content = set()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        base_content = ' '.join(soup.stripped_strings)
        seen_content.add(base_content)
        print(f"Scraped base URL: {url}")
    except requests.RequestException as e:
        print(f"Error loading URL {url}: {e}")
        return ""
    
    # Scraping other paths
    for path in paths:
        full_url = f"{url.rstrip('/')}/{path.lstrip('/')}"
        try:
            response = requests.get(full_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            path_content = ' '.join(soup.stripped_strings)
            if path_content and path_content not in seen_content:
                print(f"Scraping content from Path {full_url}")
                base_content += "\n" + path_content
                seen_content.add(path_content)
            else:
                print(f"Skipping {full_url} as content is already in the data") 
        except requests.RequestException as e:
            print(f"Error loading URL {full_url}: {e}")
    
    # Ensure content is valid (non-empty, non-redundant)
    if base_content.strip() and "0" not in base_content:
        cache_data(url, {"content": base_content})
    else:
        print(f"Skipping cache for {url} as no valid content was scraped.")
    
    return base_content




# Extract emails and phone numbers
def extract_emails_contacts(text):
    # Email extraction remains the same
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    # Improved phone number regex to avoid confusion with postal codes
    phone_pattern = r"(?<!\d)(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})(?!\d)"
    phones = re.findall(phone_pattern, text)
    
    return emails, phones




# Helper function to normalize phone numbers (removing hyphens, spaces, and handling country code)
def normalize_phone_number(phone):
    # Remove non-digit characters except the leading '+' for country codes
    phone = re.sub(r"[^\d+]", "", phone)
    
    # If the phone starts with +91 (India's country code), ensure it has the correct format
    if phone.startswith("+91"):
        return "+91" + phone[-10:]  # Always return the full number with country code
    elif phone.startswith("+"):
        return phone  # Return other international numbers as is
    else:
        return phone[-10:]  # For local numbers, return the last 10 digits
    



def handle_contact_query(company_name, website_content):
    # Extract emails and phone numbers from the content
    emails, phones = extract_emails_contacts(website_content)
    
    # Remove duplicates by converting to a set and then back to a list (to maintain order)
    emails = list(set(emails))
    
    # Normalize phone numbers and remove duplicates
    normalized_phones = list(set([normalize_phone_number(phone) for phone in phones]))
    
    # Further deduplicate the entire phone numbers
    unique_phones = []
    seen_numbers = set()  # To track full phone numbers, not just last 10 digits
    
    for phone in normalized_phones:
        if phone not in seen_numbers:  # Deduplicate based on full number
            unique_phones.append(phone)
            seen_numbers.add(phone)
    
    # Instead of filtering by 10 digits, keep all valid phone numbers (including international ones)
    valid_phones = [phone for phone in unique_phones if re.match(r"^\+?\d[\d -]{8,15}\d$", phone)]
    
    # Prepare response based on extracted information
    response = f"Contact information for {company_name}:\n"
    
    if emails:
        response += f"Emails: {', '.join(emails)}\n"
    else:
        response += "No email addresses found.\n"
    
    if valid_phones:
        response += f"Phone Numbers: {', '.join(valid_phones)}\n"
    else:
        response += "No correct number exists.\n"
    
    return response




# Function to save company data into individual files
def save_to_individual_files(company_data):
    #print("Running save_to_individual_files...")
    
    # Define the directory to save files (current directory or specify a path)
    output_dir = "./output"
    #output_dir = os.path.abspath("./output")
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} not found, creating it.")
        os.makedirs(output_dir)
    else:
        print(f"Output directory {output_dir} already exists.")
    
    # Iterate over the company data and write it to individual files
    for company_name, data in company_data.items():
        # Sanitize the company name to avoid issues with special characters in filenames
        filename = os.path.join(output_dir, f"{company_name.replace('/', '_')}.txt")
        
        print(f"Attempting to save file for company: {company_name}")
        print(f"File path: {filename}")
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Company: {company_name}\n")
                f.write("Content:\n")
                f.write(data["content"] + "\n\n")
                f.write("Emails:\n")
                f.write("\n".join(data["emails"]) + "\n\n")
                f.write("Phones:\n")
                f.write("\n".join(data["phones"]) + "\n\n")
                # Add additional information (sector, founding year, classification) if available
                if "sector" in data:
                    f.write(f"Business Sector: {data['sector']}\n\n")
                if "founding_year" in data:
                    f.write(f"Founding Year: {data['founding_year']}\n\n")
                if "classification" in data:
                    f.write(f"Classification: {data['classification']}\n\n")
                
            print(f"Successfully saved file: {filename}")
        
        except Exception as e:
            print(f"Error saving file for {company_name}: {e}")
    
    print(f"Finished saving company data to individual text files in {output_dir}")




# Load documents from the content
def load_documents(content):
    #print("Running load_documents")
    documents = text_splitter.split_documents([Document(page_content=content)])
    return documents


# Function to create a hash of a chunk
def hash_chunk(chunk):
    #print("Running hash_chunk")
    return hashlib.md5(chunk.encode('utf-8')).hexdigest()




# Function to normalize query by removing spaces and special characters
def normalize_query(query):
    #print("Running normalize_query")
    return re.sub(r'\W+', '', query.lower())




def extract_company_names_from_urls(urls):
    company_names = []
    common_subdomains = ["www", "in", "us", "uk", "ca"]  # Common subdomains
    
    for url in urls:
        # Split the URL to get the domain name parts
        domain_parts = url.split("//")[-1].split("/")[0].split(".")
        
        # Handle URLs starting with common subdomains like 'www', 'in', 'us'
        if domain_parts[0] in common_subdomains and len(domain_parts) > 1:
            domain_name = domain_parts[1]  # Take the second part as the domain name
        elif len(domain_parts) > 2 and domain_parts[0].isdigit():
            # Handle cases where the first part is a number (to avoid invalid names)
            domain_name = domain_parts[1]
        else:
            domain_name = domain_parts[0]  # For normal cases, take the first part
        
        # Normalize the domain name (convert to lowercase, replace hyphens with spaces)
        company_name = domain_name.lower().replace('-', ' ')
        company_names.append(company_name)
    
    return company_names




# Match normalized query with company names extracted from URLs
def match_company_with_query(company_name, normalized_query):
    #print("Running match_company_with_query")
    
    # Normalize the company name (remove spaces, hyphens, make lowercase)
    normalized_company_name = re.sub(r'\W+', '', company_name.lower())
    
    # Check if the normalized company name is part of the normalized query
    return normalized_company_name in normalized_query




# Extract company names from the query based on known URLs
def extract_company_names_from_query(query, urls):
    #print("Running extract_company_names_from_query")
    
    # Normalize the query
    normalized_query = normalize_query(query)
    
    # Extract company names from URLs and normalize them
    company_names = extract_company_names_from_urls(urls)
    
    companies_in_query = []
    for company_name in company_names:
        # Use the match_company_with_query for normalized matching
        if match_company_with_query(company_name, normalized_query):
            companies_in_query.append(company_name)
    return companies_in_query




# Function to load the existing FAISS index (if available)
def load_faiss_index(file_path='faiss_index.pkl'):
    try:
        with open(file_path, 'rb') as f:
            faiss_data = pickle.load(f)
        print("FAISS index loaded successfully.")
        return faiss_data['index'], faiss_data['documents']
    except FileNotFoundError:
        print("FAISS index not found. Please create it first.")
        return None, None




# Function to generate embeddings and store in FAISS (in-memory only, no file saving)
def create_faiss_index(documents):
    print("Running create_faiss_index to create in-memory FAISS index.")
    
    # Generate embeddings for each document chunk
    embeddings = embedding_model.encode([doc.page_content for doc in documents])
    # Define the FAISS index (indexing method can vary based on use case)
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Using L2 distance for similarity search
    index.add(np.array(embeddings))  # Add embeddings to the index
    # Store the original documents along with their embeddings for later retrieval
    faiss_data = {
        'index': index,
        'documents': documents,
        'embeddings': embeddings
    }
    print("FAISS index created in-memory successfully.")
    return index, documents




# Function to search FAISS for the most relevant chunks based on query
def search_faiss(query, index, documents, top_k = 5):
    # Convert query to embedding
    query_embedding = embedding_model.encode([query])
    # Search the index for top K similar embeddings
    distances, indices = index.search(np.array(query_embedding), top_k)
    # Retrieve the corresponding documents
    results = [documents[i] for i in indices[0]]
    return results




# Function to create and store FAISS index in-memory without saving to file (only if not exists)
def create_and_store_faiss_index(url, website_content):
    print(f"Checking if FAISS index already exists for URL: {url}")
    
    # Check if the URL index is already in the cache
    url_hash = hashlib.md5(url.encode()).hexdigest()
    if url_hash in url_cache:
        print(f"FAISS index already exists in-memory for {url}. Skipping creation.")
        return url_cache[url_hash]['index'], url_cache[url_hash]['documents']
    
    # Create a Document object from the website content
    print(f"Splitting the content into chunks using the text splitter for URL: {url}")
    documents = text_splitter.split_documents([Document(page_content=website_content, metadata={"url": url})])
    # Print debug information about chunking
    print(f"Total chunks created for URL {url}: {len(documents)}")  # Print the number of chunks
    # for i, doc in enumerate(documents[:3]):  # Print first 3 chunks as a sample (avoid printing all if too many)
    #     print(f"Chunk {i + 1}: {doc.page_content[:200]}...")  # Print the first 200 characters of each chunk for verification
    # Generate and store FAISS index for the current URL's content (in-memory)
    index, documents = create_faiss_index(documents)  # Use the modified in-memory function
    
    # Store in the in-memory cache
    url_cache[url_hash] = {'index': index, 'documents': documents}
    print(f"FAISS index created and stored in-memory for {url}.")
    
    return index, documents  # Return in-memory index and documents instead of saving




# Function to load the specific FAISS index for a given URL (in-memory support only)
def load_faiss_index_for_url(url):
    print(f"Attempting to load FAISS index for URL: {url}")
    
    # Instead of loading from a file, check if the index exists in-memory
    url_hash = hashlib.md5(url.encode()).hexdigest()
    if url_hash in url_cache:
        print(f"Loading in-memory FAISS index for URL: {url}")
        return url_cache[url_hash]['index'], url_cache[url_hash]['documents']
    else:
        print(f"FAISS index for URL {url} not found in-memory.")
        return None, None
    



# Function to count tokens using the specified model
def count_tokens(text, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))




def match_query_with_sub_chunks(query, sub_chunks, top_k):
    """
    Match the query with sub-chunks using IndexFlatL2 (no training required).
    """
    print(f"Matching query with {len(sub_chunks)} sub-chunks to find the top {top_k} relevant sub-chunks.")
    
    # Create sub-chunk embeddings
    sub_chunk_embeddings = embedding_model.encode(sub_chunks)
    # Create a FAISS index using IndexFlatL2 (no training required)
    sub_chunk_index = faiss.IndexFlatL2(sub_chunk_embeddings.shape[1])  # Dimension of the embeddings
    # Directly add embeddings to the index without training
    sub_chunk_index.add(np.array(sub_chunk_embeddings))
    print(f"Sub-chunk embeddings added to IndexFlatL2.")
    # Convert query to embedding and search for relevant sub-chunks
    query_embedding = embedding_model.encode([query])
    distances, indices = sub_chunk_index.search(np.array(query_embedding), top_k)  # Top `top_k` most relevant sub-chunks
    # Retrieve and return the corresponding sub-chunks
    relevant_sub_chunks = [sub_chunks[i] for i in indices[0]]
    print(f"Found {len(relevant_sub_chunks)} relevant sub-chunks for the query.")
    return relevant_sub_chunks




def create_sub_chunks_optimized(chunk, sub_chunk_size=1000, max_sub_chunks=5):
    """
    Create sub-chunks with optimizations:
    1. Use smaller sub-chunk size.
    2. Limit the number of sub-chunks created.
    """
    # Create smaller sub-chunks to reduce number of chunks processed
    sub_chunks = [chunk[i:i+sub_chunk_size] for i in range(0, len(chunk), sub_chunk_size)]
    # Limit the number of sub-chunks to process
    if len(sub_chunks) > max_sub_chunks:
        sub_chunks = sub_chunks[:max_sub_chunks]
    print(f"Created {len(sub_chunks)} optimized sub-chunks.")
    return sub_chunks




'''
Function with no sitemap scraping
def extract_paths_from_sitemap(url):
    """
    Fetches and parses the sitemap XML to extract all paths.
    Args:
        url (str): The base URL of the website.
    Returns:
        list: A list of paths extracted from the sitemap.
    """
    sitemap_url = urljoin(url, "/sitemap.xml")
    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        
        # Check if the response content is not empty
        if not response.content.strip():
            print(f"Empty content received from the sitemap URL: {sitemap_url}")
            return []
        # Attempt to parse the XML content of the sitemap
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as parse_error:
            print(f"Failed to parse XML from {sitemap_url}: {parse_error}")
            print(f"Response content: {response.content[:200]}")  # Print first 200 characters of content
            return []
        # Define namespace for parsing
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        # Extract all <loc> tags from the sitemap
        paths = []
        for loc in root.findall('.//ns:loc', namespaces=namespace):
            loc_url = loc.text.strip()
            # Get the path part of the URL
            if loc_url.startswith(url):
                path = loc_url.replace(url, "")
                paths.append(path)
        return paths
    except requests.RequestException as e:
        print(f"Failed to fetch sitemap from {sitemap_url}: {e}")
        return []
'''




def extract_paths_from_sitemap(url):
    """
    Fetches and parses the sitemap XML to extract all paths, including nested sub-sitemaps.
    Args:
        url (str): The base URL of the website.
    Returns:
        list: A list of paths extracted from the sitemap and sub-sitemaps.
    """
    sitemap_url = urljoin(url, "/sitemap.xml")  # Construct the URL to the main sitemap.xml
    all_paths = []
    try:
        # Request the main sitemap
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        
        # Check if the response content is not empty
        if not response.content.strip():
            print(f"Empty content received from the sitemap URL: {sitemap_url}")
            return []
        # Parse the main sitemap
        root = ET.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        # Collect all sub-sitemap URLs
        sub_sitemap_urls = [loc.text.strip() for loc in root.findall('.//ns:sitemap/ns:loc', namespaces=namespace)]
        # Fetch and parse each sub-sitemap
        for sub_sitemap_url in sub_sitemap_urls:
            try:
                sub_response = requests.get(sub_sitemap_url, timeout=10)
                sub_response.raise_for_status()
                
                # Parse each sub-sitemap's content
                sub_root = ET.fromstring(sub_response.content)
                sub_paths = [loc.text.strip().replace(url, "") for loc in sub_root.findall('.//ns:url/ns:loc', namespaces=namespace)]
                all_paths.extend(sub_paths)  # Add extracted paths to the overall list
                print(f"Extracted {len(sub_paths)} paths from sub-sitemap: {sub_sitemap_url}")
            except requests.RequestException as sub_e:
                print(f"Failed to fetch sub-sitemap {sub_sitemap_url}: {sub_e}")
    except requests.RequestException as e:
        print(f"Failed to fetch sitemap from {sitemap_url}: {e}")
        return []
    return all_paths




# Define Strategic and Financial Paths
STRATEGIC_PATHS = [
    "/products",
    "/home",
    "/about",
    "/about-us",
    "/services",
    # "/about",
    # "/about-us",
    # "/contact",
    # "/contact-us",
    # "/careers",
    # "/testimonials",
    # "/pricing",
    # "/shop"
]




FINANCIAL_PATHS = [
    "/portfolio",
    "/approach",
    "/criteria",
    "/investment-focus",
    "/investment-criteria",
    "/about",
    "/about-us",
    "/partnerships",
    "/investing",
    "/news",
    "/strategies",
    "/strategy",
    "/companies",
    "/team",
    "/people",
    "/contact",
    "/contact-us",
    "/portfolio",
    "/investment-strategy",     
    "/investment-critieria-approach",
    "/investment-parameters/",
    "/portfolio_companies",
    "/investment-profile",
    "/firm-overview.html"
]




def ask(queries, urls, paths=None, scraping_strategy="Sitemap"):
    print("Running ask")

    if isinstance(queries, str):
        english_tokenizer = PunktSentenceTokenizer()
        queries = english_tokenizer.tokenize(queries)

    responses = []
    company_data = {}  # Dictionary to hold company data for saving

    for url_index, url in enumerate(urls, start=1):
        print(f"\nProcessing URL {url_index}/{len(urls)}: {url}")
        website_content = ""

        # If paths is None, determine paths based on the strategy
        if scraping_strategy == "Sitemap":
            paths = extract_paths_from_sitemap(url)

            if paths:
                print(f"Successfully extracted {len(paths)} paths from sitemap for URL: {url}")
            else:
                print(f"Failed to extract paths from sitemap or sitemap is empty for URL: {url}. Using Strategic Paths as fallback.")
                paths = STRATEGIC_PATHS  # Use Strategic Paths as fallback if Sitemap is not available or empty

        elif scraping_strategy == "Strategic Paths":
            paths = STRATEGIC_PATHS
            print(f"Using Strategic Paths for URL: {url}")

        elif scraping_strategy == "Financial Paths":
            paths = FINANCIAL_PATHS
            print(f"Using Financial Paths for URL: {url}")
        

        # Scrape or load from cache
        if is_url_cached(url):
            print(f"Loading cached data for {url}")
            cached_data = load_cached_data(url)
            if cached_data:
                website_content = cached_data["content"]
            else:
                print(f"Cached data for {url} is invalid or empty, re-scraping.")
                website_content = scrape_path(url, paths)
        else:
            print(f"Scraping content from: {url}")
            website_content = scrape_path(url, paths)


        # Calculate the token count for the content
        content_token_count = count_tokens(website_content)
        print(f"Total tokens for URL {url}: {content_token_count}")


        # Skip the URL if there are no valid tokens in the content
        if content_token_count == 0:
            print(f"Skipping URL {url} due to empty content or scraping error.")
            responses.append(f"URL: {url} was skipped due to empty content.")
            continue


        # Check if content is too large
        top_k = 5
        top_k = min(top_k, 2) if content_token_count > 10000 else top_k

        # First, attempt to load the FAISS index for the URL
        print(f"Loading FAISS index for {url}...")
        index, documents = load_faiss_index_for_url(url)


        # If the index is not found, create and store it
        if not index:
            print(f"FAISS index for {url} not found, creating a new one...")
            index, documents = create_and_store_faiss_index(url, website_content)


        # If the index is still None after attempting to create it, skip this URL
        if not index:
            print(f"FAISS index could not be created or loaded for {url}. Skipping.")
            continue


        # Placeholder for extracted emails and phone numbers
        emails, phones = extract_emails_contacts(website_content)
        for query_index, query in enumerate(queries, start=1):
            print(f"\nProcessing Query {query_index}/{len(queries)} for URL {url_index}: {query}")
            print(f"Searching for relevant documents in the FAISS index of {url} for Query {query_index}...")
            relevant_docs = search_faiss(query, index, documents, top_k=top_k)
            print(f"Found {len(relevant_docs)} relevant documents for the query in {url}.")


            # Join all content from relevant docs
            relevant_content = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Add this line below the above line to count and print the number of chunks used for the query
            selected_chunk_count = len(relevant_content.split('\n\n'))
            print(f"Selected {selected_chunk_count} chunks for URL {url}, Query {query_index} to send to GPT.")
            
            # Add this line below the above line
            relevant_content_token_count = count_tokens(relevant_content)
            print(f"Relevant content token count for URL {url}, Query {query_index}: {relevant_content_token_count}")
            
            
            # Step 1: If still too long, split the chunk into smaller optimized sub-chunks
            if count_tokens(relevant_content + query) > 14000:  #  limit to identify a large chunk
                print(f"Content for URL {url}, Query {query_index} is too long, creating optimized sub-chunks.")
                sub_chunks = create_sub_chunks_optimized(relevant_content, sub_chunk_size=1000, max_sub_chunks=5)
            
                # Step 2: Match the query with sub-chunks and find the most relevant sub-chunks
                print(f"Matching query {query_index} with sub-chunks to find relevant content.")
                relevant_sub_chunks = match_query_with_sub_chunks(query, sub_chunks, top_k)
            
                # Use the most relevant sub-chunks as context for GPT
                relevant_content = "\n\n".join(relevant_sub_chunks)
            
            # Step 3: If still too long after sub-chunking, skip this query
            total_tokens = count_tokens(relevant_content + query)
            if total_tokens > 16385:  # Maximum token limit for GPT (8000)
                print(f"Skipping query {query_index} for {url} due to excessive token count: {total_tokens} tokens.")
                continue
            print(f"Relevant content prepared for GPT for URL {url_index}, Query {query_index}.")


            # Send to GPT
            print(f"Sending Query {query_index} and context to GPT for response generation for URL {url_index}...")
            gpt_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that provides accurate information based on website content."},
                    {"role": "user", "content": f"Answer the following query with accuracy based on the content of the website:\n{relevant_content}\n\nQuery: {query}"}
                ],
                max_tokens=300,
                temperature=0.2
            )
            
            print(f"Received response from GPT for URL {url_index}, Query {query_index}.")
            result = gpt_response['choices'][0]['message']['content'].strip()
            print(f"GPT Response for URL {url_index}, Query {query_index}: {result}")
            responses.append(f"URL: {url}\nQuery: {query.strip()}\nResponse: {result}")
            
            # Store the company data for saving later
            company_name = url.split("//")[-1].split("/")[0]  # Extract company name from URL
            if company_name not in company_data:
            
                company_data[company_name] = {"content": website_content, "emails": emails, "phones": phones}
    
    # Call the save function to save the company data into individual files
    if company_data:
        print("Saving company data to files...")
        save_to_individual_files(company_data)
    

    if not responses:
        print("No matching data found.")
        return "No matching data found."
    print("Returning final responses for all URLs and queries.")
    return "\n\n".join(responses)
