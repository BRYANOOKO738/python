import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import json

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

# US Presidents dataset witjh key information2
presidents_data = [
    {"number": 1, "name": "George Washington", "years": "1789-1797", "party": "No Party", "vice_president": "John Adams", "preceded_by": "None (first president)", "succeeded_by": "John Adams"},
    {"number": 2, "name": "John Adams", "years": "1797-1801", "party": "Federalist", "vice_president": "Thomas Jefferson", "preceded_by": "George Washington", "succeeded_by": "Thomas Jefferson"},
    {"number": 3, "name": "Thomas Jefferson", "years": "1801-1809", "party": "Democratic-Republican", "vice_president": "Aaron Burr, George Clinton", "preceded_by": "John Adams", "succeeded_by": "James Madison"},
    {"number": 4, "name": "James Madison", "years": "1809-1817", "party": "Democratic-Republican", "vice_president": "George Clinton, Elbridge Gerry", "preceded_by": "Thomas Jefferson", "succeeded_by": "James Monroe"},
    {"number": 5, "name": "James Monroe", "years": "1817-1825", "party": "Democratic-Republican", "vice_president": "Daniel D. Tompkins", "preceded_by": "James Madison", "succeeded_by": "John Quincy Adams"},
    {"number": 6, "name": "John Quincy Adams", "years": "1825-1829", "party": "Democratic-Republican", "vice_president": "John C. Calhoun", "preceded_by": "James Monroe", "succeeded_by": "Andrew Jackson"},
    {"number": 7, "name": "Andrew Jackson", "years": "1829-1837", "party": "Democratic", "vice_president": "John C. Calhoun, Martin Van Buren", "preceded_by": "John Quincy Adams", "succeeded_by": "Martin Van Buren"},
    {"number": 8, "name": "Martin Van Buren", "years": "1837-1841", "party": "Democratic", "vice_president": "Richard Mentor Johnson", "preceded_by": "Andrew Jackson", "succeeded_by": "William Henry Harrison"},
    {"number": 9, "name": "William Henry Harrison", "years": "1841", "party": "Whig", "vice_president": "John Tyler", "preceded_by": "Martin Van Buren", "succeeded_by": "John Tyler", "notes": "Died in office after 31 days"},
    {"number": 10, "name": "John Tyler", "years": "1841-1845", "party": "Whig", "vice_president": "None", "preceded_by": "William Henry Harrison", "succeeded_by": "James K. Polk", "notes": "First vice president to succeed to presidency"},
    {"number": 11, "name": "James K. Polk", "years": "1845-1849", "party": "Democratic", "vice_president": "George M. Dallas", "preceded_by": "John Tyler", "succeeded_by": "Zachary Taylor"},
    {"number": 12, "name": "Zachary Taylor", "years": "1849-1850", "party": "Whig", "vice_president": "Millard Fillmore", "preceded_by": "James K. Polk", "succeeded_by": "Millard Fillmore", "notes": "Died in office"},
    {"number": 13, "name": "Millard Fillmore", "years": "1850-1853", "party": "Whig", "vice_president": "None", "preceded_by": "Zachary Taylor", "succeeded_by": "Franklin Pierce"},
    {"number": 14, "name": "Franklin Pierce", "years": "1853-1857", "party": "Democratic", "vice_president": "William R. King", "preceded_by": "Millard Fillmore", "succeeded_by": "James Buchanan"},
    {"number": 15, "name": "James Buchanan", "years": "1857-1861", "party": "Democratic", "vice_president": "John C. Breckinridge", "preceded_by": "Franklin Pierce", "succeeded_by": "Abraham Lincoln"},
    {"number": 16, "name": "Abraham Lincoln", "years": "1861-1865", "party": "Republican", "vice_president": "Hannibal Hamlin, Andrew Johnson", "preceded_by": "James Buchanan", "succeeded_by": "Andrew Johnson", "notes": "Assassinated"},
    {"number": 17, "name": "Andrew Johnson", "years": "1865-1869", "party": "Democratic", "vice_president": "None", "preceded_by": "Abraham Lincoln", "succeeded_by": "Ulysses S. Grant", "notes": "Impeached but not removed"},
    {"number": 18, "name": "Ulysses S. Grant", "years": "1869-1877", "party": "Republican", "vice_president": "Schuyler Colfax, Henry Wilson", "preceded_by": "Andrew Johnson", "succeeded_by": "Rutherford B. Hayes"},
    {"number": 19, "name": "Rutherford B. Hayes", "years": "1877-1881", "party": "Republican", "vice_president": "William A. Wheeler", "preceded_by": "Ulysses S. Grant", "succeeded_by": "James A. Garfield"},
    {"number": 20, "name": "James A. Garfield", "years": "1881", "party": "Republican", "vice_president": "Chester A. Arthur", "preceded_by": "Rutherford B. Hayes", "succeeded_by": "Chester A. Arthur", "notes": "Assassinated"},
    {"number": 21, "name": "Chester A. Arthur", "years": "1881-1885", "party": "Republican", "vice_president": "None", "preceded_by": "James A. Garfield", "succeeded_by": "Grover Cleveland"},
    {"number": 22, "name": "Grover Cleveland", "years": "1885-1889", "party": "Democratic", "vice_president": "Thomas A. Hendricks", "preceded_by": "Chester A. Arthur", "succeeded_by": "Benjamin Harrison", "notes": "Only president to serve non-consecutive terms (22nd and 24th)"},
    {"number": 23, "name": "Benjamin Harrison", "years": "1889-1893", "party": "Republican", "vice_president": "Levi P. Morton", "preceded_by": "Grover Cleveland", "succeeded_by": "Grover Cleveland"},
    {"number": 24, "name": "Grover Cleveland", "years": "1893-1897", "party": "Democratic", "vice_president": "Adlai Stevenson I", "preceded_by": "Benjamin Harrison", "succeeded_by": "William McKinley", "notes": "Only president to serve non-consecutive terms (22nd and 24th)"},
    {"number": 25, "name": "William McKinley", "years": "1897-1901", "party": "Republican", "vice_president": "Garret Hobart, Theodore Roosevelt", "preceded_by": "Grover Cleveland", "succeeded_by": "Theodore Roosevelt", "notes": "Assassinated"},
    {"number": 26, "name": "Theodore Roosevelt", "years": "1901-1909", "party": "Republican", "vice_president": "Charles W. Fairbanks", "preceded_by": "William McKinley", "succeeded_by": "William Howard Taft"},
    {"number": 27, "name": "William Howard Taft", "years": "1909-1913", "party": "Republican", "vice_president": "James S. Sherman", "preceded_by": "Theodore Roosevelt", "succeeded_by": "Woodrow Wilson"},
    {"number": 28, "name": "Woodrow Wilson", "years": "1913-1921", "party": "Democratic", "vice_president": "Thomas R. Marshall", "preceded_by": "William Howard Taft", "succeeded_by": "Warren G. Harding"},
    {"number": 29, "name": "Warren G. Harding", "years": "1921-1923", "party": "Republican", "vice_president": "Calvin Coolidge", "preceded_by": "Woodrow Wilson", "succeeded_by": "Calvin Coolidge", "notes": "Died in office"},
    {"number": 30, "name": "Calvin Coolidge", "years": "1923-1929", "party": "Republican", "vice_president": "Charles G. Dawes", "preceded_by": "Warren G. Harding", "succeeded_by": "Herbert Hoover"},
    {"number": 31, "name": "Herbert Hoover", "years": "1929-1933", "party": "Republican", "vice_president": "Charles Curtis", "preceded_by": "Calvin Coolidge", "succeeded_by": "Franklin D. Roosevelt"},
    {"number": 32, "name": "Franklin D. Roosevelt", "years": "1933-1945", "party": "Democratic", "vice_president": "John Nance Garner, Henry A. Wallace, Harry S. Truman", "preceded_by": "Herbert Hoover", "succeeded_by": "Harry S. Truman", "notes": "Only president to serve more than two terms; died in office"},
    {"number": 33, "name": "Harry S. Truman", "years": "1945-1953", "party": "Democratic", "vice_president": "Alben W. Barkley", "preceded_by": "Franklin D. Roosevelt", "succeeded_by": "Dwight D. Eisenhower"},
    {"number": 34, "name": "Dwight D. Eisenhower", "years": "1953-1961", "party": "Republican", "vice_president": "Richard Nixon", "preceded_by": "Harry S. Truman", "succeeded_by": "John F. Kennedy"},
    {"number": 35, "name": "John F. Kennedy", "years": "1961-1963", "party": "Democratic", "vice_president": "Lyndon B. Johnson", "preceded_by": "Dwight D. Eisenhower", "succeeded_by": "Lyndon B. Johnson", "notes": "Assassinated"},
    {"number": 36, "name": "Lyndon B. Johnson", "years": "1963-1969", "party": "Democratic", "vice_president": "Hubert Humphrey", "preceded_by": "John F. Kennedy", "succeeded_by": "Richard Nixon"},
    {"number": 37, "name": "Richard Nixon", "years": "1969-1974", "party": "Republican", "vice_president": "Spiro Agnew, Gerald Ford", "preceded_by": "Lyndon B. Johnson", "succeeded_by": "Gerald Ford", "notes": "Only president to resign"},
    {"number": 38, "name": "Gerald Ford", "years": "1974-1977", "party": "Republican", "vice_president": "Nelson Rockefeller", "preceded_by": "Richard Nixon", "succeeded_by": "Jimmy Carter", "notes": "Only president never elected as president or vice president"},
    {"number": 39, "name": "Jimmy Carter", "years": "1977-1981", "party": "Democratic", "vice_president": "Walter Mondale", "preceded_by": "Gerald Ford", "succeeded_by": "Ronald Reagan"},
    {"number": 40, "name": "Ronald Reagan", "years": "1981-1989", "party": "Republican", "vice_president": "George H. W. Bush", "preceded_by": "Jimmy Carter", "succeeded_by": "George H. W. Bush"},
    {"number": 41, "name": "George H. W. Bush", "years": "1989-1993", "party": "Republican", "vice_president": "Dan Quayle", "preceded_by": "Ronald Reagan", "succeeded_by": "Bill Clinton"},
    {"number": 42, "name": "Bill Clinton", "years": "1993-2001", "party": "Democratic", "vice_president": "Al Gore", "preceded_by": "George H. W. Bush", "succeeded_by": "George W. Bush", "notes": "Impeached but not removed"},
    {"number": 43, "name": "George W. Bush", "years": "2001-2009", "party": "Republican", "vice_president": "Dick Cheney", "preceded_by": "Bill Clinton", "succeeded_by": "Barack Obama"},
    {"number": 44, "name": "Barack Obama", "years": "2009-2017", "party": "Democratic", "vice_president": "Joe Biden", "preceded_by": "George W. Bush", "succeeded_by": "Donald Trump"},
    {"number": 45, "name": "Donald Trump", "years": "2017-2021", "party": "Republican", "vice_president": "Mike Pence", "preceded_by": "Barack Obama", "succeeded_by": "Joe Biden", "notes": "Impeached twice but not removed"},
    {"number": 46, "name": "Joe Biden", "years": "2021-2025", "party": "Democratic", "vice_president": "Kamala Harris", "preceded_by": "Donald Trump", "succeeded_by": "Donald Trump"},
    {"number": 47, "name": "Donald Trump", "years": "2025-Present", "party": "Republican", "vice_president": "J.D. Vance", "preceded_by": "Joe Biden", "succeeded_by": "Incumbent"}
]

# Generate knowledge base with additional facts and FAQs for better coverage
knowledge_base = []

# Process presidents data into a knowledge base with various question formats
def create_knowledge_base(presidents):
    kb = []
    
    for president in presidents:
        # Basic information
        kb.append(f"President number {president['number']} was {president['name']}.")
        kb.append(f"{president['name']} was the {president['number']}{'st' if president['number'] % 10 == 1 and president['number'] != 11 else 'nd' if president['number'] % 10 == 2 and president['number'] != 12 else 'rd' if president['number'] % 10 == 3 and president['number'] != 13 else 'th'} president.")
        kb.append(f"{president['name']} served as president from {president['years']}.")
        kb.append(f"{president['name']} belonged to the {president['party']} party.")
        kb.append(f"{president['name']}'s vice president was {president['vice_president']}.")
        
        # Succession information
        kb.append(f"{president['name']} was preceded by {president['preceded_by']}.")
        kb.append(f"{president['name']} was succeeded by {president['succeeded_by']}.")
        kb.append(f"Before {president['name']}, the president was {president['preceded_by']}.")
        kb.append(f"After {president['name']}, the president was {president['succeeded_by']}.")
        
        # Notes if available
        if 'notes' in president and president['notes']:
            kb.append(f"{president['name']}: {president['notes']}.")
    
    # Add temporal relations
    for i in range(len(presidents) - 1):
        kb.append(f"{presidents[i+1]['name']} came after {presidents[i]['name']}.")
        kb.append(f"{presidents[i]['name']} came before {presidents[i+1]['name']}.")
    
    # Add specific answer formats for common question types
    for i in range(len(presidents)):
        if i > 0:
            kb.append(f"The president before {presidents[i]['name']} was {presidents[i-1]['name']}.")
        if i < len(presidents) - 1:
            kb.append(f"The president after {presidents[i]['name']} was {presidents[i+1]['name']}.")
            
    return kb

# Generate and add the knowledge base entries
knowledge_base = create_knowledge_base(presidents_data)

# Add more specific facts to handle tricky questions
knowledge_base.extend([
    "Grover Cleveland was both the 22nd and 24th president, serving two non-consecutive terms.",
    "William Henry Harrison had the shortest presidency, serving only 31 days before his death.",
    "Franklin D. Roosevelt served the longest, with over 12 years in office across four terms.",
    "Four presidents have been assassinated: Lincoln, Garfield, McKinley, and Kennedy.",
    "Nixon was the only president to resign from office.",
    "The president before Joe Biden was Donald Trump.",
    "The president after Joe Biden is Donald Trump.",
    "Donald Trump served as both the 45th and 47th president of the United States.",
    "Biden's presidency was from 2021 to 2025.",
    "Before Biden was Trump's first term as president.",
    "After Biden is Trump's second term as president."
])

# Create DataFrame for easier processing
df = pd.DataFrame({'text': knowledge_base})

# Text preprocessing functions
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lemmatize words and remove stopwords
    tokens = nltk.word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])
    return text

# Apply preprocessing to knowledge base
df['processed_text'] = df['text'].apply(preprocess_text)

# Create and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed_text'])

# Helper function to normalize text for name matching
def normalize_name(name):
    """Normalize president names for matching"""
    name = name.lower()
    name = re.sub('[%s]' % re.escape(string.punctuation), '', name)
    return name

# Create a dictionary for quick president lookup
presidents_dict = {normalize_name(president['name']): president for president in presidents_data}
for president in presidents_data:
    # Add last name lookup
    last_name = normalize_name(president['name'].split()[-1])
    if last_name not in presidents_dict:
        presidents_dict[last_name] = president
    # Handle common nicknames
    if president['name'] == "William Jefferson Clinton":
        presidents_dict["bill clinton"] = president
    if president['name'] == "James Earl Carter Jr.":
        presidents_dict["jimmy carter"] = president
    if president['name'] == "Franklin Delano Roosevelt":
        presidents_dict["fdr"] = president

class PresidentsQA:
    def __init__(self, presidents_data, knowledge_base_df, vectorizer, tfidf_matrix):
        self.presidents_data = presidents_data
        self.df = knowledge_base_df
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.presidents_dict = {normalize_name(p['name']): p for p in presidents_data}
        
        # Add last names for lookup
        for p in presidents_data:
            last_name = normalize_name(p['name'].split()[-1])
            if last_name not in self.presidents_dict:
                self.presidents_dict[last_name] = p
                
        # Define patterns for question detection
        self.before_patterns = [
            r'(?:who|which president) (?:was|came) before ([a-zA-Z\s\.]+)',
            r'who preceded ([a-zA-Z\s\.]+)',
            r'before ([a-zA-Z\s\.]+) was president'
        ]
        
        self.after_patterns = [
            r'(?:who|which president) (?:was|came) after ([a-zA-Z\s\.]+)',
            r'who succeeded ([a-zA-Z\s\.]+)',
            r'after ([a-zA-Z\s\.]+) was president'
        ]
        
        self.which_number_patterns = [
            r'(?:which|what) number president was ([a-zA-Z\s\.]+)',
            r'what was ([a-zA-Z\s\.]+)(?:\'s)? number'
        ]
        
        self.which_president_number_patterns = [
            r'who was (?:the )?(\d+)(?:st|nd|rd|th)? president',
            r'(?:who|which president) was number (\d+)'
        ]
        
        self.when_served_patterns = [
            r'when (?:did|was) ([a-zA-Z\s\.]+) president',
            r'what years did ([a-zA-Z\s\.]+) serve'
        ]
        
        self.which_party_patterns = [
            r'what party was ([a-zA-Z\s\.]+)',
            r'which party did ([a-zA-Z\s\.]+) belong to'
        ]
        
    def extract_president_name(self, question, patterns):
        """Extract president name from question using regex patterns"""
        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                return match.group(1).strip()
        return None
    
    def extract_president_number(self, question):
        """Extract president number from question"""
        for pattern in self.which_president_number_patterns:
            match = re.search(pattern, question.lower())
            if match:
                return match.group(1)
        return None
    
    def find_president_by_name(self, name):
        """Find president information by name"""
        norm_name = normalize_name(name)
        
        # Direct match
        if norm_name in self.presidents_dict:
            return self.presidents_dict[norm_name]
        
        # Try to find closest match
        for president_name, president in self.presidents_dict.items():
            if norm_name in president_name or president_name in norm_name:
                return president
                
        return None
    
    def find_president_by_number(self, number):
        """Find president information by number"""
        number = int(number)
        for president in self.presidents_data:
            if president['number'] == number:
                return president
        return None
    
    def answer_before_question(self, question):
        """Answer questions about which president came before another"""
        name = self.extract_president_name(question, self.before_patterns)
        if name:
            president = self.find_president_by_name(name)
            if president:
                if president['preceded_by'] == "None (first president)":
                    return f"{president['name']} was the first president, so there was no president before them."
                return f"The president before {president['name']} was {president['preceded_by']}."
        return None
    
    def answer_after_question(self, question):
        """Answer questions about which president came after another"""
        name = self.extract_president_name(question, self.after_patterns)
        if name:
            president = self.find_president_by_name(name)
            if president:
                if president['succeeded_by'] == "Incumbent":
                    return f"{president['name']} is the current president."
                return f"The president after {president['name']} was {president['succeeded_by']}."
        return None
    
    def answer_number_question(self, question):
        """Answer questions about a president's number"""
        name = self.extract_president_name(question, self.which_number_patterns)
        if name:
            president = self.find_president_by_name(name)
            if president:
                suffix = "th"
                if president['number'] % 10 == 1 and president['number'] != 11:
                    suffix = "st"
                elif president['number'] % 10 == 2 and president['number'] != 12:
                    suffix = "nd"
                elif president['number'] % 10 == 3 and president['number'] != 13:
                    suffix = "rd"
                
                # Special case for Grover Cleveland
                if president['name'] == "Grover Cleveland":
                    return "Grover Cleveland was both the 22nd and 24th president, the only president to serve non-consecutive terms."
                    
                return f"{president['name']} was the {president['number']}{suffix} president of the United States."
        return None
    
    def answer_which_president_question(self, question):
        """Answer questions about which president had a specific number"""
        number = self.extract_president_number(question)
        if number:
            president = self.find_president_by_number(number)
            if president:
                return f"The {number}{'st' if int(number) % 10 == 1 and int(number) != 11 else 'nd' if int(number) % 10 == 2 and int(number) != 12 else 'rd' if int(number) % 10 == 3 and int(number) != 13 else 'th'} president was {president['name']}."
        return None
    
    def answer_when_served_question(self, question):
        """Answer questions about when a president served"""
        name = self.extract_president_name(question, self.when_served_patterns)
        if name:
            president = self.find_president_by_name(name)
            if president:
                return f"{president['name']} served as president from {president['years']}."
        return None
    
    def answer_party_question(self, question):
        """Answer questions about which party a president belonged to"""
        name = self.extract_president_name(question, self.which_party_patterns)
        if name:
            president = self.find_president_by_name(name)
            if president:
                return f"{president['name']} belonged to the {president['party']} party."
        return None
    
    def handle_special_cases(self, question):
        """Handle special cases and common questions"""
        q = question.lower()
        
        # Current president
        if "current president" in q or "president now" in q or "who is president" in q:
            current = next((p for p in self.presidents_data if p['succeeded_by'] == "Incumbent"), None)
            if current:
                return f"The current president is {current['name']}, the {current['number']}th president of the United States."
        
        # First president
        if "first president" in q:
            return "George Washington was the first president of the United States, serving from 1789 to 1797."
            
        # Latest president
        if "latest president" in q or "newest president" in q:
            latest = self.presidents_data[-1]
            return f"The latest president is {latest['name']}, the {latest['number']}th president, who took office in {latest['years'].split('-')[0]}."
        
        # President before Biden (common question)
        if "before biden" in q:
            biden = next((p for p in self.presidents_data if p['name'] == "Joe Biden"), None)
            if biden:
                return f"The president before Joe Biden was {biden['preceded_by']}."
                
        # President after Biden (common question)
        if "after biden" in q:
            biden = next((p for p in self.presidents_data if p['name'] == "Joe Biden"), None)
            if biden:
                return f"The president after Joe Biden is {biden['succeeded_by']}."
                
        # Trump serving twice
        if "trump twice" in q or "trump two terms" in q or "trump second term" in q:
            return "Donald Trump served as both the 45th president (2017-2021) and the 47th president (2025-Present), with Joe Biden's presidency in between."
            
        return None
        
    def get_answer_from_knowledge_base(self, question):
        """Get answer from knowledge base using TF-IDF similarity"""
        processed_question = preprocess_text(question)
        question_vector = self.vectorizer.transform([processed_question])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(question_vector, self.tfidf_matrix).flatten()
        
        # Get most similar entry if similarity is above threshold
        most_similar_idx = similarity_scores.argmax()
        if similarity_scores[most_similar_idx] > 0.2:  # Threshold for relevance
            return self.df['text'].iloc[most_similar_idx]
        
        return None
    
    def answer_question(self, question):
        """Main function to answer president-related questions"""
        # First check for special cases
        answer = self.handle_special_cases(question)
        if answer:
            return answer
            
        # Try specific question patterns
        methods = [
            self.answer_before_question,
            self.answer_after_question,
            self.answer_number_question,
            self.answer_which_president_question,
            self.answer_when_served_question,
            self.answer_party_question
        ]
        
        for method in methods:
            answer = method(question)
            if answer:
                return answer
        
        # Fallback to knowledge base similarity search
        answer = self.get_answer_from_knowledge_base(question)
        if answer:
            return answer
            
        # If no good answer found
        return "I'm not sure about that. Could you rephrase your question about US presidents?"

# Create the QA system
presidents_qa = PresidentsQA(presidents_data, df, vectorizer, tfidf_matrix)

# Function to save the model and data
def save_model_data():
    """Save the model and data for future use"""
    import pickle
    
    # Save presidents data
    with open('presidents_data.json', 'w') as f:
        json.dump(presidents_data, f)
    
    # Save the vectorizer and processed dataframe
    with open('presidents_qa_model.pkl', 'wb') as f:
        pickle.dump({
            'vectorizer': vectorizer,
            'df': df
        }, f)
    
    print("Model and data saved successfully.")

# Function to load the model and data
def load_model_data():
    """Load the model and data"""
    import pickle
    
    # Load presidents data
    with open('presidents_data.json', 'r') as f:
        loaded_presidents_data = json.load(f)
    
    # Load the vectorizer and processed dataframe
    with open('presidents_qa_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    loaded_vectorizer = model_data['vectorizer']
    loaded_df = model_data['df']
    loaded_tfidf_matrix = loaded_vectorizer.transform(loaded_df['processed_text'])
    
    # Create the QA system
    loaded_qa = PresidentsQA(loaded_presidents_data, loaded_df, loaded_vectorizer, loaded_tfidf_matrix)
    
    return loaded_qa

# Interactive demo function
def interactive_demo():
    """Interactive demo for the presidents QA system"""
    print("US Presidents Question-Answering System")
    print("Type 'exit' to quit the program\n")
    
    while True:
        question = input("\nAsk a question about US presidents: ")
        
        if question.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
            
        answer = presidents_qa.answer_question(question)
        print(f"Answer: {answer}")

# Example usage
if __name__ == "__main__":
    # Uncomment to save the model and data
    # save_model_data()
    
    # Example questions to demonstrate the system
    example_questions = [
        "Who was the first president?",
        "Who was president before Biden?",
        "Who was the president after Obama?",
        "What number president was Abraham Lincoln?",
        "Who was the 16th president?",
        "When did Franklin D. Roosevelt serve as president?",
        "What party did Richard Nixon belong to?",
        "Who is the current president?",
        "Which president served two non-consecutive terms?",
        "Who was the president before Trump's first term?",
        "Who was the president after Biden?"
    ]
    
    print("US Presidents Question-Answering System")
    print("Here are some example questions the system can answer:")
    
    for i, question in enumerate(example_questions, 1):
        print(f"{i}. {question}")
        answer = presidents_qa.answer_question(question)
        print(f"   Answer: {answer}\n")
    
    print("\nNow it's your turn to ask questions!")
    interactive_demo()