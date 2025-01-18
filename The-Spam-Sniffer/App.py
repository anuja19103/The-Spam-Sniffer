import streamlit as st
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def load_model():
    try:
        with open("vectorizer.pkl", "rb") as vec_file:
            vectorizer = pickle.load(vec_file)
        with open("model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def check_message(message, vectorizer, model):
    if not vectorizer or not model:
        st.error("Model and vectorizer are not loaded. Please check your files.")
        return None
    try:
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)[0]
        return "Spam" if prediction == 1 else "Not Spam"
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None


def log_message(message, result):
    try:
        with open("message_log.txt", "a") as log_file:
            log_file.write(f"{message}\t{result}\n")
    except Exception as e:
        st.error(f"Error logging message: {e}")


def display_message_history():
    if os.path.exists("message_log.txt"):
        with open("message_log.txt", "r") as log_file:
            history = log_file.readlines()

        # Debugging: print the raw content of the file
        st.write("Log Content:")
        st.write(history)  # Show the raw file contents to debug

        if history:
            st.write("#### Message History")
            for line in history:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    message, result = parts
                    color = 'red' if result == "Spam" else 'green'

                    st.markdown(f"""
                    <div style="padding: 10px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
                        <p style="font-size: 18px; font-weight: bold;">Message:</p>
                        <p style="font-size: 16px; color: #333;">{message}</p>
                        <p style="font-size: 16px; font-weight: bold; color: {color};">Result: {result}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No logs available.")
    else:
        st.info("No message history found.")


def display_eda_info():
    st.write("### Data Processing Steps")
    st.markdown("""
    1. **Data Cleaning**:
       - Identified and removed any duplicates to ensure data integrity.
       - Handled missing values by imputing or removing incomplete entries.
       - Renamed columns to improve readability and consistency.

       ```python
       # Data Cleaning Code Example
       df.drop_duplicates(inplace=True)
       df.fillna(method='ffill', inplace=True)
       df.rename(columns={'old_name': 'new_name'}, inplace=True)
       ```

    2. **Exploratory Data Analysis (EDA)**:
       - **Message Length**: Analyzed the distribution of message lengths to understand trends and patterns.
       - **Word Count**: Examined the distribution of word counts in messages.
       - **Character Count**: Investigated the character count distribution across messages.
       - **Word Frequency**: Generated word clouds and frequency distributions for common words in spam and non-spam messages.
       - **Bigram and Trigram Analysis**: Identified frequent pairs and triplets of words.

       ```python
       # EDA Code Example
       df['message_length'] = df['message'].apply(len)
       word_count = df['message'].apply(lambda x: len(x.split()))
       character_count = df['message'].apply(len)

       # Word Frequency
       from wordcloud import WordCloud
       spam_words = ' '.join(list(df[df['label'] == 'spam']['message']))
       spam_wordcloud = WordCloud(width=800, height=400).generate(spam_words)

       # Display WordCloud
       plt.figure(figsize=(10, 5))
       plt.imshow(spam_wordcloud, interpolation='bilinear')
       plt.axis('off')
       plt.show()
       ```

    3. **Data Preprocessing**:
       - **Lowercasing**: Converted all text to lowercase to maintain uniformity.
       - **Special Characters**: Removed punctuation, numbers, and special characters.
       - **Stopwords Removal**: Eliminated common stopwords that do not contribute to text meaning.
       - **Stemming/Lemmatization**: Reduced words to their base or root form.

       ```python
       # Data Preprocessing Code Example
       import string
       import nltk
       from nltk.corpus import stopwords
       from nltk.stem import PorterStemmer

       nltk.download('stopwords')
       stop_words = set(stopwords.words('english'))
       ps = PorterStemmer()

       def preprocess_text(text):
           text = text.lower()
           text = text.translate(str.maketrans('', '', string.punctuation))
           text = ' '.join([ps.stem(word) for word in text.split() if word not in stop_words])
           return text

       df['cleaned_message'] = df['message'].apply(preprocess_text)
       ```

    4. **Model Training**:
       - **Text Vectorization**: Used `TfidfVectorizer` to convert text data into numerical vectors that represent the importance of words.
       - **Model Selection**: Trained a `Multinomial Naive Bayes` classifier for its efficiency and accuracy in text classification tasks.
       - **Training Process**: Split the data into training and testing sets to evaluate the model’s performance.

       ```python
       # Model Training Code Example
       from sklearn.feature_extraction.text import TfidfVectorizer
       from sklearn.model_selection import train_test_split
       from sklearn.naive_bayes import MultinomialNB

       vectorizer = TfidfVectorizer()
       X = vectorizer.fit_transform(df['cleaned_message'])
       y = df['label']

       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       model = MultinomialNB()
       model.fit(X_train, y_train)
       ```

    5. **Model Evaluation**:
       - **Confusion Matrix**: Evaluated the confusion matrix to understand the number of true positives, true negatives, false positives, and false negatives.
       - **Accuracy**: Achieved high accuracy by correctly classifying a significant proportion of the messages.
       - **Precision and Recall**: Analyzed precision and recall to understand the model’s performance, particularly for spam detection.
       - **F1 Score**: Calculated the F1 score as a balance between precision and recall.

       ```python
       # Model Evaluation Code Example
       from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

       y_pred = model.predict(X_test)
       accuracy = accuracy_score(y_test, y_pred)
       precision = precision_score(y_test, y_pred, pos_label='spam')
       recall = recall_score(y_test, y_pred, pos_label='spam')
       f1 = f1_score(y_test, y_pred, pos_label='spam')

       conf_matrix = confusion_matrix(y_test, y_pred)

       # Displaying Metrics
       st.write(f"Accuracy: {accuracy}")
       st.write(f"Precision: {precision}")
       st.write(f"Recall: {recall}")
       st.write(f"F1 Score: {f1}")
       st.write("Confusion Matrix:")
       st.write(conf_matrix)
       ```
    """)


def display_model_details():
    st.markdown("""
    ### Model Overview
    - **Model Type**: Multinomial Naive Bayes
    - **Purpose**: To classify messages as 'Spam' or 'Not Spam'.
    - **Training Data**: Utilized a labeled dataset with spam and non-spam messages for training.
    - **Vectorization**: Employed `TfidfVectorizer` to convert text to numerical format.

    ### Performance Metrics
    - **Accuracy**: 96.0%
    - **Precision**: 100.0% (Spam detection)
    - **Recall**: 95.0% (Correctly identifying spam messages)
    - **F1 Score**: 97.4% (Balance between precision and recall)


    ### Model Evaluation
    - **Confusion Matrix**: Evaluates the confusion matrix to understand the number of true positives, true negatives, false positives, and false negatives.
    - **ROC Curve**: Plots the ROC curve to visualize the trade-off between sensitivity and specificity.
    - **Precision-Recall Curve**: Assesses the precision-recall curve to understand the relationship between precision and recall.

    ### Deployment
    - Relies on `model.pkl` (trained model) and `vectorizer.pkl` (text vectorizer) for predictions.
    - Integrated with the Streamlit app to provide real-time message classification.

    ### Confusion Matrixx
    """)
    st.image("confusion_matrix.png")


# Initialize Streamlit app
st.set_page_config(page_title="The Spam Sniffer", layout="wide")

# Load model and vectorizer
vectorizer, model = load_model()

# Sidebar Navigation
st.sidebar.header("The Spam Sniffer")
st.sidebar.image("logo.png")
# Create radio buttons for navigation
choice = st.sidebar.radio("Select an option:",
                          ["Welcome", "Message Check", "Message Inbox", "Model Details", "Data Processing Steps",
                           "View Documentation", "View PPT"])

# Main App Logic
if choice == "Message Check":
    st.title("The Spam Sniffer")

    # Text area for user input
    message = st.text_area("Enter the message you want to check:")

    # Button to check the message
    if st.button("Check Message"):
        if message.strip():  # Check if the message is not empty
            result = check_message(message, vectorizer, model)  # Call the function to check the message

            if result:  # If a result is returned
                log_message(message, result)  # Log the message and its result

                # Display the result with appropriate color
                st.markdown(
                    f"<h3 style='color: {'red' if result == 'Spam' else 'green'};'>{result}</h3>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning("Please enter a valid message.")  # Warning for empty input

elif choice == "Message Inbox":
    st.title("Message Inbox")
    display_message_history()  # Function to display message history

elif choice == "Model Details":
    st.title("Model Details")
    display_model_details()  # Function to display model details

elif choice == "Data Processing Steps":
    st.header("Data Processing Steps and Outputs")
    display_eda_info()  # Function to display processing steps and outputs

elif choice == "View Documentation":
    st.title("Documentation")

    # Check if the documentation PDF exists
    if os.path.exists("Documentation.pdf"):
        # Provide the option to download the PDF
        with open("Documentation.pdf", "rb") as pdf_file:
            st.download_button(
                label="Download Documentation PDF",
                data=pdf_file,
                file_name="Documentation.pdf",
                mime="application/pdf"
            )
    else:
        st.error("Documentation PDF not found.")

elif choice == "View PPT":
    st.title("Presentation (PPT)")
    # Check if the PPTX file exists
    if os.path.exists("demo.pptx"):
        with open("demo.pptx", "rb") as pptx_file:  # Correct extension here
            st.download_button(
                label="Download PPT Presentation",
                data=pptx_file,
                file_name="demo.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"  # MIME type for .pptx
            )
    else:
        st.error("PPTX file not found.")
else:
    # Welcome page
    st.title("Welcome to The Spam Sniffer")
    st.write("This application helps you identify whether a message is spam or not.")
    st.write("Use the sidebar to navigate through the different functionalities.")
    st.write("1. **Message Check**: Check a single message for spam.")
    st.write("2. **Message History**: View the history of messages checked.")
    st.write("3. **Model Details**: Learn about the underlying model.")
    st.write("4. **Processing Steps**: Understand the steps involved in processing messages.")