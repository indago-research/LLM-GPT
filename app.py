import streamlit as st
from datetime import datetime
from main import ask  # Import the ask function from main.py
from PIL import Image  # Import to handle local image loading
import os
import pandas as pd
from io import BytesIO

# Custom CSS for setting the font type to "Times New Roman"
st.markdown(
    """
    <style>
    /* Apply the font to all text elements */
    * {
        font-family: 'Playfair Display', Times, serif !important;  /* Set font type to Times New Roman */
    }
    
    /* Additional CSS to control the sidebar font */
    .css-1d391kg, .css-1e5imcs, .css-1e5imcs.e1fqkh3o3, .css-1v3fvcr, [data-testid="stSidebar"], .css-qbe2hs {
        font-family: 'Playfair Display', Times, serif !important;  /* Sidebar font */
    }

    /* Ensure form elements like text input, textarea, and buttons use the same font */
    .stTextInput div, .stTextArea textarea, .stRadio div, .stButton button, .stDownloadButton button, .stNumberInput input {
        font-family: 'Playfair Display', Times, serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the local image
image_path = './assets/indago.PNG'  # Update the path to your image

logo_image = Image.open(image_path)  # Use PIL to open the image

# Store the chat history and question inputs in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'question_inputs' not in st.session_state:
    st.session_state.question_inputs = [""]  # Start with one empty question input

# Add a logo at the top of the sidebar
st.sidebar.image(logo_image, width=200)  # Display the local image

# Center the title on the main page
st.markdown("<h1 style='text-align: center;'>DataGem by Indago</h1>", unsafe_allow_html=True)

# Sidebar Chat history section (Automatically generated titles)
st.sidebar.subheader("Chat History")

# Add a button to delete chat history
if st.sidebar.button("Delete Chat History"):
    st.session_state.chat_history = []  # Clear the chat history
    st.session_state.question_inputs = [""]
    st.success("Chat history deleted!")

# Add a button to clear FAISS cache and URL cache
if st.sidebar.button("Clear Cache"):
    try:
        import main

        if hasattr(main, 'faiss_cache') and isinstance(main.faiss_cache, dict):
            main.faiss_cache.clear()  # Clear the FAISS cache
        else:
            st.warning("FAISS cache not found or not defined as a dictionary in main.py.")

        if hasattr(main, 'url_cache') and isinstance(main.url_cache, dict):
            main.url_cache.clear()  # Clear the URL cache
        else:
            st.warning("URL cache not found or not defined as a dictionary in main.py.")

        st.success("Cache cleared successfully!")  # Confirmation message
    except ImportError:
        st.error("Unable to import main.py. Please ensure the file is present and accessible.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Display the chat history in the sidebar if available
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        # Use expander to show each chat history entry with title
        with st.sidebar.expander(f"**{chat['title']}**"):
            st.markdown(chat['response'], unsafe_allow_html=True)
            st.write("---")  # Separator for visual clarity
else:
    st.sidebar.write("No chat history yet.")

# Create a placeholder container to position the "+" button later
# Create a placeholder container to position the "-" and "+" buttons later
button_placeholder = st.empty()  # Placeholder outside the form

# Handle the "+" and "-" button click events outside the form
with button_placeholder.container():
    # Define a column layout with three columns (left: "-", middle: space, right: "+")
    col_minus, col_space, col_plus = st.columns([1, 9, 1])  # Adjust column widths as needed

    # Place the "-" button in the first column
    with col_minus:
        if st.button("➖", help="Remove Last Question", key="remove_button"):
            if len(st.session_state.question_inputs) > 1:  # Ensure at least one question box remains
                st.session_state.question_inputs.pop()  # Remove the last question field

    # Place the "+" button in the third column
    with col_plus:
        if st.button("➕", help="Add More Questions", key="add_button"):
            st.session_state.question_inputs.append("")  # Add a new empty question field


# Main page form to enter URLs and multiple queries
with st.form("url_query_form"):
    # URL Input Box
    urls_input = st.text_area("Submit the Access Points", placeholder="Your Links here")

    # Add a radio button to select the scraping strategy just below the URL input
    scraping_strategy = st.radio(
        "Select Path Strategy",
        ("Sitemap", "Strategic Paths", "Financial Paths"),
        index=0  # Default selection is Sitemap
    )

    # Create a container for Question 1 to apply full-width class
    st.session_state.question_inputs[0] = st.text_input(
        "Question 1", value=st.session_state.question_inputs[0], key="question_0", placeholder="Enter your query here"
    )

    # Use markdown to wrap the Question 1 input in a custom CSS class for full width
    # st.markdown(f'<div class="full-width-input">{st.session_state.question_inputs[0]}</div>', unsafe_allow_html=True)

    # Display the remaining question input boxes, if any, inside the form
    for i in range(1, len(st.session_state.question_inputs)):
        st.session_state.question_inputs[i] = st.text_input(f"Question {i+1}", value=st.session_state.question_inputs[i], key=f"question_{i+1}")

    # Create columns to place "Get Response" button in the same row
    col1, _ = st.columns([4, 1.6])  # Adjust column widths as needed
    with col1:
        # Place the "Get Response" button in the first column
        submit_button = st.form_submit_button("Get Response")
import re

# After the response is generated
if submit_button:
    if not urls_input.strip():                                                                                              
        st.error("Please enter at least one URL.")
    else:
        urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
        queries = [q for q in st.session_state.question_inputs if q.strip()]  # Collect all non-empty questions
        if not queries:
            st.error("Please enter at least one question.")
        else:
            with st.spinner(":gear: Generating response, please wait..."):
                # Call the ask() function with the collected queries, URLs, and scraping strategy
                response = ask(queries, urls, scraping_strategy=scraping_strategy)

            # Use regular expressions to split the response at each 'URL:'
            response_list = re.split(r'(?=URL:)', response)  # Include 'URL:' in each split part
            st.session_state['responses'] = response_list  # Store the responses in session state

# Display responses and the download button (if present in session state)
if 'responses' in st.session_state:
    response_list = st.session_state['responses']
    new_data = {"URL": []}  # Initialize with 'URL' as the first column
    for item in response_list:
        if not item.strip():
            continue  # Skip empty items
        url = None
        question = None
        answer = []
        lines = item.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("URL:"):
                url = line.replace("URL:", "").strip()
                i += 1
            elif line.startswith("Query:"):
                question = line.replace("Query:", "").strip()
                if question not in new_data:
                    new_data[question] = []  # Create a new column for each unique query
                i += 1
            elif line.startswith("Response:"):
                # Extract any text after 'Response:' on the same line
                response_text = line.replace("Response:", "").strip()
                response_lines = [response_text] if response_text else []
                i += 1
                # Start capturing any subsequent lines
                while i < len(lines) and not lines[i].startswith("URL:") and not lines[i].startswith("Query:"):
                    response_lines.append(lines[i])
                    i += 1
                final_answer = '\n'.join(response_lines).strip()
                answer.append(final_answer if final_answer else "No response found.")
            else:
                i += 1  # Skip any other lines

        if url:
            if url not in new_data["URL"]:
                new_data["URL"].append(url)  # Add URL only once
            row_index = new_data["URL"].index(url)

            # Ensure all columns have the same number of rows
            for key in new_data.keys():
                if len(new_data[key]) <= row_index:
                    new_data[key].append("")  # Fill with empty strings

            if question and answer:
                final_answer = "\n".join(answer).strip()
                new_data[question][row_index] = final_answer
                st.markdown(f"**URL:** {url}")
                st.markdown(f"**Query:** {question}")
                st.markdown(f"**Response:**\n{final_answer}")
                st.write("---")

    # Create the DataFrame for download
    new_df = pd.DataFrame(new_data)
    buffer = BytesIO()
    new_df.to_excel(buffer, index=False)
    buffer.seek(0)

    # Download button
    st.download_button(
        label="Download Updated Responses",
        data=buffer,
        file_name="output_responses.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
