#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from ipynb.fs.full. aloo import fetch_youtube_videos
#from ipynb.fs.full. datasets import combined
from ipynb.fs.full. preprocessing import *


# In[ ]:





# In[ ]:


df = pd.read_csv("data/processed_data.csv")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("models/tfidf_matrix.pkl") 


# In[ ]:





# In[ ]:


st.title("📚 Learning Resource Recommender")
st.sidebar.header("Preferences")
show_videos = st.sidebar.checkbox("Youtube Videos", value=True)
show_courses = st.sidebar.checkbox("Courses", value=True)
show_books = st.sidebar.checkbox("Books", value=True)

#st.markdown("Discover courses, books, and videos tailored to your interests!")
#user_query = st.text_input("🔍 Enter a topic (e.g., Python, Machine Learning):", "Python")


# In[ ]:


course_filters = {}
if show_courses:
    st.sidebar.subheader("Course Filters")

    # Price Filter (Free/Paid)
    course_filters['price'] = st.sidebar.radio(
        "Price",
        ["All", "Free", "Paid"]
    )

    # Difficulty Filter
    course_filters['level'] = st.sidebar.multiselect(
        "Difficulty Level",
        options=df[df['type'] == 'course']['level'].unique(),
        default=df[df['type'] == 'course']['level'].unique()
    )

course_filters['sort'] = st.sidebar.selectbox(
        "Sort Courses By",
        options=["Relevance", "Price (Low to High)", "Price (High to Low)", "Duration (Short to Long)"]
    )


# In[ ]:


user_query = st.text_input("🔍 Enter a topic (e.g., Python, Machine Learning):", "Python")


# In[ ]:


if user_query:
    # Get similarity scores
    query_vector = tfidf.transform([user_query])
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()
    df["similarity_score"] = cosine_sim  # Add scores to DataFrame

    # Split into courses and books
    courses = df[df["type"] == "course"].copy()
    books = df[df["type"] == "book"].copy()

    if show_courses and not courses.empty:
        # Price Filter
        if course_filters['price'] == "Free":
            courses = courses[courses["price"] == 0]
        elif course_filters['price'] == "Paid":
            courses = courses[courses["price"] > 0]

        # Difficulty Filter
        courses = courses[courses["level"].isin(course_filters['level'])]

        # Sorting
        if course_filters['sort'] == "Price (Low to High)":
            courses = courses.sort_values("price", ascending=True)
        elif course_filters['sort'] == "Price (High to Low)":
            courses = courses.sort_values("price", ascending=False)
        elif course_filters['sort'] == "Duration (Short to Long)":
            courses = courses.sort_values("contentduration", ascending=True)
        else:  # Default: Relevance
            courses = courses.sort_values("similarity_score", ascending=False)


    if show_videos:
        st.subheader("🎥 Recommended Videos")
        youtube_videos = fetch_youtube_videos(["AIzaSyDyQg3Yom5ttD82Yy82mUJVnopJQH7YYnU"], user_query)
        if youtube_videos:
            for video in youtube_videos:
                st.markdown(f"[{video['title']}]({video['url']})")
                st.image(video['thumbnail'], width=200)
        else:
            st.write("No videos found.")

    # Courses (With filters/sorting)
    if show_courses and not courses.empty:
        st.subheader("🎓 Recommended Courses")
        for _, row in courses.iterrows():
            st.markdown(f"[{row['title']}]({row['url']})")
            st.write(f"**Difficulty**: {row['level']}")
            st.write(f"**Duration**: {row['contentduration']} weeks")
            #st.write(f"**Price**: {'Free' if row['price'] == 0 else f'${row['price']:.2f'}")
            st.write(f"**Price**: {'Free 🎉' if row['price'] == 0 else f'${row["price"]:.2f}'}")
            st.write("---")

    if show_books and not books.empty:
        st.subheader("📚 Recommended Books")
        # Sort books by similarity score (original order)
        books = books.sort_values("similarity_score", ascending=False)
        for _, row in books.iterrows():
            st.markdown(f"[{row['title']}]({row['url']})")
            st.write(f"**Price**: ${row['price']:.2f}")
            st.write("---")   


# In[ ]:


# youtube videos
'''youtube_recs = fetch_youtube_videos(
    api_key=["AIzaSyDyQg3Yom5ttD82Yy82mUJVnopJQH7YYnU"],
    query=user_query,
    max_results=5
)

# YouTube results
st.subheader("Recommended YouTube Videos")
if youtube_recs:
    for video in youtube_recs:
        st.markdown(f"[{video['title']}]({video['url']})")
else:
    st.warning("No YouTube videos found.")'''


# In[ ]:




