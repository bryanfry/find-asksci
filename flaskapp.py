import sys
print 'RUNNING flaskapp.py'
print 'Python interpretter is ' + sys.executable

from flask import Flask, jsonify, render_template, request

import pandas as pd
import string
import numpy as np
import os
import pickle
#import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.metrics import pairwise
import re
#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
# import seaborn as sbn

bp_pickles = '/home/ubuntu/proj_asksci/files_out/pickles'

##################### TWO FUNCTIONS TO CLEAN TEXT ##########################

def separate_sentences (s):
    s = re.sub('\.[a-zA-z]', lambda x: x.group(0)[0:-1] + ' ' + x.group(0)[-1], s)
    return s

def clean_text (s):
    try:
        s = filter (lambda x: x in string.printable, s) # Remove unprintable weirdness. NO ERR
        s = s.strip() #NO ERR
        s = separate_sentences (s)
    except:
        print '******* ERR'
    return s

########### FUNCTION TO TRANSFORM AND CLASSIFY TEXT #############

#### Function to score and classify text.  Used both for redditor comment corpus and
# for user input question.  Classification is OPTIONAL.  If no classifier is passed in,
# it will not proceed

def transform_classify_text (t, vectorizer_obj, dimreduce_obj, classifier_obj = None):
    t_vec   = vectorizer_obj.transform(t)
    t_xform = dimreduce_obj.transform(t_vec)
    if not classifier_obj == None:
        topic = classifier_obj.predict (t_xform)
    else: 
        topic = None
    return t_xform, topic

 ########## FUNCTION TO BUILD DATAFRAME WITH REDDITOR COSINE SIMILARITY TO QUESTION ###########

def build_cosine_sim_df (df_reddit, q_xform):

    # Build a new 2-column dataframe, with AUTHORS and cosine similarity to question
    df_cossim = pd.DataFrame()
    df_cossim['AUTHOR']   = df_reddit.AUTHOR
    
    # Extract the SVD values from the redditor dataframe, to yield an N_redditor x 200
    # numpy array.
    SVD_array = np.array ([df_reddit.iloc[i].SVD for i in range (len (df_reddit))])
    
    cs = pairwise.cosine_similarity (q_xform, SVD_array).transpose()
    df_cossim['COS_SIM']  = cs.tolist() 
    df_cossim = df_cossim.sort_values (['COS_SIM'], ascending=False)
    return df_cossim
    
 ########  Function to Build dataframe mapping individual reddit ####
 ########  comment to cosine similarity with question ###############

 # Function ingests the SVD-transformed user question, along with the redditor info dataframe 
# that contains the SVD-transformed representation of each redditors total corpus.
#
# Then build a new dataframe giving cosine similarity between the question and each
# redditor's corpus.

def build_cosine_sim_comment_df (comments_list, q_xform):
    # Build a new 2-column dataframe, with comment and cosine similarity to question
    df_cossim = pd.DataFrame()
    df_cossim['COMMENT'] = comments_list
    
    comments_xform, topic_dummy = transform_classify_text (comments_list,\
                                             TFIDF_obj, SVD_obj, \
                                             classifier_obj = None)
    
    cs = pairwise.cosine_similarity (q_xform, comments_xform).transpose()
    df_cossim['COS_SIM'] = cs.tolist()
    df_cossim = df_cossim.sort_values (['COS_SIM'], ascending = False)
    return df_cossim



##### FUNCTION TO MATCH QUESTION TO REDDITOR / COMMENT ##########


def process_user_question (q, df_reddit, dict_reddit, TFIDF_obj, SVD_obj, \
                       classifier_obj, verbose = False):
    
    # transform and classify the question
    q_xform, q_topic = transform_classify_text (np.array([q]), TFIDF_obj, \
                                                SVD_obj, classifier_obj)
    
    q_topic = q_topic [0][0:-4]  #Remove '_ALL' tag at end of topic label.
    
    # Build dataframe mapping authors to their 'total comment corpus'
    # cosine similarity to the question asked.
    df_cossim = build_cosine_sim_df (df_reddit, q_xform)

    #Get the author at the top of the dataframe (best cosine similarity)
    u = str (df_cossim.AUTHOR.tolist()[0])
    url = 'reddit.com/u/' + u  # Url to view all comments from given user
    
    # From the comments dictionary, transfrom the individual comments for best redditor
    comments_list = [dict_reddit[u][i]['BODY'] for i in range (len(dict_reddit[u]))]

    
    # Compute cosine similarity of each comment to the question
    df_cossim_com = build_cosine_sim_comment_df (comments_list, q_xform) 
    
    # Extract the top comment (best cosine similarity) from the df
    comment_match = df_cossim_com.COMMENT.tolist()[0]
    
    n = len (comments_list) #Number of comments for best-matched redditor
    
    if verbose == True:
        print 'THIS IS A QUESTION ABOUT ' + q_topic
        print 'BEST REDDITOR MATCH IS ' + u
        print 'URL FOR REDDITOR IS ' + url
        print 'THIS USER HAS ' + str (n) + ' COMMENTS IN /r/askscience' 
        print
        print 'COMMENT BEST MATCHING QUESTION IS... '
        print 
        print comment_match

    return q_topic, u, comment_match, n, url




#################################
############ MAIN ###############
#################################

app = Flask(__name__)

# Load the pickles... there are 5 of them.
# -- Redditor dataframe (containing SVD-transformed values for total corpuses)
# -- Total redditor comment archive
# -- TFIDF Vectorizer
# -- SVD Transformer
# -- Support Vector Classifier (SVC) for topic

# Load the dataframe with the consolidated redditor info.
# This object includes the SVD representation of the redditor's
# entire comment corpus

print 'Loading Redditor dataframe....'
with open (os.path.join (bp_pickles, 'df_redditors.pkl')) as f:
    df_reddit = pickle.load (f)  

print 'Loading Redditor comment archive....'
with open (os.path.join (bp_pickles, 'prolific_ask_reddit_archive.pkl')) as f:
    dict_reddit = pickle.load (f)

print 'Loading TFIDF Vectorizer...'
with open (os.path.join (bp_pickles, 'tfidf_obj.pkl')) as f:
    TFIDF_obj = pickle.load (f) 
    
print 'Loading Truncated SVD Transformer....'
with open (os.path.join (bp_pickles, 'TruncSVD_200_comp.pkl')) as f:
    SVD_obj = pickle.load (f) 
    
print 'Loading SVC....'   
with open (os.path.join (bp_pickles, 'app_SVC.pkl')) as f:
    classifier_obj = pickle.load (f) 




# This function designates a page
#@app.route ('/interactive/')
@app.route ('/')
def interactive():
	print '\n###### CALLBACK INTERACTIVE ########\n'	
	try:
		return render_template ('interactive.html')
	except Exception,e :
		return (str(e))

# This function designates a process to run in the background
@app.route('/background_process')
def background_process():
	print '\n######### CALLBACK BACKGROUND_PROCESS ##########\n'
	#try:
	q = request.args.get('question', 0, type=str)
	#DGB

	print 'TEXT = ' 
	print q
	
	if len (q.strip()) > 1:
		topic, redditor, best_comment, n_comments, url = process_user_question (q, df_reddit, \
                                dict_reddit, TFIDF_obj, SVD_obj, classifier_obj, \
                                verbose = False)
	else:
		topic = ''
		redditor = ''
		best_comment = ''
		n_comments = ''  # Normally n_comments is  a numeric value, but passing as empty string to clear field in form
        url = ''

	return jsonify (topic=topic, redditor=redditor, n_comments=n_comments,\
                    best_comment=best_comment, url=url)

	#except Exception as e:
		#print 'ERROR'
		#return str(e)

if __name__ == '__main__':
    app.run()

# jsonify -- import from flask

# Removed from TOP of interactive.html... the app did not run when it was present, compaining that 'header.html' was not found....

# (was present in the interactive.html code from the tutorial website)
# {% extends "header.html" %}

# If 

#{% block body %}
#{% endblock %}


