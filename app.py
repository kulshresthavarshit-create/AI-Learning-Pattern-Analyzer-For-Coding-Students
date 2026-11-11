from flask import Flask, render_template, request

import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
import os
matplotlib.use('Agg')
app = Flask('__name__')

# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv("processed_data.csv")

# ==========================================
# LOAD MODEL
# ==========================================

rf_model = joblib.load("learning_pattern_model.pkl")
lr_model= joblib.load("success_rate_model.pkl")


# ==========================================
# TOPIC NAME CONVERSION
# ==========================================

topic_map = {
    0: "Arrays",
    1: "Strings",
    2: "Recursion",
    3: "Dynamic Programming",
    4: "Graphs"
}

difficulty_map = {
    0: "Easy",
    1: "Medium",
    2: "Hard"
}

# Create static folder automatically
if not os.path.exists('static'):
    os.makedirs('static')

# ====================================
# Attempts Graph
# ====================================
plt.figure(figsize=(5,4))
data['attempts'].value_counts().plot(kind='bar')
plt.title("Attempts Graph")
plt.xlabel("Attempts")
plt.ylabel("Count")
plt.savefig('static/attempts_chart.png')
plt.close()

# ====================================
# Weak Topics Graph
# ====================================
weak_topics = data.groupby('topic')['success_rate'].mean()

plt.figure(figsize=(6,4))
weak_topics.plot(kind='bar')
plt.title("Weak Topics Graph")
plt.xlabel("Topics")
plt.ylabel("Average Success Rate")
plt.savefig('static/weak_topics_chart.png')
plt.close()

# ====================================
# Success Rate Graph
# ====================================
plt.figure(figsize=(5,4))
data['success_rate'].hist()
plt.title("Success Rate Graph")
plt.xlabel("Success Rate")
plt.ylabel("Students")
plt.savefig('static/success_rate_chart.png')
plt.close()

# ====================================
# Learning Pattern Graph
# ====================================
if 'learning_pattern' in data.columns:

    plt.figure(figsize=(5,4))
    data['learning_pattern'].value_counts().plot(kind='bar')
    plt.title("Learning Pattern Graph")
    plt.xlabel("Learning Pattern")
    plt.ylabel("Count")
    plt.savefig('static/learning_pattern_chart.png')
    plt.close()
# ==========================================
# ROUTE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():
    student_data=pd.DataFrame()
    prediction = None
    learning_pattern = None
    success_rate = None
    recommendation = None
    weak_topics=[]
    weak_topic_score=0
    topic_recommendation=""
    rank=None
    tables=data.head(10).to_html(
       classes='table table-bordered table-striped',
       index=False
    )

    if request.method == "POST":
        student_name = request.form.get("student_name","")

        student_data = data[
           data['student_name'].str.lower() == student_name.lower()
] 
    # Calculate rank based on success rate

    data['rank'] = data['success_rate'].rank(
      ascending=False,
      method='dense'
)

    rank = "Not Found"

    if not student_data.empty:

       rank =int(student_data.iloc[0]['rank'])


       attempts = float(student_data.iloc[0]['attempts'])

       time_taken = float(student_data.iloc[0]['time_taken'])

       errors = float(student_data.iloc[0]['errors_count'])

       success_rate = float(student_data.iloc[0]['success_rate'])

    else:

     attempts = float(request.form.get("attempts",0))

     time_taken = float(request.form.get("time_taken",0))

     errors = float(request.form.get("errors",0))

     print("Attempts=",attempts)
     print("Time Taken=",time_taken)
     print("Errors=",errors)
     
     

     rf_features = [[0,attempts,1, time_taken, errors,0.75]]
     lr_features=[[attempts,time_taken,errors]]

     prediction = rf_model.predict(rf_features)[0]
     success_rate=max(0,min(1,(attempts*10-errors*5)/(time_taken + 50)))
     success_rate=round(success_rate,2)
    
     print("Prediction:",prediction)
     print("Success Rate:",success_rate)
        

        
           
    filtered_students = data[
            (data['attempts'] >= attempts - 2) &
            (data['attempts'] <= attempts + 2)
        ]

    topic_performance = filtered_students.groupby(
          'topic'
        )['success_rate'].mean()

    weak_topics = topic_performance.nsmallest(3)

    recommendations = []

    for topic in weak_topics.index:

             recommendations.append(
            f"Practice more in {topic}"
        )

    if success_rate > 1:
              success_rate = 1.0

    if success_rate < 0:
              success_rate = 0.0
        


    if success_rate >= 0.75:
              learning_pattern = "Fast Learner"

    elif success_rate >= 0.40:
                learning_pattern = "Average Learner"

    else:
             learning_pattern = "Needs Improvement"
    print("Success Rate=",success_rate)
    print("Learning Pattern=",learning_pattern)         
        
    if success_rate >= 0.75:
              recommendation = "Excellent performance. Keep practicing advanced problems."

    elif success_rate >= 0.40:
                recommendation = "Good performance. Focus more on weak topics."

    else:
                recommendation = "Needs improvement. Practice basics and reduce errors."
    weak_topics=[]

# FAST LEARNER
    if learning_pattern == "Fast Learner" and success_rate >= 0.8:

             weak_topics.append({
            "topic": "No Major Weak Topic",
            "score": 0.95,
            "recommendation": "Excellent coding performance."
    })

# AVERAGE LEARNER
    elif learning_pattern == "Average Learner" and success_rate >= 0.4:

             weak_topics.append({
              "topic": "Graphs",
              "score": 0.45,
              "recommendation": "Practice graph traversal algorithms."
    })

# SLOW LEARNER
    elif learning_pattern == "Slow Learner" or success_rate < 0.4:

        weak_topics.append({
              "topic": "Dynamic Programming",
              "score": 0.35,
              "recommendation": "Practice memorization and tabulation."
})


    return render_template(
    "index.html",
    learning_pattern=learning_pattern,
    tables=tables,
    success_rate=success_rate,
    recommendation=recommendation,
    prediction=prediction,
    weak_topics=weak_topics,
    weak_topic_score=weak_topic_score,
    topic_recommendation=topic_recommendation,
    rank=rank
)


# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)