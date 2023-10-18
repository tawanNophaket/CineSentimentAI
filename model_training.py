import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

def train_model():

	#โหลดข้อมูลจาก 'my_data.csv'
    data = pd.read_csv('my_data.csv', encoding='utf-8')

	#ทำความสะอาดข้อมูล
    data.dropna(inplace=True)  

	#เเบ่งข้อมูลเป็น feature เเละ class
    X = data['review']
    y = data['sentiment']

	#แบ่งข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	#ประมวณผลชุดข้อมูลเเละสร้างโมเดล ด้วย pipeline
	#make_pipeline คือประมวณผลที่มีลำดับขั้นตอนการทำงานหลายลำดับเข้าด้วยกัน ในที่นี้คือ TfidfVectorizer() ซึ่งจะทำการประมวณผลชุดข้อมูล เเละ LogisticRegression() ซึ่งจะทำการสร้างโมเดล
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())

	#ฝึกสอนโมเดล
    model.fit(X_train, y_train)
    
    #ประเมินผลลัพธ์ด้วย accuracy
    test_accuracy = model.score(X_test, y_test)
    print(f"Accuracy บน testing set: {test_accuracy*100:.2f}%")

    #ประเมินประสิทธิภาพโมเดลบน testing set
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['negative', 'positive'])
    print(report)

    #ประเมินด้วย Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)  # ใช้ 5-fold cross validation
    print(f"Average accuracy from cross-validation: {cv_scores.mean()*100:.2f}%")

    #บันทึกโมเดล
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

train_model()