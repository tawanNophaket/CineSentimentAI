# Flask เป็น micro web framework ที่เขียนด้วยภาษา Python มันเป็น "micro" framework ที่หมายความว่าไม่ต้องใช้เครื่องมือหรือไลบรารีเฉพาะเจาะจงที่ต้องใช้กับการพัฒนาเว็บแอปพลิเคชัน แต่ให้ความอิสระในการเลือกเครื่องมือและการปรับแต่งตามที่นักพัฒนาเห็นสมควร
# Flask รองรับการทำงานกับ Jinja2 templating, ซึ่งทำให้การสร้างหน้า HTML
# Flask สามารถใช้งาน Python กับ template อื่นๆ เช่น HTML ได้สะดวกสบาย
# route เปรียบเสมือนการเรียกใช้งาน website. ฟังก์ชันที่อยู่ในเว็บไซต์จะทำงานก็ต่อเมื่อมีการร้องขอ route ซึ่งเป็น URL ของเว็บไซต์

from flask import Flask, render_template, request, jsonify
import pandas as pd
import csv
from model_training import train_model
import pickle

#สร้าง object flask ในชื่อตัวเเปรว่า app ทำหน้าที่เป็นกลางในการจัดการกับการตั้งค่า, URL routing, การรับคำขอจาก client และการส่งคำตอบกลับไปยัง client และอื่นๆ
app = Flask(__name__)

#โหลดโมเดลที่บันทึกไว้
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

#กำหนด route หลักที่จะแสดงหน้า index.html โดยใช้ render_template ซึ่งเป็นฟังก์ชันใน Flask ที่ใช้เพื่อแสดงผลหน้า HTML
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    prediction = model.predict([review])[0]
    return jsonify({'prediction': prediction})

# เมื่อผู้ใช้ป้อนข้อความรีวิวและส่งไปยังเซิร์ฟเวอร์, มันจะไปยัง route '/predict' ที่มีวิธีการ POST (เป็นการส่งข้อมูลกลับมาที่เซิฟเวอโดยไม่ยุ่งกับ url ของเว็บไซต์. ข้อความรีวิวจะถูกส่งไปยังโมเดล, และโมเดลจะทำนายว่ามีทัศนคติเชิงบวกหรือเชิงลบ. ผลลัพธ์จะถูกส่งกลับเป็น JSON ไปยังผู้ใช้.
# request.form ใช้เพื่อเข้าถึงข้อมูลที่ส่งมาจาก front-end ของเว็บแอปพลิเคชัน
# jsonify คือฟังก์ชันที่ใช้สำหรับส่งข้อมูลกลับไปยัง front-end ในรูปแบบ JSON, ซึ่งสามารถใช้งานได้ง่ายกับ JavaScript ที่ใช้งานอยู่ที่ client-side.

@app.route('/feedback', methods=['POST'])
def feedback():
    review = request.form['review']
    predicted_sentiment = request.form['predicted_sentiment']
    correctness = request.form['correctness']

    print(f"Review: {review}, Predicted: {predicted_sentiment}, Correctness: {correctness}")
    
    if correctness == "โมเดลทำนายไม่ถูกต้อง":
        true_sentiment = "positive" if predicted_sentiment == "negative" else "negative"

        with open('my_data.csv', 'a', encoding="utf-8") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
            writer.writerow([review, true_sentiment])
        
        return jsonify(success=True, message="บันทึก feedback สำเร็จ!")

    return jsonify(success=True, message="ไม่มีการบันทึก feedback")

if __name__ == '__main__':
    app.run(debug=True)