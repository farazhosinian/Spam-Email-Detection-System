import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# دانلود stopwords
nltk.download('stopwords')

# 1. بارگذاری داده‌ها
df = pd.read_csv('SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'message'])

# نمایش چند خط اول داده‌ها
print(df.head())

# 2. پردازش متن (حذف نشانه‌های نگارشی، کلمات بی‌معنی، ریشه‌یابی کلمات)
def preprocess_text(text):
    # تبدیل به حروف کوچک
    text = text.lower()
    # حذف نشانه‌های نگارشی
    text = ''.join([char for char in text if char not in string.punctuation])
    # حذف کلمات بی‌معنی
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # ریشه‌یابی کلمات
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

# اعمال پردازش روی تمام پیام‌ها
df['processed_message'] = df['message'].apply(preprocess_text)

# 3. تقسیم داده‌ها به ویژگی‌ها و برچسب‌ها
X = df['processed_message']
y = df['label'].apply(lambda x: 1 if x == 'spam' else 0)  # تبدیل برچسب‌ها به 1 و 0

# تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. نمایش داده‌ها به صورت برداری (Vectorization)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. ساخت مدل Naive Bayes
model = MultinomialNB()

# آموزش مدل
model.fit(X_train_tfidf, y_train)

# پیش‌بینی روی داده‌های تست
y_pred = model.predict(X_test_tfidf)

# ارزیابی مدل
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))

# 6. استفاده از مدل برای پیش‌بینی ایمیل‌های جدید
def predict_spam(text):
    text = preprocess_text(text)  # پردازش متن جدید
    text_tfidf = vectorizer.transform([text])  # تبدیل به ویژگی‌های عددی
    prediction = model.predict(text_tfidf)
    return 'Spam' if prediction == 1 else 'Ham'

# تست پیش‌بینی
new_email = "Free money! Win a prize now!"
print(predict_spam(new_email))  # خروجی: 'Spam'
