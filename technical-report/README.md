# Technical Report

Back to the main report: [README.md](../README.md)
Blog draft: [blog/index.html](../blog/index.html)

## What I Built
For this project, I developed a Streamlit-based web application that predicts the risk of heart disease using machine learning.

The application allows users to enter medical information and receive a prediction based on a trained model. To ensure accurate results, the application uses the same preprocessing steps that were applied during model training. This helps maintain consistency between the training process and real-time predictions.

The project combines data preprocessing, model training, model deployment, and user interface design into a single application.

## Input Features
The prediction model uses the following medical attributes:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol Level
- Fasting Blood Sugar
- Resting ECG Results
- Maximum Heart Rate Achieved
- Exercise-Induced Angina
- Oldpeak (ST Depression)
- Slope of Peak Exercise ST Segment
- Number of Major Vessels
- Thalassemia

These features are commonly used in heart disease assessment and help the model identify potential risk patterns.

## Development Process
The project followed a standard machine learning workflow:

1. Data collection and exploration
2. Data cleaning and preprocessing
3. Feature encoding and transformation
4. Feature scaling
5. Model training and evaluation
6. Model selection
7. Application development using Streamlit
8. Testing and validation

Although the overall workflow was straightforward, each stage required careful implementation to ensure the final application worked correctly.

## Challenges Faced
Several challenges were encountered during development.

One of the major challenges was ensuring that the feature set used during prediction exactly matched the feature set used during training. Even small differences could affect the model's performance and lead to incorrect predictions.

Other challenges included:

- Organizing notebook code into a structured application
- Managing preprocessing pipelines and saved model files
- Ensuring correct feature encoding and scaling
- Debugging prediction-related issues
- Designing a clear and user-friendly interface

These challenges highlighted the importance of maintaining consistency throughout the machine learning pipeline.

## Key Learnings
Through this project, I learned several important lessons:

- Data preprocessing is just as important as model selection.
- Consistency between training and prediction pipelines is critical.
- Machine learning models should be tested thoroughly before deployment.
- A simple and clear user interface improves the overall usability of an application.
- Proper project organization makes debugging and maintenance easier.

I also gained practical experience in integrating machine learning models into real-world applications using Streamlit.

## Project Outcome

The final application successfully predicts heart disease risk based on user-provided medical information.

The project demonstrated the complete machine learning workflow, from data preparation and model training to deployment and user interaction. It also helped me understand the challenges involved in transforming a machine learning experiment into a usable software application.


## Conclusion

This project was an important learning experience that strengthened my understanding of machine learning and application development.

It showed me that building a successful project involves much more than training a model. Data preprocessing, testing, deployment, and user experience all play a significant role in creating a reliable solution.

Overall, the project helped me gain practical skills that I can apply to future AI and machine learning projects.
