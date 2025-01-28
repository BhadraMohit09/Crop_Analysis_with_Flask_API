> ### Crop Yield Prediction Web Application🌾

This project is a **web application** designed to predict crop yield based on various agricultural parameters. The app allows users to input information like farm area, fertilizer used, pesticide used, water usage, crop type, irrigation type, soil type, and season to predict the expected yield. The application uses a pre-trained **machine learning model** that processes the inputs and returns a predicted yield in tons.

### Features
- **Interactive User Interface**: The web app provides a clean, intuitive form for inputting data.
- **Prediction Display**: Once the form is submitted, the predicted yield is displayed directly on the webpage.
- **Responsive Design**: The app is built using modern frontend technologies (Tailwind CSS and Bootstrap), ensuring a smooth user experience on both mobile and desktop devices.
- **Fast Prediction**: The machine learning model makes predictions instantly after submission.
- **Open Source**: The project is open-source and can be freely modified and used for educational and personal purposes.

### Technologies Used
- **Flask**: Lightweight Python web framework to build the server-side logic and handle HTTP requests.
- **Joblib**: Used for loading the pre-trained machine learning model and making predictions.
- **Pandas**: Used to format and preprocess the input data into a DataFrame that is compatible with the model.
- **Tailwind CSS**: Utility-first CSS framework used to style the application with a clean and modern design.
- **Bootstrap**: A CSS framework used for responsive design to make sure the web app works on mobile devices.
- **HTML, CSS, and JavaScript**: For building the front-end interface, user interactions, and enhancing the overall user experience.

### Installation

#### Prerequisites
Before running the app, make sure you have the following installed:
- **Python 3.x** (for running the Flask server)
- **pip** (Python package manager)
- **Virtual Environment** (optional but recommended for isolation)

#### Step 1: Clone the repository
To get the source code, clone the repository from GitHub:
```bash
git clone https://github.com/BhadraMohit09/Crop_Analysis_with_Flask_API.git
cd Crop_Analysis_with_Flask_API
