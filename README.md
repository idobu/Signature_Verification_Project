# Signature_Verification_Project Overview
A Client-Server project using React-Native, NodeJS (mostly TS) and Python to make Registration of a user using written signature and then verify the logged-in user.

## Introduction  
This project introduces a signature authentication system that leverages a React-Native application, a NodeJS server, and multiple Python services to create a robust signature verification process. The system is designed to learn from the user's unique signature characteristics during registration and use this information to verify identity during login attempts.

# System Workflow
## Registration
During the registration phase, the user is required to provide 10 instances of their signature through the React-Native application. These samples are sent to the server, which coordinates the learning process across several Python-based services.

## Login
When logging in, the user provides a signature which is evaluated by the system. If the signature is not recognized, a message indicating the failure stage is returned. A successful match results in a success message.

# Authentication Steps
## Step 1: Raising Hands Verification
The system analyzes the signature to verify the number of times the user raises their hand. This step ensures that the hand-lifting pattern during login matches the pattern observed during training.

## Step 2: Direction Check
The system assesses the directionality of the signature strokes. It distinguishes between signatures based on the direction in which lines are drawn (e.g., top-to-bottom vs. bottom-to-top), adding a layer of complexity to the authentication process.

![image](https://github.com/idobu/Signature_Verification_Project/assets/112335952/42d3d4c6-6765-4b9f-831c-71329ed4cefe)

## Step 3: SVM Model Learning
The system employs a Support Vector Machine (SVM) model to learn the user's signature style from the 10 samples provided. It creates a template that is used to check if at least 75% of the login signature fits within the learned template. Figure_1

![image](https://github.com/idobu/Signature_Verification_Project/assets/112335952/3c3377a9-e991-49c4-836f-f2c1fe3dba95)

## Step 4: DC-GAN Model Learning
Using CUDA for accelerated processing, the system trains a Deep Convolutional Generative Adversarial Network (DC-GAN) model to generate signature forgeries. Once the model can successfully forge signatures, it uses this ability to verify the authenticity of login signatures against potential forgeries.

![image](https://github.com/idobu/Signature_Verification_Project/assets/112335952/fc0cfaee-ca96-4cd2-8073-28ac34d9787d)


![image](https://github.com/idobu/Signature_Verification_Project/assets/112335952/60ac4a5e-0943-4a0c-9682-aceb3199ba0f)


# Accuracy Ranges
The system's accuracy in identifying signatures varies, with a range of 75% to 93%. This variation is due to the unique nature of each signature and the different learning outcomes of the models.

# Running the Project
## Client-Side Setup
Utilize a temporary server such as NGROK and set the server link using the NGROK_LINK variable in App.tsx.
Activate the client using the command expo-cli start --tunnel. Scan the provided code with the Expo application.
## Server-Side Setup
Run the NodeJS server and obtain the server link to update the NGROK_LINK in the client application. Python Services
The Python services will run smoothly if the path in the Constant file is correctly set.
* It is recommended to install and use CUDA
