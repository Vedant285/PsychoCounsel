# PsycoCouncil: The Human Psychological Counselor

## Overview
PsycoCouncil is an advanced psychological counseling platform integrating cutting-edge technologies such as Computer Vision and Signal Processing for precise emotion recognition and heart rate analysis. Through the utilization of these sophisticated methods, PsycoCouncil aims to deliver comprehensive and accurate insights to enhance the counseling experience for individuals seeking psychological support.

## Cite
  ```
    TY  - CONF
    TI  - Human Psychological Counselling Framework using Computer Vision
    T2  - 2024 11th International Conference on Computing for Sustainable Global Development (INDIACom)
    SP  - 745
    EP  - 750
    AU  - S. Dutta
    AU  - V. Shukla
    AU  - Y. Pant
    AU  - V. Tripathi
    PY  - 2024
    DO  - 10.23919/INDIACom61295.2024.10498255
    Y1  - 28 Feb.-1 March 2024 
  ```

## Project Link
  [Webapp](https://council.streamlit.app/) 
   *Reboot if faced with memory issues*

## Research Paper
  [Human Psychological Counselling Framework using Computer Vision
](https://ieeexplore.ieee.org/document/10498255)


## Usage
1. Start Monitor, and get live data of emotion recognition.
2. Stop Monitor, and get emotion tracking and heart metric reports.
3. Go to Counsel Tab, Select informations that you want to give to the chatbot.
4. Fill-up required values, press counsel button, and recieve feedback from AI.

## Technology used
- Streamlit for web application
- DeepFace Facial Atrribute Analysis
- GPT-4 for LLM integration (Clarifai Model wrapped in LangChain)
- rPhotoplethysmography (rPPG) for heart rate analysis
- AssemblyAI speech recognition and GTTS



## Running the App Locally

1. **Clone the Repository**:  
   ```
   https://github.com/Vedant285/PsychoCounsel.git
   ```

2. **Install Virtual Environment**:  
   ```
   python -m venv venv
   ```
   Activate the virtual environment:
   - On Windows: `venv\Scripts\Activate`
   - On macOS and Linux: `source venv/bin/activate`

3. **Install Required Packages**:  
   ```
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**:  
   ```
   streamlit run app.py
   ```

5. **Open the App**:  
   The app should now be running. Open your web browser and go to `http://localhost:8501/` to interact with the app.

## Applications

1. **Mental Health Assessment**: Useful for psychologists and therapists.
2. **Fitness and Wellness**: Monitor cardiovascular health.
3. **Telehealth Services**: Provide real-time data to healthcare providers.

## User Trust & Ethics
- Immediate deletion after processing speech transcription.
