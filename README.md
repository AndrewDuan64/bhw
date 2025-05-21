
# install

conda create -n bhw-env python=3.10 -y
conda activate bhw-env

git clone git@github.com:AndrewDuan64/bhw.git

pip install -r requirements.txt

python -m spacy download en_core_web_sm

streamlit run app.py

